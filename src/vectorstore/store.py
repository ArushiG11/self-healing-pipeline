"""Load embedded chunks into the pgvector-backed `chunks` table.

A row's id is the sha256 content-hash of its text (the same content_hash() used
throughout the pipeline), not a surrogate key. INSERT ... ON CONFLICT (id) DO NOTHING
makes loading idempotent: re-running embed+load over the same text just no-ops on
the repeat instead of creating a duplicate row.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import psycopg
from psycopg.types.json import Jsonb
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ledger"))
from ledger import content_hash  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embedding"))
from embed import EmbeddedChunk  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inject"))
import faults  # noqa: E402

STAGE = "vectorstore"
DEFAULT_BATCH_SIZE = 100


def chunk_id(chunk: EmbeddedChunk) -> str:
    return content_hash(chunk.text)


def load_chunks(
    conn: psycopg.Connection,
    chunks: Iterable[EmbeddedChunk],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Insert chunks, skipping any whose content-hash id already exists.

    Commits every `batch_size` rows rather than per-row, but conflict resolution is
    still per-row (ON CONFLICT DO NOTHING), so a rerun over already-loaded chunks is
    safe regardless of batch boundaries. Returns the number of rows actually
    inserted (rows skipped by the conflict, whether pre-existing in the table or
    duplicated within `chunks` itself, are not counted).

    Unlike ingest/embed, this stage has no ledger tracking (see caveat when this
    module was introduced) — a raised error propagates to the caller. It's still
    left in a safe state: everything committed before the error stays committed,
    and the connection's own pending (uncommitted) work is rolled back rather than
    left dangling in an aborted transaction.
    """
    register_vector(conn)  # so `embedding` reads back as a list/array, not a raw string
    inserted = 0
    pending = 0
    with conn.cursor() as cur:
        for chunk in chunks:
            try:
                faults.trigger(STAGE)
                cur.execute(
                    """
                    INSERT INTO chunks (id, text, embedding, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    RETURNING id
                    """,
                    (chunk_id(chunk), chunk.text, chunk.embedding, Jsonb(chunk.metadata)),
                )
            except Exception:
                conn.rollback()
                raise
            if cur.fetchone() is not None:
                inserted += 1
            pending += 1
            if pending >= batch_size:
                conn.commit()
                pending = 0
    conn.commit()
    return inserted
