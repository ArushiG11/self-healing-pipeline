"""Batch embedding of cleaned records with all-MiniLM-L6-v2, ledger-tracked per batch.

Unlike ingestion (tracked per record), the unit of work here is a *batch*: one
(stage="embed", input_hash) row covers `batch_size` records embedded together in a
single model.encode() call, since that's the actual unit that succeeds or fails
together (a batch either all embeds or the whole batch's model call blew up).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ledger"))
from ledger import InvalidTransition, Ledger, content_hash  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inject"))
import faults  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "observability"))
from telemetry import stage_span  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingestion"))
from reader import ReviewRecord  # noqa: E402

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
STAGE = "embed"
DEFAULT_BATCH_SIZE = 32

_model = None  # lazy singleton: loading the model is the expensive part, do it once


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


@dataclass
class EmbeddedChunk:
    text: str
    embedding: List[float]
    metadata: dict


def _chunked(records: Iterable[ReviewRecord], size: int) -> Iterator[List[ReviewRecord]]:
    it = iter(records)
    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch


def batch_hash(batch: List[ReviewRecord]) -> str:
    """Deterministic identity for a batch, independent of the records' order within it."""
    member_hashes = sorted(content_hash(r.raw) for r in batch)
    return content_hash("\n".join(member_hashes))


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_batches(
    ledger: Ledger,
    records: Iterable[ReviewRecord],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[EmbeddedChunk]:
    """Embed records in batches, tracking each batch through the ledger the same way
    ingestion tracks each record: get_or_create -> skip if is_done -> running ->
    succeeded/failed around the actual work (here, one model.encode() call per batch).

    A batch that fails to embed yields nothing for any of its records — a stuck/failed
    batch is left for the healer to retry (failed -> retrying), not retried here, same
    policy as ingest_records().
    """
    for batch in _chunked(records, batch_size):
        input_hash = batch_hash(batch)
        ledger.get_or_create(STAGE, input_hash, rows_in=len(batch))
        if ledger.is_done(STAGE, input_hash):
            continue

        try:
            ledger.transition(STAGE, input_hash, "running")
        except InvalidTransition:
            continue

        chunks = []
        with stage_span(STAGE, input_hash=input_hash) as span:
            span.set_rows_in(len(batch))
            try:
                faults.trigger(STAGE)
                vectors = embed_texts([r.data["text"] for r in batch])
            except Exception as e:
                ledger.transition(
                    STAGE,
                    input_hash,
                    "failed",
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                span.mark_failed(type(e).__name__)
            else:
                ledger.transition(STAGE, input_hash, "succeeded", rows_out=len(vectors))
                span.set_rows_out(len(vectors))
                for record, vector in zip(batch, vectors):
                    metadata = {k: v for k, v in record.data.items() if k != "text"}
                    chunks.append(
                        EmbeddedChunk(text=record.data["text"], embedding=vector, metadata=metadata)
                    )

        # yield only after the span has closed, same reasoning as ingest_records()
        for chunk in chunks:
            yield chunk
