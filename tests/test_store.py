"""Tests for src/vectorstore/store.py against the real local pgvector-backed chunks table."""

import os
import sys
from pathlib import Path

import psycopg
import pytest
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ledger"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ingestion"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "embedding"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "vectorstore"))

from ledger import content_hash  # noqa: E402
from embed import EmbeddedChunk  # noqa: E402
from store import chunk_id, load_chunks  # noqa: E402

DSN = os.environ.get("TEST_DATABASE_DSN", "dbname=self_healing")


def make_chunk(text: str, **metadata) -> EmbeddedChunk:
    # cheap fake embedding: real dimensionality, not a real model's output
    return EmbeddedChunk(text=text, embedding=[0.01] * 384, metadata=metadata)


@pytest.fixture()
def conn():
    c = psycopg.connect(DSN)
    register_vector(c)
    c.execute("TRUNCATE TABLE chunks")
    c.commit()
    yield c
    c.close()


def test_chunk_id_is_content_hash_of_text():
    chunk = make_chunk("hello world")
    assert chunk_id(chunk) == content_hash("hello world")


def test_load_inserts_new_chunks(conn):
    chunks = [make_chunk("first review"), make_chunk("second review")]
    inserted = load_chunks(conn, chunks)

    assert inserted == 2
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        assert cur.fetchone()[0] == 2


def test_load_stores_text_embedding_and_metadata(conn):
    chunk = make_chunk("a review with metadata", asin="B123", rating=5.0)
    load_chunks(conn, [chunk])

    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute("SELECT id, text, embedding, metadata FROM chunks")
        row = cur.fetchone()

    assert row["id"] == content_hash("a review with metadata")
    assert row["text"] == "a review with metadata"
    assert row["metadata"] == {"asin": "B123", "rating": 5.0}
    assert len(row["embedding"].to_list()) == 384


def test_rerun_with_same_content_is_a_no_op(conn):
    chunk = make_chunk("duplicate content", asin="ORIGINAL")
    first = load_chunks(conn, [chunk])
    assert first == 1

    # same text (and thus same id) but different metadata, simulating a rerun where
    # the embedding pipeline recomputed the same chunk from a fresh pass
    rerun_chunk = make_chunk("duplicate content", asin="SHOULD_NOT_OVERWRITE")
    second = load_chunks(conn, [rerun_chunk])
    assert second == 0  # skipped by ON CONFLICT DO NOTHING

    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute("SELECT count(*) AS n FROM chunks")
        assert cur.fetchone()["n"] == 1
        cur.execute("SELECT metadata FROM chunks WHERE id = %s", (content_hash("duplicate content"),))
        # first write wins; the "rerun" did not overwrite it
        assert cur.fetchone()["metadata"] == {"asin": "ORIGINAL"}


def test_duplicate_within_same_batch_only_inserted_once(conn):
    chunks = [make_chunk("repeated in one call"), make_chunk("repeated in one call")]
    inserted = load_chunks(conn, chunks)

    assert inserted == 1
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        assert cur.fetchone()[0] == 1


def test_batch_size_does_not_affect_correctness(conn):
    chunks = [make_chunk(f"review {i}") for i in range(10)]
    inserted = load_chunks(conn, chunks, batch_size=3)

    assert inserted == 10
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        assert cur.fetchone()[0] == 10


def test_hnsw_similarity_query_finds_the_closest_chunk(conn):
    near = EmbeddedChunk(text="near vector", embedding=[0.1] * 384, metadata={})
    far = EmbeddedChunk(text="far vector", embedding=[-0.1] * 384, metadata={})
    load_chunks(conn, [near, far])

    query = [0.1] * 384
    with conn.cursor() as cur:
        cur.execute(
            "SELECT text FROM chunks ORDER BY embedding <=> %s::vector LIMIT 1",
            (query,),
        )
        assert cur.fetchone()[0] == "near vector"
