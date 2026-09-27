"""Query the pgvector-backed chunks table: embed the query with the same model used to
embed the chunks, run a cosine-distance vector search, and optionally narrow the top
candidates with a cross-encoder rerank for better precision.

Vector search (bi-encoder cosine similarity) is cheap and scales to the whole table via
the HNSW index, but scores the query and each chunk independently. A cross-encoder reads
the query and a candidate together in one forward pass, which is more accurate but too
slow to run over the whole table -- so reranking only narrows an already-cheap top-N.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embedding"))
from embed import embed_texts  # noqa: E402

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_MULTIPLIER = 4  # extra candidates to fetch before reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = None


def get_cross_encoder():
    """Lazy singleton, same reasoning as embed.get_model(): load the model once."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def vector_search(conn: psycopg.Connection, query_embedding, top_k: int) -> list[dict]:
    """Nearest neighbors by cosine distance, nearest first."""
    register_vector(conn)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, text, metadata, embedding <=> %s::vector AS distance
            FROM chunks
            ORDER BY distance
            LIMIT %s
            """,
            (query_embedding, top_k),
        )
        return cur.fetchall()


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Re-score candidates with a cross-encoder over (query, text) pairs, return the
    top_k highest-scoring, best first.
    """
    if not candidates:
        return []
    encoder = get_cross_encoder()
    scores = encoder.predict([(query, c["text"]) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_k]]


def retrieve(
    conn: psycopg.Connection,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    rerank_enabled: bool = False,
    candidate_k: Optional[int] = None,
) -> list[dict]:
    """Retrieve the top_k most relevant chunks for `query`.

    Without rerank: embeds the query, returns the top_k nearest chunks by cosine
    distance directly.

    With rerank_enabled=True: casts a wider net via vector search (candidate_k,
    default top_k * 4) then narrows to top_k with the cross-encoder.
    """
    query_embedding = embed_texts([query])[0]
    fetch_k = candidate_k if candidate_k is not None else (
        top_k * DEFAULT_CANDIDATE_MULTIPLIER if rerank_enabled else top_k
    )
    candidates = vector_search(conn, query_embedding, fetch_k)

    if not rerank_enabled:
        return candidates[:top_k]
    return rerank(query, candidates, top_k)
