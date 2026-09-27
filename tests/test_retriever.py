"""Tests for src/vectorstore/retriever.py, including a retrieval-quality check
against tests/gold_set.json -- a small hand-built set of query -> relevant-chunk
pairs picked from real, currently-loaded review data (see that file's _comment for
why it's tied to a specific loaded sample).

Real Postgres + pgvector + the real embedding model and cross-encoder throughout;
no mocks. If chunks is empty or has been repopulated from a different sample, the
gold-set tests are skipped rather than failing on stale expectations.
"""

import json
import os
import sys
from pathlib import Path

import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "embedding"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "vectorstore"))

from retriever import get_cross_encoder, rerank, retrieve, vector_search  # noqa: E402

DSN = os.environ.get("TEST_DATABASE_DSN", "dbname=self_healing")
GOLD_SET_PATH = Path(__file__).parent / "gold_set.json"


@pytest.fixture()
def conn():
    c = psycopg.connect(DSN)
    yield c
    c.close()


@pytest.fixture()
def gold_pairs(conn):
    pairs = json.loads(GOLD_SET_PATH.read_text())["pairs"]
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM chunks WHERE id = ANY(%s)", ([p["chunk_id"] for p in pairs],))
        present = {row[0] for row in cur.fetchall()}
    missing = [p for p in pairs if p["chunk_id"] not in present]
    if missing:
        pytest.skip(
            f"{len(missing)}/{len(pairs)} gold-set chunk ids are not in the current "
            "chunks table -- it's been repopulated from a different sample; "
            "regenerate tests/gold_set.json against current data"
        )
    return pairs


# --- basic behavior, independent of the gold set ---------------------------------


def test_vector_search_returns_nearest_first_by_distance(conn):
    results = vector_search(conn, [0.01] * 384, top_k=5)
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)


def test_retrieve_respects_top_k(conn):
    results = retrieve(conn, "anything at all", top_k=2)
    assert len(results) <= 2


def test_retrieve_without_rerank_has_no_rerank_score(conn):
    results = retrieve(conn, "anything at all", top_k=3)
    assert all("rerank_score" not in r for r in results)


def test_retrieve_with_rerank_adds_rerank_score(conn):
    results = retrieve(conn, "anything at all", top_k=3, rerank_enabled=True)
    assert all("rerank_score" in r for r in results)


def test_reranked_results_are_sorted_by_rerank_score_descending(conn):
    results = retrieve(conn, "a camera for watching pets", top_k=5, rerank_enabled=True)
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# --- retrieval quality against the hand-built gold set ----------------------------


def test_vector_search_finds_the_relevant_chunk_in_top_k(conn, gold_pairs):
    top_k = 5
    hits = 0
    misses = []
    for pair in gold_pairs:
        results = retrieve(conn, pair["query"], top_k=top_k)
        found_ids = [r["id"] for r in results]
        if pair["chunk_id"] in found_ids:
            hits += 1
        else:
            misses.append(pair["query"])

    recall_at_k = hits / len(gold_pairs)
    assert recall_at_k >= 0.75, (
        f"recall@{top_k} was {recall_at_k:.0%} ({hits}/{len(gold_pairs)}); missed: {misses}"
    )


def test_rerank_puts_the_relevant_chunk_first_when_found_in_candidates(conn, gold_pairs):
    # a stronger claim than plain recall@k: when the cross-encoder sees the right
    # chunk among its candidates, does it correctly rank it #1?
    top1_hits = 0
    for pair in gold_pairs:
        results = retrieve(conn, pair["query"], top_k=5, rerank_enabled=True, candidate_k=20)
        if results and results[0]["id"] == pair["chunk_id"]:
            top1_hits += 1

    top1_rate = top1_hits / len(gold_pairs)
    assert top1_rate >= 0.75, f"reranked top-1 accuracy was only {top1_rate:.0%}"


# --- get_cross_encoder / rerank, tested directly rather than only through retrieve ---


def test_get_cross_encoder_returns_the_same_singleton_instance():
    assert get_cross_encoder() is get_cross_encoder()


def test_rerank_of_empty_candidates_returns_empty_list():
    assert rerank("any query", [], top_k=5) == []


def test_rerank_orders_a_clear_match_above_an_unrelated_one():
    candidates = [
        {"id": "unrelated", "text": "This is a birthday card for my grandmother."},
        {"id": "relevant", "text": "This indoor security camera has great night vision."},
    ]
    results = rerank("security camera with good night vision", candidates, top_k=2)

    assert [r["id"] for r in results] == ["relevant", "unrelated"]
    assert results[0]["rerank_score"] > results[1]["rerank_score"]


def test_rerank_respects_top_k_even_with_more_candidates():
    candidates = [
        {"id": "a", "text": "first candidate text"},
        {"id": "b", "text": "second candidate text"},
        {"id": "c", "text": "third candidate text"},
    ]
    results = rerank("candidate", candidates, top_k=1)
    assert len(results) == 1
