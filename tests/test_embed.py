"""Tests for src/embedding/embed.py against a real local Postgres ledger and the real
all-MiniLM-L6-v2 model (no mocking the model; only the induced-failure test stubs
embed_texts, since there's no cheap way to make the real model raise on demand).
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ledger"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ingestion"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "embedding"))

from ledger import Ledger  # noqa: E402
from reader import ReviewRecord  # noqa: E402
import embed as embed_module  # noqa: E402
from embed import EMBEDDING_DIM, STAGE, _chunked, batch_hash, embed_batches, get_model  # noqa: E402

DSN = os.environ.get("TEST_DATABASE_DSN", "dbname=self_healing")


@pytest.fixture()
def ledger():
    led = Ledger(DSN)
    led._conn.execute("TRUNCATE TABLE job_ledger")
    led._conn.commit()
    yield led
    led.close()


def cleaned_record(line_number: int, raw: str, text: str, **extra) -> ReviewRecord:
    data = {"rating": 5.0, "title": "t", "text": text, **extra}
    return ReviewRecord(line_number=line_number, raw=raw, data=data)


def test_batch_hash_is_order_independent():
    a = cleaned_record(1, "raw-a", "text a")
    b = cleaned_record(2, "raw-b", "text b")
    assert batch_hash([a, b]) == batch_hash([b, a])


def test_batch_hash_differs_for_different_members():
    a = cleaned_record(1, "raw-a", "text a")
    b = cleaned_record(2, "raw-b", "text b")
    c = cleaned_record(3, "raw-c", "text c")
    assert batch_hash([a, b]) != batch_hash([a, c])


def test_single_batch_happy_path(ledger):
    records = [cleaned_record(i, f"raw-{i}", f"this is review number {i}") for i in range(3)]
    out = list(embed_batches(ledger, records, batch_size=10))

    assert len(out) == 3
    for chunk in out:
        assert len(chunk.embedding) == EMBEDDING_DIM
        assert "text" not in chunk.metadata  # text lives in chunk.text, not duplicated
        assert chunk.metadata["rating"] == 5.0

    h = batch_hash(records)
    row = ledger._get(STAGE, h)
    assert row["status"] == "succeeded"
    assert row["rows_in"] == 3
    assert row["rows_out"] == 3


def test_multiple_batches_tracked_as_separate_ledger_rows(ledger):
    records = [cleaned_record(i, f"raw-{i}", f"review text {i}") for i in range(5)]
    out = list(embed_batches(ledger, records, batch_size=2))

    assert len(out) == 5  # batches of 2, 2, 1

    with ledger._conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM job_ledger WHERE stage = %s", (STAGE,))
        assert cur.fetchone()["n"] == 3  # three batches -> three ledger rows


def test_already_succeeded_batch_is_skipped_on_rerun(ledger):
    records = [cleaned_record(i, f"raw-{i}", f"repeat me {i}") for i in range(2)]
    first = list(embed_batches(ledger, records, batch_size=10))
    assert len(first) == 2

    h = batch_hash(records)
    row_before = ledger._get(STAGE, h)
    assert row_before["attempt_count"] == 1

    second = list(embed_batches(ledger, records, batch_size=10))
    assert second == []

    row_after = ledger._get(STAGE, h)
    assert row_after["attempt_count"] == 1  # no re-embed attempted


def test_batch_that_fails_to_embed_is_marked_failed_and_yields_nothing(ledger, monkeypatch):
    def boom(texts):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(embed_module, "embed_texts", boom)

    records = [cleaned_record(i, f"raw-fail-{i}", f"doomed text {i}") for i in range(2)]
    out = list(embed_batches(ledger, records, batch_size=10))

    assert out == []
    h = batch_hash(records)
    row = ledger._get(STAGE, h)
    assert row["status"] == "failed"
    assert row["error_type"] == "RuntimeError"
    assert row["error_message"] == "model exploded"


def test_previously_failed_batch_is_left_for_the_healer(ledger, monkeypatch):
    def boom(texts):
        raise RuntimeError("still exploding")

    monkeypatch.setattr(embed_module, "embed_texts", boom)

    records = [cleaned_record(i, f"raw-stuck-{i}", f"stuck text {i}") for i in range(2)]
    list(embed_batches(ledger, records, batch_size=10))

    monkeypatch.undo()  # restore the real embed_texts
    out_second_pass = list(embed_batches(ledger, records, batch_size=10))

    assert out_second_pass == []  # not retried by this stage
    h = batch_hash(records)
    row = ledger._get(STAGE, h)
    assert row["status"] == "failed"
    assert row["attempt_count"] == 1


# --- get_model / _chunked, tested directly rather than only through embed_batches ---


def test_get_model_returns_the_same_singleton_instance():
    assert get_model() is get_model()


def test_chunked_splits_evenly():
    batches = list(_chunked([1, 2, 3, 4], size=2))
    assert batches == [[1, 2], [3, 4]]


def test_chunked_handles_a_remainder():
    batches = list(_chunked([1, 2, 3, 4, 5], size=2))
    assert batches == [[1, 2], [3, 4], [5]]


def test_chunked_of_empty_input_yields_nothing():
    assert list(_chunked([], size=3)) == []


def test_chunked_size_larger_than_input_yields_one_batch():
    assert list(_chunked([1, 2], size=10)) == [[1, 2]]
