"""Tests for src/ingestion/ingest.py against a real local Postgres ledger (no mocks)."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ledger"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ingestion"))

import reader as reader_module  # noqa: E402
from ledger import Ledger, content_hash  # noqa: E402
from ingest import STAGE, ingest_records, run_ingest  # noqa: E402
from reader import ReviewRecord  # noqa: E402

DSN = os.environ.get("TEST_DATABASE_DSN", "dbname=self_healing")


class _FakeFile:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return iter(self._lines)

    def __exit__(self, *exc_info):
        return False


class _FakeFileSystem:
    def __init__(self, lines):
        self._lines = lines

    def open(self, path, mode="r", encoding="utf-8"):
        return _FakeFile(self._lines)


def use_fake_lines(monkeypatch, lines):
    monkeypatch.setattr(reader_module, "HfFileSystem", lambda: _FakeFileSystem(lines))


@pytest.fixture()
def ledger():
    led = Ledger(DSN)
    led._conn.execute("TRUNCATE TABLE job_ledger")
    led._conn.commit()
    yield led
    led.close()


def raw_record(line_number: int, raw: str, data: dict) -> ReviewRecord:
    return ReviewRecord(line_number=line_number, raw=raw, data=data)


def test_valid_record_is_ingested_and_marked_succeeded(ledger):
    rec = raw_record(1, "line-1", {"rating": 5.0, "title": "x", "text": "a perfectly good review"})
    out = list(ingest_records(ledger, [rec]))

    assert len(out) == 1
    assert out[0].data["text"] == "a perfectly good review"

    row = ledger._get(STAGE, content_hash("line-1"))
    assert row["status"] == "succeeded"
    assert row["rows_out"] == 1
    assert row["attempt_count"] == 1


def test_filtered_record_marked_succeeded_with_zero_rows_out_and_not_yielded(ledger):
    rec = raw_record(1, "line-2", {"title": "x", "text": "no rating here at all"})  # missing rating
    out = list(ingest_records(ledger, [rec]))

    assert out == []
    row = ledger._get(STAGE, content_hash("line-2"))
    assert row["status"] == "succeeded"
    assert row["rows_out"] == 0


def test_record_that_raises_during_cleaning_is_marked_failed(ledger):
    # text is a list, not a str -> clean_text()'s .strip() call blows up
    rec = raw_record(1, "line-3", {"rating": 4.0, "title": "x", "text": ["not", "a", "string"]})
    out = list(ingest_records(ledger, [rec]))

    assert out == []
    row = ledger._get(STAGE, content_hash("line-3"))
    assert row["status"] == "failed"
    assert row["error_type"] is not None
    assert row["attempt_count"] == 1


def test_already_succeeded_record_is_skipped_without_reprocessing(ledger):
    rec = raw_record(1, "line-4", {"rating": 5.0, "title": "x", "text": "already handled before"})
    list(ingest_records(ledger, [rec]))  # first pass: succeeds

    row_before = ledger._get(STAGE, content_hash("line-4"))
    assert row_before["attempt_count"] == 1

    out_second_pass = list(ingest_records(ledger, [rec]))
    assert out_second_pass == []  # skipped, not re-yielded

    row_after = ledger._get(STAGE, content_hash("line-4"))
    assert row_after["attempt_count"] == 1  # untouched: no new "running" attempt


def test_previously_failed_record_is_left_for_the_healer_not_retried_here(ledger):
    rec = raw_record(1, "line-5", {"rating": 3.0, "title": "x", "text": ["bad", "shape"]})
    list(ingest_records(ledger, [rec]))  # first pass: fails

    row_after_fail = ledger._get(STAGE, content_hash("line-5"))
    assert row_after_fail["status"] == "failed"

    out_second_pass = list(ingest_records(ledger, [rec]))
    assert out_second_pass == []

    row_after_second_pass = ledger._get(STAGE, content_hash("line-5"))
    assert row_after_second_pass["status"] == "failed"
    assert row_after_second_pass["attempt_count"] == 1  # no retry attempted by ingest itself


def test_multiple_records_each_tracked_independently(ledger):
    records = [
        raw_record(1, "multi-1", {"rating": 5.0, "title": "a", "text": "good review number one"}),
        raw_record(2, "multi-2", {"title": "b", "text": "missing rating so filtered out"}),
        raw_record(3, "multi-3", {"rating": 1.0, "title": "c", "text": "too short"}),
        raw_record(4, "multi-4", {"rating": 2.0, "title": "d", "text": "another good review here"}),
    ]
    out = list(ingest_records(ledger, records))

    yielded_texts = {r.data["text"] for r in out}
    assert yielded_texts == {"good review number one", "another good review here"}

    statuses = {
        content_hash(r.raw): ledger._get(STAGE, content_hash(r.raw))["status"] for r in records
    }
    assert all(s == "succeeded" for s in statuses.values())


def test_run_ingest_streams_directly_from_the_reader(ledger, monkeypatch):
    import json

    lines = [
        json.dumps({"rating": 5.0, "title": "a", "text": "a fine review straight off the stream"}),
        json.dumps({"title": "b", "text": "missing rating so this gets filtered"}),
    ]
    use_fake_lines(monkeypatch, [line + "\n" for line in lines])

    out = list(run_ingest(ledger, path="fake"))

    assert [r.data["text"] for r in out] == ["a fine review straight off the stream"]
    with ledger._conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM job_ledger WHERE stage = %s", (STAGE,))
        assert cur.fetchone()["n"] == 2  # both raw lines got tracked, not just the survivor
