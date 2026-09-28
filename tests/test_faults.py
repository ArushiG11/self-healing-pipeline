"""Tests for src/inject/faults.py: the on-command fault-injection switch, plus its
wiring into ingest, embed, and vectorstore load (real Postgres, no mocks).
"""

import os
import sys
from pathlib import Path

import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ledger"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ingestion"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "embedding"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "vectorstore"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "inject"))

import faults  # noqa: E402
from ledger import Ledger, content_hash  # noqa: E402
from reader import ReviewRecord  # noqa: E402
from ingest import STAGE as INGEST_STAGE, ingest_records  # noqa: E402
from embed import STAGE as EMBED_STAGE, EmbeddedChunk, batch_hash, embed_batches  # noqa: E402
from store import STAGE as STORE_STAGE, load_chunks  # noqa: E402

DSN = os.environ.get("TEST_DATABASE_DSN", "dbname=self_healing")


@pytest.fixture(autouse=True)
def _clean_faults():
    faults.disarm()
    yield
    faults.disarm()  # never let a test leak an armed fault into the next one


@pytest.fixture()
def ledger():
    led = Ledger(DSN)
    led._conn.execute("TRUNCATE TABLE job_ledger")
    led._conn.commit()
    yield led
    led.close()


@pytest.fixture()
def conn():
    c = psycopg.connect(DSN)
    c.execute("TRUNCATE TABLE chunks")
    c.commit()
    yield c
    c.close()


def raw_record(n: int, text: str) -> ReviewRecord:
    return ReviewRecord(line_number=n, raw=f"raw-{n}", data={"rating": 5.0, "title": "t", "text": text})


# --- faults.py itself, no stages involved --------------------------------------


def test_trigger_is_a_noop_when_nothing_armed():
    faults.trigger("ingest")  # should not raise


def test_arm_then_trigger_raises_the_right_type():
    faults.arm("ingest", "rate_limit")
    with pytest.raises(faults.RateLimitError):
        faults.trigger("ingest")


def test_trigger_only_fires_for_the_armed_stage():
    faults.arm("embed", "dropped_connection")
    faults.trigger("ingest")  # different stage, no-op
    with pytest.raises(faults.ConnectionDroppedError):
        faults.trigger("embed")


def test_fault_disarms_itself_after_default_one_shot():
    faults.arm("ingest", "malformed_record")
    with pytest.raises(faults.MalformedRecordFault):
        faults.trigger("ingest")
    faults.trigger("ingest")  # second call: already consumed, no-op now
    assert not faults.is_armed("ingest")


def test_times_controls_how_many_triggers_fire():
    faults.arm("ingest", "rate_limit", times=2)
    with pytest.raises(faults.RateLimitError):
        faults.trigger("ingest")
    assert faults.is_armed("ingest")
    with pytest.raises(faults.RateLimitError):
        faults.trigger("ingest")
    assert not faults.is_armed("ingest")


def test_disarm_clears_a_pending_fault():
    faults.arm("ingest", "rate_limit")
    faults.disarm("ingest")
    faults.trigger("ingest")  # no-op now


def test_unknown_fault_kind_rejected():
    with pytest.raises(ValueError):
        faults.arm("ingest", "not_a_real_fault")


# --- wired into ingest_records ---------------------------------------------------


def test_injected_fault_fails_the_ingest_record_via_ledger(ledger):
    faults.arm(INGEST_STAGE, "rate_limit")
    rec = raw_record(1, "a perfectly fine review, long enough to pass")
    out = list(ingest_records(ledger, [rec]))

    assert out == []
    row = ledger._get(INGEST_STAGE, content_hash(rec.raw))
    assert row["status"] == "failed"
    assert row["error_type"] == "RateLimitError"


def test_ingest_recovers_once_fault_is_consumed(ledger):
    # arm for exactly one hit -- the record that trips it fails, is left for the
    # healer (not retried by ingest itself), matching the existing failed-record policy
    faults.arm(INGEST_STAGE, "malformed_record")
    rec1 = raw_record(1, "first review is long enough to pass filtering")
    rec2 = raw_record(2, "second review is also long enough to pass filtering")
    out = list(ingest_records(ledger, [rec1, rec2]))

    assert len(out) == 1
    assert out[0].data["text"] == rec2.data["text"]
    assert ledger._get(INGEST_STAGE, content_hash(rec1.raw))["status"] == "failed"
    assert ledger._get(INGEST_STAGE, content_hash(rec2.raw))["status"] == "succeeded"


# --- wired into embed_batches ----------------------------------------------------


def test_injected_fault_fails_the_embed_batch_via_ledger(ledger):
    faults.arm(EMBED_STAGE, "dropped_connection")
    records = [raw_record(i, f"text number {i}") for i in range(3)]
    out = list(embed_batches(ledger, records, batch_size=10))

    assert out == []
    h = batch_hash(records)
    row = ledger._get(EMBED_STAGE, h)
    assert row["status"] == "failed"
    assert row["error_type"] == "ConnectionDroppedError"


# --- wired into load_chunks (no ledger there, so it must propagate + roll back) --


def test_injected_fault_propagates_from_load_chunks_and_rolls_back(conn):
    good = EmbeddedChunk(
        text="row that should not get stuck uncommitted", embedding=[0.01] * 384, metadata={}
    )
    faults.arm(STORE_STAGE, "dropped_connection")

    with pytest.raises(faults.ConnectionDroppedError):
        load_chunks(conn, [good], batch_size=10)

    # connection must still be usable afterward -- not left in an aborted transaction
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        assert cur.fetchone()[0] == 0


def test_load_chunks_fault_only_affects_the_triggering_chunk(conn):
    first = EmbeddedChunk(text="commits before the fault", embedding=[0.01] * 384, metadata={})
    second = EmbeddedChunk(text="trips the fault", embedding=[0.02] * 384, metadata={})

    # batch_size=1 forces a commit after the first chunk before the second is attempted
    inserted = load_chunks(conn, [first], batch_size=1)
    assert inserted == 1

    faults.arm(STORE_STAGE, "rate_limit")
    with pytest.raises(faults.RateLimitError):
        load_chunks(conn, [second], batch_size=1)

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        assert cur.fetchone()[0] == 1  # the earlier, already-committed row survives
