"""End-to-end proof that the self-healing loop actually works, using real fault
injection, a real ledger, and the classifier's deterministic path (no LLM needed --
both scenarios below are named rules, not ambiguous cases, so they never touch
Gemini). Ties together: faults -> ingest_records -> poller -> classify_failure ->
heal_job -> re-run.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ledger"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ingestion"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "inject"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "healer"))

from ledger import Ledger, content_hash  # noqa: E402
from reader import ReviewRecord  # noqa: E402
from ingest import STAGE, ingest_records  # noqa: E402
from poller import poll  # noqa: E402
from classifier import classify_failure  # noqa: E402
from heal import heal_job  # noqa: E402
import faults  # noqa: E402

DSN = os.environ.get("TEST_DATABASE_DSN", "dbname=self_healing")


@pytest.fixture(autouse=True)
def _clean_faults():
    faults.disarm()
    yield
    faults.disarm()


@pytest.fixture()
def ledger():
    led = Ledger(DSN)
    led._conn.execute("TRUNCATE TABLE job_ledger")
    led._conn.commit()
    yield led
    led.close()


def make_record(raw: str, text: str) -> ReviewRecord:
    return ReviewRecord(line_number=1, raw=raw, data={"rating": 5.0, "title": "t", "text": text})


def test_rate_limit_fault_retries_and_recovers(ledger):
    record = make_record("rate-limit-e2e", "a perfectly good review, long enough to pass")
    h = content_hash(record.raw)

    # 1. inject a rate-limit fault and run it -- the record fails
    faults.arm(STAGE, "rate_limit")
    out = list(ingest_records(ledger, [record]))
    assert out == []
    row = ledger._get(STAGE, h)
    assert row["status"] == "failed"
    assert row["error_type"] == "RateLimitError"

    # 2. poller finds it as an unhealed failure
    found = []
    poll(ledger, found.append, interval_seconds=0, max_iterations=1)
    assert {r["input_hash"] for r in found} == {h}

    # 3. the deterministic rule alone (no LLM) classifies it as "retry"
    assert classify_failure(row["error_type"], row["error_message"]) == "retry"

    # 4. heal_job applies that: backoff, then failed -> retrying
    decision = heal_job(ledger, row, classify=classify_failure, base_backoff=0.01, max_backoff=0.01)
    assert decision == "retry"
    assert ledger._get(STAGE, h)["status"] == "retrying"

    # 5. re-running the SAME record picks up retrying -> running itself, and now
    #    succeeds (the fault was one-shot and already consumed)
    out = list(ingest_records(ledger, [record]))
    assert len(out) == 1
    final = ledger._get(STAGE, h)
    assert final["status"] == "succeeded"
    assert final["attempt_count"] == 2  # one failed attempt, one successful retry

    # 6. it's fully recovered: no unhealed failures left
    found_after = []
    poll(ledger, found_after.append, interval_seconds=0, max_iterations=1)
    assert found_after == []


def test_code_bug_fault_escalates_without_endless_retrying(ledger):
    record = make_record("schema-mismatch-e2e", "a perfectly good review, long enough to pass")
    h = content_hash(record.raw)

    # 1. inject a schema-mismatch (code bug) fault and run it -- the record fails
    faults.arm(STAGE, "schema_mismatch")
    out = list(ingest_records(ledger, [record]))
    assert out == []
    row = ledger._get(STAGE, h)
    assert row["status"] == "failed"
    assert row["error_type"] == "SchemaError"

    # 2. poller finds it
    found = []
    poll(ledger, found.append, interval_seconds=0, max_iterations=1)
    assert {r["input_hash"] for r in found} == {h}

    # 3. the deterministic rule alone (no LLM) classifies it as "escalate" --
    #    a code bug isn't fixed by retrying
    assert classify_failure(row["error_type"], row["error_message"]) == "escalate"

    # 4. heal_job applies that immediately: failed -> escalated, no backoff sleep
    sleeps = []
    decision = heal_job(ledger, row, classify=classify_failure, sleep_fn=sleeps.append)
    assert decision == "escalate"
    assert sleeps == []  # never slept -- escalation is immediate, not a retry attempt

    final = ledger._get(STAGE, h)
    assert final["status"] == "escalated"
    assert final["attempt_count"] == 1  # exactly one attempt was ever made -- no retry loop
    assert ledger.is_done(STAGE, h) is True  # terminal: nothing will pick this up again

    # 5. it does NOT reappear as an unhealed failure -- there is nothing left to
    #    endlessly retry; it's parked for a human, not looping
    found_after = []
    poll(ledger, found_after.append, interval_seconds=0, max_iterations=1)
    assert found_after == []
