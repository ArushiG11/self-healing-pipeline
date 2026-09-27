"""Tests for src/healer/heal.py against a real local Postgres ledger (no mocks).

`classify` and `sleep_fn` are injected fakes -- this module isn't testing
classification correctness (that's test_classifier.py) or real wall-clock waits,
just the retry/backoff/cap wiring itself.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ledger"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "healer"))

from ledger import Ledger  # noqa: E402
from heal import DEFAULT_MAX_ATTEMPTS, backoff_seconds, heal_job  # noqa: E402

DSN = os.environ.get("TEST_DATABASE_DSN", "dbname=self_healing")


@pytest.fixture()
def ledger():
    led = Ledger(DSN)
    led._conn.execute("TRUNCATE TABLE job_ledger")
    led._conn.commit()
    yield led
    led.close()


def fail_job(ledger: Ledger, stage: str, input_hash: str, attempts: int = 1) -> dict:
    """Drive a job through `attempts` full fail cycles and return its current row."""
    ledger.get_or_create(stage, input_hash)
    for i in range(attempts):
        if i > 0:
            ledger.transition(stage, input_hash, "retrying")
        ledger.transition(stage, input_hash, "running")
        ledger.transition(stage, input_hash, "failed", error_type="X", error_message="boom")
    return ledger._get(stage, input_hash)


def always(decision):
    return lambda error_type, error_message: decision


def recording_sleep():
    calls = []
    return calls, calls.append


# --- backoff_seconds is a pure function ------------------------------------------


def test_backoff_grows_exponentially_with_attempt_count():
    assert backoff_seconds(1, base=1.0, cap=1000) == 1.0
    assert backoff_seconds(2, base=1.0, cap=1000) == 2.0
    assert backoff_seconds(3, base=1.0, cap=1000) == 4.0
    assert backoff_seconds(4, base=1.0, cap=1000) == 8.0


def test_backoff_is_capped():
    assert backoff_seconds(10, base=1.0, cap=5.0) == 5.0


def test_backoff_handles_zero_attempts_gracefully():
    assert backoff_seconds(0, base=1.0, cap=1000) == 1.0


# --- retry path -------------------------------------------------------------------


def test_retry_decision_sleeps_the_backoff_then_marks_retrying(ledger):
    row = fail_job(ledger, "ingest", "h1", attempts=1)
    sleeps, record_sleep = recording_sleep()

    decision = heal_job(
        ledger, row, classify=always("retry"), sleep_fn=record_sleep, base_backoff=1.0
    )

    assert decision == "retry"
    assert sleeps == [1.0]  # attempt_count=1 -> base * 2^0
    assert ledger._get("ingest", "h1")["status"] == "retrying"


def test_retry_with_failover_also_marks_retrying(ledger):
    row = fail_job(ledger, "ingest", "h2", attempts=2)
    sleeps, record_sleep = recording_sleep()

    decision = heal_job(
        ledger, row, classify=always("retry_with_failover"), sleep_fn=record_sleep
    )

    assert decision == "retry_with_failover"
    assert sleeps == [2.0]  # attempt_count=2 -> base * 2^1
    assert ledger._get("ingest", "h2")["status"] == "retrying"


# --- escalate path ------------------------------------------------------------------


def test_escalate_decision_never_sleeps_and_marks_escalated(ledger):
    row = fail_job(ledger, "ingest", "h3", attempts=1)
    sleeps, record_sleep = recording_sleep()

    decision = heal_job(ledger, row, classify=always("escalate"), sleep_fn=record_sleep)

    assert decision == "escalate"
    assert sleeps == []
    updated = ledger._get("ingest", "h3")
    assert updated["status"] == "escalated"
    assert updated["error_type"] == "X"


# --- hard attempt cap ---------------------------------------------------------------


def test_cap_forces_escalate_without_consulting_classifier(ledger):
    row = fail_job(ledger, "ingest", "h4", attempts=DEFAULT_MAX_ATTEMPTS)

    def refusing_classify(error_type, error_message):
        raise AssertionError("classifier should never be consulted once capped")

    decision = heal_job(ledger, row, classify=refusing_classify, sleep_fn=lambda s: None)

    assert decision == "escalate"
    assert ledger._get("ingest", "h4")["status"] == "escalated"


def test_cap_is_configurable(ledger):
    row = fail_job(ledger, "ingest", "h5", attempts=2)

    # with max_attempts=2, a job that has already failed twice is at the cap
    decision = heal_job(
        ledger, row, max_attempts=2, classify=always("retry"), sleep_fn=lambda s: None
    )

    assert decision == "escalate"
    assert ledger._get("ingest", "h5")["status"] == "escalated"


def test_one_below_cap_still_retries(ledger):
    row = fail_job(ledger, "ingest", "h6", attempts=DEFAULT_MAX_ATTEMPTS - 1)

    decision = heal_job(ledger, row, classify=always("retry"), sleep_fn=lambda s: None)

    assert decision == "retry"
    assert ledger._get("ingest", "h6")["status"] == "retrying"


# --- concurrent-healer race is swallowed, not a crash --------------------------------


def test_already_healed_job_does_not_raise(ledger):
    row = fail_job(ledger, "ingest", "h7", attempts=1)
    # simulate another healer having already escalated it between poll and heal
    ledger.transition("ingest", "h7", "escalated", error_type="X", error_message="boom")

    # heal_job's own row is now stale (still says "failed"); applying a decision
    # against it must not raise even though the real current status has moved on
    decision = heal_job(ledger, row, classify=always("retry"), sleep_fn=lambda s: None)

    assert decision == "retry"  # this is what heal_job decided, even though...
    assert ledger._get("ingest", "h7")["status"] == "escalated"  # ...it didn't get applied
