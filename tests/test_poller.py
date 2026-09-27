"""Tests for src/healer/poller.py against a real local Postgres ledger (no mocks)."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ledger"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "healer"))

from ledger import Ledger  # noqa: E402
from poller import poll  # noqa: E402

DSN = os.environ.get("TEST_DATABASE_DSN", "dbname=self_healing")


@pytest.fixture()
def ledger():
    led = Ledger(DSN)
    led._conn.execute("TRUNCATE TABLE job_ledger")
    led._conn.commit()
    yield led
    led.close()


def fail_job(ledger: Ledger, stage: str, input_hash: str) -> None:
    ledger.get_or_create(stage, input_hash)
    ledger.transition(stage, input_hash, "running")
    ledger.transition(stage, input_hash, "failed", error_type="X", error_message="x")


def test_poll_hands_each_unhealed_failure_to_the_callback(ledger):
    fail_job(ledger, "ingest", "hash-1")
    fail_job(ledger, "ingest", "hash-2")

    seen = []
    poll(ledger, seen.append, interval_seconds=0, max_iterations=1)

    assert {row["input_hash"] for row in seen} == {"hash-1", "hash-2"}


def test_poll_ignores_healthy_and_already_decided_jobs(ledger):
    fail_job(ledger, "ingest", "still-failed")

    fail_job(ledger, "ingest", "already-retrying")
    ledger.transition("ingest", "already-retrying", "retrying")

    ledger.get_or_create("ingest", "never-failed")

    seen = []
    poll(ledger, seen.append, interval_seconds=0, max_iterations=1)

    assert {row["input_hash"] for row in seen} == {"still-failed"}


def test_poll_respects_max_iterations_and_stops(ledger):
    fail_job(ledger, "ingest", "hash-1")

    call_count = {"n": 0}

    def counting_callback(row):
        call_count["n"] += 1

    poll(ledger, counting_callback, interval_seconds=0, max_iterations=3)

    # 1 failure found on each of 3 iterations, since nothing changes its status
    assert call_count["n"] == 3


def test_poll_scoped_to_a_stage(ledger):
    fail_job(ledger, "ingest", "ingest-hash")
    fail_job(ledger, "embed", "embed-hash")

    seen = []
    poll(ledger, seen.append, interval_seconds=0, max_iterations=1, stage="embed")

    assert {row["input_hash"] for row in seen} == {"embed-hash"}


def test_poll_picks_up_a_failure_that_appears_between_iterations(ledger):
    seen_per_iteration = []

    def callback(row):
        seen_per_iteration.append(row["input_hash"])
        if row["input_hash"] == "will-appear-late":
            return
        # after the first iteration's (empty) pass, plant a failure for the next one

    poll(ledger, callback, interval_seconds=0, max_iterations=1)
    assert seen_per_iteration == []  # nothing failed yet

    fail_job(ledger, "ingest", "will-appear-late")
    poll(ledger, callback, interval_seconds=0, max_iterations=1)
    assert seen_per_iteration == ["will-appear-late"]
