"""Tests for src/ledger/ledger.py against a real local Postgres instance (no mocks).

Requires a reachable Postgres server; defaults to the local `self_healing` database
(matches what `psql -l` shows on this machine). Override with TEST_DATABASE_DSN,
e.g. TEST_DATABASE_DSN="dbname=self_healing_test" pytest tests/test_ledger.py
"""

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ledger"))

from ledger import Ledger, InvalidTransition, content_hash  # noqa: E402

DSN = os.environ.get("TEST_DATABASE_DSN", "dbname=self_healing")


@pytest.fixture()
def ledger():
    led = Ledger(DSN)
    led._conn.execute("TRUNCATE TABLE job_ledger")
    led._conn.commit()
    yield led
    led.close()


def new_hash() -> str:
    """A fresh input_hash per test so rows never collide across runs."""
    return content_hash(uuid.uuid4().hex)


def test_content_hash_is_deterministic():
    assert content_hash("same input") == content_hash("same input")


def test_content_hash_differs_for_different_input():
    assert content_hash("input a") != content_hash("input b")


def test_content_hash_treats_str_and_its_utf8_bytes_as_equivalent():
    text = "café"
    assert content_hash(text) == content_hash(text.encode("utf-8"))


def test_content_hash_is_a_sha256_hex_digest():
    digest = content_hash("anything")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_ledger_usable_as_a_context_manager_and_closes_on_exit():
    with Ledger(DSN) as led:
        h = content_hash(uuid.uuid4().hex)
        row = led.get_or_create("ingest", h)
        assert row["status"] == "pending"
        assert led._conn.closed == 0

    assert led._conn.closed != 0  # connection was closed on __exit__


def test_get_or_create_is_idempotent_and_starts_pending(ledger):
    h = new_hash()
    first = ledger.get_or_create("ingest", h, rows_in=100)
    assert first["status"] == "pending"
    assert first["attempt_count"] == 0
    assert first["rows_in"] == 100

    second = ledger.get_or_create("ingest", h, rows_in=999)
    assert second["id"] == first["id"]
    assert second["rows_in"] == 100  # unchanged by the second call


def test_happy_path_pending_to_running_to_succeeded(ledger):
    h = new_hash()
    ledger.get_or_create("embed", h, rows_in=50)

    running = ledger.transition("embed", h, "running")
    assert running["status"] == "running"
    assert running["attempt_count"] == 1
    assert running["started_at"] is not None

    succeeded = ledger.transition("embed", h, "succeeded", rows_out=50)
    assert succeeded["status"] == "succeeded"
    assert succeeded["rows_out"] == 50
    assert succeeded["finished_at"] is not None
    assert ledger.is_done("embed", h) is True
    assert all(r["input_hash"] != h for r in ledger.pending_or_retrying())


def test_retry_path_failed_to_retrying_to_running_to_succeeded(ledger):
    h = new_hash()
    ledger.get_or_create("vectorstore", h)

    ledger.transition("vectorstore", h, "running")
    failed = ledger.transition(
        "vectorstore", h, "failed", error_type="TimeoutError", error_message="upstream timed out"
    )
    assert failed["status"] == "failed"
    assert failed["attempt_count"] == 1
    assert failed["error_type"] == "TimeoutError"

    retrying = ledger.transition("vectorstore", h, "retrying")
    assert retrying["status"] == "retrying"
    assert retrying in ledger.pending_or_retrying("vectorstore")

    running_again = ledger.transition("vectorstore", h, "running")
    assert running_again["attempt_count"] == 2
    # error fields are cleared once a fresh attempt starts
    assert running_again["error_type"] is None
    assert running_again["error_message"] is None

    succeeded = ledger.transition("vectorstore", h, "succeeded", rows_out=10)
    assert succeeded["status"] == "succeeded"
    assert ledger.is_done("vectorstore", h) is True


def test_escalation_path_failed_to_escalated_is_terminal(ledger):
    h = new_hash()
    ledger.get_or_create("healer", h)

    ledger.transition("healer", h, "running")
    ledger.transition("healer", h, "failed", error_type="SchemaError", error_message="bad column")

    escalated = ledger.transition(
        "healer", h, "escalated", error_type="SchemaError", error_message="bad column"
    )
    assert escalated["status"] == "escalated"
    assert escalated["finished_at"] is not None
    assert ledger.is_done("healer", h) is True

    with pytest.raises(InvalidTransition):
        ledger.transition("healer", h, "retrying")
    with pytest.raises(InvalidTransition):
        ledger.transition("healer", h, "running")


def test_unhealed_failures_returns_only_undecided_failures(ledger):
    failed_hash = new_hash()
    ledger.get_or_create("ingest", failed_hash)
    ledger.transition("ingest", failed_hash, "running")
    ledger.transition("ingest", failed_hash, "failed", error_type="X", error_message="x")

    retrying_hash = new_hash()
    ledger.get_or_create("ingest", retrying_hash)
    ledger.transition("ingest", retrying_hash, "running")
    ledger.transition("ingest", retrying_hash, "failed", error_type="X", error_message="x")
    ledger.transition("ingest", retrying_hash, "retrying")  # decided: not undecided anymore

    escalated_hash = new_hash()
    ledger.get_or_create("ingest", escalated_hash)
    ledger.transition("ingest", escalated_hash, "running")
    ledger.transition("ingest", escalated_hash, "failed", error_type="X", error_message="x")
    ledger.transition("ingest", escalated_hash, "escalated", error_type="X", error_message="x")

    pending_hash = new_hash()
    ledger.get_or_create("ingest", pending_hash)  # never even attempted

    results = {row["input_hash"] for row in ledger.unhealed_failures()}
    assert results == {failed_hash}


def test_unhealed_failures_can_be_scoped_to_a_stage(ledger):
    h1 = new_hash()
    ledger.get_or_create("ingest", h1)
    ledger.transition("ingest", h1, "running")
    ledger.transition("ingest", h1, "failed", error_type="X", error_message="x")

    h2 = new_hash()
    ledger.get_or_create("embed", h2)
    ledger.transition("embed", h2, "running")
    ledger.transition("embed", h2, "failed", error_type="X", error_message="x")

    assert {row["input_hash"] for row in ledger.unhealed_failures(stage="ingest")} == {h1}
    assert {row["input_hash"] for row in ledger.unhealed_failures(stage="embed")} == {h2}


def test_unhealed_failures_ordered_oldest_first(ledger):
    hashes = [new_hash() for _ in range(3)]
    for h in hashes:
        ledger.get_or_create("ingest", h)
        ledger.transition("ingest", h, "running")
        ledger.transition("ingest", h, "failed", error_type="X", error_message="x")

    results = [row["input_hash"] for row in ledger.unhealed_failures()]
    assert results == hashes  # inserted/failed in this order -> returned in this order


@pytest.mark.parametrize(
    "start_status,illegal_target",
    [
        ("pending", "succeeded"),
        ("pending", "failed"),
        ("pending", "escalated"),
        ("running", "pending"),
        ("running", "retrying"),
        ("running", "escalated"),
        ("succeeded", "running"),
        ("succeeded", "retrying"),
    ],
)
def test_illegal_transitions_raise(ledger, start_status, illegal_target):
    h = new_hash()
    ledger.get_or_create("ingest", h)

    # drive the job to start_status via only legal moves
    path_to = {
        "pending": [],
        "running": ["running"],
        "succeeded": ["running", "succeeded"],
        "failed": ["running", "failed"],
        "retrying": ["running", "failed", "retrying"],
        "escalated": ["running", "failed", "escalated"],
    }
    for step in path_to[start_status]:
        ledger.transition("ingest", h, step)

    with pytest.raises(InvalidTransition):
        ledger.transition("ingest", h, illegal_target)


def test_transition_on_unknown_job_raises_key_error(ledger):
    with pytest.raises(KeyError):
        ledger.transition("ingest", "does-not-exist", "running")


def test_unique_stage_input_hash_constraint(ledger):
    h = new_hash()
    ledger.get_or_create("ingest", h)
    with ledger._conn.cursor() as cur:
        with pytest.raises(Exception):
            cur.execute(
                "INSERT INTO job_ledger (stage, input_hash, status) VALUES (%s, %s, 'pending')",
                ("ingest", h),
            )
    ledger._conn.rollback()
