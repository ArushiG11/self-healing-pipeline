"""Apply a healing decision to a failed ledger job: backoff + retrying, or escalate.

This is the piece that plugs into poller.poll() as `on_failure`. It does not perform
the retrying -> running transition itself -- that already happens naturally the next
time ingest_records()/embed_batches() encounters the record/batch (TRANSITIONS
already allows "retrying" -> "running", and get_or_create()+is_done() lets a caller
safely re-feed the same raw record/batch through the stage). This module's job is
only: decide, then move failed -> retrying (with backoff) or failed -> escalated,
enforcing a hard attempt cap that overrides even an LLM "retry" recommendation --
the cap exists so a classifier can never retry a job forever.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ledger"))
from ledger import InvalidTransition, Ledger  # noqa: E402

from classifier import MENU, classify_failure  # noqa: E402

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_BACKOFF_SECONDS = 1.0
DEFAULT_MAX_BACKOFF_SECONDS = 60.0


def backoff_seconds(
    attempt_count: int,
    *,
    base: float = DEFAULT_BASE_BACKOFF_SECONDS,
    cap: float = DEFAULT_MAX_BACKOFF_SECONDS,
) -> float:
    """Exponential backoff: base * 2^(attempt_count - 1), capped at `cap`.

    attempt_count is the ledger's own count of completed attempts (1 after the
    first failed attempt), so the wait grows with how many times this job has
    already failed.
    """
    return min(base * (2 ** max(attempt_count - 1, 0)), cap)


def heal_job(
    ledger: Ledger,
    row: dict,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    classify: Callable[[str, str], str] = classify_failure,
    sleep_fn: Callable[[float], None] = time.sleep,
    base_backoff: float = DEFAULT_BASE_BACKOFF_SECONDS,
    max_backoff: float = DEFAULT_MAX_BACKOFF_SECONDS,
) -> str:
    """Decide and apply a healing action for one failed ledger row.

    Returns the decision actually applied: "retry", "retry_with_failover", or
    "escalate". If max_attempts has already been reached, the hard cap forces
    "escalate" without even consulting `classify` -- a capped job never gets
    asked "should we retry" again, no matter what a classifier might say.

    If another healer already moved this job out of "failed" (e.g. a concurrent
    poller), the ledger's own transition() raises InvalidTransition; that's caught
    here as "someone else already handled it," not an error.
    """
    stage = row["stage"]
    input_hash = row["input_hash"]
    attempt_count = row["attempt_count"]

    if attempt_count >= max_attempts:
        decision = "escalate"
    else:
        decision = classify(row["error_type"], row["error_message"])
        if decision not in MENU:  # defense in depth; classify_failure already guarantees this
            decision = "escalate"

    try:
        if decision == "escalate":
            ledger.transition(
                stage,
                input_hash,
                "escalated",
                error_type=row["error_type"],
                error_message=row["error_message"],
            )
        else:
            delay = backoff_seconds(attempt_count, base=base_backoff, cap=max_backoff)
            sleep_fn(delay)
            ledger.transition(stage, input_hash, "retrying")
    except InvalidTransition:
        pass  # already moved out of "failed" by someone else; nothing to do

    return decision
