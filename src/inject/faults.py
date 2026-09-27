"""On-command fault injection for exercising the pipeline's retry/escalate paths.

Not a production concern: lets a human or a test deliberately trigger a specific
failure class at a specific stage, without waiting for a real rate limit, a real
malformed record, or a real dropped connection to happen on its own. Each pipeline
stage calls trigger(stage) at the point where it does its real work; if nothing is
armed for that stage, trigger() is a no-op, so this is inert unless explicitly used.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Optional, Tuple


class RateLimitError(Exception):
    """Simulates an upstream API rate limit (e.g. a 429) -- transient, worth retrying."""


class MalformedRecordFault(Exception):
    """Simulates a record whose shape a stage can't handle. Whether that's worth
    retrying (a transient hiccup) or should be escalated (a genuine bad record) is
    the healer's call, same as for a real malformed-record failure.
    """


class ConnectionDroppedError(Exception):
    """Simulates the database/network connection dying mid-operation -- transient,
    worth retrying.
    """


class SchemaError(Exception):
    """Simulates a code bug: the record's shape doesn't match what the stage expects
    (e.g. a required field renamed or missing upstream). Not transient -- retrying
    changes nothing -- so this is squarely classifier.py's deterministic "code bug"
    bucket, not a case that should ever reach the LLM.
    """


FAULTS: Dict[str, Callable[[str], Exception]] = {
    "rate_limit": lambda stage: RateLimitError(f"injected rate_limit fault at stage {stage!r}"),
    "malformed_record": lambda stage: MalformedRecordFault(
        f"injected malformed_record fault at stage {stage!r}"
    ),
    "dropped_connection": lambda stage: ConnectionDroppedError(
        f"injected dropped_connection fault at stage {stage!r}"
    ),
    "schema_mismatch": lambda stage: SchemaError(
        f"injected schema_mismatch fault at stage {stage!r}"
    ),
}

_lock = threading.Lock()
_armed: Dict[str, Tuple[str, int]] = {}  # stage -> (fault_kind, remaining triggers)


def arm(stage: str, kind: str, *, times: int = 1) -> None:
    """Arm `stage` to raise the `kind` fault on its next `times` calls to trigger()."""
    if kind not in FAULTS:
        raise ValueError(f"unknown fault kind {kind!r}; choices: {sorted(FAULTS)}")
    if times < 1:
        raise ValueError("times must be >= 1")
    with _lock:
        _armed[stage] = (kind, times)


def disarm(stage: Optional[str] = None) -> None:
    """Clear a pending fault for `stage`, or every armed stage if `stage` is None."""
    with _lock:
        if stage is None:
            _armed.clear()
        else:
            _armed.pop(stage, None)


def is_armed(stage: str) -> bool:
    with _lock:
        return stage in _armed


def trigger(stage: str) -> None:
    """Raise the fault armed for `stage`, decrementing its remaining count (and
    disarming it once exhausted). No-op if nothing is armed for this stage.
    """
    with _lock:
        armed = _armed.get(stage)
        if armed is None:
            return
        kind, remaining = armed
        remaining -= 1
        if remaining <= 0:
            del _armed[stage]
        else:
            _armed[stage] = (kind, remaining)
    raise FAULTS[kind](stage)
