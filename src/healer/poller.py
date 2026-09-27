"""Poller: repeatedly finds ledger jobs that failed but have no healing decision yet.

This is discovery only, not the decision itself -- deciding retry vs. escalate for
each found job is a separate concern (not built here). `on_failure` is where that
decision-maker plugs in later; for now it can be anything from "print it" to
"transition it," but the poller doesn't know or care which.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from ledger import Ledger


def poll(
    ledger: Ledger,
    on_failure: Callable[[dict], None],
    *,
    interval_seconds: float = 5.0,
    stage: Optional[str] = None,
    max_iterations: Optional[int] = None,
) -> None:
    """Loop: each iteration, find unhealed failures and hand each one to on_failure.

    - stage: scope polling to one pipeline stage, or None for all stages.
    - max_iterations: stop after this many polls (None = run forever). Exists so
      this can be tested and run as a bounded batch job, not just a daemon.
    - interval_seconds: sleep between polls; not applied before the first poll or
      after the last one.
    """
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        for row in ledger.unhealed_failures(stage=stage):
            on_failure(row)
        iterations += 1
        if max_iterations is None or iterations < max_iterations:
            time.sleep(interval_seconds)
