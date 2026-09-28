"""Ledger-tracked ingest stage: clean + filter each raw record under (stage="ingest", input_hash).

input_hash is the hash of the *raw* JSONL line, not the cleaned text — it identifies
the unit of work ("parsing this exact review"), independent of whether cleaning it
succeeds. Dedup by cleaned-text hash (dedup.py) is a separate, unrelated concern.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ledger"))
from ledger import InvalidTransition, Ledger, content_hash  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inject"))
import faults  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "observability"))
from telemetry import stage_span  # noqa: E402

from clean import MIN_TEXT_LENGTH, clean_record
from reader import ReviewRecord, stream_reviews

STAGE = "ingest"


def ingest_records(
    ledger: Ledger,
    records: Iterable[ReviewRecord],
    *,
    min_text_length: int = MIN_TEXT_LENGTH,
) -> Iterator[ReviewRecord]:
    """Run each raw record through ledger-tracked cleaning, yielding survivors.

    Per record:
      - get_or_create("ingest", hash(raw line))
      - skip (no work attempted) if already done (succeeded/escalated from a prior run)
      - skip if the row is "failed" or stuck "running" — those aren't this stage's call
        to retry; that's the healer's job to triage (failed -> retrying) first
      - otherwise: running -> clean_record -> succeeded (rows_out=1, and yield) or
        succeeded (rows_out=0, no yield) if the record was filtered out, or failed
        (with error_type/error_message) if cleaning itself raised
    """
    for record in records:
        input_hash = content_hash(record.raw)
        ledger.get_or_create(STAGE, input_hash, rows_in=1)
        if ledger.is_done(STAGE, input_hash):
            continue

        try:
            ledger.transition(STAGE, input_hash, "running")
        except InvalidTransition:
            continue

        survivor = None
        with stage_span(STAGE, input_hash=input_hash) as span:
            try:
                faults.trigger(STAGE)
                cleaned = clean_record(record.data, min_text_length=min_text_length)
            except Exception as e:
                ledger.transition(
                    STAGE,
                    input_hash,
                    "failed",
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                span.mark_failed(type(e).__name__)
            else:
                if cleaned is None:
                    ledger.transition(STAGE, input_hash, "succeeded", rows_out=0)
                    span.set_rows_out(0)
                else:
                    ledger.transition(STAGE, input_hash, "succeeded", rows_out=1)
                    span.set_rows_out(1)
                    survivor = ReviewRecord(
                        line_number=record.line_number, raw=record.raw, data=cleaned
                    )

        # yield only after the span has closed -- otherwise its duration would
        # include however long the consumer takes between next() calls
        if survivor is not None:
            yield survivor


def run_ingest(
    ledger: Ledger, *, min_text_length: int = MIN_TEXT_LENGTH, **stream_kwargs
) -> Iterator[ReviewRecord]:
    """Convenience wrapper: ingest_records() fed directly from stream_reviews()."""
    return ingest_records(ledger, stream_reviews(**stream_kwargs), min_text_length=min_text_length)
