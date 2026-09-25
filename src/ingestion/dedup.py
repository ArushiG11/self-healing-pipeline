"""In-run dedup for cleaned review records, by hash of the cleaned text.

Dedup is on the *cleaned* `text` field (via the same content_hash() the ledger uses
for input_hash), not the raw line — two records with identical wording are duplicates
even if other fields (e.g. timestamp, helpful_vote) differ.

The `seen` set is in-memory and scoped to whatever the caller keeps it alive for —
nothing is persisted here. A fresh process, or a fresh `seen` set, sees every record
again. Cross-run dedup is a ledger concern (input_hash lookups), not this module's.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ledger"))
from ledger import content_hash  # noqa: E402

from clean import stream_clean_reviews
from reader import ReviewRecord


def dedup_records(
    records: Iterable[ReviewRecord], seen: Optional[Set[str]] = None
) -> Iterator[ReviewRecord]:
    """Filter an iterable of ReviewRecords, dropping ones whose cleaned text hash is
    already in `seen`. Mutates `seen` in place, so callers can share one set across
    multiple calls (e.g. multiple category files in the same run).
    """
    if seen is None:
        seen = set()
    for record in records:
        text_hash = content_hash(record.data["text"])
        if text_hash in seen:
            continue
        seen.add(text_hash)
        yield record


def stream_deduped_reviews(
    *, seen: Optional[Set[str]] = None, **stream_kwargs
) -> Iterator[ReviewRecord]:
    """Like stream_clean_reviews, but skips records already seen (by cleaned-text
    hash) earlier in this run.
    """
    return dedup_records(stream_clean_reviews(**stream_kwargs), seen=seen)
