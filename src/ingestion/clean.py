"""Cleaning and filtering for raw Amazon review records before they enter the pipeline.

Operates on the `data` dict of a ReviewRecord (see reader.py) — the JSON already
parsed from a JSONL line. Two separate concerns:
  - cleaning: strip HTML fragments and collapse whitespace in free-text fields
  - filtering: drop records that don't carry enough signal to be useful downstream
"""

from __future__ import annotations

import html
import re
from typing import Iterator, Optional

from reader import ReviewRecord, stream_reviews

MIN_TEXT_LENGTH = 15

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities (e.g. '&amp;' -> '&')."""
    return html.unescape(_TAG_RE.sub(" ", text))


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (spaces, newlines, tabs) into single spaces and trim."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    return normalize_whitespace(strip_html(text))


def clean_record(data: dict, min_text_length: int = MIN_TEXT_LENGTH) -> Optional[dict]:
    """Return a cleaned copy of data, or None if the record should be dropped.

    Drops records with a missing rating, or with review text shorter than
    min_text_length once HTML and whitespace noise are removed (cleaning happens
    before the length check, so e.g. "<p>ok</p>" doesn't pass on its raw length).
    """
    rating = data.get("rating")
    if rating is None:
        return None

    title = clean_text(data.get("title") or "")
    text = clean_text(data.get("text") or "")

    if len(text) < min_text_length:
        return None

    cleaned = dict(data)
    cleaned["title"] = title
    cleaned["text"] = text
    return cleaned


def stream_clean_reviews(
    *, min_text_length: int = MIN_TEXT_LENGTH, **stream_kwargs
) -> Iterator[ReviewRecord]:
    """Like stream_reviews, but yields only cleaned, filtered records.

    Kept separate from stream_reviews so the raw read and the cleaning policy can
    change independently — a stage that needs raw records (e.g. to hash the
    original line for the ledger) still has stream_reviews available.
    """
    for record in stream_reviews(**stream_kwargs):
        cleaned = clean_record(record.data, min_text_length=min_text_length)
        if cleaned is None:
            continue
        yield ReviewRecord(line_number=record.line_number, raw=record.raw, data=cleaned)
