"""Streaming line-by-line reader for the Amazon Reviews 2023 dataset on Hugging Face.

Reads JSONL over HTTP via HfFileSystem — the file is never downloaded to disk, so this
works against multi-GB dataset files (Electronics.jsonl is ~22.6GB) with flat memory use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, Optional

from huggingface_hub import HfFileSystem

ELECTRONICS_REVIEWS_PATH = (
    "datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/Electronics.jsonl"
)


@dataclass
class ReviewRecord:
    line_number: int
    raw: str
    data: dict


class MalformedRecord(ValueError):
    """A single JSONL line failed to parse; carries enough context for the caller to
    record it against the ledger (line_number as part of the input, the bad raw line)
    without killing the rest of the stream.
    """

    def __init__(self, line_number: int, raw: str, cause: Exception):
        self.line_number = line_number
        self.raw = raw
        super().__init__(f"malformed JSON on line {line_number}: {cause}")


def stream_reviews(
    path: str = ELECTRONICS_REVIEWS_PATH,
    skip: int = 0,
    limit: Optional[int] = None,
) -> Iterator[ReviewRecord]:
    """Yield one ReviewRecord per JSONL line, streamed over HTTP.

    - path: an HfFileSystem path (defaults to the Electronics category file).
    - skip: number of leading lines to discard, e.g. to resume after a prior run
      recorded progress in the ledger.
    - limit: stop after yielding this many records (None = read to end of file).

    A line that fails to parse raises MalformedRecord *from the generator* rather than
    being silently dropped — the caller (ingestion stage) is expected to catch it,
    record the failure against the ledger via its own (stage, input_hash) for that
    line, and decide retry/escalate, rather than the reader deciding that on its own.
    """
    fs = HfFileSystem()
    yielded = 0
    with fs.open(path, mode="r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            if line_number <= skip:
                continue
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError as e:
                raise MalformedRecord(line_number, raw_line, e) from e

            yield ReviewRecord(line_number=line_number, raw=raw_line, data=data)
            yielded += 1
            if limit is not None and yielded >= limit:
                return
