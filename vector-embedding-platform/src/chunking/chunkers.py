import re
from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    parent_id: str      # the review this came from
    seq: int            # position within the review
    text: str

def fixed_size_chunks(parent_id: str, text: str,
                      size: int = 500, overlap: int = 100) -> list[Chunk]:
    chunks, start, seq = [], 0, 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(Chunk(f"{parent_id}-{seq}", parent_id, seq, piece))
            seq += 1
        start += size - overlap
    return chunks

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def sentence_chunks(parent_id: str, text: str, max_chars: int = 500) -> list[Chunk]:
    sentences = _SENT_SPLIT.split(text)
    chunks, buf, seq = [], "", 0
    for sent in sentences:
        if buf and len(buf) + len(sent) + 1 > max_chars:
            chunks.append(Chunk(f"{parent_id}-{seq}", parent_id, seq, buf.strip()))
            seq += 1
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        chunks.append(Chunk(f"{parent_id}-{seq}", parent_id, seq, buf.strip()))
    return chunks

CHUNKERS = {"fixed": fixed_size_chunks, "sentence": sentence_chunks}