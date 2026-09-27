"""Tests for src/ingestion/reader.py.

Line-parsing logic (skip/limit/malformed-line handling) is tested against a fake
in-memory filesystem for speed and determinism -- the same style as testing
dedup_records() on plain lists rather than a real network stream. One real test
against the live Hugging Face dataset is included too, bounded with `limit` so it
stays fast, since this project's convention is to verify the real integration, not
just the logic around it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ingestion"))

import reader as reader_module  # noqa: E402
from reader import MalformedRecord, ReviewRecord, stream_reviews  # noqa: E402


class _FakeFile:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return iter(self._lines)

    def __exit__(self, *exc_info):
        return False


class _FakeFileSystem:
    def __init__(self, lines):
        self._lines = lines

    def open(self, path, mode="r", encoding="utf-8"):
        return _FakeFile(self._lines)


def use_fake_lines(monkeypatch, lines):
    monkeypatch.setattr(reader_module, "HfFileSystem", lambda: _FakeFileSystem(lines))


def test_yields_one_record_per_line_with_correct_line_numbers(monkeypatch):
    use_fake_lines(monkeypatch, ['{"a": 1}\n', '{"a": 2}\n', '{"a": 3}\n'])
    records = list(stream_reviews(path="fake"))

    assert [r.line_number for r in records] == [1, 2, 3]
    assert [r.data["a"] for r in records] == [1, 2, 3]
    assert all(isinstance(r, ReviewRecord) for r in records)


def test_raw_field_preserves_the_original_line_text(monkeypatch):
    use_fake_lines(monkeypatch, ['{"a": 1}\n'])
    record = next(stream_reviews(path="fake"))

    assert record.raw == '{"a": 1}'  # stripped of the trailing newline, otherwise verbatim


def test_blank_lines_are_skipped(monkeypatch):
    use_fake_lines(monkeypatch, ['{"a": 1}\n', "\n", "   \n", '{"a": 2}\n'])
    records = list(stream_reviews(path="fake"))

    assert [r.data["a"] for r in records] == [1, 2]
    assert [r.line_number for r in records] == [1, 4]  # line numbers count blank lines too


def test_skip_discards_leading_lines(monkeypatch):
    use_fake_lines(monkeypatch, ['{"a": 1}\n', '{"a": 2}\n', '{"a": 3}\n'])
    records = list(stream_reviews(path="fake", skip=2))

    assert [r.line_number for r in records] == [3]
    assert records[0].data["a"] == 3


def test_limit_stops_after_n_records(monkeypatch):
    use_fake_lines(monkeypatch, ['{"a": 1}\n', '{"a": 2}\n', '{"a": 3}\n'])
    records = list(stream_reviews(path="fake", limit=2))

    assert [r.data["a"] for r in records] == [1, 2]


def test_skip_and_limit_compose(monkeypatch):
    lines = [f'{{"a": {i}}}\n' for i in range(1, 6)]
    use_fake_lines(monkeypatch, lines)
    records = list(stream_reviews(path="fake", skip=2, limit=2))

    assert [r.data["a"] for r in records] == [3, 4]


def test_malformed_line_raises_with_line_number_and_raw_text(monkeypatch):
    use_fake_lines(monkeypatch, ['{"a": 1}\n', "not json\n", '{"a": 3}\n'])

    it = stream_reviews(path="fake")
    first = next(it)
    assert first.data["a"] == 1

    with pytest.raises(MalformedRecord) as exc_info:
        next(it)

    assert exc_info.value.line_number == 2
    assert exc_info.value.raw == "not json"


def test_generator_terminates_on_a_malformed_line_not_just_pauses(monkeypatch):
    # raising from inside a generator ends it -- next() after catching the
    # exception raises StopIteration, it does NOT resume yielding line 3
    use_fake_lines(monkeypatch, ['{"a": 1}\n', "not json\n", '{"a": 3}\n'])

    it = stream_reviews(path="fake")
    assert next(it).data["a"] == 1
    with pytest.raises(MalformedRecord):
        next(it)
    with pytest.raises(StopIteration):
        next(it)


def test_recovery_pattern_is_a_new_call_with_skip_past_the_bad_line(monkeypatch):
    # the intended recovery: catch MalformedRecord, note its line_number, then
    # start a *new* stream_reviews(skip=...) call to continue past it -- that's
    # what `skip` is for, not resuming the same exhausted generator
    use_fake_lines(monkeypatch, ['{"a": 1}\n', "not json\n", '{"a": 3}\n'])

    results = []
    it = stream_reviews(path="fake")
    try:
        for record in it:
            results.append(record.data["a"])
    except MalformedRecord as e:
        results.extend(r.data["a"] for r in stream_reviews(path="fake", skip=e.line_number))

    assert results == [1, 3]


# --- real integration: the live dataset, bounded so it stays fast -----------------


def test_real_stream_reads_actual_electronics_reviews():
    records = list(stream_reviews(limit=3))

    assert len(records) == 3
    assert all(r.line_number == i + 1 for i, r in enumerate(records))
    for r in records:
        assert "rating" in r.data
        assert "text" in r.data
        assert isinstance(r.raw, str) and r.raw  # the raw line was preserved verbatim
