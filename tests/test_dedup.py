import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ingestion"))

from dedup import dedup_records  # noqa: E402
from reader import ReviewRecord  # noqa: E402


def record(line_number: int, text: str, **extra) -> ReviewRecord:
    data = {"rating": 5.0, "title": "t", "text": text, **extra}
    return ReviewRecord(line_number=line_number, raw=text, data=data)


def test_dedup_drops_exact_repeat_text():
    records = [
        record(1, "Great product, works well"),
        record(2, "Great product, works well"),
        record(3, "A totally different review"),
    ]
    out = list(dedup_records(records))
    assert [r.line_number for r in out] == [1, 3]


def test_dedup_ignores_other_fields_when_text_matches():
    records = [
        record(1, "Same wording here", asin="A"),
        record(2, "Same wording here", asin="B"),
    ]
    out = list(dedup_records(records))
    assert len(out) == 1
    assert out[0].data["asin"] == "A"  # first occurrence wins


def test_dedup_keeps_distinct_text():
    records = [record(1, "first review"), record(2, "second review")]
    out = list(dedup_records(records))
    assert len(out) == 2


def test_dedup_is_lazy_generator():
    def records_gen():
        yield record(1, "only one")
        raise AssertionError("should not be consumed past what's needed")

    out = dedup_records(records_gen())
    first = next(out)
    assert first.line_number == 1


def test_dedup_shared_seen_set_persists_across_calls():
    seen = set()
    first_batch = list(dedup_records([record(1, "shared text")], seen=seen))
    second_batch = list(dedup_records([record(2, "shared text")], seen=seen))
    assert len(first_batch) == 1
    assert len(second_batch) == 0


def test_dedup_default_seen_is_not_shared_between_calls():
    out_a = list(dedup_records([record(1, "text a")]))
    out_b = list(dedup_records([record(2, "text a")]))
    assert len(out_a) == 1
    assert len(out_b) == 1  # fresh `seen` per call when not provided
