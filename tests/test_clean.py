import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ingestion"))

from clean import clean_record, clean_text, normalize_whitespace, strip_html  # noqa: E402


def test_strip_html_removes_tags_and_decodes_entities():
    # tags become spaces (not "") so adjacent tags like </p><p> don't glue words together;
    # trimming is normalize_whitespace's job, exercised separately below
    assert strip_html("<p>Great &amp; sturdy</p>").strip() == "Great & sturdy"
    assert strip_html("no tags here") == "no tags here"


def test_normalize_whitespace_collapses_runs():
    assert normalize_whitespace("a\n\n  b\t\tc   ") == "a b c"


def test_clean_text_combines_both():
    assert clean_text("  <br/>Works  great!<br/>\n\nWould buy   again.  ") == (
        "Works great! Would buy again."
    )


def test_clean_record_drops_missing_rating():
    assert clean_record({"title": "ok", "text": "this text is long enough to pass"}) is None


def test_clean_record_drops_too_short_text():
    assert clean_record({"rating": 5.0, "title": "meh", "text": "too short"}) is None


def test_clean_record_drops_when_html_strips_below_threshold():
    # raw text looks long enough, but it's almost entirely markup
    data = {"rating": 4.0, "title": "x", "text": "<div class='x'><span>ok</span></div>"}
    assert clean_record(data) is None


def test_clean_record_keeps_valid_record_and_cleans_fields():
    data = {
        "rating": 5.0,
        "title": "<b>Love</b> it",
        "text": "  Works great &amp; fast.\n\nWould   buy again!  ",
        "asin": "B000000000",
    }
    cleaned = clean_record(data)
    assert cleaned is not None
    assert cleaned["title"] == "Love it"
    assert cleaned["text"] == "Works great & fast. Would buy again!"
    assert cleaned["asin"] == "B000000000"  # untouched fields pass through


def test_clean_record_respects_custom_min_length():
    data = {"rating": 3.0, "title": "", "text": "exactly ten"}
    assert clean_record(data, min_text_length=5) is not None
    assert clean_record(data, min_text_length=50) is None
