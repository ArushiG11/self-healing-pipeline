"""Tests for src/healer/classifier.py.

Deterministic-rule tests need no network access at all. The ambiguous/LLM-path
tests inject a fake `llm_call` rather than mocking the anthropic SDK internals --
that's the seam classify_ambiguous()/classify_failure() expose for exactly this.
A real end-to-end call to Gemini is included but skipped unless GOOGLE_API_KEY or
GEMINI_API_KEY is set, since this sandbox has no credentials configured by default.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "healer"))

from classifier import MENU, classify_ambiguous, classify_deterministic, classify_failure  # noqa: E402


def refusing_llm_call(error_type: str, error_message: str) -> str:
    raise AssertionError("deterministic case should never call the LLM")


# --- deterministic rules ---------------------------------------------------------


@pytest.mark.parametrize(
    "error_type,error_message",
    [
        ("TimeoutError", "the request timed out"),
        ("RateLimitError", "injected rate_limit fault at stage 'ingest'"),  # our own fault type
        ("SomeGenericError", "upstream returned 429"),
        ("SomeGenericError", "rate limit exceeded, try again later"),
        ("requests.exceptions.Timeout", "Read timed out."),
    ],
)
def test_transient_errors_classified_as_retry_without_llm(error_type, error_message):
    assert classify_deterministic(error_type, error_message) == "retry"
    assert classify_failure(error_type, error_message, llm_call=refusing_llm_call) == "retry"


@pytest.mark.parametrize(
    "error_type,error_message",
    [
        ("KeyError", "'asin'"),
        ("SchemaError", "bad column"),
        ("ValueError", "schema mismatch: expected 384 dims, got 256"),
        ("ValueError", "schema error in record"),
    ],
)
def test_code_bug_errors_classified_as_escalate_without_llm(error_type, error_message):
    assert classify_deterministic(error_type, error_message) == "escalate"
    assert classify_failure(error_type, error_message, llm_call=refusing_llm_call) == "escalate"


@pytest.mark.parametrize(
    "error_type,error_message",
    [
        ("ConnectionDroppedError", "injected dropped_connection fault at stage 'embed'"),
        ("MalformedRecordFault", "injected malformed_record fault at stage 'ingest'"),
        ("TypeError", "unsupported operand type(s)"),
        ("", ""),
    ],
)
def test_unrecognized_errors_are_ambiguous(error_type, error_message):
    assert classify_deterministic(error_type, error_message) is None


# --- ambiguous path: LLM constrained to the fixed menu ---------------------------


def test_ambiguous_case_uses_the_injected_llm_and_returns_its_decision():
    calls = []

    def fake_llm(error_type, error_message):
        calls.append((error_type, error_message))
        return '{"decision": "retry_with_failover", "reason": "backup endpoint available"}'

    result = classify_failure("ConnectionDroppedError", "dropped mid-request", llm_call=fake_llm)

    assert result == "retry_with_failover"
    assert calls == [("ConnectionDroppedError", "dropped mid-request")]


@pytest.mark.parametrize("decision", MENU)
def test_classify_ambiguous_accepts_every_menu_value(decision):
    fake_llm = lambda et, em: json.dumps({"decision": decision, "reason": "x"})  # noqa: E731
    assert classify_ambiguous("X", "y", llm_call=fake_llm) == decision


def test_unparseable_llm_response_defaults_to_escalate():
    fake_llm = lambda et, em: "not json at all"  # noqa: E731
    assert classify_ambiguous("X", "y", llm_call=fake_llm) == "escalate"


def test_off_menu_decision_defaults_to_escalate():
    fake_llm = lambda et, em: '{"decision": "just_ignore_it", "reason": "x"}'  # noqa: E731
    assert classify_ambiguous("X", "y", llm_call=fake_llm) == "escalate"


def test_missing_decision_key_defaults_to_escalate():
    fake_llm = lambda et, em: '{"reason": "no decision field here"}'  # noqa: E731
    assert classify_ambiguous("X", "y", llm_call=fake_llm) == "escalate"


def test_llm_call_raising_defaults_to_escalate_not_a_crash():
    def blows_up(error_type, error_message):
        raise ConnectionError("network is down")

    assert classify_ambiguous("X", "y", llm_call=blows_up) == "escalate"


def test_valid_json_but_not_an_object_defaults_to_escalate():
    fake_llm = lambda et, em: '"retry"'  # a bare JSON string, not an object  # noqa: E731
    assert classify_ambiguous("X", "y", llm_call=fake_llm) == "escalate"


# --- real end-to-end call, gated on real credentials being available ------------


@pytest.mark.skipif(
    not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
    reason="no GOOGLE_API_KEY/GEMINI_API_KEY in this environment; set one to exercise the real API call",
)
def test_real_llm_call_returns_a_menu_decision_for_a_genuinely_ambiguous_case():
    result = classify_failure(
        "ConnectionDroppedError",
        "the database connection was reset mid-transaction",
    )
    assert result in MENU
