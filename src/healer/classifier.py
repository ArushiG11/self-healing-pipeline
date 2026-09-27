"""Classify a failed ledger job into a healing decision: retry, retry_with_failover,
or escalate.

Deterministic rules run first and never touch the network -- free, instant, and
unambiguous for the two failure classes we can name outright:
  - timeout / 429 (rate limit)  -> transient -> "retry"
  - KeyError / schema mismatch  -> code bug  -> "escalate"
Everything else (including our own faults.ConnectionDroppedError and
faults.MalformedRecordFault -- deliberately not bucketed here; see their
docstrings) is genuinely ambiguous and goes to an LLM, constrained to the fixed
three-item menu via structured outputs. If the model's answer doesn't parse as
JSON, or the decision isn't literally one of the three menu values, the result is
"escalate" -- never a guess.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

MENU = ("retry", "retry_with_failover", "escalate")

_TRANSIENT_TYPE_RE = re.compile(r"timeout|ratelimit", re.IGNORECASE)
_TRANSIENT_MESSAGE_RE = re.compile(
    r"\btimeout\b|\btimed out\b|\b429\b|rate.?limit|too many requests", re.IGNORECASE
)

_CODE_BUG_TYPE_RE = re.compile(r"^keyerror$|schemaerror", re.IGNORECASE)
_CODE_BUG_MESSAGE_RE = re.compile(r"schema mismatch|schema error", re.IGNORECASE)

_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": list(MENU)},
        "reason": {"type": "string"},
    },
    "required": ["decision", "reason"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You are triaging a failed job in a data pipeline. Given the error type and "
    "message, choose exactly one healing decision from this fixed menu: "
    "retry (a fresh attempt is likely to succeed as-is), "
    "retry_with_failover (retry but route to a backup/alternate resource), or "
    "escalate (a human needs to look at this). If you are not confident, choose "
    "escalate rather than guessing."
)


def classify_deterministic(error_type: str, error_message: str) -> Optional[str]:
    """Rule-based classification. Returns a menu decision, or None if the failure
    isn't confidently transient or a code bug -- meaning: defer to the LLM.
    """
    error_type = error_type or ""
    error_message = error_message or ""

    if _TRANSIENT_TYPE_RE.search(error_type) or _TRANSIENT_MESSAGE_RE.search(error_message):
        return "retry"

    if _CODE_BUG_TYPE_RE.search(error_type) or _CODE_BUG_MESSAGE_RE.search(error_message):
        return "escalate"

    return None


GEMINI_MODEL = "gemini-2.5-flash"


def call_gemini(error_type: str, error_message: str) -> str:
    """Real call to Gemini, constrained to the fixed menu via a JSON response schema.

    Returns the raw response text, unparsed -- classify_ambiguous() does the
    defensive parsing so a bad or unparseable response degrades to "escalate"
    rather than propagating an exception out of the healer.
    """
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GOOGLE_API_KEY, falling back to GEMINI_API_KEY
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"error_type: {error_type}\nerror_message: {error_message}",
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=_SCHEMA,
        ),
    )
    return response.text


def classify_ambiguous(
    error_type: str,
    error_message: str,
    *,
    llm_call: Callable[[str, str], str] = call_gemini,
) -> str:
    """Ask the LLM to pick from the fixed menu. Never raises: any failure to call,
    parse, or validate the response results in "escalate" -- never a guess.
    """
    try:
        raw = llm_call(error_type, error_message)
        decision = json.loads(raw).get("decision")
    except Exception:
        return "escalate"

    return decision if decision in MENU else "escalate"


def classify_failure(
    error_type: str,
    error_message: str,
    *,
    llm_call: Callable[[str, str], str] = call_gemini,
) -> str:
    """Full pipeline: deterministic rules first, LLM only for ambiguous cases."""
    decision = classify_deterministic(error_type, error_message)
    if decision is not None:
        return decision
    return classify_ambiguous(error_type, error_message, llm_call=llm_call)
