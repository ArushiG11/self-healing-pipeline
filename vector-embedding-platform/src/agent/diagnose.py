import json
from src.generation.groq_client import GroqClient

SYSTEM = (
    "You are a data-pipeline reliability assistant. Given a failed job's name and "
    "error message, classify the failure and recommend ONE action from this exact menu:\n"
    "- retry: transient failure (rate limit, timeout, network blip, connection reset)\n"
    "- retry_with_failover: the failure is specific to an external embedding API\n"
    "- escalate: code bugs, missing files, config/schema problems, or anything unclear\n\n"
    "Respond ONLY with JSON: {\"category\": \"...\", \"action\": \"retry|retry_with_failover|escalate\", "
    "\"reasoning\": \"one sentence\"}"
)

def diagnose(job_name: str, error: str) -> dict:
    llm = GroqClient()
    raw = llm.complete(SYSTEM, f"Job: {job_name}\nError: {error}")
    try:
        clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
        d = json.loads(clean)
        if d.get("action") not in {"retry", "retry_with_failover", "escalate"}:
            raise ValueError("action outside menu")
        return d
    except Exception:
        # if the LLM misbehaves, fail SAFE: escalate, never guess an action
        return {"category": "unparseable", "action": "escalate",
                "reasoning": f"could not parse diagnosis: {raw[:200]}"}