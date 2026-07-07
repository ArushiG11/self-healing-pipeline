import logging
import os
import time
import psycopg2
from dotenv import load_dotenv
from src.agent.diagnose import diagnose
from src.agent.registry import runner_for

load_dotenv()
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
log = logging.getLogger("agent")

MAX_ATTEMPTS = 2          # per failed job, then escalate
BACKOFF_SECONDS = 30

def _conn():
    return psycopg2.connect(os.environ["PG_CONN"])

def unhealed_failures():
    """Failed jobs with no healing_log entry yet."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT j.id, j.job_name, j.error FROM jobs j
            LEFT JOIN healing_log h ON h.failed_job_id = j.id
            WHERE j.status = 'failed' AND h.id IS NULL
            ORDER BY j.id;""")
        return cur.fetchall()

def record(failed_job_id, error, diagnosis, action, outcome):
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO healing_log
              (failed_job_id, observed_error, diagnosis, action_taken, outcome)
            VALUES (%s, %s, %s, %s, %s);""",
            (failed_job_id, error, diagnosis["reasoning"],
             diagnosis["action"], outcome))

def heal(job_id, job_name, error):
    d = diagnose(job_name, error)
    log.info("diagnosis | job=%s action=%s | %s", job_name, d["action"], d["reasoning"])

    if d["action"] == "escalate":
        record(job_id, error, d, "escalate", "needs_human")
        log.warning("ESCALATED | job=%s — human attention needed", job_name)
        return

    runner = runner_for(job_name)
    if runner is None:
        record(job_id, error, d, "escalate", "no_registered_runner")
        log.warning("ESCALATED | job=%s — not in registry", job_name)
        return

    for attempt in range(1, MAX_ATTEMPTS + 1):
        wait = BACKOFF_SECONDS * attempt
        log.info("healing | job=%s attempt=%d (backoff %ds)", job_name, attempt, wait)
        time.sleep(wait)
        try:
            runner()   # idempotent by design — safe to re-run
            record(job_id, error, d, f"retry(attempt={attempt})", "recovered")
            log.info("RECOVERED | job=%s", job_name)
            return
        except Exception as e:
            log.warning("retry failed | job=%s attempt=%d | %s", job_name, attempt, e)

    record(job_id, error, d, f"retry(x{MAX_ATTEMPTS})", "exhausted->needs_human")
    log.warning("ESCALATED | job=%s after %d attempts", job_name, MAX_ATTEMPTS)

def run_forever(poll_seconds: int = 60):
    log.info("agent up | polling every %ds", poll_seconds)
    while True:
        for job_id, job_name, error in unhealed_failures():
            heal(job_id, job_name, error)
        time.sleep(poll_seconds)

if __name__ == "__main__":
    run_forever()