import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def _conn():
    return psycopg2.connect(os.environ["PG_CONN"])

def job_start(job_name: str) -> int:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs (job_name, status) VALUES (%s, 'running') RETURNING id",
            (job_name,),
        )
        return cur.fetchone()[0]

def job_finish(job_id: int, records_in: int, records_out: int):
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='succeeded', finished_at=now(), "
            "records_in=%s, records_out=%s WHERE id=%s",
            (records_in, records_out, job_id),
        )

def job_fail(job_id: int, error: str):
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='failed', finished_at=now(), error=%s WHERE id=%s",
            (error[:2000], job_id),
        )