"""Postgres-backed job ledger: tracks per-(stage, input) attempts through a fixed state machine."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import psycopg
from psycopg.rows import dict_row

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# pending -> running -> succeeded
#                    -> failed -> retrying -> running
#                              -> escalated
TRANSITIONS: dict[str, set[str]] = {
    "pending": {"running"},
    "running": {"succeeded", "failed"},
    "failed": {"retrying", "escalated"},
    "retrying": {"running"},
    "succeeded": set(),
    "escalated": set(),
}

DONE_STATUSES = {"succeeded", "escalated"}
RUNNABLE_STATUSES = {"pending", "retrying"}


def content_hash(content: bytes | str) -> str:
    """Stable sha256 hex digest of input content, used as the ledger's input_hash."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


class InvalidTransition(ValueError):
    pass


class Ledger:
    def __init__(self, conninfo: str):
        self.conninfo = conninfo
        self._conn = psycopg.connect(conninfo, row_factory=dict_row, autocommit=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text())
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def get_or_create(
        self, stage: str, input_hash: str, rows_in: Optional[int] = None
    ) -> dict:
        """Return the job row for (stage, input_hash), inserting a fresh 'pending' row if absent."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO job_ledger (stage, input_hash, status, rows_in)
                VALUES (%s, %s, 'pending', %s)
                ON CONFLICT (stage, input_hash) DO NOTHING
                """,
                (stage, input_hash, rows_in),
            )
        self._conn.commit()
        return self._get(stage, input_hash)

    def transition(
        self,
        stage: str,
        input_hash: str,
        new_status: str,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        rows_out: Optional[int] = None,
    ) -> dict:
        """Move the job to new_status, enforcing the fixed state machine.

        Locks the row (SELECT ... FOR UPDATE) for the duration of the check-then-update
        so a concurrent transition() on the same job can't read a status this call is
        about to change out from under it.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM job_ledger WHERE stage = %s AND input_hash = %s FOR UPDATE",
                (stage, input_hash),
            )
            row = cur.fetchone()
        if row is None:
            self._conn.rollback()
            raise KeyError(f"no job_ledger row for stage={stage!r} input_hash={input_hash!r}")

        current = row["status"]
        allowed = TRANSITIONS.get(current, set())
        if new_status not in allowed:
            self._conn.rollback()
            raise InvalidTransition(
                f"cannot transition {stage!r}/{input_hash!r} from {current!r} to {new_status!r} "
                f"(allowed: {sorted(allowed) or 'none'})"
            )

        set_clauses = ["status = %s", "updated_at = CURRENT_TIMESTAMP"]
        params: list = [new_status]

        if new_status == "running":
            set_clauses.append("attempt_count = attempt_count + 1")
            set_clauses.append("started_at = CURRENT_TIMESTAMP")
            set_clauses.append("error_type = NULL")
            set_clauses.append("error_message = NULL")
        elif new_status in ("succeeded", "failed"):
            set_clauses.append("finished_at = CURRENT_TIMESTAMP")
            set_clauses.append("error_type = %s")
            set_clauses.append("error_message = %s")
            params.extend([error_type, error_message])
            if new_status == "succeeded":
                set_clauses.append("rows_out = %s")
                params.append(rows_out)
        elif new_status == "escalated":
            set_clauses.append("finished_at = CURRENT_TIMESTAMP")
            set_clauses.append("error_type = %s")
            set_clauses.append("error_message = %s")
            params.extend([error_type, error_message])
        # 'retrying' just flips status; attempt_count increments on the next 'running'.

        params.extend([stage, input_hash])
        with self._conn.cursor() as cur:
            cur.execute(
                f"UPDATE job_ledger SET {', '.join(set_clauses)} WHERE stage = %s AND input_hash = %s",
                params,
            )
        self._conn.commit()
        return self._get(stage, input_hash)

    def is_done(self, stage: str, input_hash: str) -> bool:
        """True if the job has reached a terminal state (succeeded or escalated)."""
        row = self._get(stage, input_hash)
        return row is not None and row["status"] in DONE_STATUSES

# hands a worker the list of jobs it should pick up next.
    def pending_or_retrying(self, stage: Optional[str] = None) -> list[dict]:
        """Jobs eligible to be picked up and run next, optionally scoped to a stage."""
        query = "SELECT * FROM job_ledger WHERE status = ANY(%s)"
        params: list = [list(RUNNABLE_STATUSES)]
        if stage is not None:
            query += " AND stage = %s"
            params.append(stage)
        query += " ORDER BY created_at"
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def _get(self, stage: str, input_hash: str) -> Optional[dict]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM job_ledger WHERE stage = %s AND input_hash = %s",
                (stage, input_hash),
            )
            return cur.fetchone()
