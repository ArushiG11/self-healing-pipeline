CREATE TABLE IF NOT EXISTS job_ledger (
    id              BIGSERIAL PRIMARY KEY,
    stage           TEXT NOT NULL,
    input_hash      TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'retrying', 'escalated')),
    attempt_count   INTEGER NOT NULL DEFAULT 0,
    error_type      TEXT,
    error_message   TEXT,
    rows_in         INTEGER,
    rows_out        INTEGER,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at      TIMESTAMP,
    finished_at     TIMESTAMP,
    UNIQUE (stage, input_hash)
);

CREATE INDEX IF NOT EXISTS idx_job_ledger_status ON job_ledger (status);
CREATE INDEX IF NOT EXISTS idx_job_ledger_stage ON job_ledger (stage);


-- stage + input_hash together identify one unit of work (e.g., "parsing this exact review").
-- status is where the job sits in its lifecycle right now.
-- attempt_count — how many times we've tried, so you can cap retries.
-- error_class / error_message — recorded only on failure, so the healer (built later) has something to classify.
-- rows_in / rows_out — lets you sanity-check that a stage didn't silently drop data.
-- created_at / updated_at — for debugging and for freshness metrics later.
-- UNIQUE(stage, input_hash) — this is the one line doing the real work. It means the database itself refuses to let two rows exist for the same piece of work at the same stage. Not "the Python code tries to check first" — the constraint makes it structurally impossible, even if two processes race each other.
