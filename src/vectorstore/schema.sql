CREATE EXTENSION IF NOT EXISTS vector;

-- id is the sha256 content-hash of `text` (content_hash() in src/ledger/ledger.py),
-- not a surrogate key -- that's what makes INSERT ... ON CONFLICT (id) DO NOTHING a
-- correct dedup/idempotency check: the same text always maps to the same id.
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    embedding   VECTOR(384) NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_metadata
    ON chunks USING gin (metadata);
