# Resume bullets — derived from README.md

Every claim below traces to something `README.md` documents as actually built and
verified — nothing here goes beyond it. Pick 3-5 depending on the role/context; they're
grouped by theme, not priority order.

**One-line project summary** (for a project title/header):
> Self-healing data pipeline (Amazon reviews → cleaning → embeddings → pgvector)
> with automatic retry/escalation via a Postgres-backed job ledger and an
> LLM-assisted failure classifier.

## Pipeline / systems design

- Designed a resumable, idempotent ETL pipeline (stream → clean → dedup → embed →
  vector store) around a Postgres-backed job ledger enforcing a strict state machine
  (`pending → running → succeeded/failed → retrying/escalated`), so any stage can be
  safely re-run without reprocessing already-completed work.
- Built a streaming ingestion layer that processes a 22.6GB dataset directly over
  HTTP with zero full-file downloads, using row-level locking (`SELECT ... FOR
  UPDATE`) to make concurrent ledger transitions race-safe.
- Implemented an idempotent vector store loader (Postgres + pgvector, HNSW index,
  content-hash-keyed `ON CONFLICT DO NOTHING` upserts) and a retriever combining
  cosine similarity search with optional cross-encoder reranking.

## Self-healing / reliability engineering

- Built a two-tier failure classifier: deterministic rules resolve clear-cut cases
  (timeouts/rate limits → retry, schema errors → escalate) instantly and for free;
  only genuinely ambiguous failures are sent to an LLM (Gemini), constrained to a
  fixed decision menu via structured output, with any unparseable or off-menu
  response defaulting to escalation rather than a guess.
- Implemented a retry engine with exponential backoff and a hard attempt cap
  enforced *before* the classifier is even consulted, guaranteeing no job can be
  retried indefinitely regardless of what a downstream LLM recommends.
- Designed and tested an on-command fault-injection framework (simulated rate
  limits, malformed records, dropped connections, schema mismatches) to validate
  the full detect → classify → retry/escalate loop against real, reproducible
  failure scenarios rather than relying on production incidents to exercise it.

## Observability

- Instrumented every pipeline stage with OpenTelemetry spans and Prometheus metrics
  (throughput, per-stage latency histograms, failure rate by error type), and built
  a real-time 4-panel Grafana dashboard on top of them.

## Testing / quality / CI

- Achieved 98% test coverage across a 139-test suite written against real
  infrastructure — a live Postgres+pgvector database, the real embedding model, the
  real dataset — rather than mocks.
- Set up a 3-stage CI pipeline (GitHub Actions: lint, test, Docker build) with tests
  run against a real Postgres+pgvector service container; diagnosed and fixed a
  dependency-resolution bug where an unpinned requirement set caused pip to
  backtrack into a pre-Python-3.13 `numpy` release requiring a native compiler.
- Built a hand-labeled retrieval-quality evaluation set and measured real
  recall@k/MRR (recall@5 = 100%, MRR improving from 0.838 → 0.875 with cross-encoder
  reranking) rather than assuming embedding quality without measurement.

## Skills / stack keywords

Python · PostgreSQL · pgvector · psycopg · sentence-transformers · LLM structured
output (Gemini) · OpenTelemetry · Prometheus · Grafana · Docker · GitHub Actions ·
pytest
