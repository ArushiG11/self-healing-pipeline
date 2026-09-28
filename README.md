# Self-Healing Pipeline

A data pipeline that streams Amazon product reviews, cleans and deduplicates them,
embeds them, and loads them into a vector store for semantic search — instrumented
end-to-end with a Postgres-backed job ledger so that failures are automatically
retried, retried with backoff, or escalated, using deterministic rules first and an
LLM only for genuinely ambiguous cases.

Everything described below is implemented, tested, and has been run for real against
live systems (a real Postgres+pgvector database, the real Hugging Face dataset, the
real embedding model) — nothing here is aspirational. See `PROJECT_LOG.md` for the
full build history, including every design decision and every bug found along the way.

## What it does

Source data: [`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
(Electronics category, `Electronics.jsonl`, ~22.6GB), streamed over HTTP — never
downloaded in full.

```
reader → clean/filter → dedup → ingest (ledger-tracked)
  → embed (ledger-tracked, batched) → vectorstore (pgvector, idempotent load)
  → retriever (vector search + optional cross-encoder rerank)

failures anywhere above → ledger → poller → classifier → heal (backoff / escalate)
```

### Pipeline stages (`src/`)

| Module | What it does |
|---|---|
| `ledger/ledger.py` | The core state machine: `pending → running → succeeded/failed → retrying/escalated`. Postgres-backed, `UNIQUE(stage, input_hash)`, row-locked with `SELECT ... FOR UPDATE` during transitions to prevent concurrent double-processing. |
| `ingestion/reader.py` | Streams the dataset line-by-line via `HfFileSystem` (no full download). Raises `MalformedRecord` with line context on a bad line rather than silently dropping it. |
| `ingestion/clean.py` | Strips HTML, normalizes whitespace, drops records with a missing rating or text too short *after* cleaning. |
| `ingestion/dedup.py` | In-run dedup by content-hash of cleaned text. |
| `ingestion/ingest.py` | Wraps read→clean under ledger tracking, one ledger row per raw line. A record already `succeeded` is skipped; a `failed` record is left for the healer, not retried here. |
| `embedding/embed.py` | Batches cleaned records through `all-MiniLM-L6-v2` (`sentence-transformers`), ledger-tracked per *batch* (the unit that succeeds/fails together). |
| `vectorstore/store.py` | Loads embeddings into a `chunks` table (Postgres + pgvector). Row `id` is the content-hash of the text; `ON CONFLICT (id) DO NOTHING` makes reloading the same content a no-op. |
| `vectorstore/retriever.py` | Cosine-distance vector search (HNSW-indexed) with an optional cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rerank over a wider candidate set. |
| `healer/poller.py` | Finds ledger rows with `status='failed'` — "no healing decision yet" is exactly that status, not separate state. |
| `healer/classifier.py` | Deterministic rules first (`timeout`/`429` → retry, `KeyError`/schema mismatch → escalate); only ambiguous cases go to Gemini, constrained to a fixed JSON menu (`retry` / `retry_with_failover` / `escalate`). Any unparseable or off-menu LLM response defaults to `escalate` — it never guesses. |
| `healer/heal.py` | Applies the decision: exponential backoff (`base * 2^(attempt_count-1)`, capped) before moving a job to `retrying`, or straight to `escalated`. A hard attempt cap forces escalation *before the classifier is even called*, once exhausted. |
| `inject/faults.py` | An on-command switch (`arm`/`trigger`/`disarm`) that injects a rate limit, a malformed record, a dropped connection, or a schema mismatch at any stage — for testing the retry/escalate paths without waiting for a real failure. |
| `observability/telemetry.py` | An OpenTelemetry span + a set of Prometheus metrics (`rows_in`/`rows_out`/`attempts`/`failures` counters, a latency histogram with buckets sized to this workload) around every unit of work in every stage. |

### Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Postgres with the pgvector extension, then:
psql -d <your_db> -f src/ledger/schema.sql
psql -d <your_db> -f src/vectorstore/schema.sql

# a long-running worker: reader -> ingest -> embed -> store, looped, with telemetry
python deploy/run_worker.py --batch-size 50 --sleep-seconds 10
```

`run_worker.py` exposes Prometheus metrics on `:9464/metrics`. `deploy/prometheus.yml`
scrapes it; `deploy/grafana/self_healing_pipeline_dashboard.json` is a 4-panel
dashboard (rows in/out, stage latency p50/p95/p99, failure rate by stage, failures by
error type) provisioned against that Prometheus instance.

## Testing

```bash
pytest tests/ -m "not live" --cov=src --cov-report=term-missing
```

Tests run against real systems, not mocks: a real local Postgres+pgvector database,
the real live Hugging Face dataset (bounded with `limit` where used), the real
`all-MiniLM-L6-v2` model and cross-encoder. The two tests marked `@pytest.mark.live`
(one real dataset network stream, one real Gemini API call) are excluded by default
and only run explicitly or in an environment with credentials configured.

**Last run of the command above:**

```
135 passed, 2 skipped, 2 deselected
TOTAL   573 stmts   9 miss   98% coverage
```

The 2 skips are conditional, not broken: one needs `GOOGLE_API_KEY`/`GEMINI_API_KEY`
set, the other needs the vectorstore populated with the specific sample
`tests/gold_set.json`'s 8 hand-picked query→chunk pairs were built from (it checks
for this and skips cleanly rather than failing on stale data). In a run against that
data, retrieval quality measured: **recall@5 = 8/8**, cross-encoder reranked
**top-1 accuracy = 6/8**, **MRR improving from 0.838 → 0.875** with reranking — a
small (8-pair) hand-built set, useful as a regression check, not a statistically
powered eval.

## CI (`.github/workflows/ci.yml`)

Three jobs on every push: **lint** (`ruff`), **test** (the command above, against a
real `pgvector/pgvector:pg16` Postgres service container), **build** (the Docker
image, depends on the first two passing). Dependencies in `requirements.txt` are
pinned to exact versions — an earlier unpinned version made a clean dependency
resolve backtrack into a `numpy` release with no Python 3.13 wheel, requiring a
compiler the build image doesn't have.

## Known limitations

These are real, current gaps, not hidden:

- `vectorstore/store.py` has no ledger stage of its own — a failure there propagates
  to the caller rather than going through the retry/escalate machinery.
- A record whose *ingest* already succeeded isn't re-yielded on a resumed run, so a
  crash between ingest and embed can strand it.
- `retry` and `retry_with_failover` currently do the same thing — there's no actual
  failover target wired into ingest/embed yet.
- Spans go to console output only; there's no trace-backend UI (Jaeger/Tempo).
- The observability stack (worker, Prometheus, Grafana) runs as manually-started
  local processes, not a persistent service.

Full history and reasoning for every one of these: `PROJECT_LOG.md`.

## Tech stack

Python 3.13 · PostgreSQL 16 + pgvector · psycopg3 · `sentence-transformers`
(`all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) · Google Gemini
(`google-genai`) · OpenTelemetry · Prometheus · Grafana · Docker · GitHub Actions ·
pytest + `ruff`
