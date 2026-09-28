# Self-Healing Pipeline — Build Log

This documents everything built in this project so far: every instruction given, what was
built or changed in response, why, and how it was verified. It's written chronologically so
the reasoning and the order of decisions stay visible, not just the end state of the code.

The project is a self-healing data pipeline: it streams Amazon review data, cleans it,
embeds it, stores it in a vector database, and — the "self-healing" part — tracks every
unit of work through a ledger so that failures can be automatically retried, retried with
a fallback, or escalated to a human, using a mix of deterministic rules and an LLM for
genuinely ambiguous cases.

---

## Conventions that hold across the whole project

These weren't stated once and forgotten — they were established early and then applied
consistently every time new code was added, so it's worth naming them up front:

- **Test against the real thing, never mocks.** Every test that touches the ledger runs
  against a real local Postgres database (`self_healing`), not SQLite or an in-memory
  fake. The Hugging Face reader is tested against the real, live 22.6GB dataset file. The
  embedding stage is tested against the real `all-MiniLM-L6-v2` model. The vectorstore is
  tested against a real pgvector-backed table with a real HNSW index.
- **One hash function, reused everywhere.** `content_hash()` (sha256) in
  `src/ledger/ledger.py` is the single hashing primitive for: the ledger's `input_hash`
  (per raw line for ingestion, per batch composition for embedding), the in-run dedup key
  (hash of cleaned text), and the vectorstore row's identity (`chunks.id` = hash of text).
- **Same shape for every pipeline stage.** Ingestion and embedding both follow:
  `get_or_create(stage, hash) → skip if is_done() → transition to "running" → do the real
  work → transition to "succeeded"/"failed"`. A failed job is deliberately *not* retried
  by the stage itself — that decision belongs to the healer, not the worker.
- **The ledger's state machine is the source of truth**, not a separate flag:
  `pending → running → succeeded | failed → retrying | escalated → running (again)`.
  "No healing decision yet" isn't tracked separately — it's just `status == 'failed'`,
  because the state machine only allows leaving `failed` via a decision.
- **Verify with real numbers, not just "tests pass."** Nearly every stage was run against
  real data after being built, with the resulting row counts cross-checked directly in
  Postgres via SQL — not just trusted from the Python script's own counters.

---

## 1. Project scaffolding

**Instruction:** *"create folders src/ledger, src/ingestion, src/embedding, src/vectorstore, src/healer, src/inject, tests, deploy"*

Created the empty directory structure. Each folder maps to one pipeline concern:
`ledger` (state tracking), `ingestion` (read + clean + dedup), `embedding`, `vectorstore`,
`healer` (retry/escalate decisions), `inject` (fault injection for testing self-healing),
plus `tests` and `deploy`.

---

## 2. The ledger schema

**Instruction:** *"Write schema.sql: one job_ledger table — stage, input hash, status, attempt count, error info, row counts, timestamps, with a UNIQUE(stage, input_hash) constraint"*

Created `src/ledger/schema.sql` with a single `job_ledger` table: `stage`, `input_hash`,
`status`, `attempt_count`, `error_type`/`error_message`, `rows_in`/`rows_out`,
`created_at`/`updated_at`/`started_at`/`finished_at`, and `UNIQUE(stage, input_hash)` so
the database itself — not application code — refuses to let two rows exist for the same
unit of work.

---

## 3. The Ledger class

**Instruction:** *"Write ledger.py: a content_hash() function, and a Ledger class with get_or_create, transition (enforcing a fixed state machine: pending→running→succeeded/failed→retrying/escalated), is_done, pending_or_retrying."*

Created `src/ledger/ledger.py` (initially against SQLite): `content_hash()`, and a
`Ledger` class with `get_or_create`, `transition` (enforcing the exact state machine
named in the instruction — `InvalidTransition` raised otherwise), `is_done`, and
`pending_or_retrying`. Added `retrying` to the schema's status `CHECK` constraint, since
the state machine needs it but the original schema only listed four statuses.

---

## 4. Testing the ledger against real Postgres — and migrating to it

**Instruction:** *"Write tests/test_ledger.py against a real local Postgres — not a mock — covering the happy path, retry path, escalation path, and that illegal transitions raise an error."*

This instruction couldn't be satisfied without first **migrating the whole `Ledger` class
off SQLite and onto Postgres** — you can't test a `sqlite3`-backed class against a real
Postgres server. So:

- Rewrote `ledger.py` to use `psycopg` (v3) instead of `sqlite3`: `%s` placeholders instead
  of `?`, `status = ANY(%s)` instead of an `IN` clause (psycopg doesn't expand tuples the
  way `sqlite3`/psycopg2 do), `BIGSERIAL` instead of `AUTOINCREMENT`.
- Set up a project `.venv` and `requirements.txt` (`psycopg[binary]`, `pytest`) — neither
  existed yet.
- Found the machine already had a local Postgres server running with an empty `self_healing`
  database — used that as the target.
- Wrote `tests/test_ledger.py`: 14 tests covering the happy path, retry path, escalation
  path, illegal transitions (parametrized across 8 invalid edges), and the `UNIQUE`
  constraint itself.

All 14 passed.

---

## 5. Explaining and fixing a real concurrency bug

**Instruction:** *"Run it. Don't move on until they're green and you can explain, out loud, what SELECT ... FOR UPDATE is doing in transition() and why the unique constraint matters."*

Ran the tests (green). Then, asked to explain `SELECT ... FOR UPDATE` in `transition()` —
**it didn't exist yet.** `transition()` was doing a plain read, then a separate write: a
genuine check-then-act race where two concurrent workers could both read `status='failed'`,
both independently decide `retrying` was a legal next state, and both apply it —
double-dispatching the same job.

Fixed it for real, not just explained the concept:
- Added `SELECT ... FOR UPDATE` inside `transition()`, in the same transaction as the
  subsequent `UPDATE`, so a second caller touching the same row blocks until the first
  commits or rolls back.
- Found and fixed a second bug while doing this: the `InvalidTransition` raise path
  wasn't rolling back, which would have held the row lock open indefinitely on that
  connection.
- **Proved** the fix worked with two real threads and two real connections (not just
  asserted it): Worker A holds the lock for 1.5s, Worker B's `transition()` call visibly
  blocks until A commits, then proceeds with A's already-applied state.
- Explained `UNIQUE(stage, input_hash)` separately: it protects *creation* (two racing
  `get_or_create()` calls), a different race from the one `SELECT ... FOR UPDATE` protects
  (mutating an existing row).

---

## 6. Streaming reader for the Hugging Face dataset

**Instruction:** *"Write a streaming reader against the Hugging Face dataset - McAuley-Lab/Amazon-Reviews-2023 - hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/Electronics.jsonl"*

Created `src/ingestion/reader.py`. `Electronics.jsonl` is 22.6GB, so this reads over HTTP
via `HfFileSystem` (installed `huggingface_hub[hf_xet]`, `fsspec`) rather than downloading
the file — flat memory use regardless of file size.

- `stream_reviews(path, skip, limit)` — a generator yielding `ReviewRecord(line_number,
  raw, data)` per line. `skip` supports resuming from a ledger-recorded offset later;
  `limit` bounds it for testing.
- `MalformedRecord` — a bad JSONL line raises this *out of* the generator (with line
  number and the raw text) instead of being silently dropped, so a later ingestion stage
  can decide what to do with it rather than the reader deciding on its own.

Verified against the real, live dataset (no auth needed, confirmed public) — streamed real
records, confirmed field shapes, confirmed `skip`+`limit` line up correctly, and confirmed
a malformed line raises with the right line number without killing the rest of the stream.

---

## 7. Cleaning and filtering

**Instruction:** *"Clean each record (strip HTML fragments, normalize whitespace), filter out too-short text or missing ratings."*

Created `src/ingestion/clean.py`:
- `strip_html()` — replaces tags with a space (not empty string, so `</p><p>` doesn't glue
  adjacent words together), decodes HTML entities.
- `normalize_whitespace()` — collapses any run of whitespace to one space, trims.
- `clean_record()` — drops records with a missing rating, or whose *cleaned* text (after
  stripping markup) is under a length threshold — checked after cleaning, so markup-heavy
  filler like `<div><span>ok</span></div>` doesn't pass on raw length alone.
- `stream_clean_reviews()` — wraps `stream_reviews` with cleaning + filtering applied.

`tests/test_clean.py` (8 tests, pure functions, no external dependencies). Verified live:
of the first 200 raw records, 9 were dropped (missing rating or too short) and the rest
passed the invariants.

---

## 8. In-run dedup

**Instruction:** *"Hash each record's cleaned text for dedup; skip anything already seen in this run."*

Created `src/ingestion/dedup.py`:
- `dedup_records(records, seen=None)` — filters any iterable of `ReviewRecord` by
  `content_hash()` of the cleaned `text` field (the same hash function the ledger uses).
  `seen` defaults to a fresh set per call but can be shared across calls (e.g. multiple
  category files in one run) and is mutated in place.
- `stream_deduped_reviews()` — `stream_clean_reviews` with dedup layered on top.

Deliberately decoupled from the network-dependent stream so it's testable on plain lists.
`tests/test_dedup.py` (6 tests). Verified live: 22 duplicate-text records dropped out of
the first ~4,700 cleaned reviews.

---

## 9. Wiring the ledger into ingestion

**Instruction:** *"Before processing each record, call ledger.get_or_create("ingest", hash); skip if is_done. Write running → succeeded/failed around the actual work."*

Created `src/ingestion/ingest.py`. Key design decision: `input_hash` is the hash of the
**raw** JSONL line, not the cleaned text — it identifies the *input* to the stage
("parsing this exact review"), independent of whether cleaning it succeeds. (Cleaned-text
hashing is dedup's separate concern.)

`ingest_records()`:
- `get_or_create("ingest", hash(raw)) → skip if is_done()`.
- `transition(..., "running") → clean_record() → transition(..., "succeeded", rows_out=1)`
  and yield, or `rows_out=0` and don't yield if filtered, or `transition(...,
  "failed", ...)` if cleaning itself raised.
- **A judgment call:** a row already sitting in `failed` (from a prior run) is *not*
  retried here — `InvalidTransition` is caught and the record is skipped. Forcing
  `failed → retrying` isn't ingestion's call to make; that's the healer's job.

`tests/test_ingest.py` (6 tests, real Postgres): happy path, filtered-but-not-failed, an
actual failure (text as a list crashes `.strip()`), skip-when-already-succeeded, leave-
failed-for-the-healer, and independent tracking across a mixed batch.

---

## 10. Running it for real numbers

**Instruction:** *"Run it. Record the real numbers: raw records seen, kept after filtering, duplicates dropped. These are the numbers that go on the resume later — not before."*

Ran a real 50,000-line sample (truncated the ledger first for a clean run):

| Metric | Count |
|---|---|
| raw records seen | 50,000 |
| kept after filtering | 46,897 |
| duplicates dropped | 766 |
| final (kept, deduped) | 46,131 |

Cross-checked against the `job_ledger` table directly via SQL (`sum(rows_out)` matched
"kept" exactly), not just the Python script's own counters. Flagged the scope honestly:
this was the first 50,000 lines of a 22.6GB file, not a full-file run — a full run would
take on the order of hours at the measured throughput (~1,110 records/sec).

---

## 11. Confirming existing test coverage

**Instruction:** *"Write tests for the cleaning function and the dedup logic."*

These already existed from steps 7–8 (`test_clean.py`, `test_dedup.py`). Re-ran them (14
tests, still green) rather than writing duplicates, and asked whether there was a specific
gap in mind.

---

## 12. pgvector setup

**Instruction:** *"Set up Postgres with the pgvector extension, a chunks table (id, text, embedding vector(384), metadata)."*

pgvector wasn't available for this Postgres install — Homebrew's `pgvector` bottle only
targets postgresql@17/18, but the running server is postgresql@16. Rather than upgrading
the Postgres major version (which would touch other databases on the machine, not just
`self_healing`), **built pgvector 0.8.0 from source** against `postgresql@16`'s
`pg_config` and installed it into that install's extension directory — no server restart,
no impact on existing data.

Created `src/vectorstore/schema.sql`: `chunks` table (`id`, `text`, `embedding
VECTOR(384)`, `metadata JSONB`), an HNSW index on `embedding` (cosine ops), a GIN index on
`metadata`. Applied it, then verified for real: inserted 3 test vectors, ran a `<=>`
cosine-distance query, confirmed correct ranking, deleted the test rows.

---

## 13. Explaining the index choices

**Instruction:** *"why we used hnsw and gin and what were the alternate options?"*

Explained (no code changes, pure Q&A):
- **HNSW vs. IVFFlat vs. no index** for `embedding`: IVFFlat needs representative data at
  build time to compute good cluster centroids, which the table didn't have (it was
  empty). HNSW builds incrementally with no training step, so it was the only ANN option
  that fit an empty-then-growing table. Also flagged `vector_cosine_ops` as a modeling
  choice (matches normalized sentence-transformer output) that should change if the
  embedding model ever changes.
- **GIN vs. BTREE-expression vs. `jsonb_path_ops`** for `metadata`: plain GIN supports
  both containment and key-existence queries on arbitrary keys, which matters because
  nothing had defined yet what metadata would actually contain or how it'd be queried.
  Flagged that a targeted BTREE index would be cheaper once real query patterns emerge.

---

## 14. Batch embedding

**Instruction:** *"Batch-embed cleaned records with all-MiniLM-L6-v2, each batch tracked through the ledger the same way as ingestion."*

Installed `sentence-transformers` (pulled in `torch`). Created `src/embedding/embed.py`:
- `get_model()` — lazy singleton, so the ~10s model load happens once.
- `embed_texts()` — wraps `model.encode(..., normalize_embeddings=True)` (normalized
  because the HNSW index uses cosine ops), returns plain Python lists.
- **Key design decision:** unlike ingestion (tracked per record), the ledger unit here is
  the **batch** — one `model.encode()` call either succeeds or fails as a whole.
  `batch_hash()` hashes the *sorted* per-record raw hashes, so a batch's identity doesn't
  depend on iteration order.
- `embed_batches()` — same pattern as `ingest_records`: `get_or_create → skip if is_done
  → running → embed_texts → succeeded/failed`, same "leave failed batches for the healer"
  policy.

`tests/test_embed.py` (7 tests, real Postgres + the real model; only the induced-failure
tests stub `embed_texts`, since there's no cheap way to make the real model error on
demand). Verified live end-to-end: 100 raw → 95 cleaned → 6 batches, all succeeded,
producing real normalized 384-dim vectors.

---

## 15. Loading into pgvector, idempotently

**Instruction:** *"Load embeddings into pgvector with an HNSW index. ON CONFLICT DO NOTHING on the row's content-hash ID, so a rerun is safe."*

This required changing `chunks.id` from `BIGSERIAL` to `TEXT PRIMARY KEY` — "content-hash
ID" means the hash *is* the identity, not a separate dedup column next to a surrogate key.
The table was empty, so this was a safe drop-and-recreate.

Created `src/vectorstore/store.py`:
- `chunk_id()` = `content_hash(chunk.text)` — same hash function as everywhere else.
- `load_chunks()` — per-row `INSERT ... ON CONFLICT (id) DO NOTHING RETURNING id`,
  counting only actual inserts; commits every `batch_size` rows for throughput, but
  conflict resolution is per-row so a rerun is always safe regardless of batch boundaries.

**Found and fixed a real bug**, not just a theoretical one: reading `embedding` back out
without an adapter gave a raw string (`"[0.01,0.01,...]"`), not a list — a test caught
this immediately. Installed the `pgvector` PyPI package and called `register_vector(conn)`
so vector columns round-trip properly.

`tests/test_store.py` (7 tests, real Postgres + pgvector). Verified live end-to-end,
including a genuine idempotency check: reloading the exact same 191 chunks inserted 0 new
rows, and a real HNSW-indexed similarity query returned the correct nearest neighbor.

---

## 16. Confirming row counts end to end

**Instruction:** *"Run it, confirm row counts in Postgres match what you expect."*

Ran a fresh 500-record pass and independently verified every number via `SELECT`/`GROUP
BY` in `psql` (not the Python script's own counters):

| stage | status | rows | sum(rows_in) | sum(rows_out) |
|---|---|---|---|---|
| ingest | succeeded | 500 | 500 | 487 |
| embed | succeeded | 16 | 487 | 487 |

`chunks`: 487 rows, 487 distinct ids. Every relationship checked out: 500 raw → 487 kept
→ ⌈487/32⌉ = 16 embed batches → 487 chunks, with 16 = ⌈487/32⌉ confirmed by hand.

---

## 17. Fault injection

**Instruction:** *"Add a switch that deliberately raises: a simulated rate-limit error, a malformed record, a dropped connection — at any stage, on command."*

Created `src/inject/faults.py`: three exception types (`RateLimitError`,
`MalformedRecordFault`, `ConnectionDroppedError`), `arm(stage, kind, times=1)` /
`trigger(stage)` / `disarm()` / `is_armed()`, thread-safe. `trigger()` is a no-op unless
something is armed, so it's inert by default.

Wired `faults.trigger(stage)` into `ingest.py`, `embed.py`, and `store.py` at the exact
point each stage's real error handling already lived, so an injected fault is
indistinguishable from a genuine one.

**Found and fixed two real bugs while wiring this in:**
1. `embed.py` had `from reader import ReviewRecord` *before* the `sys.path.insert` for
   the `ingestion` directory — it only worked because every caller happened to insert
   that path first. Fixed the ordering.
2. `store.py`'s insert loop had **no exception handling at all** — a fault mid-batch
   would have left the connection in an aborted-transaction state, unusable until
   manually rolled back. Added `except Exception: conn.rollback(); raise`.

`tests/test_faults.py` (12 tests): the switch in isolation, then integration through all
three stages, including a test that proves the connection stays usable after a fault and
another proving only the triggering chunk is lost (earlier committed rows survive).
Verified live: armed a rate-limit fault mid-run against real HF data — exactly 1 of 50
records failed with the injected error, the rest succeeded normally.

---

## 18. The poller

**Instruction:** *"Write a poller that finds failed jobs in the ledger with no healing decision yet."*

Recognized that "no healing decision yet" isn't new state to track — the ledger's own
state machine already encodes it as `status == 'failed'` (a decision means transitioning
to `retrying` or `escalated`, at which point the row leaves that set on its own).

Added `Ledger.unhealed_failures(stage=None)` (symmetric with the existing
`pending_or_retrying()`). Created `src/healer/poller.py`: `poll(ledger, on_failure,
interval_seconds, stage, max_iterations)` — loops, finds unhealed failures, hands each to
a callback. Deliberately decides nothing itself; `on_failure` is where the actual
retry/escalate logic plugs in.

Added 3 tests to `test_ledger.py`, created `tests/test_poller.py` (6 tests). Verified
live: injected a real `ConnectionDroppedError` into an ingest run, confirmed the poller
found exactly that job.

---

## 19. The classifier (deterministic rules + LLM fallback)

**Instruction:** *"Write a classifier: deterministic rules first (timeout/429 → transient, KeyError/schema mismatch → code bug); only ambiguous cases go to an LLM constrained to a fixed menu (retry / retry_with_failover / escalate), and anything the LLM returns that doesn't parse or match the menu defaults to escalate — never guess."*

This was an LLM-shaped task, so the Claude API reference was consulted before writing any
code that calls an LLM (rather than guessing at SDK usage from memory).

Created `src/healer/classifier.py`:
- `classify_deterministic()` — regex rules matching **exactly** what was specified:
  `timeout`/`ratelimit` in the error type, or `timeout`/`429`/`rate limit` in the message
  → `"retry"`; `KeyError`/`SchemaError` in the type, or `schema mismatch`/`schema error`
  in the message → `"escalate"`. Deliberately did **not** bucket `ConnectionDroppedError`
  or `MalformedRecordFault` (from step 17) here, even though they might seem transient —
  the instruction named exactly two rules, and those fault types' own docstrings already
  said "the healer decides," so they fall through to the LLM as genuinely ambiguous cases.
- `classify_ambiguous()` — calls an **injectable** `llm_call` function (so tests don't
  need real credentials), constrained to the fixed menu via structured output. Any
  failure — unparseable JSON, a missing `decision` key, an off-menu value, or the call
  itself raising — defaults to `"escalate"`. Never guesses.
- `classify_failure()` — runs the deterministic rules first; only calls the LLM path if
  they return `None` (ambiguous).

Initially built against **Claude** (Anthropic SDK), using `output_config.format` with a
JSON schema constraining `decision` to an enum of the three menu values.

`tests/test_classifier.py` (23 tests): every deterministic rule case (proven to *never*
call the LLM, via a stub that raises if invoked), every defensive-fallback case for the
ambiguous path, and a live end-to-end test gated on real credentials being present.
22 passed, 1 skipped (no credentials available in this environment at the time).

---

## 20. Security incident: a pasted API key, and switching providers to Gemini

**What happened:** the user pasted a live-looking Google/Gemini API key directly into the
chat. This was flagged immediately as a credential leak — the key was never used anywhere,
and the user was told to treat it as compromised: revoke/rotate it via Google AI Studio or
Cloud Console, and use `! export GEMINI_API_KEY=...` in future (runs in the session's
shell, never appears in chat text).

**Instruction (via a clarifying question, confirmed by the user):** switch the classifier
from Claude to Gemini.

- Installed `google-genai`, then **inspected the actually-installed SDK** (its real
  `GenerateContentConfig` fields, its real API-key environment-variable priority) rather
  than writing the integration from memory, since no bundled skill covered Gemini.
- Rewrote `call_claude()` → `call_gemini()`: `google.genai.Client()` (auto-resolves
  `GOOGLE_API_KEY`, falling back to `GEMINI_API_KEY`), model `gemini-2.5-flash`,
  `response_json_schema` + `response_mime_type="application/json"` in place of Claude's
  `output_config.format` — same enum-constrained schema, different provider's mechanism.
- Everything else in the classifier was untouched — the deterministic rules, the
  defensive-parsing/never-guess logic, and the `llm_call` injection seam are all
  provider-agnostic, which is why all 22 non-live tests kept passing unmodified.
- `requirements.txt`: `anthropic` → `google-genai`; uninstalled the unused package.
- The live test still awaits a rotated, real key to actually execute.

---

## 21. Wiring retries: backoff and a hard attempt cap

**Instruction:** *"Wire retries through the ledger's retrying → running transition, with backoff and a hard attempt cap."*

Recognized that the `retrying → running` transition **already happens automatically** —
`ingest_records`/`embed_batches` pick up any row in `retrying` and transition it to
`running` themselves the next time they're fed that record/batch (this was already tested
back in step 4's retry-path test). So this step's actual job was the piece before that:
deciding *whether* to retry, and moving `failed → retrying` (not `→ running`) with backoff
and a cap.

Created `src/healer/heal.py`:
- `backoff_seconds(attempt_count, base=1.0, cap=60.0)` — exponential:
  `base * 2^(attempt_count - 1)`, capped. Reuses the ledger's existing `attempt_count`
  column — no new counter needed.
- `heal_job()` — **the hard cap is checked before the classifier is even called**: if
  `attempt_count >= max_attempts` (default 5), the decision is forced to `"escalate"`
  regardless of what an LLM might recommend. A capped job never gets asked "should we
  retry" again. Otherwise: classify, then either sleep the backoff and transition
  `failed → retrying`, or transition `failed → escalated`. Catches `InvalidTransition` so
  a concurrent healer racing on the same row is a no-op, not a crash.

`tests/test_heal.py` (10 tests, real Postgres, injected `classify`/`sleep_fn` so no real
waiting or LLM calls): backoff math and capping, both retry variants, escalate never
sleeping, the cap overriding the classifier (and refusing to even call it), a configurable
cap, one-below-cap still retrying, and the concurrent-race guard.

**Verified with a full live demo**, not just unit tests: fault-injected a real failure into
a real ingest run → poller found it → `heal_job` applied backoff and moved it to
`retrying` → the *same record* was re-fed through `ingest_records` → it picked up
`retrying → running` on its own, exactly as designed → `succeeded`, `attempt_count`
incremented from 1 to 2.

Full suite at this point: **100 passed, 1 skipped** (the still-pending live Gemini call).

---

## 22. Proving the self-healing loop works, with real injected failures

**Instruction:** *"Test it against injected failures: inject a rate limit, confirm it retries and recovers; inject a code bug, confirm it escalates without endless retrying."*

The existing fault switch (rate-limit, malformed-record, dropped-connection) had nothing
matching the classifier's "code bug" rule (`KeyError`/schema mismatch) — `malformed_record`
was deliberately built to be *ambiguous* (goes to the LLM), not a deterministic code-bug
case. Added a fourth fault kind to `src/inject/faults.py`: `schema_mismatch → SchemaError`,
so this scenario could be tested the same "on command" way as the others, without touching
the still-unverified LLM path at all.

Wrote and ran, end to end against real data:
- **Rate limit → retries and recovers:** inject → `ingest_records` fails
  (`error_type='RateLimitError'`) → poller finds it → `classify_failure` (deterministic
  rule, no LLM) returns `"retry"` → `heal_job` backs off and moves it to `retrying` →
  re-feeding the *same record* through `ingest_records` picks up `retrying → running`
  itself → `succeeded`, `attempt_count` 1→2 → zero unhealed failures remain.
- **Code bug → escalates without looping:** inject `schema_mismatch` → fails
  (`error_type='SchemaError'`) → poller finds it → `classify_failure` returns
  `"escalate"` → `heal_job` moves it straight to `escalated` with **zero backoff
  sleeps** and `attempt_count` staying at exactly 1 → zero unhealed failures remain
  (parked for a human, not retried again).

Made this a permanent regression test, not just a one-off trace: `tests/test_self_healing_e2e.py`
(2 tests, real Postgres + real fault injection), asserting the same invariants shown live
(including that `attempt_count` never exceeds 1 in the escalate case — proof there's no
hidden retry loop). Full suite: 102 passed, 1 skipped.

---

## 23. Auditing test coverage — "did we test everything?"

**Instruction:** *"did we test everything?"*

Cross-checked every file under `src/` against `tests/` and found a real gap:
**`src/ingestion/reader.py` had zero pytest coverage** — `stream_reviews()` and
`MalformedRecord` had only ever been verified with one-off scripts in chat, which don't
persist as regression tests.

Wrote `tests/test_reader.py` (10 tests): fast, deterministic tests against a fake
in-memory filesystem for `skip`/`limit`/blank-line/malformed-line behavior, plus one real
test against the live dataset (bounded with `limit=3`).

Writing the malformed-line recovery test caught a real wrong assumption, not just added
coverage: the first draft assumed you could catch `MalformedRecord` and call `next()`
again to resume the same generator past the bad line. That's incorrect Python semantics —
raising an exception from inside a generator terminates it; a second `next()` call raises
`StopIteration`, not the next item. Rewrote the test to assert *that* explicitly, and added
a second test showing the actually-correct recovery pattern: catch the exception, then
start a **new** `stream_reviews(skip=line_number)` call — which is exactly what `skip` was
built for. Full suite after this fix: 112 passed, 1 skipped.

---

## 24. Retriever and a hand-built gold set

**Instruction:** *"Build a retriever (vector search + optional cross-encoder rerank) and a small hand-built gold set of query→relevant-chunk pairs."*

The `chunks` table only had 2 leftover synthetic rows from a prior test run — not enough
real content to hand-pick meaningful gold pairs from. Repopulated it for real: ran
`reader → ingest → embed → store` over 800 real records, yielding 776 real chunks.

Created `src/vectorstore/retriever.py`:
- `vector_search(conn, query_embedding, top_k)` — cosine-distance (`<=>`) nearest-neighbor
  query against the HNSW index.
- `rerank(query, candidates, top_k)` — cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`,
  same MiniLM family as the bi-encoder) rescoring, lazy-loaded singleton like
  `embed.get_model()`.
- `retrieve(conn, query, top_k=5, rerank_enabled=False, candidate_k=None)` — embeds the
  query with the **same** `embed_texts()` used for chunks (critical: a different model
  would put the query in a different vector space), then optionally narrows a wider
  candidate set (`top_k × 4` by default) down to `top_k` with the cross-encoder.
- Hit and fixed the exact same `%s::vector` cast issue found back in step 12 — `register_vector`
  doesn't help inside an operator expression like `<=>`, only for round-tripping values.

Built `tests/gold_set.json`: 8 hand-picked query→chunk pairs from the real, currently-loaded
data (indoor camera, iPad screen protector, hearing-aid case, Kindle cover, noise-filtering
headset mic, broken lenses, cable clips, smart TV review), with an explicit note that it's
tied to a specific loaded sample and needs regenerating if `chunks` is repopulated from
different data.

`tests/test_retriever.py` (7 tests): basic retriever behavior, plus two gold-set-driven
quality checks that **skip gracefully** (rather than failing on stale expectations) if the
gold chunk ids aren't currently in the table. Standalone run against real data: vector
search recall@5 = 8/8, reranked top-1 accuracy = 6/8.

**Found a real cross-file interaction while running the full suite**, not just this file's
own tests: `test_store.py`'s fixture truncates `chunks` before its own tests, so running
`pytest tests/` as a whole wipes the real gold-set data before `test_retriever.py` gets to
it — which is exactly why its gold-set tests correctly *skip* in a full-suite run rather
than pass or fail on stale data. They only run for real when `chunks` currently holds the
data the gold set was built from.

---

## 25. Computing real recall@k and MRR

**Instruction:** *"Compute recall@k and MRR against that gold set. These numbers are only real once you've run this — not estimated."*

The chunks table had, in fact, been wiped again by the full-suite run at the end of step
24 (confirmed via a direct query: 0/8 gold ids present) — so before computing anything,
re-ran the exact same `reader → ingest → embed → store` pipeline over the same 800 records
to restore it, and re-confirmed all 8 gold ids were present before measuring.

Computed real per-query ranks and aggregate metrics for both vector-search-only and
reranked retrieval:

| Metric | Vector search | +Rerank |
|---|---|---|
| recall@1 | 0.750 | 0.750 |
| recall@3 | 0.875 | **1.000** |
| recall@5 | 1.000 | 1.000 |
| recall@10 | 1.000 | 1.000 |
| MRR | 0.838 | **0.875** |

Reranking didn't change recall@1 but pulled two queries' correct chunks closer to the top
(rank 2→1 and 5→2), closing the gap by recall@3 and lifting MRR — while very slightly
hurting one other query (rank 1→2). Flagged explicitly: with only 8 pairs, these deltas
are indicative, not statistically robust — this isn't a large enough eval to confidently
claim reranking is a net win, just a directional signal plus a regression check.

This was reported as a one-off measurement, not wired into a persisted/repeatable eval
script — offered to add real recall@k/MRR assertions into `test_retriever.py` if wanted
for ongoing tracking.

---

## 26. Filling out the test suite against a real, grep-verified gap count

**Instruction:** *"Fill out the test suite (target a real count, don't invent one)."*

Rather than picking an arbitrary number of tests to add, enumerated every public
function/class across all 12 `src/` modules (via `grep -nE "^def |^    def |^class "`
over each file), then grepped the entire test suite for each function's name to find
which ones had **zero** direct reference anywhere — not "seems undertested," literally
never called by name in any test.

Found exactly 9 such functions and added tests for each:

| Gap | File | Tests added |
|---|---|---|
| `content_hash()` never unit-tested on its own | `src/ledger/ledger.py` | 4 (determinism, different-input, str/bytes equivalence, sha256-hex-digest shape) |
| `Ledger` context manager (`with Ledger(...) as x`) never exercised | `src/ledger/ledger.py` | 1 |
| `stream_clean_reviews()` wrapper never tested | `src/ingestion/clean.py` | 2 |
| `stream_deduped_reviews()` wrapper never tested | `src/ingestion/dedup.py` | 2 |
| `run_ingest()` wrapper never tested | `src/ingestion/ingest.py` | 1 |
| `get_model()` singleton behavior never tested | `src/embedding/embed.py` | 1 |
| `_chunked()` batching helper never tested directly | `src/embedding/embed.py` | 4 (even split, remainder, empty, size-larger-than-input) |
| `get_cross_encoder()` singleton behavior never tested | `src/vectorstore/retriever.py` | 1 |
| `rerank()` never tested as a standalone function | `src/vectorstore/retriever.py` | 3 (empty candidates, clear-match ordering, top_k respected) |

**Real count: 19 new tests**, verified against the actual before/after collected-test
delta (120 → 139 collected, 117 → 136 passed), not just the addition arithmetic.

The wrapper-function tests (`stream_clean_reviews`, `stream_deduped_reviews`,
`run_ingest`) all reused the fake-in-memory-filesystem monkeypatch pattern already
established in `test_reader.py`, since they all ultimately call `stream_reviews()`
internally.

**One real bug caught while writing these**, not just added coverage: the `run_ingest`
test asserted `cur.fetchone()[0] == 2` for a row count, which raised `KeyError: 0` — the
`Ledger`'s connection uses `row_factory=dict_row`, so `fetchone()` returns a dict, not a
tuple. Fixed by aliasing the count column (`SELECT count(*) AS n ...`) and indexing by
name, matching the convention already used elsewhere in the suite.

Restored `chunks` to 776 real rows afterward, since running the full suite (via
`test_store.py`'s own truncating fixture) wipes it, same known interaction as step 24.

---

## 27. Observability: OpenTelemetry spans, Prometheus metrics, Grafana dashboard

**Instruction:** *"Add OpenTelemetry spans around each stage; export metrics (rows in/out, latency, failure rate) to Prometheus; build a Grafana dashboard."*

The largest single task so far — three real pieces of infrastructure, not just
instrumentation code, each verified live rather than just written.

### Instrumentation code — `src/observability/telemetry.py` (new module)

- `setup_telemetry(prometheus_port=9464)` — idempotent (safe to call more than once
  per process). Configures an OpenTelemetry `TracerProvider` with a
  `ConsoleSpanExporter` (spans print to stdout — no separate trace backend like
  Jaeger was stood up, since the instruction only asked for spans to exist and
  metrics to reach Prometheus), and a `MeterProvider` wired to
  `PrometheusMetricReader`, which starts a real `prometheus_client` HTTP server.
- Five instruments: `pipeline_rows_in_total`, `pipeline_rows_out_total`,
  `pipeline_attempts_total`, `pipeline_failures_total` (all counters), and
  `pipeline_stage_duration_seconds` (a histogram). "Failure rate" is deliberately
  **not** its own stored metric — it's the standard Prometheus pattern of exposing
  `attempts`/`failures` counters and deriving the ratio in PromQL:
  `sum(rate(pipeline_failures_total[5m])) by (stage) / sum(rate(pipeline_attempts_total[5m])) by (stage)`.
- **Metric label cardinality was a deliberate design constraint**: labels are only
  `stage` (a handful of fixed values) and, for failures, `error_type` (a bounded set
  of exception class names). `input_hash` is a **span attribute only** — one label
  value per record processed would be a cardinality explosion in a continuously
  scraped Prometheus series.
- `stage_span(stage, input_hash=None)` — a context manager yielding a handle with
  `set_rows_in(n)`, `set_rows_out(n)`, and `mark_failed(error_type)`. Handles both an
  uncaught exception propagating through the block (recorded on the span, failure
  counter incremented, re-raised unchanged) and a stage that catches its own
  exception to record it against the ledger and continue (via explicit
  `mark_failed()`, since `continue` means the exception never reaches the span).

### Wiring into the three stages

- **`src/ingestion/ingest.py`** — wrapped the per-record `running → clean_record →
  succeeded/failed` block in `stage_span("ingest", input_hash=...)`. Restructured the
  control flow so the span always closes *before* any `yield` — otherwise a span's
  recorded duration would include however long the generator's consumer takes
  between `next()` calls, which would make the latency metric measure the caller,
  not the stage.
- **`src/embedding/embed.py`** — same pattern per batch, `span.set_rows_in(len(batch))`
  since the ledger unit here is a batch, not a record.
- **`src/vectorstore/store.py`** — wrapped the *entire* `load_chunks()` call in one
  span (not per-chunk), since this stage has no per-row ledger tracking — the call
  itself is the unit of work. `rows_in`/`rows_out` are the totals, set right before
  the span closes.
- **A pre-existing bug fixed in passing**: `embed.py` had
  `from reader import ReviewRecord` positioned before the `sys.path.insert` for the
  `ingestion` directory (this was actually fixed back in step 17 while wiring
  faults — noted here again since it's the same import-ordering discipline this
  step's wiring depended on).

### Test-suite compatibility

Every existing test calls the pipeline stages without ever calling
`setup_telemetry()` first, and `stage_span()` deliberately **fails loudly**
(`RuntimeError`) rather than silently no-op'ing if telemetry isn't initialized —
observability silently being broken in a real run is worse than an explicit setup
requirement. Rather than editing every test file, added `tests/conftest.py` with a
session-scoped `autouse=True` fixture that calls `setup_telemetry()` once for the
whole pytest session. Full suite after wiring: 136 passed, 3 skipped (unchanged from
before this step — instrumentation didn't change any behavior, only added
observability around it).

### Standing up real Prometheus and Grafana

Docker's daemon wasn't running, so used Homebrew instead (`brew install prometheus
grafana`) — the same approach as the pgvector install in step 12.

- `deploy/prometheus.yml` — scrapes `localhost:9464` every 5s.
- `deploy/run_worker.py` (new) — a genuine **long-running** worker (reader → ingest →
  embed → store, looped, with an increasing line offset each iteration), because a
  one-shot script exits and leaves nothing for Prometheus to scrape. Added
  `--fail-every N` to periodically arm a real fault (not fabricated data) so the
  failure-rate panel would have something real to show.
- `deploy/grafana/self_healing_pipeline_dashboard.json` — 4 panels: **Rows In/Out by
  Stage** (rate of the two counters), **Stage Latency p50/p95/p99**
  (`histogram_quantile` over the duration histogram), **Failure Rate by Stage** (the
  derived ratio above), **Failures by Error Type**. Provisioned via Grafana's HTTP
  API (`POST /api/dashboards/db`), not the UI — datasource created via
  `POST /api/datasources` pointing at `http://localhost:9090`.

### Two real bugs found and fixed while verifying this live, not just written and assumed correct

1. **Histogram buckets were wrong for the actual data.** The OTel SDK's default
   histogram boundaries (`0, 5, 10, 25 ... 10000`) are calibrated for a much coarser
   scale than these stages' real latencies (single-digit milliseconds to low
   single-digit seconds) — nearly every sample landed in the same bucket, making
   `histogram_quantile` a near-meaningless interpolation (initial readings showed
   implausible p50 ≈ p95 ≈ 4.75s for a per-record clean that actually takes
   milliseconds). Caught by comparing the numbers against what the stages should
   plausibly take, not by a test. Fixed with an explicit `View` +
   `ExplicitBucketHistogramAggregation` using boundaries sized to the real workload
   (5ms–60s). Verified the fix by checking real bucket population after restarting
   the worker: `embed` batches spread realistically across 0.25s–10s buckets instead
   of piling into one.
2. **A tooling mistake, not a code bug**: manually backgrounding a script with `&`
   inside a command passed to `run_in_background: true` double-backgrounds it — the
   "task completed" notification fires for the wrapper shell exiting immediately
   (since it just launches the process and returns), not for the actual long-running
   script. This caused a second demo process to crash with "Address already in use"
   on port 9464 because the first one was still alive. Fixed by passing the bare
   command to `run_in_background` directly and letting the tool manage the
   backgrounding itself.

### A real infrastructure interruption, handled by stopping rather than routing around it

Mid-setup, the machine's disk filled to **177MB free out of 228GB**, which crashed
the worker process and caused an `ENOSPC` error on a Grafana API call. This was
flagged immediately rather than silently retried or worked around — killed all
telemetry/Prometheus/Grafana processes to stop further writes, identified (without
deleting) the largest reclaimable caches (`~/Library/Caches/Google` 2.9GB,
`BraveSoftware` 898MB, `Homebrew` 728MB, `pip` 302MB), and asked how to proceed
rather than assuming permission to delete anything. The user freed space themselves;
confirmed 2.2GB free before resuming, then rebuilt the worker → Prometheus → Grafana
chain from scratch (fresh TSDB, fresh Grafana data dir) rather than assuming the
pre-crash state was still consistent — which is what surfaced the stale-histogram-
bucket artifact above (old and new bucket boundaries briefly coexisted in
Prometheus's data until a fresh TSDB cleared it).

### Final live verification

With the worker, Prometheus, and Grafana all running fresh, queried every one of the
dashboard's panel expressions **through Grafana's own datasource proxy** (not just
Prometheus directly), confirming the whole chain end to end:

| Query | Real result |
|---|---|
| `rate(pipeline_rows_in_total[2m])` by stage | ~1.6–1.7 rows/sec across ingest/embed/vectorstore |
| `rate(pipeline_rows_out_total[2m])` by stage | ~1.58 rows/sec, consistent across stages |
| p95 latency by stage | ingest ≈4.8ms, embed ≈437ms, vectorstore ≈240ms — plausible given real model inference vs. pure-Python cleaning |
| failure rate (ingest) | 1.5%, matching the `--fail-every 3` injected rate |
| failures by error type | `RateLimitError` on `ingest`, the only fault armed in this run |

### Clarifying what was actually built (follow-up questions)

The user asked how to view the dashboard, then — reasonably — asked how this was
implemented without a Grafana account. Clarified: this is **self-hosted Grafana**,
installed via Homebrew directly on the machine, running on `localhost:3000` with its
own local built-in `admin/admin` default login stored in a local SQLite file
(`/tmp/grafana-data`) — no grafana.com account, no cloud service, nothing left the
machine. Same category of thing as "how was Postgres implemented" — local software,
not a hosted product requiring credentials.

**Currently running** (all background processes in this session, not permanent
installs — they stop if the session ends): the worker (`deploy/run_worker.py`),
Prometheus (`:9090`), Grafana (`:3000`). *(Update: macOS killed both Grafana and
Prometheus shortly after this for system-wide low memory — see step 28. The worker
was separately stopped by hand before step 28's test run, to free the port it
needed.)*

---

## 28. Running the coverage command for real

**Instruction:** *"pytest tests/ --cov=src --cov-report=term-missing run this and explain the result."*

First attempt: **all 138 tests errored**, `OSError: [Errno 48] Address already in
use`. Cause: `deploy/run_worker.py` from step 27 was still running in the background,
holding port 9464 — the same port `tests/conftest.py`'s session-scoped
`setup_telemetry()` fixture binds for every test run. Stopped that process and reran.

Clean result: **136 passed, 3 skipped, 98% coverage** (573 statements, 9 missed).
Looked up the exact missed lines rather than describing them vaguely:

| File | Coverage | What's missed, precisely |
|---|---|---|
| `classifier.py` | 87% (5 lines) | The entire body of `call_gemini()` — needs real credentials, gated by a `skipif`. |
| `heal.py` | 97% (1 line) | The `decision = "escalate"` defense-in-depth branch for a hypothetical off-menu `classify` return — unreachable in practice since `classify_failure` already guarantees validity. |
| `faults.py` | 97% (1 line) | `arm()`'s `times < 1` validation guard — no test calls it with an invalid `times`. A real, small, closeable gap. |
| `telemetry.py` | 97% (2 lines) | `setup_telemetry()`'s idempotent early-return, and `stage_span()`'s fail-loud guard for being called before setup — both correct, neither triggered by this suite's structure (`conftest.py` always sets up exactly once, first). |

Offered (not yet done, since only asked to run and explain): a test for
`arm(times=0)`, and repopulating `chunks` so the gold-set tests run instead of skip.

Immediately after, an automated notification reported that macOS killed the
still-running Grafana and Prometheus processes from step 27 for system-wide low
memory. Reported this to the user rather than silently restarting them, and declined
to auto-restart given the machine was already showing resource pressure.

---

## 29. GitHub Actions: lint, test, build the Docker image on every push

**Instruction:** *"GitHub Actions: lint, test, build the Docker image on every push."*

Found an empty `Dockerfile` and a `pytest.ini` already sitting in the repo (added
externally, not by me) — the `pytest.ini` had a `live` marker already registered:
*"hits a real external service (network dataset stream, live LLM call) — excluded
from default CI runs."* This set the whole design: real Postgres+pgvector in CI (a
service container, deterministic, no external-network flakiness — nothing to
exclude), but the truly external/credentialed tests (`test_reader.py`'s real dataset
stream, `test_classifier.py`'s real Gemini call) marked `@pytest.mark.live` and
deselected via `pytest -m "not live"`. Added the marker to the classifier's test too,
for consistency (it already self-skipped via `skipif`, but wasn't marked).

### Lint

Ran `ruff check .` — 158 errors, but nearly all of them in `vector-embedding-platform/`,
an unrelated project living in the same git repo. Scoped to `ruff check src tests
deploy` instead: still 133 errors, mostly `I001` (import-sort) and `RUF100` (unused
`noqa`) conflicting with this codebase's deliberate `sys.path.insert()` +
`# noqa: E402` pattern used throughout every module. Traced this to the installed
ruff version's zero-config defaults being far broader than documented — wrote an
explicit `pyproject.toml` selecting only `select = ["E", "F"]`, which both fixed the
false-positive noise **and** made the existing `# noqa: E402` comments meaningful
(E402 is genuinely selected, not silently ignored). Down to 6 genuine issues — one
real unused import (`content_hash` in `test_embed.py`), five lines over 110 chars.
Fixed all 6 by hand so the lint job starts green, not red.

### Test

Verified the real Postgres image before trusting it in CI: `docker pull
pgvector/pgvector:pg16` (confirmed the tag exists — didn't guess it), started it
locally on a throwaway port, applied both `schema.sql` files against it, then ran
the **entire non-live suite against that fresh, empty, password-authenticated
container** — not the local trust-auth dev database — since that's what CI actually
provides. Passed clean: 135 passed, 2 skipped, 2 deselected. This was the real
verification that mattered, not just "the YAML looks right."

### Build — a real dependency bug, not a config problem

Wrote the `Dockerfile` (`python:3.13-slim`, install `requirements.txt`, run
`deploy/run_worker.py`) and a `.dockerignore` (excluding `.venv/`, `.git/`, the
unrelated `vector-embedding-platform/`, etc.). First local `docker build` failed —
not on disk, on a real bug: `requirements.txt` was **unpinned**, so a clean,
single-shot `pip install -r requirements.txt` (which Docker and CI both do) made
pip's resolver backtrack through years of old `sentence-transformers`/`transformers`
releases hunting for a compatible set, landed on `numpy<2.0` (predates Python 3.13
wheels), and tried to build it from source — which needs a C compiler that
`python:3.13-slim` doesn't have. The local `.venv` never hit this because it was
built incrementally across dozens of separate `pip install` calls over the whole
session, never one coherent resolve. **Fixed by pinning every dependency to the
exact versions already verified working** (via `pip freeze` from the working venv) —
the correct fix regardless of Docker, not a workaround.

### The disk, again — twice

Attempting to verify the fix with a real local build, disk approached zero **twice**:
- First attempt: asked before running it at all, given the machine's history this
  session (recommended skipping local verification; user chose "try it, abort if
  disk gets low"). Monitored disk during the build; it hit 0GB free, aborted the
  build, pruned. Only reclaimed 123KB — the actual consumption was 2GB in orphaned
  Docker volumes (from the earlier `pgvector/pgvector` test container), reclaimed via
  `docker volume prune`. Host disk recovered to 3GB, later settling at 3GB after
  Docker Desktop's own VM disk file finished a lazy reclaim.
- Retried with the pinned versions and real headroom: this time the *dependency*
  problem was fixed, but the build kept running in the background after a failed
  `kill $(pgrep ...)` call (empty pgrep match, silently did nothing) while it
  downloaded `torch`'s large wheel, driving disk down to 144MB before it was found
  (via `ps aux`) and killed by PID directly.

**Decision: stopped attempting local `docker build` after that**, rather than a
third try. Diagnosed why pruning wasn't fully working: Docker Desktop's own VM disk
file (`Docker.raw`) is 2.1GB and doesn't shrink after internal pruning on macOS,
regardless of what's actually stored inside it — a platform quirk, not something
fixable from inside the containers. The dependency-resolution fix itself was
confirmed (the exact failure mode reproduced, then stopped reproducing after
pinning); the *complete* image build was left to get its first full run on GitHub's
runners, which have ~14GB+ free.

### Final CI workflow — `.github/workflows/ci.yml`

Three jobs on every push: **lint** (`ruff check src tests deploy`), **test** (real
`pgvector/pgvector:pg16` service container, schema applied, `pytest -m "not live"
--cov=src --cov-report=term-missing`, pip cache + a Hugging Face model cache keyed
on `requirements.txt`), **build** (Docker image via `docker/build-push-action`, not
pushed anywhere, gated on lint+test passing first).

Verified locally one more time end-to-end after all fixes: `ruff check` clean,
`pytest -m "not live"` green (135 passed, 2 skipped, 2 deselected) — with the newly
pinned `requirements.txt` in place, confirming the pin didn't break anything that
worked before.

---

## 30. README and resume bullets, in that order

**Instruction:** *"Write the README from what actually runs. Update your resume bullets from the README, not the other way around."*

Re-verified ground truth before writing anything, rather than working from memory:
re-ran `pytest tests/ -m "not live" --cov=src --cov-report=term-missing` to get the
*current* real numbers (136 → 135 passed this time, since the gold-set tests were
skipping again — `chunks` had been repopulated differently since step 28), and
`find`'d the actual current file tree across `src/`, `tests/`, `deploy/`, `.github/`
rather than reconstructing it from what I remembered building.

**`README.md`** (new): a pipeline-stage table with real file paths, the actual
`run_worker.py` invocation, the actual test command, and its just-measured output
inline (`135 passed, 2 skipped, 2 deselected`, `98% coverage`) rather than a
remembered number. The retrieval-quality numbers (recall@5, MRR) are explicitly
framed as "measured in a run against that data," not stated as a standing fact,
since those specific gold-set tests are conditional on `chunks` currently holding
the sample they were built from. The "Known limitations" section is copied from
this log's own gaps list, not softened for a README audience.

**`RESUME_BULLETS.md`** (new): every bullet checked against a specific README claim
before being written — nothing added that the README doesn't already document. Caught
and fixed one bad first draft: a bullet originally read "...(98 of it exercised, per
the measured run)," a garbled fragment left over from editing, corrected before
finalizing.

---

## Known gaps, flagged honestly along the way (not yet addressed)

- **`store.py` has no ledger stage.** A fault or real error during vectorstore loading
  still just propagates to the caller — safely (rollback added in step 17), but with no
  retry/escalate wiring of its own.
- **A record ingest already marked `succeeded` isn't re-yielded on a resumed run.** Fine
  within one run, but if a future run resumes after a mid-pipeline crash, a record whose
  ingest succeeded but whose embedding never ran won't be handed to embedding again unless
  something durably persisted it first.
- **`retry` and `retry_with_failover` currently produce identical ledger behavior** — both
  just move the job to `retrying`. The distinction only becomes meaningful once an actual
  failover target exists somewhere in ingest/embed, which hasn't been built.
- **The live Gemini call in the classifier is still unverified** — pending a rotated,
  working API key.
- **`test_store.py` and `test_retriever.py` share `chunks` and step on each other.**
  Running the full suite truncates the real gold-set data before the gold-set tests run;
  they skip cleanly rather than failing, but only ever run for real in a standalone
  invocation after repopulating the vectorstore.
- **The gold set is small (8 pairs).** Useful as a regression check and a rough quality
  signal, not as a statistically powered retrieval eval.
- **recall@k/MRR are computed ad hoc, not as a persisted/repeatable eval.** Nothing
  currently tracks these numbers over time or fails CI if they regress.
- **The observability stack (worker, Prometheus, Grafana) is not persistent.** All
  three are background processes started manually in this session — they stop when
  the session ends or the machine restarts. There's no `brew services start`, launch
  agent, or process supervisor keeping them alive, and no provisioning-as-code
  (Grafana's datasource/dashboard were created via one-off API calls, not files
  Grafana auto-loads on startup) beyond the JSON dashboard file itself being saved
  under `deploy/grafana/`.
- **No distributed trace backend.** Spans go to `ConsoleSpanExporter` (stdout) only —
  there's no Jaeger/Tempo to browse traces in a UI, just structured console output.
  Swapping in an OTLP exporter later is a small, isolated change in
  `setup_telemetry()`.
- **`store.py`'s telemetry span is coarser than ingest/embed's.** One span per
  `load_chunks()` call rather than per-chunk, inherited from the same "no per-chunk
  ledger tracking" gap noted above — a large chunk list means one long span/latency
  sample rather than many short ones.
- **`faults.arm()`'s `times < 1` validation guard has no test.** Found by the
  coverage run in step 28; small and easy to close, just not done yet.
- **The full Docker image build was never completed locally.** The dependency-
  resolution bug it surfaced (unpinned `requirements.txt` → pip backtracks into a
  `numpy` release needing a compiler) is fixed and verified as a root cause, but the
  *complete* build (through the `torch`/`sentence-transformers` install) has only
  ever been attempted on this machine, twice, both times stopped for disk safety
  before finishing. Its first full completion will be on GitHub's own CI runners.
- **This machine's disk is unstable independent of anything being run.** Free space
  has swung between 144MB and 4.4GB+ across these sessions without a consistent
  cause traced — Docker Desktop's own VM disk file (`Docker.raw`, 2.1GB, doesn't
  shrink after pruning on macOS) explains part of it, but a drop from 604MB to
  389MB was observed with no disk-intensive command run in between. Worth the
  user's own investigation outside of this project.
- **No persisted, repeatable retrieval eval, still.** `README.md` now states the
  recall@5/MRR numbers as "measured in a specific run," precisely because nothing
  guarantees they're reproducible without first confirming the gold-set chunks are
  loaded — this is the same gap as above, now visible to anyone reading the README
  rather than only this log.
