# Retrieval Quality Upgrades: Corpus Cleaning, Product-Context Embeddings, Cross-Encoder Reranking

Picks up where the [project overview](README.md) leaves off (chunking-strategy A/B, sentence chunking winning at recall@5 0.393 / MRR 0.311). This phase asked: without changing chunking strategy, how much further can retrieval quality move via (1) cleaning the corpus harder, (2) giving embeddings product context, and (3) reranking?

---

## 1. Corpus cleaning, applied where it actually takes effect

**What was found:** `src/ingestion/load.py` already had a `clean_text()` helper (strips `<br/>` tags, `[[VIDEOID:...]]` markers) — but it was never actually called. Raw text went straight into the length filter and the hash used to derive chunk/parent IDs. Sampling live corpus text also turned up two patterns it didn't handle: `[[ASIN:B00RPJOQQ8 ...]]` product-reference tags, and undecoded HTML entities (`&#34;`, `&amp;`, `&#8211;`, `&lt;`, ...).

**What changed:**
- Generalized the video-only regex into one `BRACKET_TAG_RE` that strips any `[[...]]` markup (covers both `VIDEOID` and `ASIN` tags, and any future tag of the same shape).
- Added `html.unescape()` ahead of tag-stripping.
- Wired `clean_text()` in *before* the length filter and the hash, not after.

**Why before the hash specifically matters:** `text_hash()` derives both `parent_id` and `chunk_id`. Hashing pre-cleaning text meant two reviews with identical visible content but different markup wouldn't dedupe, and a review that was mostly markup could pass the length filter on bulk that a reader would never see. Cleaning before hashing fixes both — but it also means chunk IDs are now derived from *cleaned* text, so every downstream artifact (embeddings, pgvector tables, gold set) keyed on the old hashes became silently incompatible with anything built after the change. (Concretely: evaluating the new gold set against the old `chunks` table scored recall@5 = 0.067 — not a retrieval regression, just two artifacts from different corpus versions being compared as if they were the same. The old table was dropped once this was confirmed.)

**Effect on the corpus:** 50,001 raw reviews → 45,272 kept (vs. 45,293 before extended cleaning — the ~21 delta is reviews that either dropped below the length floor once markup no longer padded them out, or now dedupe against an already-kept review).

---

## 2. Product-context embeddings

**Idea:** `bge-small` embeds review text with no signal about what product it's about. "Battery life is great" carries the same vector whether it's a review of headphones or a router. Prepending product identity should sharpen that.

**Implementation:**
- During ingestion, join `raw/meta_categories/meta_Electronics.jsonl` (the metadata companion to the reviews dataset, 5.2 GB) by `parent_asin`, adding `product_title` / `category` columns to `reviews_clean.parquet`.
- At embedding time, build `"Product: {title or category} | Review: {chunk text}"`, embed *that* string, but store the original chunk text for display/citations. Users and the LLM never see the synthetic prefix — only the retrieval math does.
- The query side is **not** enriched — at ask-time the system doesn't know which product the user means (that's what the question is asking), so enrichment is passage-side only. This is the standard asymmetric pattern for this kind of retrieval augmentation.

**Fallback:** only ~70% of reviews' `parent_asin`s matched a metadata record (31,897 / 45,272) — some ASINs in the reviews aren't present in this metadata snapshot (deprecated/delisted listings). The other 30% fall back to `category` (from the review's own ingestion category, `"Electronics"`), so every chunk gets *some* domain grounding rather than none.

**Engineering snag:** `datasets.load_dataset("json", data_files=meta_file, streaming=True)` crashed with `TypeError: Couldn't cast array of type struct<avatar: string, name: string, about: list<item: string>> to null` — the metadata file has inconsistent per-record schema (an `author` field that's a struct in some records, absent/null in others), which breaks the library's Arrow-based schema inference partway through the stream. Downloading the full 5.2 GB file wasn't a reasonable alternative just to extract three fields per record. Fix: bypass `datasets`/Arrow entirely — stream the raw JSONL line-by-line over HTTP via `huggingface_hub.HfFileSystem`, `json.loads` each line directly, and stop early once every needed `parent_asin` has been found.

---

## 3. Cross-encoder reranking

**Idea:** `bge-small` is a bi-encoder — query and passage are embedded independently, and "relevance" is just distance in a shared vector space. A cross-encoder instead reads `(query, passage)` together in one forward pass and outputs a direct relevance score. Strictly more expressive, at the cost of not being pre-indexable — it has to run at query time, over a small candidate set.

**Implementation:** new `src/retrieval/` module:
- `retriever.py` — `Retriever.retrieve()`: embed the query, pull the top 20 candidates from pgvector (cheap ANN search), then (if a reranker is configured) rerank and keep the top 5.
- `reranker.py` — `CrossEncoderReranker`, wrapping `cross-encoder/ms-marco-MiniLM-L-6-v2` (a small MS MARCO-tuned cross-encoder).

**Wired into both consumers**, not just the eval script: `generation/rag.py` (the real `/ask` path) and `eval/retrieval_eval.py` both go through the same `Retriever`. This was deliberate — an eval harness that measures a different code path than production is measuring the wrong thing.

---

## Test results

Same methodology as the main README: recall@k / MRR against an LLM-generated gold set (150 questions this round), hits judged at the parent-review level so chunking strategies stay comparable. The gold set is regenerated every time the corpus or embeddings change, since chunk/parent IDs are content-hash-derived and a corpus change invalidates old labels.

| Stage | `chunks_sentence` recall@5 / MRR | `chunks_fixed` recall@5 / MRR |
|---|---|---|
| Baseline (README, basic cleaning) | 0.393 / 0.311 | 0.327 / 0.230 |
| + extended cleaning only | 0.40 / 0.318 | 0.36 / 0.248 |
| + metadata enrichment + reranking (combined) | **0.360 / 0.335** | **0.367 / 0.336** |

### Reading this honestly, not just as a win

- **MRR improved for both strategies**, most visibly for fixed-size chunking (0.248 → 0.336, +35% relative). This is the expected reranking effect: when the right chunk is retrieved at all, it now tends to rank higher.
- **Recall@5 is mixed** — up slightly for fixed, down for sentence (0.40 → 0.360). Reranking draws from a *wider* top-20 pool before cutting to 5, so in principle it should only ever help recall. A drop means the cross-encoder is occasionally demoting the correct chunk out of the top 5 despite it being present in the top 20 — a known reranker failure mode (the cross-encoder's judgment isn't infallible), not a wiring bug. Confirmed by checking the retriever/reranker code path directly, not just the aggregate number.
- **The two chunking strategies are now close to tied.** Reranking appears to be absorbing most of the retrieval-quality gap that chunking strategy used to control — the earlier A/B result mattered more before reranking existed than it does now.
- **Caveat:** this run combines two changes (enrichment + reranking) at once, so their individual contributions aren't isolated. Attributing the recall dip specifically to reranking (vs. enrichment) would need an ablation — see follow-ups.

---

## Known follow-ups

- **Ablation** to separate enrichment's contribution from reranking's (e.g., rerank on/off with enrichment held fixed).
- ~~**Agent registry gap:**~~ Fixed — `src/agent/registry.py` now also registers `embed:fixed` and `vecload:pg:chunks_fixed`, so a failure in either is auto-retried like its `sentence` counterpart instead of always escalating.
- **Metadata fill rate** is ~70%; the remaining 30% of chunks embed with only the category-level fallback (`"Electronics"`) rather than a real product title.
