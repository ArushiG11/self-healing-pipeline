# Review-RAG: A Self-Healing RAG Data Platform

A production-style Retrieval-Augmented Generation system over **45,000+ Amazon product reviews**, built with a vendor-agnostic architecture, a retrieval evaluation harness, and an agentic self-healing layer that reads pipeline failures and retries or escalates them on its own.

**Stack:** Python · pgvector (Postgres) · sentence-transformers (bge-small + cross-encoder reranking) · Groq (Llama 3.1) · FastAPI · Streamlit

---

## Documentation

All project documentation lives in [`docs/`](docs/):

| Doc | What's in it |
|---|---|
| [`docs/README.md`](docs/README.md) | Full project narrative: architecture, retrieval evaluation methodology, the debugging story behind the eval harness, observability, API reference, quickstart, roadmap |
| [`docs/process.md`](docs/process.md) | **Process doc** — chronological build log: what was done, in what order, and why, phase by phase |
| [`docs/solution-design.md`](docs/solution-design.md) | **Solution design doc** — the system by architectural layer: what technology is used at each stage, why it was chosen, and how it's implemented, plus a full tech stack reference |
| [`docs/retrieval-quality-upgrades.md`](docs/retrieval-quality-upgrades.md) | Deep dive on the most recent phase: corpus cleaning, product-context embeddings, cross-encoder reranking — implementation detail and test results |

If you're new to this project, read `docs/solution-design.md` for *what exists and why*, and `docs/process.md` for *how it got built*.

---

## Quickstart

```bash
# 1. Postgres + pgvector
docker run -d --name pgvector -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag \
  -e POSTGRES_DB=ragdb -p 5432:5432 pgvector/pgvector:pg16
docker exec -it pgvector psql -U rag -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 2. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY (free at console.groq.com)

# 3. Pipeline (each step is an observable, restartable job)
python -m src.ingestion.load
python -c "from src.embedding.run import run; run(strategy='sentence')"
python -c "from src.vectorstore.load import run; run('data/chunks_sentence.parquet', table='chunks_sentence')"

# 4. Evaluate
python -m src.eval.goldset
python -c "from src.eval.retrieval_eval import run; run(tables=['chunks_sentence'])"

# 5. Serve
uvicorn src.api.main:app --reload   # then open http://localhost:8000/docs
```

See [`docs/README.md`](docs/README.md) for the full architecture and API reference.
