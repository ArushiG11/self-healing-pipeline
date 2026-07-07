"""Maps job_name prefixes to re-runnable callables. The agent may ONLY invoke these."""
from src.ingestion.load import ingest
from src.embedding.run import run as embed_run
from src.vectorstore.load import run as vecload_run
from src.eval.goldset import run as goldset_run

def _embed_sentence(**kw): return embed_run(strategy="sentence", **kw)
def _vecload_sentence(**kw): return vecload_run("data/chunks_sentence.parquet", table="chunks_sentence", **kw)

JOB_REGISTRY = {
    "ingest": lambda: ingest(),
    "embed:sentence": _embed_sentence,
    "vecload:pg:chunks_sentence": _vecload_sentence,
    "goldset:generate": lambda: goldset_run(),
}

def runner_for(job_name: str):
    for prefix, fn in JOB_REGISTRY.items():
        if job_name.startswith(prefix):
            return fn
    return None