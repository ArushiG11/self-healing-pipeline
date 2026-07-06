import json
import logging
import numpy as np
import pandas as pd
from src.embedding.bge import BgeEmbeddingClient
from src.vectorstore.pg import PgVectorStore
from src.ingestion.jobs import job_start, job_finish, job_fail

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
log = logging.getLogger("eval")

def evaluate(table: str, gold_path: str = "data/goldset.jsonl",
             k: int = 5) -> dict:
    gold = [json.loads(l) for l in open(gold_path)]
    client = BgeEmbeddingClient()
    store = PgVectorStore(table=table)

    hits, rr_sum = 0, 0.0
    for g in gold:
        vec = np.array(client.embed([g["question"]])[0])
        results = store.query(vec, k=k)
        rank = next(
            (i + 1 for i, h in enumerate(results)
             if h.metadata["parent_asin"] is not None
             and h.id.split("-")[0] == g["gold_parent_id"]),
            None,
        )
        if rank:
            hits += 1
            rr_sum += 1.0 / rank

    n = len(gold)
    return {"table": table, "k": k, "n": n,
            "recall_at_k": round(hits / n, 3),
            "mrr": round(rr_sum / n, 3)}

def run(tables: list[str] = ["chunks_sentence", "chunks_fixed"]):
    job_id = job_start("eval:retrieval")
    try:
        results = [evaluate(t) for t in tables]
        df = pd.DataFrame(results)
        print("\n" + df.to_string(index=False))
        df.to_csv("data/eval_results.csv", index=False)
        job_finish(job_id, records_in=len(results), records_out=len(results))
    except Exception as e:
        log.error("eval failed | %s", e)
        job_fail(job_id, str(e))
        raise

if __name__ == "__main__":
    run()