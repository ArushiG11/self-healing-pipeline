import json
import logging
import time
import pandas as pd
from src.generation.groq_client import GroqClient
from src.ingestion.jobs import job_start, job_finish, job_fail

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
log = logging.getLogger("goldset")

SYSTEM = (
    "You write realistic customer questions. Given a product review excerpt, "
    "write ONE short question a shopper might ask that this excerpt answers. "
    "Use ONLY details explicitly stated in the excerpt - do NOT add product "
    "names, timeframes, measurements, or scenarios that are not in the text. "
    "Include the most identifying details the excerpt actually contains. "
    "Paraphrase the wording - do not copy exact phrases. "
    "Return ONLY the question text."
)

def run(n: int = 300, in_path: str = "data/chunks_sentence.parquet",
        out_path: str = "data/goldset.jsonl"):
    job_id = job_start("goldset:generate")
    try:
        df = pd.read_parquet(in_path)
        # informative chunks only: enough text to ask about
        pool = df[df.text.str.len() > 150].sample(n, random_state=42)

        llm = GroqClient()
        rows = []
        for i, row in enumerate(pool.itertuples(), 1):
            q = llm.complete(SYSTEM, f"Review excerpt:\n{row.text}").strip()
            rows.append({"question": q, "gold_chunk_id": row.chunk_id,
                         "gold_parent_id": row.parent_id})
            if i % 25 == 0:
                log.info("progress | %d/%d", i, n)
            time.sleep(0.5)   # stay friendly with the free tier

        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log.info("done | %d questions -> %s", len(rows), out_path)
        job_finish(job_id, records_in=n, records_out=len(rows))
    except Exception as e:
        log.error("goldset failed | %s", e)
        job_fail(job_id, str(e))
        raise

if __name__ == "__main__":
    run()