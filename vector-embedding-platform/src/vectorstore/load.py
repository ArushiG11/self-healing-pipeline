import logging
import pandas as pd
from tqdm import tqdm
from src.vectorstore.pg import PgVectorStore
from src.ingestion.jobs import job_start, job_finish, job_fail

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
log = logging.getLogger("vecload")

def run(in_path: str = "data/chunks_sentence.parquet", batch: int = 500):
    job_id = job_start("vecload:pg")
    try:
        df = pd.read_parquet(in_path)
        store = PgVectorStore()
        for i in tqdm(range(0, len(df), batch), desc="loading"):
            b = df.iloc[i:i + batch]
            store.upsert(
                ids=b.chunk_id.tolist(),
                vectors=b.embedding.tolist(),
                documents=b.text.tolist(),
                metadatas=b[["rating", "asin", "parent_asin", "helpful_vote"]]
                          .to_dict("records"),
            )
        n = store.count()
        log.info("done | rows_in_store=%d", n)
        job_finish(job_id, records_in=len(df), records_out=n)
    except Exception as e:
        log.error("vecload failed | %s", e)
        job_fail(job_id, str(e))
        raise

if __name__ == "__main__":
    run()