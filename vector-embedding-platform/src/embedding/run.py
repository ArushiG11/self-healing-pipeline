import logging
import pandas as pd
from tqdm import tqdm
from src.chunking.chunkers import CHUNKERS
from src.embedding.bge import BgeEmbeddingClient
from src.ingestion.jobs import job_start, job_finish, job_fail

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
log = logging.getLogger("embed")

def run(strategy: str = "sentence",
        in_path: str = "data/reviews_clean.parquet",
        out_path: str | None = None,
        batch_size: int = 64) -> str:
    out_path = out_path or f"data/chunks_{strategy}.parquet"
    job_id = job_start(f"embed:{strategy}")
    try:
        df = pd.read_parquet(in_path)
        chunker = CHUNKERS[strategy]

        chunks = []
        for row in df.itertuples():
            chunks.extend(chunker(row.id, row.text))
        log.info("chunked | reviews=%d chunks=%d", len(df), len(chunks))

        client = BgeEmbeddingClient()
        texts = [c.text for c in chunks]
        vectors = []
        for i in tqdm(range(0, len(texts), batch_size), desc="embedding"):
            vectors.extend(client.embed(texts[i:i + batch_size]))

        meta = df.set_index("id")[["rating", "asin", "parent_asin", "helpful_vote"]]
        out = pd.DataFrame({
            "chunk_id": [c.chunk_id for c in chunks],
            "parent_id": [c.parent_id for c in chunks],
            "seq": [c.seq for c in chunks],
            "text": texts,
            "embedding": vectors,
        }).join(meta, on="parent_id")

        out.to_parquet(out_path, index=False)
        log.info("done | chunks=%d dim=%d -> %s", len(out), client.dim(), out_path)
        job_finish(job_id, records_in=len(df), records_out=len(out))
        return out_path
    except Exception as e:
        log.error("embed failed | %s", e)
        job_fail(job_id, str(e))
        raise

if __name__ == "__main__":
    run()