import hashlib
import html
import logging
import pandas as pd
from datasets import load_dataset
from src.ingestion.jobs import job_start, job_finish, job_fail
import re
TAG_RE = re.compile(r"<[^>]+>")
BRACKET_TAG_RE = re.compile(r"\[\[[^\]]*\]\]")  # e.g. [[VIDEOID:...]], [[ASIN:...]]

def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = BRACKET_TAG_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ingestion")

MIN_TEXT_LEN = 20

def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()

def ingest(category: str = "raw_review_Electronics", limit: int = 50_000,
           out_path: str = "data/reviews_clean.parquet") -> str:
    job_id = job_start(f"ingest:{category}")
    raw_seen = after_filter = 0
    seen_hashes: set[str] = set()
    rows = []
    try:
        review_category = category.removeprefix("raw_review_")
        data_file = (
            "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/"
            f"raw/review_categories/{review_category}.jsonl"
        )
        ds = load_dataset("json", data_files=data_file, split="train", streaming=True)
        for rec in ds:
            raw_seen += 1
            if raw_seen > limit:
                break

            text = clean_text((rec.get("text") or "").strip())
            rating = rec.get("rating")
            if len(text) < MIN_TEXT_LEN or rating is None:
                continue
            after_filter += 1

            h = text_hash(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            rows.append({
                "id": h[:16],
                "text": text,
                "rating": float(rating),
                "asin": rec.get("asin"),
                "parent_asin": rec.get("parent_asin"),
                "helpful_vote": rec.get("helpful_vote", 0),
                "timestamp": rec.get("timestamp"),
            })

            if raw_seen % 10_000 == 0:
                log.info("progress | raw_seen=%d kept=%d", raw_seen, len(rows))

        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)

        log.info("done | raw_seen=%d after_filter=%d kept=%d -> %s",
                 raw_seen, after_filter, len(df), out_path)
        job_finish(job_id, records_in=raw_seen, records_out=len(df))
        return out_path

    except Exception as e:
        log.error("ingestion failed | %s", e)
        job_fail(job_id, str(e))
        raise

if __name__ == "__main__":
    ingest()