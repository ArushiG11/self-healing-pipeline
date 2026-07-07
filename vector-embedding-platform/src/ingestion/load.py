import hashlib
import html
import json
import logging
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfFileSystem
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

def load_product_meta(review_category: str, parent_asins: set[str]) -> dict[str, dict]:
    """title/main_category by parent_asin, restricted to the asins we actually kept.

    Reads the raw jsonl line-by-line via HfFileSystem instead of the `datasets`
    JSON loader: the metadata file has inconsistent per-record schemas (e.g. an
    `author` field that's sometimes a struct, sometimes null) that break the
    latter's Arrow-based schema casting.
    """
    fs = HfFileSystem()
    path = (
        "datasets/McAuley-Lab/Amazon-Reviews-2023/"
        f"raw/meta_categories/meta_{review_category}.jsonl"
    )
    meta = {}
    remaining = set(parent_asins)
    with fs.open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            pa = rec.get("parent_asin")
            if pa in remaining:
                meta[pa] = {"title": rec.get("title"), "category": rec.get("main_category")}
                remaining.discard(pa)
                if not remaining:
                    break
    return meta

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

        parent_asins = set(df["parent_asin"].dropna())
        meta = load_product_meta(review_category, parent_asins)
        df["product_title"] = df["parent_asin"].map(lambda pa: meta.get(pa, {}).get("title"))
        df["category"] = df["parent_asin"].map(
            lambda pa: meta.get(pa, {}).get("category") or review_category
        )

        df.to_parquet(out_path, index=False)

        log.info("done | raw_seen=%d after_filter=%d kept=%d matched_meta=%d -> %s",
                 raw_seen, after_filter, len(df), len(meta), out_path)
        job_finish(job_id, records_in=raw_seen, records_out=len(df))
        return out_path

    except Exception as e:
        log.error("ingestion failed | %s", e)
        job_fail(job_id, str(e))
        raise

if __name__ == "__main__":
    ingest()