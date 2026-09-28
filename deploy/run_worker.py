"""Long-running pipeline worker: reader -> ingest -> embed -> store, in a loop, with
telemetry active throughout. This is the actual thing Prometheus scrapes and Grafana
visualizes -- a one-shot script has nothing to scrape once it exits.

Usage:
    python deploy/run_worker.py [--batch-size N] [--sleep-seconds N] [--prometheus-port N]

Each iteration reads the next BATCH_SIZE raw lines (advancing a persistent line offset
so successive iterations see new data, not the same lines re-skipped by the ledger),
runs them through ingest -> embed -> store, then sleeps before the next iteration.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import psycopg

SRC = Path(__file__).resolve().parent.parent / "src"
for module_dir in ("observability", "ledger", "ingestion", "embedding", "vectorstore", "inject"):
    sys.path.insert(0, str(SRC / module_dir))

from telemetry import setup_telemetry  # noqa: E402
from ledger import Ledger  # noqa: E402
from reader import stream_reviews  # noqa: E402
from ingest import ingest_records, STAGE as INGEST_STAGE  # noqa: E402
from embed import embed_batches  # noqa: E402
from store import load_chunks  # noqa: E402
import faults  # noqa: E402

DSN = "dbname=self_healing"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=10.0)
    parser.add_argument("--prometheus-port", type=int, default=9464)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="stop after N iterations (default: run forever)",
    )
    parser.add_argument(
        "--fail-every",
        type=int,
        default=0,
        help=(
            "arm a rate_limit fault on the ingest stage every N iterations "
            "(0 = never; for demoing the failure-rate metric with real, non-fabricated data)"
        ),
    )
    args = parser.parse_args()

    setup_telemetry(prometheus_port=args.prometheus_port)
    print(
        f"telemetry live; Prometheus metrics at http://localhost:{args.prometheus_port}/metrics",
        flush=True,
    )

    ledger = Ledger(DSN)
    conn = psycopg.connect(DSN)

    offset = 0
    iteration = 0
    try:
        while args.iterations is None or iteration < args.iterations:
            if args.fail_every and iteration % args.fail_every == 0:
                faults.arm(INGEST_STAGE, "rate_limit")

            raw = stream_reviews(skip=offset, limit=args.batch_size)
            cleaned = ingest_records(ledger, raw)
            chunks = list(embed_batches(ledger, cleaned, batch_size=args.batch_size))
            inserted = load_chunks(conn, chunks)

            offset += args.batch_size
            iteration += 1
            print(
                f"[iter {iteration}] lines {offset - args.batch_size}-{offset}: "
                f"{len(chunks)} embedded, {inserted} inserted",
                flush=True,
            )
            time.sleep(args.sleep_seconds)
    finally:
        ledger.close()
        conn.close()


if __name__ == "__main__":
    main()
