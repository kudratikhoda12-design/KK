#!/usr/bin/env python3
"""Stage 2: data acquisition + data-quality pipeline.

Usage (from the repo root):
    python scripts/run_stage2.py                       # full period from config/config.yaml
    python scripts/run_stage2.py --start 2024-06-01 --end 2024-06-03 --data-dir /tmp/ethof_test
    python scripts/run_stage2.py --steps report        # rebuild the report only

Every step is idempotent: verified files and processed days are reused on re-runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ethof import pipeline, quality  # noqa: E402
from ethof.config import Paths, load_config, setup_logging  # noqa: E402

STEPS = ["download_small", "book", "funding", "metrics", "trades", "assemble", "report"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--symbol")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--data-dir")
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--report-md", default=None, help="markdown report path (default docs/stage2_data_quality_report.md for the full run)")
    a = ap.parse_args()

    if a.data_dir:
        os.environ["ETHOF_DATA_DIR"] = a.data_dir
    cfg = load_config(a.config)
    if a.start: cfg["period"]["start"] = a.start
    if a.end: cfg["period"]["end"] = a.end
    symbol = a.symbol or cfg["source"]["symbol"]
    paths = Paths(cfg["data_dir"]).make()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    setup_logging(paths.logs / f"stage2_{symbol}_{stamp}.log")
    log = logging.getLogger("stage2")
    log.info("config: period %s..%s symbol %s data_dir %s", cfg["period"]["start"], cfg["period"]["end"],
             symbol, paths.root)
    steps = a.steps.split(",")
    t0 = time.time()

    if "download_small" in steps:
        for ds in ("bookDepth", "fundingRate", "metrics"):
            if cfg["datasets"][ds]["enabled"]:
                pipeline.fetch_small(cfg, paths, ds, symbol)
    book = book_qc = None
    if "book" in steps:
        book, book_qc = pipeline.build_book(cfg, paths, symbol)
    if "funding" in steps:
        (paths.reports / "funding_qc.json").write_text(json.dumps(pipeline.build_funding(cfg, paths, symbol), indent=1, default=str))
    if "metrics" in steps:
        (paths.reports / "metrics_qc.json").write_text(json.dumps(pipeline.build_metrics(cfg, paths, symbol), indent=1, default=str))
    if "trades" in steps:
        qcs = pipeline.build_trade_days(cfg, paths, symbol)
        (paths.reports / "trades_daily_qc.json").write_text(json.dumps(qcs))
    if "assemble" in steps:
        if book is None:
            import polars as pl
            book = pl.read_parquet(paths.processed / "book_snapshots" / f"{symbol}-book_snapshots.parquet")
            book_qc = pl.read_csv(paths.reports / "book_daily_qc.csv", try_parse_dates=True)
        pipeline.assemble_months(cfg, paths, symbol, book, book_qc)
    if "report" in steps:
        md = Path(a.report_md) if a.report_md else None
        quality.build_report(cfg, paths, symbol, md)
    log.info("done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
