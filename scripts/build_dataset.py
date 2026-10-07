#!/usr/bin/env python3
"""Stage 3: research dataset = 1-minute bars + causal features + forward-return targets.

    python scripts/build_dataset.py [--symbol ETHUSDT] [--data-dir DIR]

Writes data/processed/research/<SYMBOL>-research.parquet and reports/tables/feature_summary_dev.csv.
The large-trade threshold is chosen by a rule fixed in advance and applied only to the initial training window
(2023-03-01 .. 2024-02-29): the size bucket closest to the median per-minute 95th-percentile aggTrade size.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ethof import features as F  # noqa: E402
from ethof import targets as T  # noqa: E402
from ethof.config import Paths, load_config, setup_logging  # noqa: E402

INITIAL_TRAIN_END = "2024-03-01"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol")
    ap.add_argument("--data-dir")
    a = ap.parse_args()
    if a.data_dir:
        os.environ["ETHOF_DATA_DIR"] = a.data_dir
    cfg = load_config()
    sym = a.symbol or cfg["source"]["symbol"]
    paths = Paths(cfg["data_dir"]).make()
    setup_logging(paths.logs / f"build_dataset_{sym}.log")
    log = logging.getLogger("build_dataset")

    bars = pl.read_parquet(sorted((paths.processed / "bars_1m").glob(f"{sym}-bars_1m-*.parquet"))).sort("ts")
    funding = pl.read_parquet(paths.processed / "funding" / f"{sym}-funding.parquet")
    metrics = pl.read_parquet(paths.processed / "metrics" / f"{sym}-metrics.parquet")
    log.info("bars %s, funding %s, metrics %s", bars.shape, funding.shape, metrics.shape)

    # large-trade threshold rule (initial training window only)
    init = bars.filter(pl.col("ts") < pl.lit(INITIAL_TRAIN_END).str.to_datetime(time_zone="UTC"))
    med_p95 = float(init["trade_size_p95"].median())
    buckets = cfg["aggregation"]["size_buckets_eth"]
    large = min(buckets, key=lambda k: abs(k - med_p95))
    log.info("median per-minute p95 trade size (initial window) = %.3f ETH -> large-trade bucket >= %g ETH", med_p95, large)

    df = F.build_features(bars, funding, metrics, large_trade_eth=large)
    df = T.add_targets(df, neutral_band=cfg["research"]["neutral_band"])
    keep = (["ts", "px", "exec_px", "exec_px_from_trades", "has_trades", "book_stale", "book_short_day",
             "volume", "n_trades", "ofi_1"] + sorted(set(sum(F.GROUPS.values(), [])) - {"ofi_1"})
            + [c for c in df.columns if c.startswith(("fwd_ret_", "up_", "cls3_", "label_end_"))])
    df = df.select(keep)
    out = paths.processed / "research" / f"{sym}-research.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="zstd")
    meta = {"symbol": sym, "rows": df.height, "columns": df.width, "first_ts": str(df["ts"].min()),
            "last_ts": str(df["ts"].max()), "large_trade_eth": large, "median_p95_trade_size_initial": med_p95,
            "bytes": out.stat().st_size}
    (paths.reports / f"research_dataset_{sym}.json").write_text(json.dumps(meta, indent=1))
    log.info("research dataset: %s", meta)


if __name__ == "__main__":
    main()
