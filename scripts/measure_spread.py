#!/usr/bin/env python3
"""Measure the quoted bid-ask spread of ETHUSDT perpetual from Binance `bookTicker` files.

bookTicker (tick-by-tick best bid/ask) is archived only until 2024-03-30 and each day is ~3.4 GB
uncompressed, so we measure a few development-period days (one weekday per period) and use the result
as the half-spread input of the cost model. Files are SHA-256 verified, processed, then deleted.

    python scripts/measure_spread.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ethof import data as dl  # noqa: E402
from ethof.config import Paths, load_config  # noqa: E402

DAYS = ["2023-06-14", "2023-11-15", "2024-03-13"]


def main() -> None:
    cfg = load_config()
    paths = Paths(cfg["data_dir"]).make()
    rows = []
    for day in DAYS:
        rf = dl.RemoteFile("bookTicker", "daily", "ETHUSDT", day, cfg["source"]["base_url"])
        rec = dl.download(rf, paths.raw)
        dl.append_manifest(paths.manifest, [rec])
        if rec.status not in ("ok", "cached"):
            rows.append({"day": day, "status": rec.status})
            continue
        zp = rf.local_path(paths.raw)
        with tempfile.TemporaryDirectory(dir=paths.root) as td:
            with zipfile.ZipFile(zp) as zf:
                name = zf.namelist()[0]
                zf.extract(name, td)
            lf = pl.scan_csv(Path(td) / name, schema_overrides={"best_bid_price": pl.Float64,
                                                                 "best_ask_price": pl.Float64})
            q = lf.select(
                sec=(pl.col("transaction_time") // 1000),
                bid=pl.col("best_bid_price"), ask=pl.col("best_ask_price"),
            ).with_columns(spr=pl.col("ask") - pl.col("bid"),
                           spr_bps=(pl.col("ask") - pl.col("bid")) / ((pl.col("ask") + pl.col("bid")) / 2) * 1e4)
            # one observation per second (last quote in the second) => time-weighted at 1-s resolution
            per_sec = q.group_by("sec").agg(pl.col("spr").last(), pl.col("spr_bps").last()).collect(engine="streaming")
            stats = per_sec.select(
                n_seconds=pl.len(),
                share_crossed_or_locked=(pl.col("spr") <= 0).mean(),
                share_one_tick=((pl.col("spr") - 0.01).abs() < 1e-9).mean(),
                median_spread_usd=pl.col("spr").median(),
                mean_spread_bps=pl.col("spr_bps").mean(),
                median_spread_bps=pl.col("spr_bps").median(),
                p99_spread_bps=pl.col("spr_bps").quantile(0.99),
            ).row(0, named=True)
            rows.append({"day": day, "status": "ok", "sha256": rec.sha256_actual, **stats})
        zp.unlink()
        print(rows[-1], flush=True)
    res = pl.DataFrame(rows)
    out = ROOT / "reports" / "tables" / "spread_measurement.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.write_csv(out)
    ok = res.filter(pl.col("status") == "ok")
    summary = {"days": ok["day"].to_list(), "mean_of_daily_mean_spread_bps": float(ok["mean_spread_bps"].mean()),
               "half_spread_bps_used": round(float(ok["mean_spread_bps"].mean()) / 2, 3)}
    (ROOT / "reports" / "tables" / "spread_summary.json").write_text(json.dumps(summary, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
