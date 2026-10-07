"""Stage 2 orchestration: download -> verify -> clean -> aggregate -> Parquet, with QC records.

Layout under cfg['data_dir'] (git-ignored):

    raw/<dataset>/<SYMBOL>/<file>.zip          raw archive files (aggTrades zips deleted after use
                                               unless download.keep_raw_aggtrades is true)
    interim/trades_1m_daily/<SYMBOL>-<day>.parquet + .qc.json   per-day bars and QC (resumable cache)
    processed/bars_1m/<SYMBOL>-bars_1m-<YYYY-MM>.parquet         final 1-minute bars (+ as-of book)
    processed/book_snapshots/<SYMBOL>-book_snapshots.parquet     cleaned 30-s depth-band snapshots
    processed/funding/<SYMBOL>-funding.parquet
    processed/metrics/<SYMBOL>-metrics.parquet
    reports/                                                     QC tables (CSV/JSON)
    manifest.csv                                                 every file: url, bytes, sha256, status
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import polars as pl

from . import data as dl
from . import preprocessing as pp
from .config import Paths

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- small datasets

def fetch_small(cfg: dict, paths: Paths, dataset: str, symbol: str) -> list[dl.DownloadRecord]:
    files = dl.remote_files(cfg, dataset, symbol)
    log.info("%s: %d files to fetch/verify", dataset, len(files))
    recs = dl.download_many(files, paths.raw, workers=cfg["download"]["max_workers_small"],
                            timeout=cfg["download"]["timeout_s"], retries=cfg["download"]["retries"])
    dl.append_manifest(paths.manifest, recs)
    by = {}
    for r in recs:
        by[r.status] = by.get(r.status, 0) + 1
    log.info("%s download status: %s", dataset, by)
    return recs


def build_book(cfg: dict, paths: Paths, symbol: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Clean every daily bookDepth file -> (wide snapshots, per-day QC table)."""
    frames, qcs = [], []
    for rf in dl.remote_files(cfg, "bookDepth", symbol):
        p = rf.local_path(paths.raw)
        if not p.exists():
            qcs.append({"period": rf.period, "file_present": False, "n_snapshots": 0})
            continue
        wide, qc = pp.clean_bookdepth(pp.read_bookdepth(p), rf.period)
        qc["file_present"] = True
        frames.append(wide)
        qcs.append(qc)
    book = pl.concat(frames).sort("ts")
    n0 = book.height
    book = book.unique(subset=["ts"], keep="first", maintain_order=True)  # cross-file overlap guard
    qc_df = pl.DataFrame(qcs, infer_schema_length=None)
    q = cfg["quality"]
    qc_df = qc_df.with_columns(
        date=pl.col("period").str.to_date(),
        book_short_day=pl.col("n_snapshots") < q["book_expected_snapshots_per_day"] * q["book_short_day_ratio"],
    )
    log.info("book: %d snapshots (%d cross-file duplicates dropped), %d short/missing days",
             book.height, n0 - book.height, int(qc_df["book_short_day"].sum()))
    out = paths.processed / "book_snapshots" / f"{symbol}-book_snapshots.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    book.write_parquet(out, compression="zstd")
    qc_df.write_csv(paths.reports / "book_daily_qc.csv")
    return book, qc_df


def build_funding(cfg: dict, paths: Paths, symbol: str) -> dict:
    frames = [pp.read_funding(rf.local_path(paths.raw)) for rf in dl.remote_files(cfg, "fundingRate", symbol)
              if rf.local_path(paths.raw).exists()]
    f = pl.concat(frames)
    start, _ = pp.period_bounds(cfg["period"]["start"])
    _, end = pp.period_bounds(cfg["period"]["end"])
    f = f.filter((pl.col("ts") >= start) & (pl.col("ts") < end))
    n0 = f.height
    f = f.unique(keep="first", maintain_order=True).sort("ts")
    gap_h = f["ts"].diff().dt.total_seconds() / 3600
    # expected spacing = previous row's funding interval; allow +-5 min jitter in calc_time
    irregular = f.with_columns(gap_h=gap_h, exp=pl.col("funding_interval_hours").shift(1)) \
                 .filter((pl.col("gap_h") - pl.col("exp")).abs() > 5 / 60)
    qc = {"n_rows": f.height, "n_duplicates_removed": n0 - f.height,
          "interval_hours_values": sorted(f["funding_interval_hours"].unique().to_list()),
          "n_irregular_spacing": irregular.height,
          "rate_min": f["last_funding_rate"].min(), "rate_max": f["last_funding_rate"].max(),
          "rate_mean": f["last_funding_rate"].mean(),
          "first_ts": str(f["ts"].min()), "last_ts": str(f["ts"].max())}
    irregular.write_csv(paths.reports / "funding_irregular_spacing.csv")
    out = paths.processed / "funding" / f"{symbol}-funding.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    f.write_parquet(out, compression="zstd")
    return qc


def build_metrics(cfg: dict, paths: Paths, symbol: str) -> dict:
    files = dl.remote_files(cfg, "metrics", symbol)
    frames = [pp.read_metrics(rf.local_path(paths.raw)) for rf in files if rf.local_path(paths.raw).exists()]
    m = pl.concat(frames)
    n_bad_ts = int(m["ts"].is_null().sum())
    n0 = m.height
    m = m.filter(pl.col("ts").is_not_null()).unique(keep="first", maintain_order=True)
    n_exact = n0 - n_bad_ts - m.height
    m = m.unique(subset=["ts"], keep="first", maintain_order=True).sort("ts")
    gaps = m.with_columns(gap_min=pl.col("ts").diff().dt.total_seconds() / 60).filter(pl.col("gap_min") > 5)
    qc = {"n_files": len(frames), "n_rows": m.height, "n_unparseable_ts": n_bad_ts,
          "n_exact_duplicates_removed": n_exact,
          "n_expected_rows": len(files) * 288,
          "n_gaps_gt_5min": gaps.height, "n_minutes_in_gaps": float((gaps["gap_min"] - 5).sum() or 0),
          "n_nonpositive_open_interest": int((m["sum_open_interest"] <= 0).sum()),
          "n_null_open_interest": int(m["sum_open_interest"].is_null().sum())}
    gaps.select("ts", "gap_min").write_csv(paths.reports / "metrics_gaps.csv")
    out = paths.processed / "metrics" / f"{symbol}-metrics.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.drop("create_time").write_parquet(out, compression="zstd")
    return qc


# --------------------------------------------------------------------------- trades (parallel per day)

def _process_trade_day(rf: dl.RemoteFile, raw_dir: Path, interim: Path, bad_tick: float,
                       buckets: list[float], keep_raw: bool, timeout: float, retries: int) -> dict:
    """Worker: one day of aggTrades -> interim 1-minute bars + QC json. Returns manifest record + qc."""
    logging.basicConfig(level=logging.INFO)
    rec = dl.download(rf, raw_dir, timeout=timeout, retries=retries)
    out = {"record": asdict(rec), "qc": {"period": rf.period, "download_status": rec.status}}
    if rec.status not in ("ok", "cached"):
        return out
    path = rf.local_path(raw_dir)
    trades, qc = pp.clean_aggtrades(pp.read_aggtrades(path), rf.period, bad_tick)
    bars = pp.aggregate_1min(trades, rf.period, buckets)
    qc.update(download_status=rec.status, sha256=rec.sha256_actual, zip_bytes=rec.bytes,
              n_minutes=bars.height, n_minutes_with_trades=int(bars["has_trades"].sum()))
    stem = interim / f"{rf.symbol}-{rf.period}"
    bars.write_parquet(stem.with_suffix(".parquet"), compression="zstd")
    stem.with_suffix(".qc.json").write_text(json.dumps(qc))
    if not keep_raw:
        path.unlink()
    out["qc"] = qc
    return out


def build_trade_days(cfg: dict, paths: Paths, symbol: str) -> list[dict]:
    interim = paths.root / "interim" / "trades_1m_daily"
    interim.mkdir(parents=True, exist_ok=True)
    files = dl.remote_files(cfg, "aggTrades", symbol)
    todo, qcs = [], []
    for rf in files:
        qcp = interim / f"{rf.symbol}-{rf.period}.qc.json"
        if qcp.exists() and (interim / f"{rf.symbol}-{rf.period}.parquet").exists():
            qcs.append(json.loads(qcp.read_text()))     # resumable: already processed
        else:
            todo.append(rf)
    log.info("aggTrades: %d days total, %d cached, %d to process", len(files), len(qcs), len(todo))
    d = cfg["download"]
    args = (paths.raw, interim, cfg["quality"]["bad_tick_rel_dev"],
            cfg["aggregation"]["size_buckets_eth"], d["keep_raw_aggtrades"], d["timeout_s"], d["retries"])
    with ProcessPoolExecutor(max_workers=d["trade_workers"], mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_process_trade_day, rf, *args): rf for rf in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rf = futs[fut]
            try:
                res = fut.result()
            except Exception:
                log.exception("trade day %s failed", rf.period)
                res = {"record": asdict(dl.DownloadRecord(rf.dataset, rf.period, rf.filename, rf.url,
                                                          "error", message="processing failed")),
                       "qc": {"period": rf.period, "download_status": "processing_error"}}
            dl.append_manifest(paths.manifest, [dl.DownloadRecord(**res["record"])])
            qcs.append(res["qc"])
            if i % 25 == 0 or i == len(todo):
                log.info("aggTrades progress: %d/%d (last %s, %s rows)", i, len(todo), rf.period,
                         res["qc"].get("n_raw"))
    return sorted(qcs, key=lambda q: q["period"])


def assemble_months(cfg: dict, paths: Paths, symbol: str, book: pl.DataFrame, book_qc: pl.DataFrame) -> list[Path]:
    """Concatenate daily bars per month, attach as-of book snapshot and data-quality flags, write Parquet."""
    interim = paths.root / "interim" / "trades_1m_daily"
    out_dir = paths.processed / "bars_1m"
    out_dir.mkdir(parents=True, exist_ok=True)
    short = book_qc.select("date", "book_short_day", book_day_snapshots="n_snapshots")
    written = []
    for month in dl.periods(cfg["period"]["start"], cfg["period"]["end"], "monthly"):
        days = sorted(interim.glob(f"{symbol}-{month}-??.parquet"))
        if not days:
            log.warning("no trade bars for %s", month)
            continue
        bars = pl.concat([pl.read_parquet(p) for p in days], how="vertical").sort("ts")
        bars = pp.asof_book_to_bars(bars, book, cfg["quality"]["book_stale_seconds"])
        bars = bars.with_columns(date=pl.col("ts").dt.date()).join(short, on="date", how="left") \
                   .with_columns(pl.col("book_short_day").fill_null(True),
                                 pl.col("book_day_snapshots").fill_null(0)).drop("date")
        p = out_dir / f"{symbol}-bars_1m-{month}.parquet"
        bars.write_parquet(p, compression="zstd")
        written.append(p)
    log.info("wrote %d monthly bar files", len(written))
    return written
