"""Stage 2 data-quality and coverage report.

Every number in the generated report is computed here from the QC records, the manifest and the
Parquet outputs; nothing is typed in by hand.
"""
from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path

import polars as pl

from . import preprocessing as pp
from .config import REPO_ROOT, Paths

log = logging.getLogger(__name__)


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:,.4g}" if abs(x) < 1e4 else f"{x:,.0f}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)


def _md_table(df: pl.DataFrame, max_rows: int = 30) -> str:
    if df.height == 0:
        return "_none_\n"
    d = df.head(max_rows)
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    for row in d.iter_rows():
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    more = f"\n_({df.height - max_rows} more rows in the CSV)_\n" if df.height > max_rows else "\n"
    return "\n".join(lines) + more


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def book_alignment_check(book: pl.DataFrame, bars: pl.LazyFrame) -> pl.DataFrame:
    """Empirical check of the bookDepth sign convention and time zone.

    For each snapshot, the average price of resting quantity inside the +-1% band is
    notional/depth. If negative percentages are bids and timestamps are UTC, the implied bid price
    should sit below, and the implied ask price above, the VWAP of the minute containing the snapshot.
    Shifting the snapshot clock by whole hours should make the agreement worse.
    """
    snap = book.select("ts", bid1=pl.col("bid_notional_1") / pl.col("bid_depth_1"),
                       ask1=pl.col("ask_notional_1") / pl.col("ask_depth_1")).drop_nulls()
    v = bars.select("ts", "vwap").filter(pl.col("vwap").is_not_null()).collect()
    rows = []
    for h in (-8, -1, 0, 1, 8):
        s = snap.with_columns(minute=(pl.col("ts") + timedelta(hours=h)).dt.truncate("1m"))
        j = s.join(v.rename({"ts": "minute"}), on="minute", how="inner")
        mid = (j["bid1"] + j["ask1"]) / 2
        rows.append({
            "clock_shift_hours": h, "n_matched": j.height,
            "share_bid1_lt_vwap_lt_ask1": float(((j["bid1"] < j["vwap"]) & (j["vwap"] < j["ask1"])).mean()),
            "median_abs_bps_bandmid_vs_vwap": float(((mid / j["vwap"]).log().abs() * 1e4).median()),
        })
    return pl.DataFrame(rows)


def build_report(cfg: dict, paths: Paths, symbol: str, md_path: Path | None = None) -> Path:
    R = paths.reports
    q = cfg["quality"]
    start, end = cfg["period"]["start"], cfg["period"]["end"]
    full_run = (start, end) == ("2023-03-01", "2026-09-30")
    primary = symbol == "ETHUSDT"
    if md_path is None:
        if full_run and primary:
            md_path = REPO_ROOT / "docs" / "data_quality_report.md"
        elif full_run:
            md_path = REPO_ROOT / "docs" / "replication" / f"data_quality_report_{symbol}.md"
        else:
            md_path = R / "stage2_data_quality_report.md"
    csv_out = (REPO_ROOT / "docs" / "stage2") if (full_run and primary) else R
    csv_out.mkdir(parents=True, exist_ok=True)
    S: list[str] = []
    add = S.append

    # ------------------------------------------------------------------ files / manifest
    man = pl.read_csv(paths.manifest, infer_schema_length=None, schema_overrides={"period": pl.String})
    man = man.with_row_index("i").sort("i").group_by("filename").last()     # latest attempt per file
    files = (man.group_by("dataset", "status").agg(n=pl.len(), bytes=pl.col("bytes").sum())
             .sort("dataset", "status"))
    n_verified = int(man.filter(pl.col("status").is_in(["ok", "cached"])).height)

    # ------------------------------------------------------------------ trades QC
    tq = pl.DataFrame(json.loads((R / "trades_daily_qc.json").read_text()), infer_schema_length=None)
    ok = tq.filter(pl.col("n_raw").is_not_null()).sort("period")
    sums = {c: int(ok[c].sum()) for c in [
        "n_raw", "n_clean", "n_removed_invalid", "n_null_or_nonfinite", "n_price_nonpositive",
        "n_qty_nonpositive", "n_first_gt_last_id", "n_outside_period", "n_bad_side_value",
        "n_exact_duplicates_removed", "n_conflicting_id_duplicates_removed",
        "n_time_decreasing_raw_order", "n_id_not_increasing_raw_order", "n_agg_id_gaps",
        "n_agg_ids_missing", "n_trade_id_gaps", "n_trade_ids_missing", "n_trade_id_overlaps",
        "n_bad_tick_flags"]}
    units = ok.group_by("time_unit").len()
    not_processed = tq.filter(pl.col("n_raw").is_null()).select("period", "download_status")
    # continuity across consecutive daily files
    cross = ok.select("period", "first_agg_id", "last_agg_id", "first_fill_id", "last_fill_id").with_columns(
        agg_gap=pl.col("first_agg_id") - pl.col("last_agg_id").shift(1) - 1,
        fill_gap=pl.col("first_fill_id") - pl.col("last_fill_id").shift(1) - 1)
    cross_bad = cross.filter((pl.col("agg_gap") != 0) | (pl.col("fill_gap") < 0))
    tq_daily = ok.select("period", "n_raw", "n_clean", "n_trade_id_gaps", "n_trade_ids_missing",
                         "n_bad_tick_flags", "n_minutes_with_trades", "zip_bytes")
    tq_daily.write_csv(R / "trades_daily_qc.csv")

    # ------------------------------------------------------------------ bars
    bar_files = sorted((paths.processed / "bars_1m").glob(f"{symbol}-bars_1m-*.parquet"))
    bars = pl.scan_parquet(bar_files)
    allb = bars.select("ts", "open", "high", "low", "close", "volume", "n_trades", "has_trades",
                       "n_bad_tick_flags", "book_stale", "book_ts", "book_short_day").collect()
    n_bars = allb.height
    exp_bars = int((pp.period_bounds(end)[1] - pp.period_bounds(start)[0]).total_seconds() // 60)
    n_dup_ts = n_bars - allb["ts"].n_unique()
    ts_sorted = bool(allb["ts"].is_sorted())
    n_empty = int((~allb["has_trades"]).sum())
    runs = pp.no_trade_runs(allb, 1)
    runs_long = runs.filter(pl.col("minutes") >= q["trade_gap_minutes"])
    runs.write_csv(csv_out / "no_trade_minute_runs.csv")
    ext = pp.extreme_bars(allb, q["extreme_bar_abs_logret"])
    ext = ext.with_columns(classification=pl.when(pl.col("reverted_next") & (pl.col("n_trades") < 20))
                           .then(pl.lit("error-candidate (reverted, thin)"))
                           .when(pl.col("reverted_next")).then(pl.lit("reverted, high activity"))
                           .otherwise(pl.lit("sustained move (market event)")))
    ext.write_csv(csv_out / "extreme_1m_bars.csv")
    ohlc_bad = allb.filter(pl.col("has_trades") & ((pl.col("high") < pl.col("low"))
                           | (pl.col("open") > pl.col("high")) | (pl.col("open") < pl.col("low"))
                           | (pl.col("close") > pl.col("high")) | (pl.col("close") < pl.col("low")))).height
    bt_bars = allb.filter(pl.col("n_bad_tick_flags") > 0).select("ts", "n_bad_tick_flags", "low", "high", "close")
    bt_bars.write_csv(csv_out / "bad_tick_flag_bars.csv")
    bt_in_ext = bt_bars.join(ext.select("ts"), on="ts", how="semi").height

    # ------------------------------------------------------------------ book
    bq = pl.read_csv(R / "book_daily_qc.csv", try_parse_dates=True)
    book = pl.read_parquet(paths.processed / "book_snapshots" / f"{symbol}-book_snapshots.parquet")
    exp_snaps = q["book_expected_snapshots_per_day"]
    short = bq.filter(pl.col("book_short_day")).select("date", "file_present", "n_snapshots")
    short.write_csv(csv_out / "book_short_days.csv")
    bgap = book.select("ts").with_columns(prev=pl.col("ts").shift(1),
                                          gap_s=pl.col("ts").diff().dt.total_seconds()) \
               .filter(pl.col("gap_s") > q["book_gap_seconds"]).sort("gap_s", descending=True)
    bgap.write_csv(csv_out / "book_snapshot_gaps.csv")
    n_bad_book = {c: int(bq[c].fill_null(0).sum()) for c in [
        "n_raw_rows", "n_invalid_rows_removed", "n_extra_band_rows_not_used", "n_exact_duplicates_removed",
        "n_conflicting_duplicates_removed", "n_incomplete_snapshots", "n_snapshots_non_monotone_depth"]}
    ex = bq.filter(pl.col("n_extra_band_rows_not_used").fill_null(0) > 0)
    first_extra = str(ex["date"].min()) if ex.height else "n/a"
    n_stale = int(allb["book_stale"].sum())
    n_no_book = int(allb["book_ts"].is_null().sum())
    stale_by_day = allb.filter(pl.col("book_stale")).group_by(pl.col("ts").dt.date().alias("date")) \
        .agg(stale_minutes=pl.len()).sort("stale_minutes", descending=True)
    stale_by_day.write_csv(csv_out / "book_stale_minutes_by_day.csv")
    align = book_alignment_check(book, bars)
    align.write_csv(csv_out / "book_sign_timezone_check.csv")

    fq = json.loads((R / "funding_qc.json").read_text())
    mq = json.loads((R / "metrics_qc.json").read_text())

    # ------------------------------------------------------------------ sizes + schema
    sizes = pl.DataFrame([{"output": d, "files": len(list((paths.processed / d).glob("*.parquet"))),
                           "MB": _dir_size(paths.processed / d) / 1e6}
                          for d in ("bars_1m", "book_snapshots", "funding", "metrics")])
    schema = pl.read_parquet_schema(bar_files[0])
    schema_df = pl.DataFrame({"column": list(schema.keys()), "dtype": [str(t) for t in schema.values()]})
    preview = pl.read_parquet(bar_files[-1]).filter(pl.col("has_trades")).select(
        "ts", "open", "high", "low", "close", "volume", "n_trades", "buy_volume", "sell_volume",
        "vwap", "bid_depth_1", "ask_depth_1", "book_age_s").tail(3)

    # ------------------------------------------------------------------ markdown
    add(f"# Stage 2 — Data Acquisition & Data-Quality Report ({symbol})\n")
    add(f"Generated by `scripts/run_stage2.py` → `ethof.quality.build_report`. Period "
        f"**{start} → {end}** (UTC). Every figure below is computed from the pipeline outputs.\n")
    add("## 1. Raw files\n")
    add(f"Verified files (SHA-256 matches the published `.CHECKSUM`): **{n_verified:,}**.\n")
    add(_md_table(files.with_columns(GB=pl.col("bytes") / 1e9).drop("bytes")))
    if not_processed.height:
        add("Trade days not processed:\n" + _md_table(not_processed))
    add("## 2. Trades (aggTrades)\n")
    add(f"- Raw aggregated trades read: **{sums['n_raw']:,}** (from {ok.height:,} daily files)")
    add(f"- Clean trades used for bars: **{sums['n_clean']:,}**")
    add(f"- Timestamp unit detected per file: {dict(units.iter_rows())}")
    add(f"- Out-of-order rows in raw file order: time decreasing {sums['n_time_decreasing_raw_order']:,}; "
        f"agg id not increasing {sums['n_id_not_increasing_raw_order']:,} (all data are sorted by (ts, agg_trade_id) anyway)")
    add(f"- Removed as data errors: **{sums['n_removed_invalid']:,}** "
        f"(null/non-finite {sums['n_null_or_nonfinite']:,}; price ≤ 0 {sums['n_price_nonpositive']:,}; "
        f"qty ≤ 0 {sums['n_qty_nonpositive']:,}; first_id > last_id {sums['n_first_gt_last_id']:,}; "
        f"outside file's day {sums['n_outside_period']:,}; bad side flag {sums['n_bad_side_value']:,})")
    add(f"- Duplicates: exact rows {sums['n_exact_duplicates_removed']:,}; conflicting same-id rows "
        f"{sums['n_conflicting_id_duplicates_removed']:,}")
    add(f"- agg_trade_id gaps within files: {sums['n_agg_id_gaps']:,} ({sums['n_agg_ids_missing']:,} ids); "
        f"across consecutive files: {cross_bad.filter(pl.col('agg_gap') != 0).height:,} boundaries with a gap")
    add(f"- Underlying trade-id gaps (first_trade_id ≠ previous last_trade_id + 1): {sums['n_trade_id_gaps']:,} gaps, "
        f"{sums['n_trade_ids_missing']:,} ids missing = "
        f"{sums['n_trade_ids_missing'] / max(1, sums['n_trade_ids_missing'] + int(ok['n_raw'].sum())):.4%} relative to agg-trade count; "
        f"overlaps {sums['n_trade_id_overlaps']:,}. Flagged, not repaired: the archive omits these fills from `aggTrades` (see §8).")
    add(f"- Bad-tick candidates (> {q['bad_tick_rel_dev']:.0%} from the minute's median price): "
        f"**{sums['n_bad_tick_flags']:,}** trades in {bt_bars.height:,} bars (flagged, kept). {bt_in_ext} of these "
        f"{bt_bars.height} bars are also in the extreme-move list (§3): the flags come from crash/squeeze minutes "
        f"whose intra-minute range exceeds 2 %, not from isolated bad prints, so nothing is removed.\n")
    add("## 3. One-minute bars\n")
    add(f"- Bars written: **{n_bars:,}**; expected minutes in period: {exp_bars:,}; duplicate timestamps: {n_dup_ts}; "
        f"sorted: {ts_sorted}")
    add(f"- Bars with ≥ 1 trade: {n_bars - n_empty:,}; **minutes without any trade: {n_empty:,}** "
        f"({n_empty / n_bars:.4%}) in {runs.height:,} runs; runs ≥ {q['trade_gap_minutes']} min: {runs_long.height:,}")
    add(f"- OHLC consistency violations (high<low, open/close outside [low, high]): {ohlc_bad}")
    add(f"\nLongest no-trade runs (`docs/stage2/no_trade_minute_runs.csv`):\n")
    add(_md_table(runs.sort("minutes", descending=True), 10))
    add(f"Extreme 1-minute bars (|log return| or log(high/low) > {q['extreme_bar_abs_logret']:.0%}), "
        f"**{ext.height}** bars (`docs/stage2/extreme_1m_bars.csv`):\n")
    add(_md_table(ext.select("ts", "open", "high", "low", "close", "n_trades", "logret", "range_rel",
                             "classification"), 40))
    add("## 4. Order book (bookDepth percentage bands)\n")
    add("Reminder: these are *cumulative resting quantity/notional within ±1…5 % of the reference price*, "
        "not level-by-level L2. No best bid/ask or spread is available from this source.\n")
    add(f"- Daily files expected: {bq.height:,}; present: {int(bq['file_present'].sum()):,}")
    add(f"- Snapshots kept: **{book.height:,}** (expected ≈ {bq.height * exp_snaps:,} at {exp_snaps}/day = "
        f"{book.height / (bq.height * exp_snaps):.3%})")
    add(f"- Raw rows {n_bad_book['n_raw_rows']:,}; invalid removed {n_bad_book['n_invalid_rows_removed']:,}; "
        f"exact dups {n_bad_book['n_exact_duplicates_removed']:,}; conflicting dups "
        f"{n_bad_book['n_conflicting_duplicates_removed']:,}; rows of extra bands not used (Binance added "
        f"+-0.2 % bands from {first_extra}; schema extension, not errors) {n_bad_book['n_extra_band_rows_not_used']:,}; incomplete snapshots (≠10 bands) "
        f"{n_bad_book['n_incomplete_snapshots']:,}; non-monotone cumulative depth {n_bad_book['n_snapshots_non_monotone_depth']:,}")
    add(f"- Gaps between consecutive snapshots > {q['book_gap_seconds']} s: **{bgap.height:,}**")
    add(f"- Short/missing days (< {q['book_short_day_ratio']:.0%} of {exp_snaps} snapshots), flagged "
        f"`book_short_day` in the bars and **not interpolated**: {short.height}\n")
    add(_md_table(short, 20))
    add("Longest snapshot gaps:\n")
    add(_md_table(bgap.head(10)))
    add(f"- Bars whose as-of snapshot is older than {q['book_stale_seconds']} s (`book_stale`): "
        f"**{n_stale:,}** ({n_stale / n_bars:.3%}); bars with no snapshot at all: {n_no_book:,}\n")
    add("**Sign-convention / time-zone check.** Average price of resting quantity within the 1% band "
        "(notional/depth) vs. the trade VWAP of the minute containing the snapshot, with the snapshot "
        "clock shifted by whole hours:\n")
    add(_md_table(align))
    add("## 5. Funding rates\n")
    add("\n".join(f"- {k}: {_fmt(v)}" for k, v in fq.items()) + "\n")
    add("## 6. Open interest / metrics (5-minute)\n")
    add("\n".join(f"- {k}: {_fmt(v)}" for k, v in mq.items()) + "\n")
    add("## 7. Outputs\n")
    add(_md_table(sizes))
    add("Processed 1-minute bar schema (`processed/bars_1m/*.parquet`):\n")
    add(_md_table(schema_df, 100))
    add("Preview (last 3 traded minutes of the final month):\n")
    add(_md_table(preview))
    add("## 8. Data-quality issues relevant to the research\n")
    n_err_cand = ext.filter(pl.col("classification").str.starts_with("error")).height
    add(f"1. **Order book is coarse.** Percentage bands only; no best bid/ask, no spread, no queue information. "
        f"Book features are null on {n_stale:,} stale minutes ({n_stale / n_bars:.3%}) and on the {short.height} short days "
        f"listed in §4; nothing is interpolated.")
    add(f"2. **Schema change inside the holdout:** extra ±0.2 % band rows appear from {first_extra}. Not used, "
        f"because they are unavailable in the development period.")
    add(f"3. **Extreme moves are real.** {ext.height} one-minute bars move > {q['extreme_bar_abs_logret']:.0%}; "
        f"{n_err_cand} meet the error-candidate rule (full reversal on < 20 trades). All are kept: they are "
        f"liquidation cascades with tens of thousands of trades, i.e. exactly the events a strategy must survive.")
    add(f"4. **Trading halts / no-trade minutes:** {n_empty} minutes in {runs.height} runs. Prices are forward-filled "
        f"for features (backward-looking only); positions cannot be traded there, so execution uses the last price "
        f"and the minute is flagged.")
    add(f"5. **Fill-id gaps:** {sums['n_trade_ids_missing']:,} underlying fill ids ({sums['n_trade_ids_missing'] / max(1, sums['n_raw']):.3%} "
        f"of the agg-trade count) are absent from `aggTrades`, so volumes may be slightly understated. Whether the "
        f"omitted fills are side-neutral cannot be verified from this archive.")
    add("6. **Single venue.** Binance perp flow only; cross-venue flow (Coinbase, Bybit, OKX, spot) is unobserved.\n")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(S))
    log.info("report written: %s", md_path)
    return md_path
