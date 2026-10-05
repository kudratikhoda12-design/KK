"""Data-health audit for Binance 1-minute klines.

``audit(df)`` runs every check on the *interim* (combined, uncleaned) table
and returns a dict of detail tables plus a one-page data-quality table.
``build_processed(df)`` applies ONLY the documented, rule-based treatments
and writes the regular 1-minute grid used for research.

Treatment rules (decided before seeing the data)
-----------------------------------------------
1. Exact duplicate rows             -> keep one copy (no information lost).
2. Conflicting duplicate timestamps -> minute set to missing (we cannot know which is right).
3. Impossible candles (high < low, high < open/close, low > open/close,
   non-positive price, VWAP outside [low, high])  -> minute set to missing.
4. Missing minutes                   -> stay missing (NaN) on the regular grid.
                                        Never forward-filled; features require
                                        >= 90% window coverage.
5. Extreme but internally consistent candles -> KEPT and flagged. Crashes and
   spikes are real risk; deleting them would make the strategy look safer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

PRICE_COLS = ["open", "high", "low", "close"]
DATA_COLS = PRICE_COLS + ["volume", "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote"]
ONE_MIN = pd.Timedelta(minutes=1)


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------
def file_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-file rows, timestamp unit, header, date range and expected rows."""
    g = df.groupby("source_file", observed=True)
    out = g.agg(rows=("open_time", "size"), ts_unit=("ts_unit", "first"),
                has_header=("has_header", "first"), first=("open_time", "min"),
                last=("open_time", "max"))
    # Expected rows for a monthly file = minutes in that month; for a daily file = 1440
    period = out.index.str.extract(r"-(\d{4}-\d{2}(?:-\d{2})?)\.zip$", expand=False)
    out["expected_rows"] = [
        1440 if len(p) == 10 else pd.Period(p).days_in_month * 1440 for p in period]
    out["coverage"] = out["rows"] / out["expected_rows"]
    # Ordering inside the original file
    ordered = df.sort_values(["source_file", "row_in_file"]).groupby("source_file", observed=True)[
        "open_time"].apply(lambda s: bool(s.is_monotonic_increasing))
    out["ordered_in_file"] = ordered
    return out.reset_index()


def timestamp_checks(df: pd.DataFrame, fs: pd.DataFrame) -> dict:
    t = df["open_time"]
    misaligned = int(((t.dt.second != 0) | (t.dt.microsecond != 0) | (t.dt.nanosecond != 0)).sum())
    # Compare as Timedeltas (resolution-safe: pandas may store ms/us/ns internally)
    dur = (df["close_time"] - df["open_time"]).astype("timedelta64[ns]")
    is_us = (df["ts_unit"].astype(str) == "us").to_numpy()
    expected_dur = pd.Series(np.where(is_us, 59_999_999_000, 59_999_000_000),
                             index=df.index).astype("timedelta64[ns]")
    bad_close = int((dur != expected_dur).sum())
    # Documented change: spot files from 2025-01-01 use microseconds
    fs = fs.copy()
    fs["expected_unit"] = np.where(fs["first"] >= pd.Timestamp("2025-01-01", tz="UTC"), "us", "ms")
    unit_mismatch = fs.loc[fs["ts_unit"].astype(str) != fs["expected_unit"], "source_file"].tolist()
    # Rows whose timestamp falls outside the period named in their file
    # (vectorised: parse each file name once, compare year*100+month integers)
    sf = df["source_file"].astype("category")
    file_ym = sf.cat.categories.str.extract(r"-(\d{4})-(\d{2})", expand=True).astype(int)
    file_ym = (file_ym[0] * 100 + file_ym[1]).to_numpy()[sf.cat.codes.to_numpy()]
    row_ym = (df["open_time"].dt.year * 100 + df["open_time"].dt.month).to_numpy()
    outside = int((row_ym != file_ym).sum())
    return dict(
        first=t.min(), last=t.max(), timezone=str(t.dt.tz),
        misaligned_to_minute=misaligned, bad_close_time=bad_close,
        files_us=int((fs["ts_unit"].astype(str) == "us").sum()),
        files_ms=int((fs["ts_unit"].astype(str) == "ms").sum()),
        unit_mismatch_files=unit_mismatch, rows_outside_file_period=outside,
        files_not_ordered=fs.loc[~fs["ordered_in_file"], "source_file"].tolist(),
    )


def duplicate_checks(df: pd.DataFrame) -> dict:
    exact = df.duplicated(subset=["open_time"] + DATA_COLS, keep="first")
    dup_ts = df.duplicated(subset=["open_time"], keep=False)
    # Conflicting = same timestamp, different data after removing exact copies
    dedup = df.loc[~exact]
    conflicting_ts = dedup.loc[dedup.duplicated(subset=["open_time"], keep=False), "open_time"].unique()
    return dict(exact_duplicate_rows=int(exact.sum()),
                rows_with_duplicate_timestamp=int(dup_ts.sum()),
                conflicting_timestamps=pd.DatetimeIndex(conflicting_ts))


def ohlc_checks(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = (df[k] for k in PRICE_COLS)
    vwap = df["quote_volume"] / df["volume"].where(df["volume"] > 0)
    tol = 1e-6 * c
    flags = pd.DataFrame({
        "high_lt_low": h < l,
        "high_lt_open_or_close": h < np.maximum(o, c),
        "low_gt_open_or_close": l > np.minimum(o, c),
        "nonpositive_price": (df[PRICE_COLS] <= 0).any(axis=1),
        "vwap_outside_range": (vwap < l - tol) | (vwap > h + tol),
    })
    return flags


def volume_checks(df: pd.DataFrame) -> dict:
    v, n = df["volume"], df["n_trades"].astype("float64")
    rel = v / v.rolling(1440, min_periods=720).median()
    return dict(
        negative_volume=int((v < 0).sum()),
        zero_volume=int((v == 0).sum()),
        zero_trades=int((n == 0).sum()),
        volume_without_trades=int(((v > 0) & (n == 0)).sum()),
        trades_without_volume=int(((v == 0) & (n > 0)).sum()),
        taker_buy_gt_volume=int((df["taker_buy_base"] > v * (1 + 1e-9)).sum()),
        volume_gt_50x_daily_median=int((rel > 50).sum()),
    )


def missing_value_checks(df: pd.DataFrame) -> pd.Series:
    return df[DATA_COLS].isna().sum()


def gap_analysis(open_times: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Missing minutes on the full grid; returns (gap table, missing by year, n_missing)."""
    t = pd.DatetimeIndex(open_times.drop_duplicates().sort_values())
    grid = pd.date_range(t[0], t[-1], freq="1min")
    present = grid.isin(t)
    missing = ~present
    # Run-length encode missing stretches
    m = missing.astype(np.int8)
    edges = np.diff(np.concatenate([[0], m, [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    gaps = pd.DataFrame({"gap_start": grid[starts], "gap_end": grid[ends - 1],
                         "missing_minutes": ends - starts})
    gaps = gaps.sort_values("missing_minutes", ascending=False).reset_index(drop=True)
    by_year = pd.DataFrame({"year": grid.year, "missing": missing}).groupby("year").agg(
        expected_minutes=("missing", "size"), missing_minutes=("missing", "sum"))
    by_year["missing_pct"] = 100 * by_year["missing_minutes"] / by_year["expected_minutes"]
    return gaps, by_year.reset_index(), int(missing.sum())


def outlier_scan(df: pd.DataFrame, z_thresh: float = 15.0, abs_thresh: float = 0.03) -> pd.DataFrame:
    """Flag extreme 1-minute moves and gather evidence on whether each is genuine.

    Robust scale = 1.4826 x rolling (1-day, centred) median of |r|. Centred
    windows are fine here: this is a data-quality diagnostic, never a feature.
    """
    d = df.drop_duplicates("open_time").set_index("open_time").sort_index()
    r = np.log(d["close"]).diff()
    scale = 1.4826 * r.abs().rolling(1441, center=True, min_periods=300).median()
    z = r / scale
    wick = np.log(d["high"] / d["low"])
    flag = (z.abs() > z_thresh) | (r.abs() > abs_thresh) | (wick > abs_thresh)
    ev = d.loc[flag, ["open", "high", "low", "close", "volume", "n_trades"]].copy()
    ev["ret_1m"] = r[flag]
    ev["robust_z"] = z[flag]
    ev["wick_range"] = wick[flag]
    # Evidence: does the price level persist over the next 10 minutes, and was
    # there real trading activity? A one-print spike on tiny volume that fully
    # reverses is suspicious; a high-volume move that persists is a market event.
    fwd = np.log(d["close"]).shift(-10) - np.log(d["close"])
    ev["ret_next_10m"] = fwd[flag]
    med_vol = d["volume"].rolling(1441, center=True, min_periods=300).median()
    ev["volume_vs_daily_median"] = (d["volume"] / med_vol)[flag]
    reverted = (np.sign(ev["ret_1m"]) != np.sign(ev["ret_next_10m"])) & (
        ev["ret_next_10m"].abs() > 0.8 * ev["ret_1m"].abs())
    thin = (ev["n_trades"].astype("float64") < 10) | (ev["volume_vs_daily_median"] < 0.5)
    ev["assessment"] = np.where(reverted & thin, "suspect (thin + full reversal)",
                                "consistent with genuine market move")
    return ev.sort_values("robust_z", key=np.abs, ascending=False)


def stale_checks(df: pd.DataFrame, run_minutes: int = 30) -> dict:
    d = df.drop_duplicates("open_time").sort_values("open_time")
    same = d["close"].diff().eq(0).astype(np.int8).to_numpy()
    # longest run of unchanged closes
    edges = np.diff(np.concatenate([[0], same, [0]]))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return dict(longest_unchanged_close_run=int(runs.max()) if len(runs) else 0,
                runs_ge_threshold=int((runs >= run_minutes).sum()))


def monthly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage and activity by month, to detect changes in data coverage/regimes."""
    d = df.drop_duplicates("open_time")
    m = d.groupby(d["open_time"].dt.tz_localize(None).dt.to_period("M")).agg(
        rows=("open_time", "size"), volume_btc=("volume", "sum"),
        quote_volume_usdt=("quote_volume", "sum"), trades=("n_trades", "sum"),
        mean_price=("close", "mean"))
    m["avg_trade_size_btc"] = m["volume_btc"] / m["trades"].astype("float64")
    m.index = m.index.astype(str)
    m.index.name = "month"
    return m.reset_index()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def audit(df: pd.DataFrame) -> dict:
    """Run all checks. Returns detail tables + the summary DQ table."""
    expected_cols = {"open_time", "close_time", "raw_open_time", *DATA_COLS, "ts_unit", "source_file"}
    schema = dict(n_rows=len(df), n_cols=df.shape[1],
                  missing_columns=sorted(expected_cols - set(df.columns)),
                  unexpected_columns=sorted(set(df.columns) - expected_cols
                                            - {"has_header", "row_in_file"}),
                  dtypes={c: str(t) for c, t in df.dtypes.items()})
    fs = file_summary(df)
    ts = timestamp_checks(df, fs)
    dup = duplicate_checks(df)
    ohlc = ohlc_checks(df)
    vol = volume_checks(df)
    na = missing_value_checks(df)
    gaps, miss_year, n_missing = gap_analysis(df["open_time"])
    outl = outlier_scan(df)
    stale = stale_checks(df)
    monthly = monthly_profile(df)

    n_unique = df["open_time"].nunique()
    expected = n_unique + n_missing
    rows = [
        ("Rows / columns", f"{schema['n_rows']:,} rows, {schema['n_cols']} cols", "info",
         "None"),
        ("Schema", "missing cols: %s; unexpected: %s" % (schema["missing_columns"] or "none",
                                                          schema["unexpected_columns"] or "none"),
         "high" if schema["missing_columns"] else "info", "Stop if any expected column is missing"),
        ("Timestamp units", f"{ts['files_ms']} files ms, {ts['files_us']} files us; "
                            f"unexpected unit in {len(ts['unit_mismatch_files'])} files",
         "high" if ts["unit_mismatch_files"] else "info",
         "Unit detected per file and converted to UTC datetime"),
        ("Timestamp alignment", f"{ts['misaligned_to_minute']} not on minute boundary; "
                                f"{ts['bad_close_time']} with close_time != open+59.999s",
         "medium" if ts["misaligned_to_minute"] or ts["bad_close_time"] else "info",
         "Misaligned rows reported; grid uses open_time"),
        ("Rows outside their file's month", f"{ts['rows_outside_file_period']}",
         "medium" if ts["rows_outside_file_period"] else "info", "Reported"),
        ("Ordering within files", f"{len(ts['files_not_ordered'])} files not time-ordered",
         "low" if ts["files_not_ordered"] else "info", "Data sorted by open_time"),
        ("Missing values (cells)", f"{int(na.sum())}", "medium" if na.sum() else "info",
         "Minute treated as missing"),
        ("Duplicate rows (exact)", f"{dup['exact_duplicate_rows']}",
         "low" if dup["exact_duplicate_rows"] else "info", "Keep one copy"),
        ("Duplicate timestamps", f"{dup['rows_with_duplicate_timestamp']} rows",
         "low" if dup["rows_with_duplicate_timestamp"] else "info", "See conflicting"),
        ("Conflicting duplicates", f"{len(dup['conflicting_timestamps'])} timestamps",
         "high" if len(dup["conflicting_timestamps"]) else "info", "Minute set to missing"),
        ("Missing 1-min intervals", f"{n_missing:,} of {expected:,} "
                                    f"({100 * n_missing / expected:.3f}%) in {len(gaps)} gaps; "
                                    f"longest {int(gaps['missing_minutes'].max()) if len(gaps) else 0} min",
         "medium" if n_missing else "info", "Left missing; features need >=90% window coverage"),
        ("Gaps longer than 60 min", f"{int((gaps['missing_minutes'] > 60).sum())}",
         "medium" if (gaps["missing_minutes"] > 60).any() else "info",
         "Decisions whose windows/labels touch them are dropped"),
        ("Invalid OHLC", "; ".join(f"{k}={int(v)}" for k, v in ohlc.sum().items()),
         "high" if ohlc.to_numpy().any() else "info", "Minute set to missing"),
        ("Invalid volume", f"negative={vol['negative_volume']}, taker>volume={vol['taker_buy_gt_volume']}, "
                           f"volume w/o trades={vol['volume_without_trades']}",
         "high" if vol["negative_volume"] else "info", "Minute set to missing if negative"),
        ("Zero-volume minutes", f"{vol['zero_volume']} (zero trades: {vol['zero_trades']})",
         "low" if vol["zero_volume"] else "info", "Kept and flagged (no trading is real)"),
        ("Abnormal volume (>50x daily median)", f"{vol['volume_gt_50x_daily_median']}",
         "info", "Kept; investigated in EDA"),
        ("Extreme 1-min moves", f"{len(outl)} flagged; "
                                f"{int((outl['assessment'] != 'consistent with genuine market move').sum())} suspect",
         "medium" if len(outl) else "info", "Kept and flagged; listed individually"),
        ("Stale prices", f"longest unchanged-close run {stale['longest_unchanged_close_run']} min; "
                         f"{stale['runs_ge_threshold']} runs >= 30 min",
         "low" if stale["runs_ge_threshold"] else "info", "Reported"),
    ]
    dq = pd.DataFrame(rows, columns=["check", "result", "severity", "action"])
    return dict(schema=schema, file_summary=fs, timestamps=ts, duplicates=dup, ohlc_flags=ohlc,
                volume=vol, missing_values=na, gaps=gaps, missing_by_year=miss_year,
                outliers=outl, stale=stale, monthly=monthly, dq_table=dq)


def build_processed(df: pd.DataFrame, outliers: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the documented treatments; return (regular 1-min grid, treatment log)."""
    log = []
    exact = df.duplicated(subset=["open_time"] + DATA_COLS, keep="first")
    d = df.loc[~exact].copy()
    log.append(("drop exact duplicate rows", int(exact.sum())))

    conflicting = d["open_time"].duplicated(keep=False)
    conflict_ts = d.loc[conflicting, "open_time"].unique()
    d = d.loc[~conflicting]
    log.append(("conflicting duplicate timestamps -> missing", len(conflict_ts)))

    flags = ohlc_checks(d)
    bad = flags.any(axis=1) | (d["volume"] < 0) | d[DATA_COLS].isna().any(axis=1)
    log.append(("impossible / incomplete candles -> missing", int(bad.sum())))
    d = d.loc[~bad]

    d = d.set_index("open_time").sort_index()[DATA_COLS]
    grid = pd.date_range(d.index.min(), d.index.max(), freq="1min", name="open_time")
    g = d.reindex(grid)                       # missing minutes become NaN, NOT filled
    g["present"] = g["close"].notna()
    g["n_trades"] = g["n_trades"].astype("float64")
    g["zero_volume"] = g["volume"].eq(0)
    g["outlier_flag"] = False
    if outliers is not None and len(outliers):
        g.loc[g.index.isin(outliers.index), "outlier_flag"] = True
    log.append(("minutes missing on regular grid (left as NaN)", int((~g["present"]).sum())))
    return g, pd.DataFrame(log, columns=["treatment", "count"])


def save_processed(grid: pd.DataFrame, symbol: str = config.SYMBOL) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(config.PROCESSED_DIR / f"{symbol}_1m_grid.parquet")
