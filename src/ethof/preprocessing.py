"""Parsing, data-quality checks, cleaning and 1-minute aggregation.

Cleaning policy (every decision is counted and written to the QC record):

* REMOVED (data errors): rows with nulls / non-finite values, price <= 0, quantity <= 0,
  first_trade_id > last_trade_id, timestamps outside the file's nominal period, exact duplicate rows
  (one copy kept), conflicting duplicates of the same agg_trade_id (first kept, count reported).
* FLAGGED, NOT REMOVED (possibly legitimate): trades far from their minute's median price
  ("bad-tick candidates"), extreme 1-minute returns, gaps in trade-id sequences, minutes without
  trades, short book-depth days, stale book snapshots.

Trade direction: Binance `is_buyer_maker == True` means the *seller* was the aggressor (taker),
so buyer-initiated volume is `is_buyer_maker == False`. No inference (tick/Lee-Ready) is needed.

Time convention: a 1-minute bar labelled `ts` covers [ts, ts + 1 min). Every field in the bar uses
only trades with timestamps inside that interval, and book information is joined as-of the last
snapshot stamped *strictly before* ts + 1 min, so a bar never contains information from after its close.
"""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

UTC = "UTC"

# --------------------------------------------------------------------------- timestamps

_UNIT_BOUNDS = {  # plausible epoch magnitudes for dates 2001..2286
    "ms": (1e12, 1e13),
    "us": (1e15, 1e16),
    "ns": (1e18, 1e19),
}


def detect_time_unit(values: pl.Series) -> str:
    """Infer the epoch unit (ms/us/ns) of an integer timestamp column.

    Both the minimum and maximum must fall in the same unit's plausible range; otherwise the file
    mixes units (or is corrupt) and we raise instead of guessing.
    """
    v = values.drop_nulls()
    if v.len() == 0:
        raise ValueError("empty timestamp column")
    lo, hi = float(v.min()), float(v.max())
    for unit, (a, b) in _UNIT_BOUNDS.items():
        if a <= lo < b and a <= hi < b:
            return unit
    raise ValueError(f"cannot determine epoch unit: min={lo}, max={hi}")


def epoch_to_utc(col: str, unit: str) -> pl.Expr:
    return pl.from_epoch(pl.col(col), time_unit=unit).dt.cast_time_unit("us").dt.replace_time_zone(UTC)


def period_bounds(period: str) -> tuple[datetime, datetime]:
    """[start, end) of a 'YYYY-MM' or 'YYYY-MM-DD' period in UTC."""
    if len(period) == 7:
        y, m = map(int, period.split("-"))
        s = datetime(y, m, 1, tzinfo=timezone.utc)
        e = datetime(y + (m == 12), 1 if m == 12 else m + 1, 1, tzinfo=timezone.utc)
    else:
        s = datetime.fromisoformat(period).replace(tzinfo=timezone.utc)
        e = s + timedelta(days=1)
    return s, e


# --------------------------------------------------------------------------- csv reading

def _read_zipped_csv(zip_path: Path, columns: list[str], schema: dict) -> pl.DataFrame:
    """Read the single CSV inside a Binance zip; handles files with or without a header row."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{zip_path.name}: expected 1 csv, found {names}")
        raw = zf.read(names[0])
    first = raw[: raw.find(b"\n")].decode().strip()
    has_header = not first[:1].isdigit()
    if has_header and first.split(",")[: len(columns)] != columns:
        log.warning("%s: header %s differs from expected %s", zip_path.name, first, columns)
    return pl.read_csv(io.BytesIO(raw), has_header=has_header, new_columns=columns,
                       schema_overrides=schema)


AGG_COLUMNS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
               "transact_time", "is_buyer_maker"]
AGG_SCHEMA = {"agg_trade_id": pl.Int64, "price": pl.Float64, "quantity": pl.Float64,
              "first_trade_id": pl.Int64, "last_trade_id": pl.Int64, "transact_time": pl.Int64,
              "is_buyer_maker": pl.String}


def read_aggtrades(zip_path: Path) -> pl.DataFrame:
    return _read_zipped_csv(zip_path, AGG_COLUMNS, AGG_SCHEMA)


# --------------------------------------------------------------------------- trades QC + cleaning

def clean_aggtrades(raw: pl.DataFrame, period: str, bad_tick_rel_dev: float = 0.02
                    ) -> tuple[pl.DataFrame, dict]:
    """Validate and clean one file of aggregated trades. Returns (clean trades, qc dict)."""
    qc: dict = {"period": period, "n_raw": raw.height}
    start, end = period_bounds(period)

    unit = detect_time_unit(raw["transact_time"])
    qc["time_unit"] = unit

    # Raw-order diagnostics (before any sorting).
    qc["n_time_decreasing_raw_order"] = int((raw["transact_time"].diff() < 0).sum())
    qc["n_id_not_increasing_raw_order"] = int((raw["agg_trade_id"].diff() <= 0).sum())

    side = raw["is_buyer_maker"].str.to_lowercase()
    qc["n_bad_side_value"] = int((~side.is_in(["true", "false"]) & side.is_not_null()).sum())
    df = raw.with_columns(is_buyer_maker=pl.col("is_buyer_maker").str.to_lowercase())

    # 1) nulls / non-finite / bad side flag -> removed
    invalid_null = (pl.any_horizontal(pl.all().is_null())
                    | ~pl.col("price").is_finite() | ~pl.col("quantity").is_finite()
                    | ~pl.col("is_buyer_maker").is_in(["true", "false"]))
    # 2) impossible values -> removed
    invalid_price = pl.col("price") <= 0
    invalid_qty = pl.col("quantity") <= 0
    invalid_ids = pl.col("first_trade_id") > pl.col("last_trade_id")
    df = df.with_columns(ts=epoch_to_utc("transact_time", unit))
    outside = (pl.col("ts") < start) | (pl.col("ts") >= end)
    flags = df.select(
        n_null_or_nonfinite=invalid_null.sum(),
        n_price_nonpositive=invalid_price.fill_null(False).sum(),
        n_qty_nonpositive=invalid_qty.fill_null(False).sum(),
        n_first_gt_last_id=invalid_ids.fill_null(False).sum(),
        n_outside_period=outside.fill_null(False).sum(),
    ).row(0, named=True)
    qc.update({k: int(v) for k, v in flags.items()})
    bad = (invalid_null | invalid_price.fill_null(False) | invalid_qty.fill_null(False)
           | invalid_ids.fill_null(False) | outside.fill_null(False))
    df = df.filter(~bad)
    qc["n_removed_invalid"] = raw.height - df.height

    # 3) duplicates: exact rows, then conflicting rows sharing an agg_trade_id
    n0 = df.height
    df = df.unique(subset=AGG_COLUMNS, keep="first", maintain_order=True)
    qc["n_exact_duplicates_removed"] = n0 - df.height
    n1 = df.height
    df = df.unique(subset=["agg_trade_id"], keep="first", maintain_order=True)
    qc["n_conflicting_id_duplicates_removed"] = n1 - df.height

    # 4) chronological order (stable: ties broken by exchange id)
    df = df.sort(["ts", "agg_trade_id"])

    # 5) sequence gaps (flag only): missing agg ids and missing underlying trade ids
    seq = df.select(
        agg_gap=(pl.col("agg_trade_id").diff() - 1).clip(lower_bound=0),
        fill_gap=(pl.col("first_trade_id") - pl.col("last_trade_id").shift(1) - 1),
    )
    qc["n_agg_id_gaps"] = int((seq["agg_gap"] > 0).sum())
    qc["n_agg_ids_missing"] = int(seq["agg_gap"].sum() or 0)
    qc["n_trade_id_gaps"] = int((seq["fill_gap"] > 0).sum())
    qc["n_trade_id_overlaps"] = int((seq["fill_gap"] < 0).sum())
    qc["n_trade_ids_missing"] = int(seq["fill_gap"].clip(lower_bound=0).sum() or 0)
    # ids at the file edges, so continuity *across* files can be checked in the report
    if df.height:
        qc["first_agg_id"], qc["last_agg_id"] = int(df["agg_trade_id"][0]), int(df["agg_trade_id"][-1])
        qc["first_fill_id"], qc["last_fill_id"] = int(df["first_trade_id"][0]), int(df["last_trade_id"][-1])

    # 6) bad-tick candidates (flag only): deviation from the same minute's median trade price.
    #    A median within the same minute is a within-bar statistic, used only for QC, not as a feature.
    df = df.with_columns(minute=pl.col("ts").dt.truncate("1m"))
    df = df.with_columns(
        bad_tick_flag=((pl.col("price") / pl.col("price").median().over("minute") - 1).abs()
                       > bad_tick_rel_dev)
    )
    qc["n_bad_tick_flags"] = int(df["bad_tick_flag"].sum())

    df = df.select(
        "ts", "minute", "agg_trade_id", "price", "quantity",
        is_buy=(pl.col("is_buyer_maker") == "false"),          # aggressor = buyer
        n_fills=(pl.col("last_trade_id") - pl.col("first_trade_id") + 1),
        bad_tick_flag="bad_tick_flag",
    )
    qc["n_clean"] = df.height
    qc["first_ts"] = str(df["ts"].min()) if df.height else None
    qc["last_ts"] = str(df["ts"].max()) if df.height else None
    return df, qc


# --------------------------------------------------------------------------- 1-minute bars

def aggregate_1min(trades: pl.DataFrame, period: str, size_buckets: list[float]) -> pl.DataFrame:
    """Aggregate clean trades to a *complete* 1-minute grid for the period.

    Minutes without trades are kept (has_trades=False, volumes/counts 0, OHLC null). Prices are
    deliberately NOT forward-filled here; how to treat empty minutes is a feature-stage decision.
    """
    q, buy = pl.col("quantity"), pl.col("is_buy")
    aggs = [
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        q.sum().alias("volume"),
        (pl.col("price") * q).sum().alias("notional"),
        pl.len().cast(pl.Int64).alias("n_trades"),
        pl.col("n_fills").sum().alias("n_fills"),
        q.filter(buy).sum().alias("buy_volume"),
        q.filter(~buy).sum().alias("sell_volume"),
        (pl.col("price") * q).filter(buy).sum().alias("buy_notional"),
        (pl.col("price") * q).filter(~buy).sum().alias("sell_notional"),
        buy.sum().cast(pl.Int64).alias("buy_trades"),
        (~buy).sum().cast(pl.Int64).alias("sell_trades"),
        q.mean().alias("trade_size_mean"),
        q.median().alias("trade_size_median"),
        q.max().alias("trade_size_max"),
        q.quantile(0.90, "linear").alias("trade_size_p90"),
        q.quantile(0.95, "linear").alias("trade_size_p95"),
        pl.col("ts").last().alias("last_trade_ts"),
        pl.col("bad_tick_flag").sum().cast(pl.Int64).alias("n_bad_tick_flags"),
    ]
    for k in size_buckets:
        tag = f"{k:g}"
        aggs += [q.filter(buy & (q >= k)).sum().alias(f"buy_volume_ge{tag}"),
                 q.filter(~buy & (q >= k)).sum().alias(f"sell_volume_ge{tag}")]

    bars = trades.group_by("minute", maintain_order=True).agg(aggs).rename({"minute": "ts"})
    bars = bars.with_columns(vwap=pl.col("notional") / pl.col("volume"))

    start, end = period_bounds(period)
    grid = pl.DataFrame({"ts": pl.datetime_range(start, end, "1m", closed="left",
                                                 time_unit="us", time_zone=UTC, eager=True)})
    out = grid.join(bars, on="ts", how="left")
    zero_cols = [c for c in out.columns
                 if c.startswith(("volume", "notional", "n_", "buy_", "sell_"))]
    out = out.with_columns([pl.col(c).fill_null(0) for c in zero_cols])
    return out.with_columns(has_trades=pl.col("n_trades") > 0)


def extreme_bars(bars: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """List 1-minute bars with |log close-to-close return| > threshold, with context to classify them.

    `reverted_next` is True if the next traded bar's close undoes more than half of the move:
    a reverted single-bar spike on few trades is a data-error *candidate*; a sustained move is
    treated as a legitimate market event. This is a triage aid; nothing is deleted.
    """
    b = bars.filter(pl.col("has_trades")).select("ts", "open", "high", "low", "close",
                                                 "n_trades", "volume", "n_bad_tick_flags")
    b = b.with_columns(prev_close=pl.col("close").shift(1), next_close=pl.col("close").shift(-1))
    b = b.with_columns(logret=(pl.col("close") / pl.col("prev_close")).log(),
                       range_rel=(pl.col("high") / pl.col("low")).log())
    b = b.with_columns(reverted_next=((pl.col("next_close") / pl.col("close")).log().abs()
                                      > 0.5 * pl.col("logret").abs())
                       & ((pl.col("next_close") / pl.col("close")).log().sign()
                          != pl.col("logret").sign()))
    return b.filter((pl.col("logret").abs() > threshold) | (pl.col("range_rel") > threshold))


def no_trade_runs(bars: pl.DataFrame, min_minutes: int) -> pl.DataFrame:
    """Runs of consecutive minutes without any trade (missing periods), length >= min_minutes."""
    b = bars.select("ts", "has_trades").with_columns(run=(pl.col("has_trades") != pl.col("has_trades").shift(1)).fill_null(True).cum_sum())
    runs = (b.filter(~pl.col("has_trades")).group_by("run")
            .agg(start=pl.col("ts").min(), end=pl.col("ts").max(), minutes=pl.len()))
    return runs.filter(pl.col("minutes") >= min_minutes).drop("run").sort("start")


# --------------------------------------------------------------------------- book depth

BOOK_COLUMNS = ["timestamp", "percentage", "depth", "notional"]
BOOK_SCHEMA = {"timestamp": pl.String, "percentage": pl.Float64, "depth": pl.Float64,
               "notional": pl.Float64}
BOOK_PCTS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]


def read_bookdepth(zip_path: Path) -> pl.DataFrame:
    return _read_zipped_csv(zip_path, BOOK_COLUMNS, BOOK_SCHEMA)


def clean_bookdepth(raw: pl.DataFrame, period: str) -> tuple[pl.DataFrame, dict]:
    """Validate one day of percentage-band depth snapshots (long format).

    `percentage` = -k means cumulative resting BID quantity within k% below the reference price;
    +k = cumulative ASK quantity within k% above. (Sign convention is verified empirically against
    trade prices in the Stage 2 report: implied average bid prices sit below trade prices.)
    These are aggregated bands, NOT level-by-level L2 data.
    """
    qc: dict = {"period": period, "n_raw_rows": raw.height}
    start, end = period_bounds(period)
    df = raw.with_columns(ts=pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us",
                                                                   time_zone=UTC, strict=False))
    # Bands outside +-1..5 % (Binance added +-0.2 % rows on 2026-01-15) are a schema extension, not errors.
    # They are counted separately and not used: they do not exist for most of the sample.
    extra = pl.col("percentage").is_not_null() & ~pl.col("percentage").is_in([float(p) for p in BOOK_PCTS])
    qc["n_extra_band_rows_not_used"] = int(df.select(extra.sum()).item())
    qc["extra_band_values"] = sorted(set(df.filter(extra)["percentage"].to_list()))
    df = df.filter(~extra)
    bad = (pl.col("ts").is_null() | pl.any_horizontal(pl.all().is_null())
           | ~pl.col("depth").is_finite() | ~pl.col("notional").is_finite()
           | (pl.col("depth") <= 0) | (pl.col("notional") <= 0)
           | (pl.col("ts") < start) | (pl.col("ts") >= end))
    qc["n_invalid_rows_removed"] = int(df.select(bad.fill_null(True).sum()).item())
    df = df.filter(~bad.fill_null(True))
    n0 = df.height
    df = df.unique(subset=BOOK_COLUMNS, keep="first", maintain_order=True)
    qc["n_exact_duplicates_removed"] = n0 - df.height
    n1 = df.height
    df = df.unique(subset=["ts", "percentage"], keep="first", maintain_order=True)
    qc["n_conflicting_duplicates_removed"] = n1 - df.height
    df = df.with_columns(pct=pl.col("percentage").cast(pl.Int8)).sort(["ts", "pct"])

    counts = df.group_by("ts").len()
    qc["n_snapshots"] = counts.height
    qc["n_incomplete_snapshots"] = int((counts["len"] != len(BOOK_PCTS)).sum())

    wide = to_wide_book(df)
    # cumulative depth must be non-decreasing in band width on each side
    mono = pl.lit(False)
    for s in ("bid", "ask"):
        for k in range(1, 5):
            mono = mono | (pl.col(f"{s}_depth_{k+1}") < pl.col(f"{s}_depth_{k}"))
    qc["n_snapshots_non_monotone_depth"] = int(wide.select(mono.fill_null(False).sum()).item())
    gaps = wide["ts"].diff().dt.total_seconds()
    qc["max_snapshot_gap_s"] = float(gaps.max()) if wide.height > 1 else None
    return wide, qc


def to_wide_book(long: pl.DataFrame) -> pl.DataFrame:
    """Long (ts, pct, depth, notional) -> one row per snapshot with bid/ask depth & notional per band."""
    key = (pl.when(pl.col("pct") < 0).then(pl.lit("bid")).otherwise(pl.lit("ask"))
           + pl.lit("_") + pl.col("pct").abs().cast(pl.String))
    l = long.with_columns(key=key)
    d = l.pivot(on="key", index="ts", values="depth", aggregate_function="first")
    n = l.pivot(on="key", index="ts", values="notional", aggregate_function="first")
    d = d.rename({c: c.replace("_", "_depth_") for c in d.columns if c != "ts"})
    n = n.rename({c: c.replace("_", "_notional_") for c in n.columns if c != "ts"})
    cols = ["ts"] + [f"{s}_{m}_{k}" for m in ("depth", "notional") for s in ("bid", "ask")
                     for k in range(1, 6)]
    out = d.join(n, on="ts", how="left")
    for c in cols:  # an incomplete snapshot keeps null for its missing bands (never imputed)
        if c not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    return out.select(cols).sort("ts")


def asof_book_to_bars(bars: pl.DataFrame, book: pl.DataFrame, stale_seconds: float) -> pl.DataFrame:
    """Attach to each bar [ts, ts+1m) the last book snapshot stamped strictly before the bar close.

    Book timestamps have 1-second resolution, so "strictly before ts+60s" == "<= ts+59s".
    If the snapshot table carries `last_change_ts` (time of the last snapshot whose values changed), the
    book age is measured from it, so a frozen feed (identical repeated snapshots) is flagged stale.
    """
    b = book.rename({"ts": "book_ts"}).sort("book_ts")
    left = bars.with_columns(_key=pl.col("ts") + pl.duration(seconds=59)).sort("_key")
    out = left.join_asof(b, left_on="_key", right_on="book_ts", strategy="backward")
    ref = "last_change_ts" if "last_change_ts" in out.columns else "book_ts"
    out = out.with_columns(book_age_s=((pl.col("ts") + pl.duration(minutes=1)) - pl.col(ref))
                           .dt.total_seconds())
    out = out.with_columns(book_stale=pl.col("book_ts").is_null() | (pl.col("book_age_s") > stale_seconds))
    return out.drop("_key")


# --------------------------------------------------------------------------- funding / metrics

FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
FUNDING_SCHEMA = {"calc_time": pl.Int64, "funding_interval_hours": pl.Int64,
                  "last_funding_rate": pl.Float64}


def read_funding(zip_path: Path) -> pl.DataFrame:
    df = _read_zipped_csv(zip_path, FUNDING_COLUMNS, FUNDING_SCHEMA)
    unit = detect_time_unit(df["calc_time"])
    return df.with_columns(ts=epoch_to_utc("calc_time", unit))


METRICS_COLUMNS = ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
                   "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                   "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
METRICS_SCHEMA = {c: pl.Float64 for c in METRICS_COLUMNS[2:]} | {"create_time": pl.String,
                                                                 "symbol": pl.String}


def read_metrics(zip_path: Path) -> pl.DataFrame:
    df = _read_zipped_csv(zip_path, METRICS_COLUMNS, METRICS_SCHEMA)
    return df.with_columns(ts=pl.col("create_time").str.to_datetime("%Y-%m-%d %H:%M:%S",
                                                                   time_unit="us", time_zone=UTC,
                                                                   strict=False))
