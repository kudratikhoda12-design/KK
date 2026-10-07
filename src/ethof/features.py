"""Feature engineering on the 1-minute bar grid.

Timeline (see docs/final_methodology.md):

    bar t covers [ts_t, ts_t + 1 min).  Decision time  tau_t = ts_t + 1 min  (the bar close).
    Every feature in row t uses only data with timestamps < tau_t:
      * trade aggregates of bars t, t-1, ...            (strictly inside [.., tau_t))
      * the last bookDepth snapshot stamped < tau_t       (as-of join done in Stage 2)
      * funding rates with calc_time <= tau_t             (as-of backward join)
      * open-interest rows with create_time <= tau_t - 5 min   (one extra period of lag, because
        the archive does not document whether a 5-min metrics row is published at or after
        create_time)
    All rolling windows are trailing windows ending at bar t (Polars rolling_* are trailing).
    No centred windows, no backward fills, no full-sample normalisation.

Feature groups (cumulative ablation sets are built in `FEATURE_SETS`):
    price   : returns, volatility, momentum, distance from moving averages, range, intraday clock
    volume  : volume / trade-count levels and trailing z-scores, intensity
    flow    : taker order-flow imbalance, trade-count imbalance, signed flow, trade size, large trades
    book    : percentage-band depth imbalance (1/2/3/5 %), its change, total depth z-score
    funding : last realised funding rate, its trailing mean, time to next funding; open interest change
"""
from __future__ import annotations

import logging

import polars as pl

log = logging.getLogger(__name__)

DAY = 1440


def _logret(c: pl.Expr, k: int) -> pl.Expr:
    return (c / c.shift(k)).log()


def _z(x: pl.Expr, window: int) -> pl.Expr:
    """Trailing z-score using a window that ends at (and includes) the current row."""
    m = x.rolling_mean(window, min_samples=window // 2)
    s = x.rolling_std(window, min_samples=window // 2)
    return (x - m) / s


def _imb(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """(a-b)/(a+b) with a safe zero denominator -> null (no information, not 0 imbalance)."""
    den = a + b
    return pl.when(den > 0).then((a - b) / den).otherwise(None)


def prepare_price(bars: pl.DataFrame) -> pl.DataFrame:
    """Add a causal, forward-filled price series for minutes without trades.

    `px` = last traded price at or before the bar close (forward fill only ever looks backwards).
    `exec_px` = the bar's VWAP, used as the execution price of orders submitted at the previous close;
    for a bar without trades the last known price is used and the bar is flagged.
    """
    return bars.sort("ts").with_columns(
        px=pl.col("close").forward_fill(),
    ).with_columns(
        exec_px=pl.coalesce(pl.col("vwap"), pl.col("px")),
        exec_px_from_trades=pl.col("vwap").is_not_null(),
    )


def build_features(bars: pl.DataFrame, funding: pl.DataFrame | None = None,
                   metrics: pl.DataFrame | None = None, large_trade_eth: float = 10.0) -> pl.DataFrame:
    """Compute all features. `bars` must be the complete, sorted 1-minute grid (Stage 2 output)."""
    b = prepare_price(bars)
    px, v = pl.col("px"), pl.col("volume")
    r1 = _logret(px, 1)
    b = b.with_columns(r1=r1)
    r = pl.col("r1")

    def rsum(e: pl.Expr, w: int) -> pl.Expr:
        return e.rolling_sum(w, min_samples=w)

    # ------------------------------------------------------------------ price
    price = [
        *[_logret(px, k).alias(f"ret_{k}") for k in (1, 5, 15, 30, 60)],
        *[r.rolling_std(w, min_samples=w).alias(f"vol_{w}") for w in (15, 60, 240)],
        r.rolling_std(DAY, min_samples=DAY // 2).alias("vol_1440"),
        (_logret(px, 240) / (r.rolling_std(240, min_samples=240) * 240 ** 0.5)).alias("mom_240_riskadj"),
        (px / px.rolling_mean(60, min_samples=60)).log().alias("dist_ma_60"),
        (px / px.rolling_mean(240, min_samples=240)).log().alias("dist_ma_240"),
        (pl.col("high").fill_null(px).rolling_max(15, min_samples=15)
         / pl.col("low").fill_null(px).rolling_min(15, min_samples=15)).log().alias("range_15"),
        (px / (rsum(pl.col("notional"), 15) / rsum(v, 15))).log().alias("dist_vwap_15"),
        ((pl.col("ts").dt.hour().cast(pl.Float64) + pl.col("ts").dt.minute() / 60) * (2 * 3.141592653589793 / 24)).sin().alias("hour_sin"),
        ((pl.col("ts").dt.hour().cast(pl.Float64) + pl.col("ts").dt.minute() / 60) * (2 * 3.141592653589793 / 24)).cos().alias("hour_cos"),
    ]
    # ------------------------------------------------------------------ volume
    lv = (v + 1e-9).log()
    volume = [
        lv.alias("log_volume_1"),
        _z(lv, DAY).alias("volume_z_1440"),
        (rsum(v, 15) / (v.rolling_mean(DAY, min_samples=DAY // 2) * 15)).alias("volume_ratio_15"),
        (rsum(v, 60) / (v.rolling_mean(DAY, min_samples=DAY // 2) * 60)).alias("volume_ratio_60"),
        (rsum(pl.col("n_trades"), 15) / 15).log1p().alias("trade_intensity_15"),
        _z((pl.col("n_trades") + 1).log(), DAY).alias("trade_count_z_1440"),
    ]
    # ------------------------------------------------------------------ trade flow
    bv, sv = pl.col("buy_volume"), pl.col("sell_volume")
    bt, st = pl.col("buy_trades"), pl.col("sell_trades")
    lb, ls = pl.col(f"buy_volume_ge{large_trade_eth:g}"), pl.col(f"sell_volume_ge{large_trade_eth:g}")
    flow = [
        _imb(bv, sv).alias("ofi_1"),
        *[_imb(rsum(bv, w), rsum(sv, w)).alias(f"ofi_{w}") for w in (5, 15, 60)],
        _imb(bt, st).alias("trade_imb_1"),
        _imb(rsum(bt, 15), rsum(st, 15)).alias("trade_imb_15"),
        # signed flow scaled by typical volume (raw OF_t = buy - sell, made scale-free)
        (rsum(bv - sv, 15) / (v.rolling_mean(DAY, min_samples=DAY // 2) * 15)).alias("signed_flow_15"),
        _z(_imb(rsum(bv, 15), rsum(sv, 15)), DAY).alias("ofi_15_z_1440"),
        _z(_imb(bv, sv), DAY).alias("ofi_1_z_1440"),
        (rsum(v, 15) / rsum(pl.col("n_trades"), 15)).log().alias("avg_trade_size_15"),
        _z((v / pl.col("n_trades")).log(), DAY).alias("avg_trade_size_z_1440"),
        (rsum(lb + ls, 15) / rsum(v, 15)).alias("large_share_15"),
        _imb(rsum(lb, 15), rsum(ls, 15)).alias("large_ofi_15"),
    ]
    # ------------------------------------------------------------------ book (percentage bands)
    def obi(k):  # stale/missing snapshots -> null (never interpolated)
        e = _imb(pl.col(f"bid_depth_{k}"), pl.col(f"ask_depth_{k}"))
        return pl.when(pl.col("book_stale")).then(None).otherwise(e)
    book = [obi(k).alias(f"obi_{k}pct") for k in (1, 2, 3, 5)]
    book_extra = [
        (pl.col("obi_1pct") - pl.col("obi_1pct").shift(15)).alias("obi_1pct_chg_15"),
        pl.col("obi_1pct").rolling_mean(15, min_samples=10).alias("obi_1pct_mean_15"),
        _z(pl.when(pl.col("book_stale")).then(None)
           .otherwise((pl.col("bid_depth_1") + pl.col("ask_depth_1")).log()), DAY).alias("depth_1pct_z_1440"),
    ]

    out = b.with_columns(price + volume + flow + book).with_columns(book_extra)

    # ------------------------------------------------------------------ funding + open interest
    fund_cols: list[str] = []
    if funding is not None:
        f = funding.sort("ts").select(
            fund_ts="ts", funding_rate_last="last_funding_rate",
            funding_interval_h="funding_interval_hours",
        ).with_columns(funding_rate_mean_3=pl.col("funding_rate_last").rolling_mean(3, min_samples=1))
        out = out.with_columns(_tau=pl.col("ts") + pl.duration(minutes=1)).sort("_tau") \
                 .join_asof(f, left_on="_tau", right_on="fund_ts", strategy="backward")
        out = out.with_columns(
            minutes_to_funding=(pl.col("fund_ts") + pl.duration(hours=1) * pl.col("funding_interval_h")
                                - pl.col("_tau")).dt.total_minutes().cast(pl.Float64),
        )
        fund_cols = ["funding_rate_last", "funding_rate_mean_3", "minutes_to_funding"]
    if metrics is not None:
        m = metrics.sort("ts").select(oi_ts="ts", oi="sum_open_interest") \
                   .with_columns(oi_avail=pl.col("oi_ts") + pl.duration(minutes=5))
        if "_tau" not in out.columns:
            out = out.with_columns(_tau=pl.col("ts") + pl.duration(minutes=1))
        out = out.sort("_tau").join_asof(m, left_on="_tau", right_on="oi_avail", strategy="backward")
        out = out.with_columns(
            oi_chg_60=(pl.col("oi") / pl.col("oi").shift(60)).log(),
            oi_chg_1440=(pl.col("oi") / pl.col("oi").shift(DAY)).log(),
        )
        fund_cols += ["oi_chg_60", "oi_chg_1440"]
    out = out.sort("ts")
    # stale OI (archive gaps) -> null rather than a silently old value
    if metrics is not None:
        stale = (pl.col("_tau") - pl.col("oi_avail")).dt.total_minutes() > 15
        out = out.with_columns([pl.when(stale).then(None).otherwise(pl.col(c)).alias(c)
                                for c in ("oi_chg_60", "oi_chg_1440")])
    return out.drop([c for c in ("_tau",) if c in out.columns])


PRICE = ["ret_1", "ret_5", "ret_15", "ret_30", "ret_60", "vol_15", "vol_60", "vol_240", "vol_1440",
         "mom_240_riskadj", "dist_ma_60", "dist_ma_240", "range_15", "dist_vwap_15", "hour_sin", "hour_cos"]
VOLUME = ["log_volume_1", "volume_z_1440", "volume_ratio_15", "volume_ratio_60", "trade_intensity_15",
          "trade_count_z_1440"]
FLOW = ["ofi_1", "ofi_5", "ofi_15", "ofi_60", "trade_imb_1", "trade_imb_15", "signed_flow_15",
        "ofi_15_z_1440", "ofi_1_z_1440", "avg_trade_size_15", "avg_trade_size_z_1440",
        "large_share_15", "large_ofi_15"]
BOOK = ["obi_1pct", "obi_2pct", "obi_3pct", "obi_5pct", "obi_1pct_chg_15", "obi_1pct_mean_15",
        "depth_1pct_z_1440"]
FUNDING = ["funding_rate_last", "funding_rate_mean_3", "minutes_to_funding", "oi_chg_60", "oi_chg_1440"]

FEATURE_SETS = {
    "A_price": PRICE,
    "B_price_volume": PRICE + VOLUME,
    "C_plus_flow": PRICE + VOLUME + FLOW,
    "D_plus_book": PRICE + VOLUME + FLOW + BOOK,
    "E_all": PRICE + VOLUME + FLOW + BOOK + FUNDING,
}
GROUPS = {"price": PRICE, "volume": VOLUME, "flow": FLOW, "book": BOOK, "funding_oi": FUNDING}
