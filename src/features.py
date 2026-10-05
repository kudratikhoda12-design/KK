"""Target and feature construction on an hourly (or h-minute) decision grid.

Timing convention (the core of the leakage control)
---------------------------------------------------
* The 1-minute grid row with index tau is the candle that OPENS at tau and
  CLOSES at tau + 1 min.
* ``minute_features`` row tau therefore uses information up to tau + 1 min.
* At decision time t we may use only candles closed by t, i.e. rows tau <= t - 1 min.
  So decision features = ``minute_features.shift(1)`` evaluated at t.
* Execution at the OPEN of the candle starting at t + lag; exit at the OPEN of
  the candle starting at t + lag + h.  Target r = ln(exit / entry).

Nothing here is ever forward-filled. Rolling statistics require at least
``MIN_WINDOW_COVERAGE`` of the window to be present, otherwise NaN.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config

FEATURE_GROUPS: dict[str, list[str]] = {
    "returns": ["ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_60m"],
    "momentum": ["ret_4h", "ret_24h", "ma_dist_24h"],
    "volatility": ["log_rv_60m", "log_rv_24h", "vol_ratio", "range_rel_60m"],
    "volume": ["rel_volume_60m", "volume_chg_60m", "taker_imb_60m"],
    "candle": ["clv_60m"],
}
FEATURES: list[str] = [f for g in FEATURE_GROUPS.values() for f in g]

FEATURE_DOCS = {
    "ret_1m": "log return over the last 1 minute",
    "ret_5m": "log return over the last 5 minutes",
    "ret_15m": "log return over the last 15 minutes",
    "ret_30m": "log return over the last 30 minutes",
    "ret_60m": "log return over the last 60 minutes",
    "ret_4h": "log return over the last 4 hours",
    "ret_24h": "log return over the last 24 hours",
    "ma_dist_24h": "log distance of price from its 24h moving average",
    "log_rv_60m": "log realised volatility of the last 60 minutes",
    "log_rv_24h": "log realised volatility of the last 24 hours (hourly units)",
    "vol_ratio": "log_rv_60m - log_rv_24h: is the last hour unusually volatile?",
    "range_rel_60m": "last-hour high-low range relative to 24h hourly volatility",
    "rel_volume_60m": "log of last-hour volume relative to the 7-day hourly average",
    "volume_chg_60m": "log change of volume: last hour vs the hour before",
    "taker_imb_60m": "taker-buy minus taker-sell volume, as share of volume (order flow)",
    "clv_60m": "where the price closed within the last-hour range (0 = low, 1 = high)",
}

EPS = 1e-12


def _min_periods(window: int) -> int:
    return int(math.ceil(config.MIN_WINDOW_COVERAGE * window))


def _roll(s: pd.Series, window: int):
    return s.rolling(window, min_periods=_min_periods(window))


def minute_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Features on the 1-minute grid. Row tau uses candles with open_time <= tau ONLY.

    pandas rolling windows are right-aligned (window ends at the current row),
    and shift(k) with k > 0 only looks backwards, so no future rows are used.
    """
    if not isinstance(grid.index, pd.DatetimeIndex) or grid.index.freq is None:
        grid = grid.asfreq("1min")       # regular grid required; missing minutes become NaN
    lc = np.log(grid["close"])
    r1 = lc.diff()                        # NaN if either minute is missing
    vol, tb = grid["volume"], grid["taker_buy_base"]

    f = pd.DataFrame(index=grid.index)
    for k, name in [(1, "ret_1m"), (5, "ret_5m"), (15, "ret_15m"), (30, "ret_30m"),
                    (60, "ret_60m"), (240, "ret_4h"), (1440, "ret_24h")]:
        f[name] = lc - lc.shift(k)
    f["ma_dist_24h"] = lc - np.log(_roll(grid["close"], 1440).mean())

    # Realised volatility: sqrt(window * mean squared 1-min return) so a few
    # missing minutes do not bias the level downwards.
    rv60 = np.sqrt(60 * _roll(r1 ** 2, 60).mean())
    rv24h_hourly = np.sqrt(60 * _roll(r1 ** 2, 1440).mean())
    f["log_rv_60m"] = np.log(rv60 + EPS)
    f["log_rv_24h"] = np.log(rv24h_hourly + EPS)
    f["vol_ratio"] = f["log_rv_60m"] - f["log_rv_24h"]
    hi60, lo60 = _roll(grid["high"], 60).max(), _roll(grid["low"], 60).min()
    f["range_rel_60m"] = np.log(hi60 / lo60) / (rv24h_hourly + EPS)

    vol60 = 60 * _roll(vol, 60).mean()
    vol_week = 60 * _roll(vol, 7 * 1440).mean()
    f["rel_volume_60m"] = np.log((vol60 + EPS) / (vol_week + EPS))
    f["volume_chg_60m"] = np.log((vol60 + EPS) / (vol60.shift(60) + EPS))
    tb60 = 60 * _roll(tb, 60).mean()
    f["taker_imb_60m"] = ((2 * tb60 - vol60) / vol60).where(vol60 > 0)
    rng = hi60 - lo60
    f["clv_60m"] = ((grid["close"] - lo60) / rng).where(rng > 0, 0.5)
    f["clv_60m"] = f["clv_60m"].where(hi60.notna() & grid["close"].notna())
    return f


def decision_times(index: pd.DatetimeIndex, horizon: int, offset: int = 0) -> pd.DatetimeIndex:
    """Non-overlapping decision grid: every `horizon` minutes, shifted by `offset` minutes."""
    minutes = index.hour * 60 + index.minute
    return index[((minutes - offset) % horizon) == 0]


def build_dataset(grid: pd.DataFrame, horizon: int = config.PRIMARY_HORIZON_MIN,
                  lag: int = config.EXECUTION_LAG_MIN, offset: int = config.DECISION_OFFSET_MIN,
                  mfeat: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per decision time t with features (info <= t) and the forward target."""
    if grid.index.freq is None:
        grid = grid.asfreq("1min")
    if mfeat is None:
        mfeat = minute_features(grid)
    # Features known at t are those of the candle that closed at t (row t - 1 min).
    known_at = mfeat.shift(1)
    t = decision_times(grid.index, horizon, offset)

    o = grid["open"]
    entry = o.shift(-lag)                 # open of candle starting at t + lag
    exit_ = o.shift(-(lag + horizon))     # open of candle starting at t + lag + h
    d = known_at.loc[t].copy()
    d["entry_time"] = t + pd.Timedelta(minutes=lag)
    d["exit_time"] = t + pd.Timedelta(minutes=lag + horizon)
    d["entry_price"] = entry.loc[t].to_numpy()
    d["exit_price"] = exit_.loc[t].to_numpy()
    d["fwd_logret"] = np.log(d["exit_price"] / d["entry_price"])
    d["fwd_ret"] = d["exit_price"] / d["entry_price"] - 1.0      # simple return for P&L
    d["y"] = (d["fwd_logret"] > 0).astype("float64").where(d["fwd_logret"].notna())

    # Polymarket-style settlement event (only meaningful for h=60, offset 0):
    # did the 1-hour candle starting at t close >= its open?
    if horizon == 60:
        c_end = grid["close"].shift(-(horizon - 1))
        d["pm_up"] = (c_end.loc[t].to_numpy() >= o.loc[t].to_numpy()).astype("float64")
        d.loc[c_end.loc[t].isna().to_numpy() | o.loc[t].isna().to_numpy(), "pm_up"] = np.nan

    d = add_regimes(d, periods_per_day=int(24 * 60 / horizon))
    d.index.name = "decision_time"
    return d


def add_regimes(d: pd.DataFrame, periods_per_day: int) -> pd.DataFrame:
    """Volatility and trend regime labels using only information available at t."""
    lrv = d["log_rv_24h"]
    win = 90 * periods_per_day
    lo = lrv.rolling(win, min_periods=30 * periods_per_day).quantile(1 / 3)
    hi = lrv.rolling(win, min_periods=30 * periods_per_day).quantile(2 / 3)
    reg = np.select([lrv > hi, lrv < lo], ["high_vol", "low_vol"], "mid_vol")
    d["vol_regime"] = pd.Series(reg, index=d.index).where(hi.notna() & lrv.notna())
    daily_sigma = np.sqrt(24) * np.exp(lrv)
    trend = d["ret_24h"].abs() > daily_sigma
    d["trend_regime"] = trend.map({True: "trending", False: "range"}).where(
        d["ret_24h"].notna() & lrv.notna())
    return d


# --------------------------------------------------------------------------
# Leakage audit
# --------------------------------------------------------------------------
def perturbation_leakage_check(grid: pd.DataFrame, check_times, horizon: int = 60, lag: int = 1,
                               lookback_days: int = 9, lookahead_days: int = 2,
                               seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """For each t: replace ALL data at or after t with random garbage and recompute.

    If any feature at t changes, that feature used future information.
    Works on a local window [t - lookback, t + lookahead] so it is cheap on
    the full 4.8M-row grid; the lookback (9 days) exceeds the longest feature
    window (7 days), so the local computation equals the global one.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for t in pd.DatetimeIndex(check_times):
        lo, hi = t - pd.Timedelta(days=lookback_days), t + pd.Timedelta(days=lookahead_days)
        local = grid.loc[lo:hi].copy()
        base = build_dataset(local, horizon, lag)
        if t not in base.index:
            continue
        fut = local.index >= t
        bad = local.copy()
        n = int(fut.sum())
        noise = rng.uniform(0.5, 2.0, n)
        for c in ["open", "high", "low", "close"]:
            bad.loc[fut, c] = bad.loc[fut, c].to_numpy() * noise
        bad.loc[fut, ["volume", "taker_buy_base"]] = rng.uniform(0, 1e4, (n, 2))
        pert = build_dataset(bad, horizon, lag)
        a, b = base.loc[t, FEATURES].astype(float), pert.loc[t, FEATURES].astype(float)
        changed = [f for f in FEATURES if not (np.isclose(a[f], b[f], equal_nan=True))]
        target_changed = not np.isclose(base.loc[t, "fwd_logret"], pert.loc[t, "fwd_logret"],
                                        equal_nan=True)
        rows.append(dict(decision_time=t, features_changed=len(changed),
                         changed_list=",".join(changed), target_changed=target_changed))
    return pd.DataFrame(rows)


def timing_table(horizon: int = 60, lag: int = 1) -> pd.DataFrame:
    """Feature -> latest information timestamp -> prediction timestamp, for the report."""
    rows = []
    windows = {"ret_1m": 1, "ret_5m": 5, "ret_15m": 15, "ret_30m": 30, "ret_60m": 60,
               "ret_4h": 240, "ret_24h": 1440, "ma_dist_24h": 1440, "log_rv_60m": 60,
               "log_rv_24h": 1440, "vol_ratio": 1440, "range_rel_60m": 1440,
               "rel_volume_60m": 10080, "volume_chg_60m": 120, "taker_imb_60m": 60, "clv_60m": 60}
    for f in FEATURES:
        rows.append(dict(feature=f, meaning=FEATURE_DOCS[f], lookback_minutes=windows[f],
                         information_window=f"candles opened in [t-{windows[f]}m, t-1m]",
                         latest_information="close of candle t-1m (closes exactly at t)",
                         prediction_time="t", execution_time=f"open of candle t+{lag}m",
                         target_window=f"[t+{lag}m, t+{lag + horizon}m]"))
    return pd.DataFrame(rows)
