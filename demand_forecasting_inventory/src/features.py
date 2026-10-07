"""Phases 14-16 - leakage-safe feature engineering for the direct multi-horizon models.

Row = (series, forecast origin t).  Information set at t = sales/prices of days <= t plus the *published calendar*.

Feature groups (``FEATURE_GROUPS``)
-----------------------------------
history          lag_k(t) = y[t+1-k]  (k = 1,7,14,21,28; lag_1 is the last observed day, so lags are measured from the
                 first forecast day t+1), rolling mean/std over y[t-w+1..t] (window INCLUDES the origin day, never the future),
                 price at t, week-over-week price change, price relative to its trailing 91-day mean.
calendar_origin  deterministic functions of the origin date (day of week/month, ISO week, month, quarter, year, weekend)
                 and the event / SNAP flags of the origin day.
known_ahead      counts over the forecast window t+1..t+H of SNAP days, event days and Christmas (store closure).  These are
                 deterministic functions of the published calendar - never of sales - and are the only features whose
                 timestamp is later than the origin.  They are tested to be invariant to any change of the sales data.

Target_H(t) = sum(y[t+1..t+H]).  Rows whose target window leaves the data are kept for prediction (target NaN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import config as C
from .forecast_core import Context, truth_sums

LAGS = (1, 7, 14, 21, 28)
ROLL_MEAN = (7, 14, 28)
ROLL_STD = (7, 28)
PRICE_REL_WINDOW = 91

FEATURE_GROUPS = {
    "history": [f"lag_{k}" for k in LAGS] + [f"rolling_mean_{w}" for w in ROLL_MEAN] + [f"rolling_std_{w}" for w in ROLL_STD]
               + ["price", "price_change", "relative_price"],
    "calendar_origin": ["day_of_week", "day_of_month", "week_of_year", "month", "quarter", "year", "weekend",
                        "event_flag", "snap_flag"],
    "known_ahead": ["snap_days_window", "event_days_window", "christmas_in_window"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
FIRST_ORIGIN = C.MIN_HISTORY - 1            # lag_28 and rolling_28 need 28 observed days


def trailing_stat(x: np.ndarray, w: int, kind: str) -> np.ndarray:
    """Mean / sample std (ddof=1) of x[t-w+1..t] for every t (NaN for t < w-1).  Exact (no online-update round-off)."""
    out = np.full(len(x), np.nan)
    win = sliding_window_view(np.asarray(x, float), w)
    out[w - 1:] = win.mean(axis=1) if kind == "mean" else win.std(axis=1, ddof=1)
    return out


def build_features(Y: np.ndarray, price: np.ndarray, dates: pd.DatetimeIndex, snap_full: np.ndarray,
                   event_full: np.ndarray, christmas_full: np.ndarray, H: int) -> pd.DataFrame:
    """Feature matrix for ALL origins t >= FIRST_ORIGIN of every series (target NaN where t + H > N - 1).

    ``snap_full`` / ``event_full`` / ``christmas_full`` cover N + MAX_HORIZON calendar days (the calendar is published ahead).
    """
    n_s, N = Y.shape
    if snap_full.shape[1] < N + H:
        raise ValueError("calendar arrays must extend H days beyond the last observed day")
    t_all = np.arange(FIRST_ORIGIN, N)
    dts = dates[t_all]
    iso = dts.isocalendar()
    cal = {
        "day_of_week": dts.dayofweek.to_numpy(), "day_of_month": dts.day.to_numpy(),
        "week_of_year": iso["week"].to_numpy().astype(int), "month": dts.month.to_numpy(),
        "quarter": dts.quarter.to_numpy(), "year": dts.year.to_numpy(), "weekend": (dts.dayofweek.to_numpy() >= 5).astype(int),
    }
    target = truth_sums(Y, H)
    frames = []
    for s in range(n_s):
        y = pd.Series(Y[s])
        p = pd.Series(price[s])
        f = {"series_idx": np.full(len(t_all), s), "origin": t_all}
        for k in LAGS:
            f[f"lag_{k}"] = y.shift(k - 1).to_numpy()[t_all]                   # y[t-(k-1)] = y[t+1-k]
        for w in ROLL_MEAN:
            f[f"rolling_mean_{w}"] = trailing_stat(Y[s], w, "mean")[t_all]     # y[t-w+1..t]
        for w in ROLL_STD:
            f[f"rolling_std_{w}"] = trailing_stat(Y[s], w, "std")[t_all]
        f["price"] = p.to_numpy()[t_all]
        f["price_change"] = (p / p.shift(7) - 1.0).to_numpy()[t_all]
        f["relative_price"] = (p / p.rolling(PRICE_REL_WINDOW, min_periods=1).mean()).to_numpy()[t_all]
        f.update(cal)
        f["event_flag"] = event_full[t_all]
        f["snap_flag"] = snap_full[s, t_all]
        for name, arr in (("snap_days_window", snap_full[s]), ("event_days_window", event_full), ("christmas_in_window", christmas_full)):
            cs = np.concatenate([[0.0], np.cumsum(arr)])                       # cs[j] = sum arr[:j]
            f[name] = cs[t_all + H + 1] - cs[t_all + 1]                         # sum arr[t+1 .. t+H]
        f["target"] = target[s, t_all]
        frames.append(pd.DataFrame(f))
    df = pd.concat(frames, ignore_index=True)
    return df


def dataset_for_horizon(ctx: Context, H: int) -> pd.DataFrame:
    df = build_features(ctx.Y, ctx.price, ctx.dates, ctx.snap_full, ctx.event_full, ctx.christmas_full, H)
    df["date"] = ctx.dates[df["origin"].to_numpy()]
    return df


# --------------------------------------------------------------------------------------
# Row selectors (chronology enforced here, in one place)
# --------------------------------------------------------------------------------------
def rows_for_fit(df: pd.DataFrame, H: int, fit_end: int) -> pd.DataFrame:
    """Rows usable for fitting: the target window must end on or before day fit_end-1 (no look into the next period)."""
    sub = df[(df["origin"] + H <= fit_end - 1)]
    assert sub["target"].notna().all()
    assert (sub["origin"] + H).max() <= fit_end - 1
    return sub


def rows_for_origins(df: pd.DataFrame, origins: np.ndarray) -> pd.DataFrame:
    """Rows at the requested origins in (series, origin) order - used for prediction."""
    sub = df[df["origin"].isin(origins)].sort_values(["series_idx", "origin"])
    return sub
