"""Event-level backtest with explicit timing and costs.

One row per decision time t:

    decision_time  t           information: candles closed by t
    entry_time     t + lag     execution at that candle's open (+ half-spread + slippage)
    exit_time      t + lag + h position re-decided; costs only on |change in position|
    gross_ret      position * simple return(entry -> exit)
    cost           turnover * cost_per_side
    net_ret        gross_ret - cost (+ funding if supplied)

Assumptions (all visible in the report): constant 1x notional, no leverage,
no compounding of position size within a period, shorting via the
perpetual future using spot prices as the price proxy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def positions_from_probs(p: pd.Series, scale: pd.Series | float, k: float, mode: str) -> pd.Series:
    """Long if p > 0.5 + k*scale, short if p < 0.5 - k*scale (long_short mode), else flat."""
    up = p > 0.5 + k * scale
    down = p < 0.5 - k * scale
    pos = np.where(up, 1.0, np.where(down & (mode == "long_short"), -1.0, 0.0))
    return pd.Series(pos, index=p.index, name="position")


def positions_from_score(score: pd.Series, mode: str = "long_short") -> pd.Series:
    """Baseline rules: sign of a feature (e.g. past return)."""
    pos = np.sign(score).fillna(0.0)
    if mode == "long_flat":
        pos = pos.clip(lower=0)
    return pos.rename("position")


def funding_between(entry: pd.Series, exit_: pd.Series, funding: pd.DataFrame | None) -> np.ndarray:
    """Sum of funding rates with funding_time in (entry, exit]."""
    if funding is None or funding.empty:
        return np.zeros(len(entry))
    ft = funding["funding_time"].to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(funding["funding_rate"].to_numpy())])
    lo = np.searchsorted(ft, entry.to_numpy(), side="right")
    hi = np.searchsorted(ft, exit_.to_numpy(), side="right")
    return cum[hi] - cum[lo]


def run_backtest(events: pd.DataFrame, position: pd.Series, cost_per_side: float,
                 extra_slippage: float = 0.0, funding: pd.DataFrame | None = None) -> pd.DataFrame:
    """events: output of features.build_dataset (needs entry/exit times and prices)."""
    e = events.loc[position.index, ["entry_time", "exit_time", "entry_price", "exit_price",
                                    "fwd_ret"]].copy()
    e = e.dropna(subset=["entry_price", "exit_price"])
    pos = position.loc[e.index].astype(float)
    e["position"] = pos

    # A position carries over only if the previous holding period ends exactly
    # when this one starts; otherwise it was closed at its own exit.
    contiguous_prev = e["entry_time"].eq(e["exit_time"].shift(1))
    contiguous_next = e["exit_time"].eq(e["entry_time"].shift(-1))
    prev = pos.shift(1).where(contiguous_prev, 0.0).fillna(0.0)
    open_turn = (pos - prev).abs()
    close_turn = pos.abs().where(~contiguous_next, 0.0)      # closed at exit if no continuation
    e["prev_position"] = prev
    e["turnover"] = open_turn + close_turn
    per_side = cost_per_side + extra_slippage
    e["cost"] = e["turnover"] * per_side
    e["gross_ret"] = pos * e["fwd_ret"]
    # Longs pay positive funding, shorts receive it.
    e["funding_ret"] = -pos * funding_between(e["entry_time"], e["exit_time"], funding)
    e["net_ret"] = e["gross_ret"] - e["cost"] + e["funding_ret"]
    e.index.name = "decision_time"
    return e


def buy_and_hold(events: pd.DataFrame, cost_per_side: float) -> pd.DataFrame:
    pos = pd.Series(1.0, index=events.dropna(subset=["fwd_ret"]).index)
    return run_backtest(events, pos, cost_per_side)


def capacity_check(grid: pd.DataFrame, events: pd.DataFrame, notional_usdt: float) -> dict:
    """Assumed order size as a share of the traded USDT volume in the execution minute."""
    qv = grid["quote_volume"].reindex(events["entry_time"]).to_numpy()
    share = notional_usdt / qv
    return dict(notional_usdt=notional_usdt,
                median_exec_minute_quote_volume=float(np.nanmedian(qv)),
                median_participation=float(np.nanmedian(share)),
                p95_participation=float(np.nanquantile(share, 0.95)))
