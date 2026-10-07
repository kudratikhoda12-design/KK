"""Vectorised cost-aware backtest with staggered (overlapping) holding periods.

Timeline for decision bar t (decision time tau_t = close of bar t):
    signal s_t in [-1, 1] is computed from features at tau_t
    the order executes during bar t+1 at that bar's VWAP  e_{t+1}  (one-bar latency; we never trade at the
      price that generated the signal)
    each signal opens a sub-position of size s_t / H held for H bars, i.e. it is closed at e_{t+1+H}
    net position held from e_{t+1} to e_{t+2} is  W_t = (1/H) * sum_{j=0}^{H-1} s_{t-j}

Per-bar P&L (fraction of capital; gross notional <= 1, no leverage):
    gross_t   = W_t * (e_{t+2} / e_{t+1} - 1)
    cost_t    = |W_t - W_{t-1}| * c          c = per-side cost (fee + slippage + half spread), on traded notional
    funding_t = -W_t * f   if a funding settlement falls inside the holding interval of W_t (longs pay f > 0)
    net_t     = gross_t - cost_t + funding_t
Netting overlapping sub-positions only saves costs when the signal persists; this is how a real
account with one net position behaves.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

MIN_PER_YEAR = 365 * 1440


@dataclass(frozen=True)
class Costs:
    fee_bps: float
    slippage_bps: float
    half_spread_bps: float

    @property
    def per_side(self) -> float:
        return (self.fee_bps + self.slippage_bps + self.half_spread_bps) * 1e-4


def staggered_position(signal: np.ndarray, H: int) -> np.ndarray:
    s = np.nan_to_num(signal.astype(float), nan=0.0)
    c = np.cumsum(s)
    w = c.copy()
    w[H:] = c[H:] - c[:-H]
    return w / H


def funding_vector(ts: np.ndarray, funding_ts: np.ndarray, funding_rate: np.ndarray) -> np.ndarray:
    """f_t = funding rate settled inside the holding interval of W_t, i.e. (approximately) at the start of
    bar t+2: the interval runs from the middle of bar t+1 to the middle of bar t+2. ts are bar starts (datetime64[us])."""
    f = np.zeros(len(ts))
    bar = ts.astype("datetime64[m]")
    settle_bar = funding_ts.astype("datetime64[m]")       # a settlement at hh:00:00.xxx falls in bar hh:00
    idx = np.searchsorted(bar, settle_bar) - 2
    ok = (idx >= 0) & (idx < len(ts))
    np.add.at(f, idx[ok], funding_rate[ok])
    return f


def run(df: pl.DataFrame, signal: np.ndarray, H: int, costs: Costs, funding: pl.DataFrame | None) -> pl.DataFrame:
    """df: complete, sorted minute grid with columns ts, exec_px. Returns per-bar P&L components."""
    e = df["exec_px"].to_numpy()
    ts = df["ts"].to_numpy()
    W = staggered_position(signal, H)
    nxt = np.full(len(e), np.nan)
    nxt[:-2] = e[2:] / e[1:-1] - 1                      # return from e_{t+1} to e_{t+2}
    gross = W * np.nan_to_num(nxt)
    dW = np.abs(np.diff(W, prepend=0.0))
    cost = dW * costs.per_side
    if funding is not None:
        f = funding_vector(ts, funding["ts"].dt.replace_time_zone(None).to_numpy(),
                           funding["last_funding_rate"].to_numpy())
        fund = -W * f
    else:
        fund = np.zeros(len(e))
    net = gross - cost + fund
    return pl.DataFrame({"ts": df["ts"], "signal": np.nan_to_num(signal.astype(float)), "position": W,
                         "turnover": dW, "gross": gross, "cost": cost, "funding": fund, "net": net})


def trade_returns(df: pl.DataFrame, signal: np.ndarray, H: int, costs: Costs) -> np.ndarray:
    """Per sub-trade return (each nonzero signal = one trade of H bars, entry e_{t+1}, exit e_{t+1+H}),
    net of a full round-trip cost (no netting credit). Used for win rate / profit factor / average trade."""
    e = df["exec_px"].to_numpy()
    s = np.nan_to_num(signal.astype(float))
    r = np.full(len(e), np.nan)
    r[: -(H + 1)] = e[H + 1:] / e[1:-H] - 1
    m = (s != 0) & ~np.isnan(r)
    return s[m] * r[m] - 2 * costs.per_side * np.abs(s[m])


def summarize(bt: pl.DataFrame, trades: np.ndarray | None = None) -> dict:
    """Performance statistics. Daily aggregation (UTC days) for Sharpe/Sortino/vol; 365-day year (crypto trades 24/7)."""
    d = bt.group_by(pl.col("ts").dt.date().alias("day")).agg(
        pl.col("net").sum(), pl.col("gross").sum(), pl.col("cost").sum(), pl.col("funding").sum(),
        pl.col("turnover").sum()).sort("day")
    net = d["net"].to_numpy()
    n_days = len(net)
    eq = np.cumprod(1 + bt["net"].to_numpy())
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min()) if len(eq) else float("nan")
    total = float(eq[-1] - 1) if len(eq) else float("nan")
    ann_ret = (1 + total) ** (365 / n_days) - 1 if n_days and total > -1 else float("nan")
    sd = net.std(ddof=1)
    dn = np.sqrt(np.mean(np.minimum(net, 0) ** 2))
    gross_sum, turn = float(bt["gross"].sum()), float(bt["turnover"].sum())
    out = {
        "days": n_days,
        "total_return": total,
        "sum_net": float(bt["net"].sum()),
        "sum_gross": gross_sum,
        "sum_cost": float(bt["cost"].sum()),
        "sum_funding": float(bt["funding"].sum()),
        "ann_return": ann_ret,
        "ann_vol": float(sd * np.sqrt(365)),
        "sharpe": float(net.mean() / sd * np.sqrt(365)) if sd > 0 else float("nan"),
        "sortino": float(net.mean() / dn * np.sqrt(365)) if dn > 0 else float("nan"),
        "max_drawdown": mdd,
        "calmar": float(ann_ret / abs(mdd)) if mdd < 0 else float("nan"),
        "turnover_per_day": turn / n_days if n_days else float("nan"),
        "exposure": float(np.abs(bt["position"].to_numpy()).mean()),
        "breakeven_cost_bps_per_side": float((gross_sum + float(bt["funding"].sum())) / turn * 1e4) if turn > 0 else float("nan"),
    }
    if trades is not None and len(trades):
        wins, losses = trades[trades > 0], trades[trades < 0]
        out.update(n_trades=int(len(trades)), win_rate=float((trades > 0).mean()),
                   avg_trade_bps=float(trades.mean() * 1e4),
                   profit_factor=float(wins.sum() / -losses.sum()) if len(losses) else float("nan"))
    else:
        out.update(n_trades=0, win_rate=float("nan"), avg_trade_bps=float("nan"), profit_factor=float("nan"))
    return out


# ----------------------------------------------------------------------------- signal rules

def prob_signal(p: np.ndarray, delta: float) -> np.ndarray:
    """LONG if P(up) > 0.5 + delta, SHORT if P(up) < 0.5 - delta, else FLAT. NaN probability -> FLAT."""
    s = np.zeros(len(p))
    s[p > 0.5 + delta] = 1.0
    s[p < 0.5 - delta] = -1.0
    return s


def momentum_signal(ret_past: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.sign(ret_past))


def flow_signal(ofi_z: np.ndarray, z: float = 1.0) -> np.ndarray:
    s = np.zeros(len(ofi_z))
    s[ofi_z > z] = 1.0
    s[ofi_z < -z] = -1.0
    return s


def random_signal(n: int, seed: int, p_active: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=n)
    s[rng.random(n) > p_active] = 0.0
    return s
