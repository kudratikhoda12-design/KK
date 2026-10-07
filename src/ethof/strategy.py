"""Strategy evaluation shared by the development (out-of-fold) backtest and the final holdout.

All strategies hold H = primary horizon minutes, execute at the next bar's VWAP and pay the same costs.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from . import backtest as B


def cost_scenarios(cfg: dict) -> dict[str, B.Costs]:
    c = cfg["research"]["costs"]
    hs = c["half_spread_bps"]
    out = {"gross": B.Costs(0.0, 0.0, 0.0)}
    for k in ("low", "base", "high"):
        out[k] = B.Costs(c[k]["fee_bps"], c[k]["slippage_bps"], hs)
    return out


def grid_with_preds(research: pl.DataFrame, preds: pl.DataFrame, start, end) -> pl.DataFrame:
    """Complete minute grid on [start, end) with the model probability attached (null where no prediction)."""
    g = research.filter((pl.col("ts") >= start) & (pl.col("ts") < end)).select(
        "ts", "exec_px", "px", "ret_15", "ofi_15_z_1440", "vol_1440", "fwd_ret_15")
    return g.join(preds.select("ts", "p"), on="ts", how="left").sort("ts")


def tradable_auc(g: pl.DataFrame, H: int) -> float:
    """AUC of the model probability against the return actually capturable: entry at the next bar's VWAP,
    exit H bars later at VWAP (instead of last-trade price at t -> t+H)."""
    e = g["exec_px"].to_numpy()
    r = np.full(len(e), np.nan)
    r[: -(H + 1)] = e[H + 1:] / e[1:-H] - 1
    p = g["p"].to_numpy().astype(float)
    m = np.isfinite(p) & np.isfinite(r) & (r != 0)
    return float(roc_auc_score(r[m] > 0, p[m]))


def signals(g: pl.DataFrame, delta: float, flow_z: float, seed: int = 0) -> dict[str, np.ndarray]:
    p = g["p"].to_numpy().astype(float)
    s_ml = B.prob_signal(np.nan_to_num(p, nan=0.5), delta)
    active = float((s_ml != 0).mean())
    return {
        "ml": s_ml,
        "momentum": B.momentum_signal(g["ret_15"].to_numpy()),
        "flow_rule": B.flow_signal(np.nan_to_num(g["ofi_15_z_1440"].to_numpy()), flow_z),
        "random": B.random_signal(g.height, seed, max(active, 0.01)),
    }


def buy_and_hold(g: pl.DataFrame, costs: B.Costs, funding: pl.DataFrame | None) -> pl.DataFrame:
    s = np.ones(g.height)
    return B.run(g, s, 1, costs, funding)  # H = 1 with constant signal => constant position 1, one entry


def evaluate(g: pl.DataFrame, H: int, delta: float, flow_z: float, costs: dict[str, B.Costs],
             funding: pl.DataFrame | None, seed: int = 0) -> tuple[pl.DataFrame, dict]:
    """Returns (summary table, {(strategy, cost): per-bar backtest frame})."""
    sig = signals(g, delta, flow_z, seed)
    rows, frames = [], {}
    for cname, c in costs.items():
        for sname, s in sig.items():
            bt = B.run(g, s, H, c, funding)
            st = B.summarize(bt, B.trade_returns(g, s, H, c))
            rows.append({"strategy": sname, "cost": cname, "delta": delta if sname == "ml" else None, **st})
            frames[(sname, cname)] = bt
        bh = buy_and_hold(g, c, funding)
        st = B.summarize(bh)
        st["n_trades"] = 1
        rows.append({"strategy": "buy_and_hold", "cost": cname, "delta": None, **st})
        frames[("buy_and_hold", cname)] = bh
    return pl.DataFrame(rows, infer_schema_length=None), frames
