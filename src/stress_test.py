"""Stress tests: how do results and risk change under adverse conditions?

Stress testing differs from robustness: robustness asks "is the result real?",
stress asks "what happens to P&L and risk if conditions get worse?".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .backtest import run_backtest
from .risk import daily_returns, max_drawdown, performance


def _row(name: str, bt: pd.DataFrame, ppy: float) -> dict:
    perf = performance(bt, ppy)
    d = daily_returns(bt)
    var99 = -np.quantile(d, 0.01)
    return dict(scenario=name, net_sharpe=perf["sharpe"], annualized_return=perf["annualized_return"],
                cumulative_return=perf["cumulative_return"], max_drawdown=perf["max_drawdown"],
                daily_VaR99=var99, worst_day=float(d.min()))


def cost_scenarios(events, position, ppy, funding=None) -> pd.DataFrame:
    rows = []
    for m in config.COST_MULTIPLIERS:
        bt = run_backtest(events, position, m * config.BASE_COST_PER_SIDE, funding=funding)
        rows.append(_row(f"costs x{m:g} ({1e4 * m * config.BASE_COST_PER_SIDE:.0f} bp/side)", bt, ppy))
    for s in config.EXTRA_SLIPPAGE_SCENARIOS:
        bt = run_backtest(events, position, config.BASE_COST_PER_SIDE, extra_slippage=s, funding=funding)
        rows.append(_row(f"base + {1e4 * s:.0f} bp extra slippage", bt, ppy))
    return pd.DataFrame(rows)


def volatility_shock(events, position, ppy, factor: float = 2.0) -> pd.DataFrame:
    """Same signals, every price move scaled by `factor` (costs unchanged)."""
    base = run_backtest(events, position, config.BASE_COST_PER_SIDE)
    shocked = base.copy()
    shocked["gross_ret"] = shocked["gross_ret"] * factor
    shocked["net_ret"] = shocked["gross_ret"] - shocked["cost"] + shocked["funding_ret"]
    return pd.DataFrame([_row("base", base, ppy), _row(f"volatility x{factor:g}", shocked, ppy)])


def signal_degradation(events, position, ppy, flip_fracs=(0.1, 0.25, 0.5), n_sims: int = 200,
                       seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Randomly flip the direction of a fraction of non-zero positions.

    50% flips = a signal with no directional information but the same trading
    frequency - the cost of trading noise.
    """
    rng = np.random.default_rng(seed)
    nz = np.flatnonzero(position.to_numpy() != 0)
    rows = []
    for q in flip_fracs:
        srs = []
        for _ in range(n_sims):
            pos = position.copy()
            flip = rng.choice(nz, size=int(q * len(nz)), replace=False)
            pos.iloc[flip] = -pos.iloc[flip]
            bt = run_backtest(events, pos, config.BASE_COST_PER_SIDE)
            srs.append(performance(bt, ppy)["sharpe"])
        srs = np.asarray(srs)
        rows.append(dict(scenario=f"{int(100 * q)}% of signals flipped", sims=n_sims,
                         mean_net_sharpe=srs.mean(), p5_net_sharpe=np.quantile(srs, 0.05),
                         p95_net_sharpe=np.quantile(srs, 0.95), share_sharpe_positive=(srs > 0).mean()))
    return pd.DataFrame(rows)


def historical_worst_days(bt: pd.DataFrame, bh: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Strategy vs buy-and-hold on the n worst BTC days inside the evaluated period."""
    d_bh, d_st = daily_returns(bh), daily_returns(bt)
    worst = d_bh.nsmallest(n).index
    return pd.DataFrame({"btc_buy_hold": d_bh[worst], "strategy": d_st.reindex(worst)}).sort_index()


def gap_shock(bt: pd.DataFrame, shocks=(-0.10, -0.20, -0.30)) -> pd.DataFrame:
    """Instant price gap while positioned: loss = position x shock (no chance to exit).

    Reported with how often the strategy was long or short, i.e. how likely it is
    to be exposed when such a gap happens.
    """
    long_share = float((bt["position"] > 0).mean())
    short_share = float((bt["position"] < 0).mean())
    d = daily_returns(bt)
    var99 = -np.quantile(d, 0.01)
    rows = []
    for s in shocks:
        rows.append(dict(shock=s, loss_if_long=-s, loss_if_short=s, share_time_long=long_share,
                         share_time_short=short_share,
                         expected_loss=long_share * -s + short_share * s,
                         loss_if_long_vs_daily_VaR99=(-s) / var99 if var99 > 0 else np.nan))
    return pd.DataFrame(rows)


def by_regime(bt: pd.DataFrame, regimes: pd.Series, ppy: float, bh: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for name, idx in bt.groupby(regimes.reindex(bt.index)).groups.items():
        sub = bt.loc[idx]
        if len(sub) < 50:
            continue
        p = performance(sub, ppy)
        row = dict(regime=name, periods=len(sub), net_sharpe=p["sharpe"], mean_net_bp=1e4 * sub["net_ret"].mean(),
                   hit_rate=p["win_rate"], exposure=p["exposure"])
        if bh is not None:
            row["buy_hold_mean_bp"] = 1e4 * bh["net_ret"].reindex(idx).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def drawdown_table(bt: pd.DataFrame) -> dict:
    eq = (1 + bt["net_ret"]).cumprod()
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    rec = eq.loc[trough:][eq.loc[trough:] >= eq.loc[peak]]
    return dict(max_drawdown=float(dd.min()), peak=peak, trough=trough,
                recovered=rec.index[0] if len(rec) else None, max_dd_check=max_drawdown(bt["net_ret"]))
