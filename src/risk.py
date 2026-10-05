"""Performance, tail risk, VaR backtesting and Monte Carlo for strategy returns.

Method choices (and why they differ from the credit module)
-----------------------------------------------------------
* The credit module simulated correlated DEFAULTS with a one-factor Gaussian
  model because loan losses are rare binary events whose dependence is not
  observed. Market strategy returns ARE observed every hour, so we use the
  empirical distribution directly (historical VaR/ES), compare it with
  parametric normal and Student-t VaR, and use a block bootstrap - not a
  Gaussian copula - for Monte Carlo, because it keeps fat tails and
  volatility clustering without assuming a distribution.
* VaR/ES are computed on DAILY strategy returns (aggregated from hourly
  trades), 1-day horizon, 95% and 99%. Losses are reported as positive numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .stats import bootstrap_stat, percentile_ci, stationary_bootstrap_indices

EULER_GAMMA = 0.5772156649


# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------
def equity_curve(net: pd.Series) -> pd.Series:
    return (1 + net).cumprod()


def max_drawdown(net: pd.Series) -> float:
    eq = equity_curve(net)
    return float((eq / eq.cummax() - 1).min())


def daily_returns(bt: pd.DataFrame, col: str = "net_ret") -> pd.Series:
    """Compound period returns into calendar-day (UTC) returns; days with no events are 0."""
    d = (1 + bt[col]).groupby(bt["exit_time"].dt.floor("D")).prod() - 1
    full = pd.date_range(d.index.min(), d.index.max(), freq="D")
    return d.reindex(full, fill_value=0.0)


def performance(bt: pd.DataFrame, periods_per_year: float) -> dict:
    net, gross = bt["net_ret"], bt["gross_ret"]
    years = (bt["exit_time"].max() - bt["entry_time"].min()) / pd.Timedelta(days=365.25)
    cum = float(equity_curve(net).iloc[-1] - 1)
    sd = net.std()
    downside = np.sqrt(np.mean(np.minimum(net, 0) ** 2))
    in_mkt = bt["position"] != 0
    wins, losses = net[in_mkt & (net > 0)], net[in_mkt & (net < 0)]
    ann_ret = (1 + cum) ** (1 / years) - 1 if years > 0 and cum > -1 else np.nan
    mdd = max_drawdown(net)
    return dict(
        periods=len(bt), years=float(years),
        cumulative_return=cum, annualized_return=float(ann_ret),
        annualized_vol=float(sd * np.sqrt(periods_per_year)),
        sharpe=float(net.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan,
        sortino=float(net.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else np.nan,
        max_drawdown=mdd, calmar=float(ann_ret / abs(mdd)) if mdd < 0 else np.nan,
        gross_sum=float(gross.sum()), cost_sum=float(bt["cost"].sum()),
        funding_sum=float(bt["funding_ret"].sum()), net_sum=float(net.sum()),
        gross_sharpe=float(gross.mean() / gross.std() * np.sqrt(periods_per_year))
        if gross.std() > 0 else np.nan,
        exposure=float(in_mkt.mean()), n_position_changes=int((bt["turnover"] > 0).sum()),
        turnover_per_year=float(bt["turnover"].sum() / years) if years > 0 else np.nan,
        # position_changes_per_year counts every entry, exit and flip (an isolated
        # one-hour trade = 2 changes); round_trips_per_year = turnover / 2.
        trades_per_year=float((bt["turnover"] > 0).sum() / years) if years > 0 else np.nan,
        round_trips_per_year=float(bt["turnover"].sum() / 2 / years) if years > 0 else np.nan,
        win_rate=float((net[in_mkt] > 0).mean()) if in_mkt.any() else np.nan,
        avg_win=float(wins.mean()) if len(wins) else np.nan,
        avg_loss=float(losses.mean()) if len(losses) else np.nan,
        profit_factor=float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else np.nan,
    )


def sharpe_ci(net: np.ndarray, periods_per_year: float, mean_block: float = config.BOOTSTRAP_BLOCK_HOURS,
              n_boot: int = config.BOOTSTRAP_N) -> tuple[float, float]:
    def _sr(x):
        s = x.std()
        return x.mean() / s * np.sqrt(periods_per_year) if s > 0 else np.nan
    return percentile_ci(bootstrap_stat((np.asarray(net, float),), _sr, mean_block, n_boot))


def probabilistic_sharpe(net: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark), Bailey & Lopez de Prado (2012)."""
    x = np.asarray(net, float)
    sr = x.mean() / x.std()
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(len(x) - 1) / denom))


def deflated_sharpe(net: np.ndarray, trial_sharpes: np.ndarray) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    trial_sharpes: per-period Sharpe of every configuration tried (from the
    experiment log). The benchmark is the Sharpe expected from the BEST of N
    skill-less trials; DSR = P(true Sharpe > that benchmark).
    """
    sr_trials = np.asarray(trial_sharpes, float)
    sr_trials = sr_trials[np.isfinite(sr_trials)]
    n = len(sr_trials)
    if n < 2:
        return dict(n_trials=n, sr0=np.nan, dsr=np.nan)
    v = sr_trials.var(ddof=1)
    sr0 = np.sqrt(v) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
                        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e)))
    return dict(n_trials=n, sr0_per_period=float(sr0), dsr=probabilistic_sharpe(net, sr0))


# --------------------------------------------------------------------------
# VaR / ES
# --------------------------------------------------------------------------
def var_es(daily: pd.Series, levels=config.VAR_LEVELS) -> pd.DataFrame:
    x = daily.dropna().to_numpy()
    mu, sd = x.mean(), x.std(ddof=1)
    df_t, loc_t, scale_t = stats.t.fit(x)
    rows = []
    for a in levels:
        q = 1 - a
        h_var = -np.quantile(x, q)
        h_es = -x[x <= -h_var].mean()
        n_var = -(mu + sd * stats.norm.ppf(q))
        n_es = -(mu - sd * stats.norm.pdf(stats.norm.ppf(q)) / q)
        if df_t > 2:
            t_var = -stats.t.ppf(q, df_t, loc_t, scale_t)
            # ES of a location-scale t: closed form
            tq = stats.t.ppf(q, df_t)
            t_es = -(loc_t - scale_t * (df_t + tq ** 2) / (df_t - 1) * stats.t.pdf(tq, df_t) / q)
        else:
            # Fit is degenerate (e.g. a mostly-flat strategy with many exact-zero days):
            # df <= 2 implies infinite variance, so the parametric t answer is not meaningful.
            t_var = t_es = np.nan
        rows += [dict(level=a, method="historical", VaR=h_var, ES=h_es),
                 dict(level=a, method="parametric normal", VaR=n_var, ES=n_es),
                 dict(level=a, method=f"Student-t (df={df_t:.1f})" + ("" if df_t > 2 else " - fit degenerate, n/a"),
                      VaR=t_var, ES=t_es)]
    return pd.DataFrame(rows)


def kupiec_pof(daily: pd.Series, level: float = 0.99, window: int = 365) -> dict:
    """Rolling historical VaR forecast for day d from the previous `window` days; count exceptions.

    Kupiec proportion-of-failures LR test: H0 exception rate = 1 - level.
    """
    x = daily.dropna()
    var_fc = -x.rolling(window).quantile(1 - level).shift(1)    # uses days before d only
    m = var_fc.notna()
    exc = (x[m] < -var_fc[m])
    n, k, p = int(m.sum()), int(exc.sum()), 1 - level
    if n == 0:
        return dict(level=level, days=0)
    ph = k / n
    ll0 = (n - k) * np.log(1 - p) + k * np.log(p)
    ll1 = (n - k) * np.log(1 - ph) + (k * np.log(ph) if k > 0 else 0.0) if 0 < ph < 1 else 0.0
    lr = -2 * (ll0 - ll1)
    return dict(level=level, window_days=window, days=n, exceptions=k, expected=n * p,
                exception_rate=ph, lr_stat=float(lr), p_value=float(stats.chi2.sf(lr, 1)))


# --------------------------------------------------------------------------
# Monte Carlo: block bootstrap of daily strategy returns
# --------------------------------------------------------------------------
def bootstrap_paths(daily: pd.Series, horizon_days: int = 365, n_paths: int = 5000,
                    mean_block: float = config.BOOTSTRAP_BLOCK_DAYS,
                    seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Simulate one-year outcomes by resampling blocks of real daily returns.

    Answers: given the realised return process, how uncertain is the one-year
    result, and how deep can drawdowns plausibly get? It inherits the sample's
    regimes - it cannot invent a crisis that never happened (stress tests do that).
    """
    x = daily.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for start in range(0, n_paths, 500):
        k = min(500, n_paths - start)
        idx = stationary_bootstrap_indices(len(x), mean_block, k, rng)[:, :horizon_days]
        paths = x[idx]
        eq = np.cumprod(1 + paths, axis=1)
        dd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
        rows.append(pd.DataFrame({"one_year_return": eq[:, -1] - 1, "max_drawdown": dd}))
    return pd.concat(rows, ignore_index=True)


def summarize_paths(paths: pd.DataFrame) -> pd.DataFrame:
    r, dd = paths["one_year_return"], paths["max_drawdown"]
    return pd.DataFrame([dict(
        paths=len(paths), median_one_year_return=r.median(), p_loss_over_year=float((r < 0).mean()),
        one_year_VaR95=-np.quantile(r, 0.05), one_year_ES95=-r[r <= np.quantile(r, 0.05)].mean(),
        median_max_drawdown=dd.median(), p5_max_drawdown=np.quantile(dd, 0.05))])
