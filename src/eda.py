"""Exploratory data analysis: each function answers one stated question.

Predictive relationships (autocorrelation, feature-vs-future-return) are
computed on DEVELOPMENT data only (see the pre-registration). Descriptive
price/volatility views may use the full period.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from . import config

# Reference palette (validated: categorical slots 1-2 pass all CVD/contrast checks on light surface)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ACCENT, ACCENT2 = "#2a78d6", "#eb6834"   # series 1 (blue), series 2 (orange)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.labelsize": 9, "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "lines.linewidth": 1.2})


def resample_bars(grid: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate the 1-min grid into bars; a bar is NaN if any minute is missing."""
    r = grid.resample(rule, label="left", closed="left")
    out = pd.DataFrame({
        "open": r["open"].first(), "high": r["high"].max(), "low": r["low"].min(),
        "close": r["close"].last(), "volume": r["volume"].sum(min_count=1),
        "taker_buy_base": r["taker_buy_base"].sum(min_count=1),
        "minutes_present": r["present"].sum(),
    })
    full = pd.Timedelta(rule) / pd.Timedelta("1min")
    out.loc[out["minutes_present"] < full, ["open", "high", "low", "close"]] = np.nan
    return out


def log_returns(bars: pd.DataFrame) -> pd.Series:
    return np.log(bars["close"]).diff()


# Q: How are returns distributed at different horizons? Are tails heavy?
def return_distribution_table(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, rule in [("1 min", "1min"), ("15 min", "15min"), ("1 hour", "1h"), ("1 day", "1D")]:
        r = log_returns(grid if rule == "1min" else resample_bars(grid, rule)).dropna()
        jb = stats.jarque_bera(r)
        rows.append(dict(horizon=label, n=len(r), mean_pct=100 * r.mean(), std_pct=100 * r.std(),
                         skew=stats.skew(r), excess_kurtosis=stats.kurtosis(r),
                         min_pct=100 * r.min(), max_pct=100 * r.max(),
                         share_beyond_4sd=float((np.abs(r - r.mean()) > 4 * r.std()).mean()),
                         normal_share_beyond_4sd=2 * stats.norm.sf(4), jarque_bera_p=jb.pvalue))
    return pd.DataFrame(rows)


# Q: Are hourly returns autocorrelated (predictable) and do volatilities cluster?
def dependence_tests(hourly_ret: pd.Series, lags=(1, 2, 3, 6, 12, 24)) -> pd.DataFrame:
    r = hourly_ret.dropna()
    rows = []
    for name, x in [("return", r), ("|return|", r.abs()), ("return^2", r ** 2)]:
        acf = [x.autocorr(l) for l in lags]
        lb = acorr_ljungbox(x, lags=[max(lags)], return_df=True)
        rows.append(dict(series=name, **{f"acf_lag{l}": a for l, a in zip(lags, acf)},
                         ljung_box_lag=max(lags), ljung_box_p=float(lb["lb_pvalue"].iloc[0])))
    return pd.DataFrame(rows)


def acf_values(x: pd.Series, max_lag: int) -> np.ndarray:
    x = x.dropna()
    return np.array([x.autocorr(l) for l in range(1, max_lag + 1)])


# Q: Is there a contemporaneous / lead-lag link between volume and returns?
def volume_return_relation(hourly: pd.DataFrame) -> pd.DataFrame:
    r = np.log(hourly["close"]).diff()
    lv = np.log(hourly["volume"].where(hourly["volume"] > 0))
    rel_v = lv - lv.rolling(168, min_periods=120).mean()   # vs trailing week (past only)
    df = pd.DataFrame({"r": r, "abs_r": r.abs(), "rel_v": rel_v,
                       "next_r": r.shift(-1), "next_abs_r": r.abs().shift(-1)}).dropna()
    out = []
    for a, b, q in [("rel_v", "abs_r", "same-hour volume vs |return|"),
                    ("rel_v", "next_abs_r", "volume now vs next-hour |return|"),
                    ("rel_v", "next_r", "volume now vs next-hour return"),
                    ("r", "next_r", "return now vs next-hour return")]:
        rho, p = stats.spearmanr(df[a], df[b])
        out.append(dict(question=q, spearman=rho, naive_p=p, n=len(df)))
    return pd.DataFrame(out)


# Q: Does activity/volatility depend on the time of day (UTC)?
def intraday_profile(hourly: pd.DataFrame) -> pd.DataFrame:
    r = np.log(hourly["close"]).diff()
    d = pd.DataFrame({"hour": hourly.index.hour, "abs_r": r.abs(), "r": r,
                      "volume": hourly["volume"]}).dropna()
    # Normalise volume by its trailing-30-day mean so growth over years does not dominate
    d["rel_volume"] = d["volume"] / d["volume"].rolling(24 * 30, min_periods=24 * 7).mean()
    return d.groupby("hour").agg(mean_abs_ret_pct=("abs_r", lambda x: 100 * x.mean()),
                                 mean_ret_bp=("r", lambda x: 1e4 * x.mean()),
                                 rel_volume=("rel_volume", "mean"))


# Q: Which volatility / trend regimes occurred, and when?
def regime_labels(hourly: pd.DataFrame) -> pd.DataFrame:
    r = np.log(hourly["close"]).diff()
    rv24 = np.sqrt((r ** 2).rolling(24, min_periods=20).sum() / 24)       # hourly units
    lo = rv24.rolling(24 * 90, min_periods=24 * 30).quantile(1 / 3)
    hi = rv24.rolling(24 * 90, min_periods=24 * 30).quantile(2 / 3)
    vol_regime = np.select([rv24 > hi, rv24 < lo], ["high", "low"], "mid")
    vol_regime = pd.Series(vol_regime, index=hourly.index).where(hi.notna())
    ret24 = np.log(hourly["close"]).diff(24)
    trending = (ret24.abs() > np.sqrt(24) * rv24).where(rv24.notna())
    return pd.DataFrame({"rv24_hourly": rv24, "vol_regime": vol_regime,
                         "trending": trending.map({True: "trending", False: "range"})})


def extreme_moves(hourly: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    r = np.log(hourly["close"]).diff()
    top = r.abs().nlargest(n).index
    return pd.DataFrame({"ret_pct": 100 * r[top], "close": hourly.loc[top, "close"],
                         "volume": hourly.loc[top, "volume"]}).sort_index()


# --------------------------------------------------------------------------
# Figures (each answers one question; saved to figures/)
# --------------------------------------------------------------------------
def fig_price_and_volatility(daily: pd.DataFrame, path) -> None:
    """Q: How did the price level and volatility regime evolve?"""
    r = np.log(daily["close"]).diff()
    vol30 = r.rolling(30, min_periods=20).std() * np.sqrt(365)
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, height_ratios=[3, 2])
    ax[0].plot(daily.index, daily["close"], color=INK, lw=0.9)
    ax[0].set_yscale("log"); ax[0].set_ylabel("BTCUSDT close (log scale)")
    ax[0].set_title("Price spans ~30x; log scale makes percentage moves comparable")
    ax[1].plot(vol30.index, 100 * vol30, color=ACCENT, lw=0.9)
    ax[1].set_ylabel("30-day realised vol\n(annualised, %)")
    ax[1].axvline(config.TEST_START, color=MUTED, lw=0.8)
    ax[1].text(config.TEST_START, ax[1].get_ylim()[1] * 0.9, " test period", color=MUTED)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_return_tails(hourly_ret: pd.Series, path) -> None:
    """Q: Are hourly returns normal? (QQ-plot against a fitted normal)"""
    r = hourly_ret.dropna()
    z = (r - r.mean()) / r.std()
    q = np.linspace(0.0005, 0.9995, 400)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(stats.norm.ppf(q), np.quantile(z, q), ".", color=ACCENT, ms=3, label="hourly returns")
    lim = max(abs(np.quantile(z, [0.0005, 0.9995])).max(), 4)
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, label="normal")
    ax.set_xlabel("normal quantile"); ax.set_ylabel("standardised return quantile")
    ax.set_title("Hourly returns have fat tails"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_acf(hourly_ret: pd.Series, path, max_lag: int = 48) -> None:
    """Q: Are returns predictable from their own past? Does volatility cluster?"""
    r = hourly_ret.dropna()
    band = 1.96 / np.sqrt(len(r))
    lags = np.arange(1, max_lag + 1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for a, x, t in [(ax[0], r, "returns"), (ax[1], r.abs(), "|returns| (volatility)")]:
        a.bar(lags, acf_values(x, max_lag), color=ACCENT if t == "returns" else ACCENT2, width=0.7)
        a.axhspan(-band, band, color=MUTED, alpha=0.2, lw=0)
        a.set_title(f"Autocorrelation of hourly {t}"); a.set_xlabel("lag (hours)")
    ax[0].set_ylabel("autocorrelation (grey = +/-1.96/sqrt(n))")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_intraday(profile: pd.DataFrame, path) -> None:
    """Q: Does volatility/volume vary by hour of day (UTC)? Two panels, one axis each."""
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    ax[0].bar(profile.index, profile["mean_abs_ret_pct"], color=ACCENT, width=0.7)
    ax[0].set_ylabel("mean |hourly return| (%)")
    ax[0].set_title("Intraday seasonality: volatility (top) and relative volume (bottom)")
    ax[1].bar(profile.index, profile["rel_volume"], color=ACCENT2, width=0.7)
    ax[1].axhline(1.0, color=MUTED, lw=0.8)
    ax[1].set_ylabel("volume / trailing 30-day mean"); ax[1].set_xlabel("hour of day (UTC)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
