"""Statistical tests for predictive content, built for overlapping, autocorrelated, heteroskedastic data.

* HAC (Newey-West) OLS: forward returns over h minutes overlap, so the residuals are MA(h-1) by construction;
  the Bartlett kernel uses maxlags = 2h (at least h-1 plus a margin for volatility clustering).
* Moving-block bootstrap over UTC days for decile spreads (keeps intraday dependence inside each block).
* Holm (FWER) and Benjamini-Hochberg (FDR) adjustments across the whole family of tests.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats


def hac_ols(y: np.ndarray, X: np.ndarray, names: list[str], maxlags: int) -> dict:
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xc = sm.add_constant(X[m], has_constant="add")
    res = sm.OLS(y[m], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags, "kernel": "bartlett"})
    out = {"n": int(m.sum()), "r2": float(res.rsquared)}
    for i, nm in enumerate(["const"] + names):
        out[f"b_{nm}"] = float(res.params[i])
        out[f"t_{nm}"] = float(res.tvalues[i])
        out[f"p_{nm}"] = float(res.pvalues[i])
    return out


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - np.nanmean(x)) / np.nanstd(x)


def decile_table(x: np.ndarray, y: np.ndarray, q: int = 10) -> pl.DataFrame:
    m = np.isfinite(x) & np.isfinite(y)
    edges = np.quantile(x[m], np.linspace(0, 1, q + 1))
    d = np.clip(np.searchsorted(edges, x[m], side="right") - 1, 0, q - 1)
    df = pl.DataFrame({"d": d + 1, "x": x[m], "y": y[m]})
    return df.group_by("d").agg(n=pl.len(), x_mean=pl.col("x").mean(), y_mean_bps=pl.col("y").mean() * 1e4,
                                y_se_naive_bps=pl.col("y").std() / pl.len().sqrt() * 1e4).sort("d")


def block_bootstrap_spread(x: np.ndarray, y: np.ndarray, day: np.ndarray, n_boot: int = 1000,
                           seed: int = 7, q: int = 10, block_days: int = 5) -> dict:
    """Top-minus-bottom decile spread of y, with a moving-block bootstrap over days.
    Decile edges are fixed on the full sample (an estimate of the population cut-offs), then whole blocks
    of consecutive days are resampled with replacement."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y, day = x[m], y[m], day[m]
    lo, hi = np.quantile(x, [1 / q, 1 - 1 / q])
    udays, inv = np.unique(day, return_inverse=True)
    nd = len(udays)
    # per-day sums so a resample is a cheap aggregation
    top = x >= hi
    bot = x <= lo
    st = np.bincount(inv, weights=y * top, minlength=nd); nt = np.bincount(inv, weights=top, minlength=nd)
    sb = np.bincount(inv, weights=y * bot, minlength=nd); nb = np.bincount(inv, weights=bot, minlength=nd)
    point = st.sum() / nt.sum() - sb.sum() / nb.sum()
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(nd / block_days))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, nd - block_days + 1, size=nblocks)
        idx = (starts[:, None] + np.arange(block_days)).ravel()[:nd]
        boots[b] = st[idx].sum() / nt[idx].sum() - sb[idx].sum() / nb[idx].sum()
    se = boots.std(ddof=1)
    # two-sided p-value from the bootstrap-se normal approximation (centred bootstrap)
    p = 2 * stats.norm.sf(abs(point) / se) if se > 0 else float("nan")
    return {"spread_bps": point * 1e4, "boot_se_bps": se * 1e4, "ci95_lo_bps": np.quantile(boots, 0.025) * 1e4,
            "ci95_hi_bps": np.quantile(boots, 0.975) * 1e4, "p_value": float(p), "n_days": int(nd)}


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out
