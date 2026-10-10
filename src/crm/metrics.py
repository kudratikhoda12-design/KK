"""Discrimination, calibration and stability statistics with formal tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-12


# --------------------------------------------------------------------------
# Discrimination
# --------------------------------------------------------------------------
def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def _midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x, kind="mergesort")
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def _midrank_fast(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def delong_cov(y, preds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """AUCs and their DeLong covariance matrix (Sun & Xu 2014 fast algorithm)."""
    y = np.asarray(y).astype(bool)
    m, n = y.sum(), (~y).sum()
    k = len(preds)
    aucs = np.empty(k)
    v01 = np.empty((k, m))
    v10 = np.empty((k, n))
    for r, p in enumerate(preds):
        p = np.asarray(p, dtype=float)
        pos, neg = p[y], p[~y]
        tx, ty, tz = _midrank_fast(pos), _midrank_fast(neg), _midrank_fast(np.r_[pos, neg])
        aucs[r] = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v01[r] = (tz[:m] - tx) / n
        v10[r] = 1.0 - (tz[m:] - ty) / m
    s01 = np.atleast_2d(np.cov(v01))
    s10 = np.atleast_2d(np.cov(v10))
    return aucs, s01 / m + s10 / n


def auc_ci(y, p, alpha=0.05) -> tuple[float, float, float]:
    a, c = delong_cov(y, [p])
    se = np.sqrt(c[0, 0])
    z = stats.norm.ppf(1 - alpha / 2)
    return float(a[0]), float(a[0] - z * se), float(a[0] + z * se)


def delong_test(y, p1, p2) -> dict:
    """Paired two-sided DeLong test H0: AUC1 == AUC2."""
    a, c = delong_cov(y, [p1, p2])
    var = c[0, 0] + c[1, 1] - 2 * c[0, 1]
    z = (a[0] - a[1]) / np.sqrt(var)
    return {"auc1": a[0], "auc2": a[1], "diff": a[0] - a[1],
            "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}


def holm(pvals: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values."""
    order = pvals.sort_values()
    m = len(order)
    adj = np.maximum.accumulate([(m - i) * p for i, p in enumerate(order.values)])
    return pd.Series(np.minimum(adj, 1.0), index=order.index).reindex(pvals.index)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y, p) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def hosmer_lemeshow(y, p, g=10) -> dict:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), g, labels=False)
    agg = df.groupby("bin").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    pbar = agg["exp"] / agg["n"]
    chi2 = (((agg["obs"] - agg["exp"]) ** 2) / (agg["n"] * pbar * (1 - pbar))).sum()
    return {"hl_chi2": float(chi2), "hl_df": g - 2, "hl_p": float(stats.chi2.sf(chi2, g - 2))}


def spiegelhalter(y, p) -> dict:
    """Spiegelhalter (1986) Z-test of overall calibration."""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    z = num / den
    return {"spiegelhalter_z": float(z), "spiegelhalter_p": float(2 * stats.norm.sf(abs(z)))}


def calibration_slope_intercept(y, p) -> dict:
    """Logistic recalibration: logit(P(y)) = a + b*logit(p).

    Calibration-in-the-large is the intercept with slope fixed to 1 (offset).
    Ideal: intercept 0, slope 1. Wald CIs from statsmodels.
    """
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    lp = np.log(p / (1 - p))
    y = np.asarray(y)
    m1 = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    m0 = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit()
    ci1, ci0 = m1.conf_int(), m0.conf_int()
    slope_z = (m1.params[1] - 1) / m1.bse[1]
    return {
        "citl": float(m0.params[0]), "citl_lo": float(ci0[0, 0]), "citl_hi": float(ci0[0, 1]),
        "citl_p": float(m0.pvalues[0]),
        "slope": float(m1.params[1]), "slope_lo": float(ci1[1, 0]), "slope_hi": float(ci1[1, 1]),
        "slope_p_vs1": float(2 * stats.norm.sf(abs(slope_z))),
    }


def ece(y, p, bins=20) -> float:
    """Expected calibration error over equal-frequency bins."""
    df = pd.DataFrame({"y": y, "p": p})
    df["b"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    a = df.groupby("b").agg(o=("y", "mean"), e=("p", "mean"), n=("y", "size"))
    return float(np.sum(a["n"] * (a["o"] - a["e"]).abs()) / a["n"].sum())


def binomial_backtest(df: pd.DataFrame, group: str, y: str = "y", p: str = "p") -> pd.DataFrame:
    """Per-bucket calibration backtest (ECB/EBA style).

    Two-sided exact binomial test and the Jeffreys test (ECB TRIM): the
    p-value is P(DR_true <= PD) under the Beta(D+1/2, N-D+1/2) posterior;
    a small value means the PD is too low (under-estimation).
    """
    rows = []
    for g, d in df.groupby(group, observed=True):
        n, k, pd_ = len(d), int(d[y].sum()), float(d[p].mean())
        binom_p = stats.binomtest(k, n, pd_).pvalue if n else np.nan
        jeff = stats.beta.cdf(pd_, k + 0.5, n - k + 0.5)
        rows.append({group: g, "n": n, "defaults": k, "pd_mean": pd_, "odr": k / n,
                     "binom_p_two_sided": binom_p, "jeffreys_p": jeff})
    out = pd.DataFrame(rows)
    out["traffic_light"] = np.select(
        [out["jeffreys_p"] < 0.01, out["jeffreys_p"] < 0.05, out["jeffreys_p"] > 0.99],
        ["red (PD too low)", "amber (PD too low)", "conservative"], "green")
    return out


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------
def psi(expected, actual, bins=10, edges=None) -> float:
    """Population stability index on quantile bins of the expected sample."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    if edges is None:
        edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1)))
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected[~np.isnan(expected)], edges)[0].astype(float)
    a = np.histogram(actual[~np.isnan(actual)], edges)[0].astype(float)
    # Missing values form their own bin.
    e = np.r_[e, np.isnan(expected).sum()]
    a = np.r_[a, np.isnan(actual).sum()]
    e, a = e / e.sum(), a / a.sum()
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(v: float) -> str:
    return "stable" if v < 0.10 else ("moderate shift" if v < 0.25 else "significant shift")


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def summary(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    auc, lo, hi = auc_ci(y, p)
    out = {"n": len(y), "dr": float(y.mean()), "pd_mean": float(p.mean()),
           "auc": auc, "auc_lo": lo, "auc_hi": hi, "gini": 2 * auc - 1,
           "ks": ks_stat(y, p), "brier": brier(y, p), "log_loss": log_loss(y, p),
           "ece": ece(y, p)}
    out.update(hosmer_lemeshow(y, p))
    out.update(spiegelhalter(y, p))
    out.update(calibration_slope_intercept(y, p))
    return out


def bootstrap_ci(y, p, fn, n_boot=500, seed=0, alpha=0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(fn(y[i], p[i]))
    return tuple(np.quantile(vals, [alpha / 2, 1 - alpha / 2]))
