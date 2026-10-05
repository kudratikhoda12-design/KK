"""Statistical tools that respect time-series dependence.

Hourly returns are not independent (volatility clusters), so i.i.d. formulas
understate uncertainty. We use:
* the stationary bootstrap (Politis & Romano, 1994) - resamples blocks of
  random length (mean L) so short-range dependence is preserved;
* Newey-West (HAC) standard errors for means of dependent series;
* Holm's step-down correction when several hypotheses are tested together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import config


def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) array of resampled positions. Blocks wrap around the end.

    Each position starts a new block with probability 1/mean_block (position 0
    always does); within a block, positions advance by one. Vectorised: for
    every position we find where its block began and add the offset.
    """
    p = 1.0 / mean_block
    new_block = rng.random((n_boot, n)) < p
    new_block[:, 0] = True
    starts = rng.integers(0, n, (n_boot, n))
    pos = np.arange(n)
    block_begin = np.maximum.accumulate(np.where(new_block, pos, 0), axis=1)
    rows = np.arange(n_boot)[:, None]
    return (starts[rows, block_begin] + (pos - block_begin)) % n


def bootstrap_stat(arrays: tuple[np.ndarray, ...], func, mean_block: float,
                   n_boot: int = config.BOOTSTRAP_N, seed: int = config.RANDOM_SEED,
                   batch: int = 250) -> np.ndarray:
    """Distribution of func(*resampled arrays) under the stationary bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    out = []
    for start in range(0, n_boot, batch):
        idx = stationary_bootstrap_indices(n, mean_block, min(batch, n_boot - start), rng)
        out.extend(func(*(a[i] for a in arrays)) for i in idx)
    return np.asarray(out, dtype="float64")


def percentile_ci(dist: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    d = dist[np.isfinite(dist)]
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))


def hac_mean_test(x: np.ndarray, maxlags: int | None = None) -> dict:
    """Test H0: E[x] = 0 with a Newey-West standard error."""
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if maxlags is None:
        maxlags = int(np.floor(4 * (len(x) / 100) ** (2 / 9)))   # Newey-West rule of thumb
    res = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return dict(mean=float(res.params[0]), se=float(res.bse[0]), t=float(res.tvalues[0]),
                p_value=float(res.pvalues[0]), n=len(x), maxlags=maxlags)


def spearman_block_test(x: np.ndarray, y: np.ndarray, mean_block: float,
                        n_boot: int = 1000, seed: int = config.RANDOM_SEED) -> dict:
    """Spearman correlation with a block-bootstrap two-sided p-value.

    The null distribution is built by bootstrapping x and y with INDEPENDENT
    block draws, which destroys the x-y link but keeps each series' own
    dependence.
    """
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    rho = stats.spearmanr(x, y).statistic
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rng = np.random.default_rng(seed)
    n = len(x)
    null = []
    for start in range(0, n_boot, 100):
        k = min(100, n_boot - start)
        ix = stationary_bootstrap_indices(n, mean_block, k, rng)
        iy = stationary_bootstrap_indices(n, mean_block, k, rng)
        for a, b in zip(ix, iy):
            null.append(np.corrcoef(rx[a], ry[b])[0, 1])
    null = np.asarray(null)
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (len(null) + 1))
    return dict(spearman=float(rho), p_block=p, naive_p=float(stats.spearmanr(x, y).pvalue), n=n)


def holm(pvalues: pd.Series, alpha: float = config.ALPHA) -> pd.DataFrame:
    """Holm-Bonferroni step-down; returns adjusted p-values and reject flags."""
    p = pvalues.sort_values()
    m = len(p)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(p.to_numpy())])
    adj = np.minimum(adj, 1.0)
    out = pd.DataFrame({"p_value": p, "p_holm": adj}, index=p.index)
    out["reject_at_5pct"] = out["p_holm"] < alpha
    return out.loc[pvalues.index]
