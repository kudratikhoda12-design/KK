"""Survival evaluation metrics: Harrell's / Uno's C-index, Breslow baseline,
IPCW Brier score and Integrated Brier Score (IBS)."""
from __future__ import annotations

import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index


def harrell_c(t, e, risk) -> float:
    """Fraction of comparable pairs ordered correctly. Higher risk should mean an
    earlier event, so lifelines (which expects 'higher = longer survival') gets -risk."""
    return float(concordance_index(t, -np.asarray(risk), e))


def bootstrap_c(t, e, risk, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(t)
    cs = [harrell_c(t[i], e[i], risk[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))


def censoring_survival(t_train, e_train):
    """KM estimate of the censoring distribution G(t) = P(C > t) (events/censoring swapped)."""
    km = KaplanMeierFitter().fit(t_train, 1 - np.asarray(e_train))
    times = km.survival_function_.index.values
    vals = km.survival_function_.values.ravel()

    def G(s, left=False):
        s = np.asarray(s, float)
        # G(s-) for left limit (used at an individual's own event time)
        idx = np.searchsorted(times, s, side="left" if left else "right") - 1
        return np.where(idx >= 0, vals[np.clip(idx, 0, None)], 1.0)
    return G


def uno_c(t_train, e_train, t, e, risk, tau=None, chunk=2000) -> float:
    """Uno's IPCW C-index (truncated at tau): robust to the censoring distribution."""
    t, e, risk = map(np.asarray, (t, e, risk))
    tau = tau if tau is not None else np.percentile(t[e == 1], 95)
    G = censoring_survival(t_train, e_train)
    w = np.where((e == 1) & (t < tau), 1.0 / np.maximum(G(t, left=True), 1e-8) ** 2, 0.0)
    num = den = 0.0
    idx = np.where(w > 0)[0]
    for s in range(0, len(idx), chunk):
        i = idx[s:s + chunk]
        comparable = t[i, None] < t[None, :]                              # j survives longer than i
        conc = (risk[i, None] > risk[None, :]) + 0.5 * (risk[i, None] == risk[None, :])
        num += (w[i, None] * comparable * conc).sum()
        den += (w[i, None] * comparable).sum()
    return float(num / den)


def breslow_baseline(t, e, risk):
    """Breslow estimator of the baseline cumulative hazard H0(t) at the unique event times."""
    t, e, r = np.asarray(t), np.asarray(e), np.exp(np.asarray(risk))
    times = np.unique(t[e == 1])
    order = np.argsort(t)
    t_sorted, r_sorted = t[order], r[order]
    at_risk = np.cumsum(r_sorted[::-1])[::-1]                            # sum exp(risk) with T >= t
    start = np.searchsorted(t_sorted, times, side="left")
    d = np.array([((t == s) & (e == 1)).sum() for s in times])
    H0 = np.cumsum(d / at_risk[start])
    return times, H0


def survival_curves(times, H0, risk, grid):
    """S(t | x) = exp(-H0(t) * exp(risk)) evaluated on `grid`. Returns (N, len(grid))."""
    H = np.interp(grid, times, H0, left=0.0)
    return np.exp(-np.outer(np.exp(risk), H))


def brier_scores(t_train, e_train, t, e, surv, grid):
    """IPCW Brier score BS(s) (Graf et al. 1999) at each time in grid."""
    G = censoring_survival(t_train, e_train)
    t, e = np.asarray(t), np.asarray(e)
    out = []
    for k, s in enumerate(grid):
        S = surv[:, k]
        had_event = (t <= s) & (e == 1)
        still_alive = t > s
        bs = (had_event * S ** 2 / np.maximum(G(t, left=True), 1e-8)
              + still_alive * (1 - S) ** 2 / max(float(G(s)), 1e-8))
        out.append(bs.mean())
    return np.array(out)


def integrated_brier(grid, bs):
    return float(np.trapezoid(bs, grid) / (grid[-1] - grid[0]))
