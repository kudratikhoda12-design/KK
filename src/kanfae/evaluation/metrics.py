"""Evaluation metrics (docs/experiment_protocol.md section 5; validated by Q1..Q5).

Censoring conventions (docs/design.md section 2.4):
* marginal IPCW (assumes CA-2): G = reverse Kaplan-Meier of the training censoring times, weights 1/G(Y_i)
  for cases and 1/G(t) for controls (scikit-survival convention);
* conditional IPCW (sensitivity analysis): G_i(t) = P(C > t | X_i) from a Cox model for censoring fitted on
  the training split; weights truncated at their 95th percentile.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _surv_struct(y, d):
    from sksurv.util import Surv
    return Surv.from_arrays(event=np.asarray(d, bool), time=np.asarray(y, float))


def marginal_censoring_G(y_train, d_train):
    """Return a callable G(t) = reverse-KM censoring survival (scikit-survival CensoringDistributionEstimator)."""
    from sksurv.nonparametric import CensoringDistributionEstimator
    est = CensoringDistributionEstimator().fit(_surv_struct(y_train, d_train))
    return lambda t: est.predict_proba(np.atleast_1d(np.asarray(t, float)))


def brier_ipcw(S, times, y, d, G_y, G_t):
    """IPCW Brier score at each time.

    S: (N, T) predicted survival at `times`; G_y: (N,) censoring survival at Y_i; G_t: (T,) or (N, T)."""
    S, y, d = np.asarray(S, float), np.asarray(y, float), np.asarray(d, bool)
    G_y = np.where(np.asarray(G_y, float) == 0, np.inf, G_y)
    G_t = np.asarray(G_t, float)
    G_t = np.where(G_t == 0, np.inf, G_t)
    if G_t.ndim == 1:
        G_t = np.broadcast_to(G_t[None, :], S.shape)
    t = np.asarray(times, float)[None, :]
    case = (y[:, None] <= t) & d[:, None]
    ctrl = y[:, None] > t
    return np.mean(np.square(S) * case / G_y[:, None] + np.square(1 - S) * ctrl / G_t, axis=0)


def integrated(times, values):
    times = np.asarray(times, float)
    return float(np.trapezoid(values, times) / (times[-1] - times[0]))


def ibs_marginal(S, times, y, d, y_train, d_train):
    G = marginal_censoring_G(y_train, d_train)
    # G(Y_i) only enters for cases with Y_i <= t <= times[-1]; evaluating at min(Y_i, times[-1]) is exact and
    # avoids evaluating the training censoring KM beyond its support for long test follow-up.
    y_eval = np.minimum(np.asarray(y, float), np.asarray(times, float)[-1])
    bs = brier_ipcw(S, times, y, d, G(y_eval), G(times))
    return integrated(times, bs), bs


def conditional_censoring_model(y_train, d_train, Xc_train, penalizer=0.1):
    """Cox model for the censoring time (event = 1 - delta). Returns a callable G(t, Xc) -> (N, T)."""
    import pandas as pd
    from lifelines import CoxPHFitter
    cols = [f"x{j}" for j in range(Xc_train.shape[1])]
    df = pd.DataFrame(Xc_train, columns=cols)
    df["T"], df["E"] = np.asarray(y_train, float), (~np.asarray(d_train, bool)).astype(int)
    cph = CoxPHFitter(penalizer=penalizer).fit(df, "T", "E")

    def G(t, Xc):
        sf = cph.predict_survival_function(pd.DataFrame(Xc, columns=cols), times=np.atleast_1d(t))
        return sf.to_numpy().T
    return G


def ibs_conditional(S, times, y, d, Xc, G_model, trunc_q=0.95):
    G_t = G_model(times, Xc)                                       # (N, T)
    G_y = np.array([G_model(np.array([yi]), Xc[i:i + 1])[0, 0] for i, yi in enumerate(y)])
    w_all = np.r_[1 / np.clip(G_y, 1e-12, None), (1 / np.clip(G_t, 1e-12, None)).ravel()]
    cap = np.quantile(w_all, trunc_q)
    G_y_c, G_t_c = np.maximum(G_y, 1 / cap), np.maximum(G_t, 1 / cap)
    bs = brier_ipcw(S, times, y, d, G_y_c, G_t_c)
    return integrated(times, bs), bs, float(cap)


def harrell_c(y, d, risk):
    """Harrell's C: among comparable pairs (Y_i < Y_j, delta_i = 1) the fraction with risk_i > risk_j (ties 0.5)."""
    y, d, r = np.asarray(y, float), np.asarray(d, bool), np.asarray(risk, float)
    num = den = 0.0
    for i in np.nonzero(d)[0]:
        comp = y > y[i]
        den += comp.sum()
        num += (r[i] > r[comp]).sum() + 0.5 * (r[i] == r[comp]).sum()
    return num / den if den > 0 else np.nan


def uno_c(y_train, d_train, y, d, risk, tau):
    from sksurv.metrics import concordance_index_ipcw
    return float(concordance_index_ipcw(_surv_struct(y_train, d_train), _surv_struct(y, d),
                                        np.asarray(risk, float), tau=tau)[0])


def survival_curve_error(S_pred, S_true, grid):
    """mean_i (1/t_max) int_0^{t_max} |S_pred_i - S_true_i| dt on `grid` (grid[0] = 0, grid[-1] = t_max)."""
    grid = np.asarray(grid, float)
    return float(np.mean(np.trapezoid(np.abs(S_pred - S_true), grid, axis=1) / (grid[-1] - grid[0])))


def d_calibration(S_at_y, d, n_bins=10):
    """Haider et al. D-calibration with uniform spreading of censored mass below S(c_i). Returns (chi2, p, counts)."""
    s = np.clip(np.asarray(S_at_y, float), 0.0, 1.0)
    d = np.asarray(d, bool)
    edges = np.linspace(0, 1, n_bins + 1)
    counts = np.zeros(n_bins)
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1)
    np.add.at(counts, idx[d], 1.0)
    for si, b in zip(s[~d], idx[~d]):
        if si <= 0:
            counts[0] += 1.0
            continue
        counts[b] += (si - edges[b]) / si
        counts[:b] += (edges[1:b + 1] - edges[:b]) / si
    exp = len(s) / n_bins
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    return chi2, float(stats.chi2.sf(chi2, n_bins - 1)), counts


def km_curve(y, d, times):
    from sksurv.nonparametric import kaplan_meier_estimator
    t, s = kaplan_meier_estimator(np.asarray(d, bool), np.asarray(y, float))
    idx = np.searchsorted(t, times, side="right") - 1
    return np.where(idx >= 0, s[np.clip(idx, 0, None)], 1.0)


def calibration_in_the_large(S_h, y, d, horizons):
    """Mean predicted S(t_h) vs Kaplan-Meier S(t_h) on the evaluation set."""
    return np.asarray(S_h).mean(axis=0), km_curve(y, d, horizons)


def ibs_time_grid(y_test, y_train=None, n=50, lo_q=0.1, hi_q=0.9):
    """Equally spaced grid between the 10th and 90th percentiles of observed test times, capped below the
    largest test and training times (the censoring KM is undefined beyond the training support)."""
    lo, hi = np.quantile(y_test, [lo_q, hi_q])
    cap = np.max(y_test) if y_train is None else min(np.max(y_test), np.max(y_train))
    hi = min(hi, cap * (1 - 1e-9))
    return np.linspace(lo, hi, n)
