"""Censoring-adjusted least-squares estimators of the AFT location function (decision D42).

Adapted from KAN-AFT (Jose, Francis & Kattumannil, arXiv:2512.20305v2, Appendix A), with the documented
differences in D42. All three estimate eta(z) by least squares on log time; none estimates sigma or the error law.
Survival curves come from the Kaplan-Meier estimate of censored residuals (homoscedastic semiparametric AFT).

Censoring assumptions:
* IPCW and transform use the marginal KM of censoring G(t) = P(C > t): this requires C independent of (T, z) (CA-2)
  and positivity (G > 0 on the support of T).
* Buckley-James requires iid errors independent of z (homoscedastic AFT) and C independent of T given z (CA-1).
"""
from __future__ import annotations

import numpy as np


def km_survival(times, events):
    """Kaplan-Meier estimate. Returns (unique event-or-censor times t_j, S(t_j)) as a right-continuous step function."""
    times, events = np.asarray(times, float), np.asarray(events, bool)
    order = np.argsort(times, kind="mergesort")
    t, e = times[order], events[order]
    uniq = np.unique(t)
    n_at_risk = len(t) - np.searchsorted(t, uniq, side="left")
    d = np.array([e[t == u].sum() for u in uniq])
    S = np.cumprod(1.0 - d / n_at_risk)
    return uniq, S


def step_eval(knots, values, x, left=False):
    """Evaluate a right-continuous step function (value 1 before the first knot) at x; left=True gives f(x-)."""
    side = "left" if left else "right"
    idx = np.searchsorted(knots, np.asarray(x, float), side=side) - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, None)], 1.0)


def censoring_G(y_train, d_train):
    """Marginal KM of the censoring distribution (censoring indicator 1 - delta)."""
    return km_survival(y_train, ~np.asarray(d_train, bool))


def ipcw_weights(y, d, G):
    """w_i = delta_i / G(T_i-). Left limit avoids zero weights for events at the largest censoring time."""
    knots, vals = G
    g = step_eval(knots, vals, y, left=True)
    return np.where(np.asarray(d, bool), 1.0 / np.maximum(g, 1e-12), 0.0)


def _int_inv_G(knots, vals, t):
    """int_0^t du / G(u) for the step function G (right-continuous), exact."""
    t = np.asarray(t, float)
    edges = np.r_[0.0, knots]
    levels = np.r_[1.0, vals]                      # G on [edges[j], edges[j+1])
    out = np.zeros_like(t)
    for j in range(len(edges)):
        lo = edges[j]
        hi = edges[j + 1] if j + 1 < len(edges) else np.inf
        seg = np.clip(np.minimum(t, hi) - lo, 0, None)
        out += seg / max(levels[j], 1e-12)
    return out


def transform_targets(y, d, G):
    """Fan-Gijbels-type synthetic responses (KAN-AFT A.3): T* = phi1(T) delta + phi2(T)(1 - delta),
    phi1(t) = (1+alpha) int_0^t du/G - alpha t/G(t), phi2(t) = (1+alpha) int_0^t du/G,
    alpha = min_{i: delta_i=1} (int_0^Ti du/G - Ti) / (Ti/G(Ti) - int_0^Ti du/G)  so that T* >= 0."""
    knots, vals = G
    y, d = np.asarray(y, float), np.asarray(d, bool)
    I = _int_inv_G(knots, vals, y)
    g = np.maximum(step_eval(knots, vals, y), 1e-12)
    num, den = I[d] - y[d], y[d] / g[d] - I[d]
    ok = den > 1e-12
    alpha = float(np.min(num[ok] / den[ok])) if ok.any() else 0.0
    phi1 = (1 + alpha) * I - alpha * y / g
    phi2 = (1 + alpha) * I
    return np.where(d, phi1, phi2), alpha


def bj_impute_logscale(logy, d, eta):
    """Classical Buckley-James (log scale): e = log y - eta; KM of residuals; censored e replaced by E[e | e > e_i].
    The largest residual is treated as an event (Efron's redistribution) so the KM reaches zero."""
    e = np.asarray(logy, float) - np.asarray(eta, float)
    d = np.asarray(d, bool).copy()
    d[np.argmax(e)] = True
    knots, S = km_survival(e, d)
    mass = -np.diff(np.r_[1.0, S])                 # KM jump at each knot
    out = e.copy()
    for i in np.nonzero(~d)[0]:
        above = knots > e[i]
        Si = step_eval(knots, S, e[i])
        if Si > 1e-12 and above.any():
            out[i] = np.sum(knots[above] * mass[above]) / Si
    return np.asarray(eta, float) + out


def bj_impute_timescale(logy, d, eta):
    """KAN-AFT A.1 variant: r = T exp(-eta); censored r replaced by E[r | r > r_i]; pseudo log response
    eta + log r* (differs from the log-scale version by Jensen's inequality; D42)."""
    r = np.exp(np.asarray(logy, float) - np.asarray(eta, float))
    d = np.asarray(d, bool).copy()
    d[np.argmax(r)] = True
    knots, S = km_survival(r, d)
    mass = -np.diff(np.r_[1.0, S])
    rs = r.copy()
    for i in np.nonzero(~d)[0]:
        above = knots > r[i]
        Si = step_eval(knots, S, r[i])
        if Si > 1e-12 and above.any():
            rs[i] = np.sum(knots[above] * mass[above]) / Si
    return np.asarray(eta, float) + np.log(rs)


def residual_survival(logy, d, eta):
    """KM of censored residuals e = log y - eta (training data). Returns callable S(t | eta_new) -> (N, T)."""
    knots, S = km_survival(np.asarray(logy) - np.asarray(eta), d)

    def surv(t, eta_new):
        z = np.log(np.asarray(t, float))[None, :] - np.asarray(eta_new, float)[:, None]
        return step_eval(knots, S, z.ravel()).reshape(z.shape)
    return surv


def fit_censored_ls(fit_fn, predict_fn, y, d, method, G=None, max_iter=50, tol=1e-3, bj_variant="log"):
    """Generic driver. fit_fn(target, weights) fits a regressor of log-time; predict_fn() returns eta on training data.

    method: "ipcw" | "transform" | "bj". Returns dict(info...)."""
    y, d = np.asarray(y, float), np.asarray(d, bool)
    logy = np.log(y)
    if method == "ipcw":
        w = ipcw_weights(y, d, G)
        fit_fn(logy, w)
        return {"weights_max": float(w.max()), "n_weighted": int((w > 0).sum())}
    if method == "transform":
        Ts, alpha = transform_targets(y, d, G)
        if np.any(Ts <= 0):
            raise ValueError("transform produced non-positive synthetic responses")
        fit_fn(np.log(Ts), np.ones_like(y))
        return {"alpha": alpha, "min_Tstar": float(Ts.min())}
    if method == "bj":
        imp = bj_impute_logscale if bj_variant == "log" else bj_impute_timescale
        eta = np.zeros_like(y)
        hist = []
        for k in range(max_iter):
            target = imp(logy, d, eta)
            fit_fn(target, np.ones_like(y))
            new = predict_fn()
            change = float(np.mean(np.abs(new - eta)))
            hist.append(change)
            eta = new
            if change < tol:
                break
        return {"iterations": k + 1, "converged": hist[-1] < tol, "last_change": hist[-1]}
    raise ValueError(method)


class LinearLS:
    """Weighted least-squares linear regressor, for the estimator tests."""

    def __init__(self, X):
        self.X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        self.beta = np.zeros(self.X.shape[1])

    def fit(self, target, w):
        W = np.asarray(w, float)
        A = self.X.T @ (self.X * W[:, None])
        self.beta = np.linalg.solve(A + 1e-10 * np.eye(len(A)), self.X.T @ (W * target))

    def predict(self, X=None):
        Xd = self.X if X is None else np.column_stack([np.ones(len(X)), X])
        return Xd @ self.beta
