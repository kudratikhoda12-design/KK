"""Additive functional Cox model, AFCM (Cui, Crainiceanu & Leroux 2021): Python re-implementation (D112, deviation).

eta_i = sum_d int F_d(X_id(t), t) dt + z_i' gamma,  F_d(x, t) = sum_jk theta_djk B_j(x) B_k(t),
B: cubic B-splines, 10 x 10 tensor basis per channel (x basis on the training range of X_d, t basis on [0, 1]).
Penalised partial likelihood (Efron):
  mean NLPL + lam_x sum_d theta_d' (D2x'D2x (x) I) theta_d + lam_t sum_d theta_d' (I (x) D2t'D2t) theta_d
              + ridge ||theta||^2,
second-order difference penalties (P-splines). (lam_x, lam_t) chosen by validation partial likelihood on a 5 x 5 grid
instead of REML in mgcv (R not available; lambda_t grid widened to 1e-4..1e4 by D120); ridge = 1e-6 removes the direction that the baseline absorbs (any F that is
a function of t only). The objective is convex; fitted by L-BFGS with strong-Wolfe line search.
Curves enter on the 50-point grid with trapezoid quadrature, the same quadrature as the functional KAN.
"""
from __future__ import annotations

import numpy as np
import torch

from ..functional.basis import BSplineBasis
from ..survival.cox import breslow_baseline, cox_nll, cox_survival

GRID = np.linspace(0.0, 1.0, 50)
LAM_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)                  # lambda_x (and lambda_t in Round 11)
LAM_T_GRID = (1e-4, 1e-2, 1.0, 1e2, 1e4)                  # lambda_t from D120: lets AFCM shrink dF/dt towards 0
LAM_X_WIDE = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1)
# D123 (owner, L39): the repaired AFCM adopted from the development diagnosis on seed 914 (variant V3 of
# scripts/diag_round12_afcm_fix.py). Used by CK-6c and Round 12; the defaults of fit_afcm stay the Round-12a model.
AFCM_D123 = dict(lams_x=LAM_X_WIDE, lams_t=LAM_T_GRID, pen_norm=True, t_pen="d1d2")


def _trap_w(grid):
    w = np.full(len(grid), grid[1] - grid[0])
    w[[0, -1]] *= 0.5
    return w


def _d2_penalty(K):
    D2 = np.diff(np.eye(K), 2, axis=0)
    return D2.T @ D2


def _d1_penalty(K):
    D1 = np.diff(np.eye(K), 1, axis=0)
    return D1.T @ D1


class AFCM:
    """Options added by D123 (defaults reproduce the Round-11/12a model exactly):
    pen_norm  divide the x penalty by n_t and the t penalty by n_x (penalty per tensor row/column);
    t_pen     "d2" (second differences in t) or "d1d2" (first + second differences: a large lam_t then removes
              linear-in-t tilts b(x) t as well as curvature);
    lam_x, lam_t  scalars or per-channel sequences;
    xin       "raw" (x basis on the training range of X_d) or "quantile" (Cui et al. 2021: x~ = F_t(X(t)), the
              training empirical CDF at each grid time, x basis on [0, 1])."""

    def __init__(self, n_x=10, n_t=10, lam_x=1e-2, lam_t=1e-2, ridge=1e-6, ties="efron", max_iter=500,
                 pen_norm=False, t_pen="d2", xin="raw"):
        self.n_x, self.n_t, self.lam_x, self.lam_t, self.ridge, self.ties = n_x, n_t, lam_x, lam_t, ridge, ties
        self.max_iter, self.pen_norm, self.t_pen, self.xin = max_iter, pen_norm, t_pen, xin
        self.bt = BSplineBasis(n_t, 3, (0.0, 1.0))

    # ---- design --------------------------------------------------------------------------------------------------
    def _set_ranges(self, xg):
        if self.xin == "quantile":
            self.qs = np.sort(xg, axis=0)                                  # (n, G, D) training order statistics
            n = len(xg)
            self.qp = (np.arange(1, n + 1) - 0.5) / n
            self.lo, self.hi = np.zeros(xg.shape[2]), np.ones(xg.shape[2])
        else:
            self.lo, self.hi = xg.min(axis=(0, 1)), xg.max(axis=(0, 1))
        self.bx = [BSplineBasis(self.n_x, 3, (float(a), float(b))) for a, b in zip(self.lo, self.hi)]

    def _q(self, d, x, g):
        """Empirical CDF of channel d at grid index g (array, broadcast with x), linear between order statistics."""
        x, g = np.broadcast_arrays(np.asarray(x, float), np.asarray(g))
        out = np.empty(x.shape)
        for gi in np.unique(g):
            m = g == gi
            out[m] = np.interp(x[m], self.qs[:, gi, d], self.qp)
        return out

    def _bx(self, d, x, g=None):
        if self.xin == "quantile":
            x = self._q(d, x, np.arange(len(GRID))[None, :] if g is None else g)
        return self.bx[d](np.clip(x, self.lo[d], self.hi[d]))           # constant extrapolation outside the range

    def design(self, xg):
        """W (n, D, n_x * n_t): W_i,d,(j,k) = int B_j(X_id(t)) B_k(t) dt (trapezoid on GRID)."""
        n, G, D = xg.shape
        Bt = self.bt(GRID) * _trap_w(GRID)[:, None]                       # (G, n_t)
        out = np.empty((n, D, self.n_x * self.n_t))
        for d in range(D):
            Bx = self._bx(d, xg[:, :, d])                                 # (n, G, n_x)
            out[:, d] = np.einsum("ngj,gk->njk", Bx, Bt).reshape(n, -1)
        return out

    def _penalty_mats(self):
        Px, Pt = _d2_penalty(self.n_x), _d2_penalty(self.n_t)
        if self.t_pen == "d1d2":
            Pt = Pt + _d1_penalty(self.n_t)
        It, Ix = np.eye(self.n_t), np.eye(self.n_x)
        Kx, Kt = np.kron(Px, It), np.kron(Ix, Pt)
        if self.pen_norm:
            Kx, Kt = Kx / self.n_t, Kt / self.n_x
        return torch.tensor(Kx), torch.tensor(Kt)

    def _lam(self, lam, k):
        return lam[k] if np.ndim(lam) else lam

    # ---- fit -----------------------------------------------------------------------------------------------------
    def fit(self, xg, z, y, d, theta0=None):
        torch.set_default_dtype(torch.float64)
        self._set_ranges(xg)
        self.zm = z.mean(0) if z is not None and z.shape[1] else np.zeros(0)
        self.zs = np.where(z.std(0) > 0, z.std(0), 1.0) if z is not None and z.shape[1] else np.ones(0)
        return self._fit_design(self.design(xg), self._z(z, len(y)), y, d, theta0)

    def _z(self, z, n):
        return (z - self.zm) / self.zs if z is not None and z.shape[1] else np.zeros((n, 0))

    def _fit_design(self, W, Z, y, d, theta0=None):
        n, D, P = W.shape
        Wt, Zt = torch.tensor(W.reshape(n, D * P)), torch.tensor(Z)
        yt, dt = torch.tensor(y, dtype=torch.float64), torch.tensor(np.asarray(d, float))
        Px, Pt = self._penalty_mats()
        theta = torch.zeros(D * P, requires_grad=True) if theta0 is None else torch.tensor(theta0[0], requires_grad=True)
        gamma = torch.zeros(Z.shape[1], requires_grad=True) if theta0 is None else torch.tensor(theta0[1], requires_grad=True)
        opt = torch.optim.LBFGS([theta, gamma], lr=1.0, max_iter=self.max_iter, tolerance_grad=1e-9,
                                tolerance_change=1e-12, history_size=50, line_search_fn="strong_wolfe")

        def objective():
            th = theta.reshape(D, P)
            pen = sum(self._lam(self.lam_x, k) * th[k] @ Px @ th[k] + self._lam(self.lam_t, k) * th[k] @ Pt @ th[k]
                      for k in range(D))
            return cox_nll(Wt @ theta + Zt @ gamma, yt, dt, self.ties) + pen + self.ridge * (theta @ theta)

        def closure():
            opt.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        opt.step(closure)
        self.theta = theta.detach().numpy().reshape(D, self.n_x, self.n_t)
        self.gamma = gamma.detach().numpy()
        self.D = D
        eta = W.reshape(n, -1) @ self.theta.reshape(-1) + Z @ self.gamma
        self.baseline = breslow_baseline(eta, y, d)
        return self

    # ---- prediction ----------------------------------------------------------------------------------------------
    def eta(self, xg, z):
        W = self.design(xg)
        return W.reshape(len(W), -1) @ self.theta.reshape(-1) + self._z(z, len(W)) @ self.gamma

    def predict_risk(self, xg, z):
        return self.eta(xg, z)

    def predict_survival(self, xg, z, times):
        return cox_survival(self.eta(xg, z), self.baseline[None], times)

    def val_nll(self, xg, z, y, d):
        return float(cox_nll(torch.tensor(self.eta(xg, z)), torch.tensor(y, dtype=torch.float64),
                             torch.tensor(np.asarray(d, float)), self.ties))

    def effect(self, d, x, t):
        """Fitted F_d(x, t) on arrays x, t (broadcast)."""
        x = np.asarray(x, float)
        t = np.broadcast_to(np.asarray(t, float), x.shape)
        g = np.clip(np.rint(t.ravel() * (len(GRID) - 1)).astype(int), 0, len(GRID) - 1)    # quantile input: nearest grid t
        Bx = self._bx(d, x.ravel(), g) if self.xin == "quantile" else self._bx(d, x.ravel())
        Bt = self.bt(t.ravel())
        return np.einsum("nj,jk,nk->n", Bx, self.theta[d], Bt).reshape(x.shape)


def fit_afcm(train, val, lams_x=LAM_GRID, lams_t=LAM_T_GRID, per_channel=False, rule="min", tol=0.01, **kw):
    """Grid over (lam_x, lam_t), selected by validation Efron NLPL. train/val: dicts xg, z, y, d.
    per_channel (D123 option): after the common grid, one coordinate sweep over channels, each choosing its own
    (lam_x_d, lam_t_d) on the same grid with the others fixed. per_channel "t" (D133): the same sweep over lam_t only,
    lam_x staying common. rule "parsimonious" (D133 dry run; common grid only): among grid points whose validation
    NLPL is within tol of the minimum, the largest lam_t (smoothest in t), then the largest lam_x -- the D120 rule and
    tolerance. Returns (best model, records)."""
    torch.set_default_dtype(torch.float64)
    proto = AFCM(**kw)
    proto._set_ranges(train["xg"])
    proto.zm = train["z"].mean(0)
    proto.zs = np.where(train["z"].std(0) > 0, train["z"].std(0), 1.0)
    W = proto.design(train["xg"])
    Ztr = proto._z(train["z"], len(W))
    recs, best, models = [], None, []
    for lx in lams_x:
        for lt in lams_t:
            m = _fit_one(proto, kw, lx, lt, W, Ztr, train)
            v = m.val_nll(val["xg"], val["z"], val["y"], val["d"])
            recs.append(dict(lam_x=lx, lam_t=lt, val_nll=v))
            models.append(m)
            if best is None or v < best[0] - 1e-12:
                best = (v, m)
    if rule == "parsimonious":
        vmin = min(r["val_nll"] for r in recs)
        k = max((i for i, r in enumerate(recs) if r["val_nll"] <= vmin + tol), key=lambda i: (recs[i]["lam_t"], recs[i]["lam_x"]))
        recs[k]["chosen"] = True
        best = (recs[k]["val_nll"], models[k])
    if per_channel == "t":                         # D133: per-channel lambda_t only, lambda_x kept common
        D = W.shape[1]
        lt_d = [best[1].lam_t] * D
        for d in range(D):
            for lt in lams_t:
                ct = list(lt_d)
                ct[d] = lt
                m = _fit_one(proto, kw, best[1].lam_x, ct, W, Ztr, train)
                v = m.val_nll(val["xg"], val["z"], val["y"], val["d"])
                recs.append(dict(channel=d, lam_x=best[1].lam_x, lam_t=ct, val_nll=v))
                if v < best[0] - 1e-12:
                    best = (v, m)
            lt_d = list(np.broadcast_to(best[1].lam_t, D))
    elif per_channel:
        D = W.shape[1]
        lx_d, lt_d = [best[1].lam_x] * D, [best[1].lam_t] * D
        for d in range(D):
            for lx in lams_x:
                for lt in lams_t:
                    cx, ct = list(lx_d), list(lt_d)
                    cx[d], ct[d] = lx, lt
                    m = _fit_one(proto, kw, cx, ct, W, Ztr, train)
                    v = m.val_nll(val["xg"], val["z"], val["y"], val["d"])
                    recs.append(dict(channel=d, lam_x=cx, lam_t=ct, val_nll=v))
                    if v < best[0] - 1e-12:
                        best = (v, m)
            lx_d, lt_d = list(np.broadcast_to(best[1].lam_x, D)), list(np.broadcast_to(best[1].lam_t, D))
    return best[1], recs


def _fit_one(proto, kw, lx, lt, W, Ztr, train):
    m = AFCM(**{**kw, "lam_x": lx, "lam_t": lt})
    m.lo, m.hi, m.bx, m.zm, m.zs = proto.lo, proto.hi, proto.bx, proto.zm, proto.zs
    if m.xin == "quantile":
        m.qs, m.qp = proto.qs, proto.qp
    return m._fit_design(W, Ztr, train["y"], train["d"])


def fit_afcm_bagged(train, val, B=3, seed=0, **kw):
    """D133 afcm_ens: B bootstrap resamples of the training subjects, each fitted by fit_afcm's validation-selected
    grid (the same fixed validation split; x-basis ranges and z standardisation from the full training set, so all
    fits share one design); the coefficient vectors (theta, gamma) are averaged -- eta is linear in them, so this is
    the mean of the resample models' linear predictors -- and the Breslow baseline is re-estimated on the full
    training set. Returns (averaged model, records with each resample's selected lambdas)."""
    torch.set_default_dtype(torch.float64)
    lams_x, lams_t = kw.pop("lams_x", LAM_GRID), kw.pop("lams_t", LAM_T_GRID)
    proto = AFCM(**kw)
    proto._set_ranges(train["xg"])
    proto.zm = train["z"].mean(0)
    proto.zs = np.where(train["z"].std(0) > 0, train["z"].std(0), 1.0)
    W = proto.design(train["xg"])
    Ztr = proto._z(train["z"], len(W))
    rng = np.random.default_rng(seed)
    n = len(train["y"])
    thetas, gammas, recs = [], [], []
    for b in range(B):
        idx = rng.integers(0, n, n)
        sub = dict(y=train["y"][idx], d=train["d"][idx])
        best = None
        for lx in lams_x:
            for lt in lams_t:
                m = _fit_one(proto, kw, lx, lt, W[idx], Ztr[idx], sub)
                v = m.val_nll(val["xg"], val["z"], val["y"], val["d"])
                if best is None or v < best[0] - 1e-12:
                    best = (v, lx, lt, m)
        recs.append(dict(resample=b, lam_x=best[1], lam_t=best[2], val_nll=best[0]))
        thetas.append(best[3].theta)
        gammas.append(best[3].gamma)
    m = AFCM(**{**kw, "lam_x": float("nan"), "lam_t": float("nan")})
    m.lo, m.hi, m.bx, m.zm, m.zs = proto.lo, proto.hi, proto.bx, proto.zm, proto.zs
    if m.xin == "quantile":
        m.qs, m.qp = proto.qs, proto.qp
    m.theta, m.gamma, m.D = np.mean(thetas, 0), np.mean(gammas, 0), W.shape[1]
    eta = W.reshape(n, -1) @ m.theta.reshape(-1) + Ztr @ m.gamma
    m.baseline = breslow_baseline(eta, train["y"], train["d"])
    m.val_nll_bag = m.val_nll(val["xg"], val["z"], val["y"], val["d"])
    return m, recs
