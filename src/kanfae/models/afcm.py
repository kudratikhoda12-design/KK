"""Additive functional Cox model, AFCM (Cui, Crainiceanu & Leroux 2021): Python re-implementation (D112, deviation).

eta_i = sum_d int F_d(X_id(t), t) dt + z_i' gamma,  F_d(x, t) = sum_jk theta_djk B_j(x) B_k(t),
B: cubic B-splines, 10 x 10 tensor basis per channel (x basis on the training range of X_d, t basis on [0, 1]).
Penalised partial likelihood (Efron):
  mean NLPL + lam_x sum_d theta_d' (D2x'D2x (x) I) theta_d + lam_t sum_d theta_d' (I (x) D2t'D2t) theta_d
              + ridge ||theta||^2,
second-order difference penalties (P-splines). (lam_x, lam_t) chosen by validation partial likelihood on a 5 x 5 grid
instead of REML in mgcv (R not available); ridge = 1e-6 removes the direction that the baseline absorbs (any F that is
a function of t only). The objective is convex; fitted by L-BFGS with strong-Wolfe line search.
Curves enter on the 50-point grid with trapezoid quadrature, the same quadrature as the functional KAN.
"""
from __future__ import annotations

import numpy as np
import torch

from ..functional.basis import BSplineBasis
from ..survival.cox import breslow_baseline, cox_nll, cox_survival

GRID = np.linspace(0.0, 1.0, 50)
LAM_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _trap_w(grid):
    w = np.full(len(grid), grid[1] - grid[0])
    w[[0, -1]] *= 0.5
    return w


def _d2_penalty(K):
    D2 = np.diff(np.eye(K), 2, axis=0)
    return D2.T @ D2


class AFCM:
    def __init__(self, n_x=10, n_t=10, lam_x=1e-2, lam_t=1e-2, ridge=1e-6, ties="efron", max_iter=500):
        self.n_x, self.n_t, self.lam_x, self.lam_t, self.ridge, self.ties = n_x, n_t, lam_x, lam_t, ridge, ties
        self.max_iter = max_iter
        self.bt = BSplineBasis(n_t, 3, (0.0, 1.0))

    # ---- design --------------------------------------------------------------------------------------------------
    def _set_ranges(self, xg):
        self.lo, self.hi = xg.min(axis=(0, 1)), xg.max(axis=(0, 1))
        self.bx = [BSplineBasis(self.n_x, 3, (float(a), float(b))) for a, b in zip(self.lo, self.hi)]

    def _bx(self, d, x):
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
        It, Ix = np.eye(self.n_t), np.eye(self.n_x)
        return torch.tensor(np.kron(Px, It)), torch.tensor(np.kron(Ix, Pt))

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
            pen = sum(self.lam_x * th[k] @ Px @ th[k] + self.lam_t * th[k] @ Pt @ th[k] for k in range(D))
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
        Bx = self._bx(d, x.ravel())
        Bt = self.bt(t.ravel())
        return np.einsum("nj,jk,nk->n", Bx, self.theta[d], Bt).reshape(x.shape)


def fit_afcm(train, val, lams_x=LAM_GRID, lams_t=LAM_GRID, **kw):
    """5 x 5 grid over (lam_x, lam_t), selected by validation Efron NLPL. train/val: dicts xg, z, y, d.
    Returns (best model, records)."""
    torch.set_default_dtype(torch.float64)
    proto = AFCM(**kw)
    proto._set_ranges(train["xg"])
    proto.zm = train["z"].mean(0)
    proto.zs = np.where(train["z"].std(0) > 0, train["z"].std(0), 1.0)
    W = proto.design(train["xg"])
    Ztr = proto._z(train["z"], len(W))
    recs, best = [], None
    for lx in lams_x:
        for lt in lams_t:
            m = AFCM(**{**kw, "lam_x": lx, "lam_t": lt})
            m.lo, m.hi, m.bx, m.zm, m.zs = proto.lo, proto.hi, proto.bx, proto.zm, proto.zs
            m._fit_design(W, Ztr, train["y"], train["d"])
            v = m.val_nll(val["xg"], val["z"], val["y"], val["d"])
            recs.append(dict(lam_x=lx, lam_t=lt, val_nll=v))
            if best is None or v < best[0] - 1e-12:
                best = (v, m)
    return best[1], recs
