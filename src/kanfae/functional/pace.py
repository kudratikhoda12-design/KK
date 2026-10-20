"""PACE: principal components analysis through conditional expectation (Yao, Mueller & Wang 2005; D112).

One channel at a time. Mean: local linear smoother (Epanechnikov) of the pooled (t, y). Covariance: 2-D local linear
smoother of the off-diagonal raw covariances. Measurement-error variance: smoothed diagonal minus the covariance
diagonal (local quadratic perpendicular to the diagonal, as in PACE), averaged over the middle half of the domain. Eigenfunctions from the covariance on the grid (quadrature
weights); K by fraction of variance explained (FVE). Scores by conditional expectation (BLUP).
Bandwidths by leave-one-subject-out CV: exact for the mean; for the covariance, LOSO on a fixed random subsample of
2,000 raw-covariance targets (computational shortcut, documented).
"""
from __future__ import annotations

import numpy as np

GRID = np.linspace(0.0, 1.0, 50)
H_GRID = (0.05, 0.08, 0.12, 0.18, 0.25)


def _epan(u):
    return np.where(np.abs(u) < 1, 0.75 * (1 - u ** 2), 0.0)


def _ll1(t_eval, t, y, h, w_extra=None):
    """1-D local linear fit at t_eval (E,) from points (t, y) (N,)."""
    u = (t[None, :] - t_eval[:, None]) / h
    w = _epan(u) if w_extra is None else _epan(u) * w_extra
    dt = t[None, :] - t_eval[:, None]
    S0, S1, S2 = w.sum(1), (w * dt).sum(1), (w * dt ** 2).sum(1)
    T0, T1 = (w * y).sum(1), (w * dt * y).sum(1)
    den = S0 * S2 - S1 ** 2
    return np.where(np.abs(den) > 1e-12, (S2 * T0 - S1 * T1) / np.where(np.abs(den) > 1e-12, den, 1.0),
                    T0 / np.maximum(S0, 1e-12))


def _ll2(s_eval, t_eval, s, t, y, h):
    """2-D local linear fit at points (s_eval, t_eval) (E,) from (s, t, y) (N,)."""
    ds, dt = s[None, :] - s_eval[:, None], t[None, :] - t_eval[:, None]
    w = _epan(ds / h) * _epan(dt / h)
    out = np.empty(len(s_eval))
    for e in range(len(s_eval)):
        we = w[e]
        m = we > 0
        if m.sum() < 3:
            out[e] = np.nan
            continue
        A = np.column_stack([np.ones(m.sum()), ds[e, m], dt[e, m]])
        W = we[m]
        M = A.T @ (A * W[:, None]) + 1e-10 * np.eye(3)
        out[e] = np.linalg.solve(M, A.T @ (W * y[m]))[0]
    return out


def _diag_quadratic(t_eval, s, t, y, h):
    """Covariance on the diagonal C(t0, t0) by a local fit that is quadratic perpendicular to the diagonal and linear
    along it (Yao, Mueller, Clifford et al. 2003), from off-diagonal raw covariances (s, t, y)."""
    v = (s + t) / np.sqrt(2.0)                    # along the diagonal
    u = (s - t) / np.sqrt(2.0)                    # perpendicular
    out = np.empty(len(t_eval))
    for e, t0 in enumerate(t_eval):
        v0 = np.sqrt(2.0) * t0
        w = _epan((v - v0) / h) * _epan(u / h)
        m = w > 0
        if m.sum() < 5:
            out[e] = np.nan
            continue
        A = np.column_stack([np.ones(m.sum()), v[m] - v0, u[m] ** 2])
        W = w[m]
        out[e] = np.linalg.solve(A.T @ (A * W[:, None]) + 1e-10 * np.eye(3), A.T @ (W * y[m]))[0]
    return out


class PACE:
    def __init__(self, fve=0.95, max_k=5, grid=GRID, seed=0):
        self.fve, self.max_k, self.grid, self.seed = fve, max_k, np.asarray(grid, float), seed

    def fit(self, times, values):
        """times, values: lists over subjects of 1-D arrays (one channel)."""
        sid = np.concatenate([np.full(len(t), i) for i, t in enumerate(times)])
        t = np.concatenate(times).astype(float)
        y = np.concatenate(values).astype(float)
        # --- mean: exact LOSO-CV over H_GRID
        best = None
        for h in H_GRID:
            u = (t[None, :] - t[:, None]) / h
            w = _epan(u) * (sid[None, :] != sid[:, None])            # drop the target's own subject
            dt = t[None, :] - t[:, None]
            S0, S1, S2 = w.sum(1), (w * dt).sum(1), (w * dt ** 2).sum(1)
            T0, T1 = (w * y).sum(1), (w * dt * y).sum(1)
            den = S0 * S2 - S1 ** 2
            ok = (np.abs(den) > 1e-12) & (S0 > 0)
            pred = (S2 * T0 - S1 * T1) / np.where(ok, den, 1.0)
            cv = np.mean((y[ok] - pred[ok]) ** 2) if ok.mean() > 0.95 else np.inf
            if best is None or cv < best[0]:
                best = (cv, h)
        self.h_mu = best[1]
        self.mu = _ll1(self.grid, t, y, self.h_mu)
        mu_obs = np.interp(t, self.grid, self.mu)
        r = y - mu_obs
        # --- raw covariances (off-diagonal) and diagonal
        S, T, Cv, P = [], [], [], []
        for i in range(len(times)):
            idx = np.nonzero(sid == i)[0]
            for a in idx:
                for b in idx:
                    if a != b:
                        S.append(t[a]); T.append(t[b]); Cv.append(r[a] * r[b]); P.append(i)
        S, T, Cv, P = map(np.asarray, (S, T, Cv, P))
        rng = np.random.default_rng(self.seed)
        tgt = rng.choice(len(S), size=min(2000, len(S)), replace=False)
        best = None
        for h in H_GRID[1:]:
            pred = np.empty(len(tgt))
            for k, j in enumerate(tgt):
                keep = P != P[j]
                pred[k] = _ll2(np.array([S[j]]), np.array([T[j]]), S[keep], T[keep], Cv[keep], h)[0]
            ok = np.isfinite(pred)
            cv = np.mean((Cv[tgt][ok] - pred[ok]) ** 2) if ok.mean() > 0.95 else np.inf
            if best is None or cv < best[0]:
                best = (cv, h)
        self.h_cov = best[1]
        gs, gt = np.meshgrid(self.grid, self.grid, indexing="ij")
        C = _ll2(gs.ravel(), gt.ravel(), S, T, Cv, self.h_cov).reshape(len(self.grid), len(self.grid))
        C = np.nan_to_num(0.5 * (C + C.T))
        V = _ll1(self.grid, t, r ** 2, self.h_mu)
        mid = (self.grid >= 0.25) & (self.grid <= 0.75)
        Cq = _diag_quadratic(self.grid[mid], S, T, Cv, self.h_cov)
        self.sigma2 = float(max(np.nanmean(V[mid] - Cq), 1e-6))
        # --- eigen-decomposition with trapezoid quadrature weights
        wq = np.full(len(self.grid), self.grid[1] - self.grid[0]); wq[[0, -1]] *= 0.5
        Wh = np.sqrt(wq)
        lam, vec = np.linalg.eigh(Wh[:, None] * C * Wh[None, :])
        order = np.argsort(lam)[::-1]
        lam, vec = lam[order], vec[:, order]
        pos = lam > 0
        lam, vec = lam[pos], vec[:, pos]
        cum = np.cumsum(lam) / lam.sum()
        K = int(min(self.max_k, np.searchsorted(cum, self.fve) + 1))
        self.lam = lam[:K]
        self.phi = (vec[:, :K] / Wh[:, None])                         # orthonormal in L2[0, 1]
        self.K = K
        return self

    def scores(self, times, values):
        out = np.zeros((len(times), self.K))
        L = np.diag(self.lam)
        for i, (ti, yi) in enumerate(zip(times, values)):
            Phi = np.column_stack([np.interp(ti, self.grid, self.phi[:, k]) for k in range(self.K)])
            mu_i = np.interp(ti, self.grid, self.mu)
            Sig = Phi @ L @ Phi.T + self.sigma2 * np.eye(len(ti))
            out[i] = L @ Phi.T @ np.linalg.solve(Sig, np.asarray(yi, float) - mu_i)
        return out

    def reconstruct(self, times, values):
        """Curves on the grid: (n, G)."""
        return self.mu[None, :] + self.scores(times, values) @ self.phi.T
