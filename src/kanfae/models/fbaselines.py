"""B1 feature builders and simple functional-survival baselines for Rounds 11-13 (D112; design note sections 2-3).

  summaries(...)   B1-SUM: per channel last value before the landmark, mean, OLS slope, residual SD
  GridFPCA         B1-FPCA (dense): per-channel eigen-decomposition of curves on the grid G (trapezoid weights),
                   K_d by 95% FVE (max 5); sparse data use kanfae.functional.pace.PACE instead
  LinearCox        lifelines Cox (Efron) on a feature matrix: fpca_cox, sum_cox, landmark_cox (last values + z)
"""
from __future__ import annotations

import numpy as np

GRID = np.linspace(0.0, 1.0, 50)


def summaries(times, values):
    """times: list of (m,), values: list of (m, D) -> (n, 4 D) [last, mean, slope, resid SD] per channel.
    Slope and residual SD are 0 for a single observation; residual SD is 0 for two."""
    n, D = len(times), np.asarray(values[0]).shape[1]
    out = np.zeros((n, D, 4))
    for i, (t, v) in enumerate(zip(times, values)):
        t, v = np.asarray(t, float), np.asarray(v, float)
        out[i, :, 0] = v[np.argmax(t)]
        out[i, :, 1] = v.mean(0)
        if len(t) >= 2 and np.ptp(t) > 0:
            A = np.column_stack([np.ones(len(t)), t])
            coef, *_ = np.linalg.lstsq(A, v, rcond=None)
            out[i, :, 2] = coef[1]
            if len(t) >= 3:
                out[i, :, 3] = np.sqrt(((v - A @ coef) ** 2).sum(0) / (len(t) - 2))
    return out.reshape(n, D * 4)


def last_values(times, values):
    return np.vstack([np.asarray(v, float)[np.argmax(t)] for t, v in zip(times, values)])


class GridFPCA:
    def __init__(self, fve=0.95, max_k=5, grid=GRID):
        self.fve, self.max_k, self.grid = fve, max_k, np.asarray(grid, float)

    def fit(self, xg):
        """xg (n, G, D)."""
        w = np.full(len(self.grid), self.grid[1] - self.grid[0]); w[[0, -1]] *= 0.5
        self.w, Wh = w, np.sqrt(w)
        self.mu, self.phi, self.K = [], [], []
        for d in range(xg.shape[2]):
            X = xg[:, :, d]
            mu = X.mean(0)
            C = (X - mu).T @ (X - mu) / len(X)
            lam, vec = np.linalg.eigh(Wh[:, None] * C * Wh[None, :])
            lam, vec = lam[::-1], vec[:, ::-1]
            lam = np.maximum(lam, 0)
            K = int(min(self.max_k, np.searchsorted(np.cumsum(lam) / lam.sum(), self.fve) + 1))
            self.mu.append(mu); self.phi.append(vec[:, :K] / Wh[:, None]); self.K.append(K)
        return self

    def transform(self, xg):
        return np.hstack([((xg[:, :, d] - self.mu[d]) * self.w) @ self.phi[d] for d in range(xg.shape[2])])


class LinearCox:
    def __init__(self, penalizer=0.0):
        self.penalizer = penalizer

    def fit(self, X, y, d):
        import pandas as pd
        from lifelines import CoxPHFitter
        self.cols = [f"f{j}" for j in range(X.shape[1])]
        df = pd.DataFrame(X, columns=self.cols).assign(T=y, E=np.asarray(d, int))
        self.cph = CoxPHFitter(penalizer=self.penalizer).fit(df, "T", "E")
        return self

    def predict_risk(self, X):
        import pandas as pd
        return self.cph.predict_log_partial_hazard(pd.DataFrame(X, columns=self.cols)).values

    def predict_survival(self, X, grid):
        import pandas as pd
        return self.cph.predict_survival_function(pd.DataFrame(X, columns=self.cols), times=grid).values.T
