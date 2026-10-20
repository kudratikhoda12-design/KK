"""Two-stage mixed-effects + Cox baseline `lme2_cox` (D112; Python re-implementation, R/JM not available).

Stage 1, per channel: Gaussian linear mixed model with random intercept and slope,
  y_ij = (beta0 + u0i) + (beta1 + u1i) t_ij + e_ij,  u_i ~ N(0, Sigma),  e_ij ~ N(0, sigma2),
fitted by EM (Laird & Ware 1982) from the BLUP / conditional-variance updates.
Stage 2: the subject's predicted value at the landmark (beta0 + u0i + (beta1 + u1i) t_L) and slope (beta1 + u1i),
together with z, enter a Cox model (lifelines, Efron).
"""
from __future__ import annotations

import numpy as np


def blup(times, values, beta, Sigma, sigma2):
    """BLUPs u_i = Sigma Z_i' V_i^{-1} (y_i - X_i beta), V_i = Z_i Sigma Z_i' + sigma2 I (Z_i = X_i = [1, t])."""
    out = np.zeros((len(times), 2))
    for i, (t, y) in enumerate(zip(times, values)):
        Z = np.column_stack([np.ones(len(t)), t])
        V = Z @ Sigma @ Z.T + sigma2 * np.eye(len(t))
        out[i] = Sigma @ Z.T @ np.linalg.solve(V, np.asarray(y, float) - Z @ beta)
    return out


class LME:
    def __init__(self, max_iter=500, tol=1e-8):
        self.max_iter, self.tol = max_iter, tol

    def fit(self, times, values):
        Zs = [np.column_stack([np.ones(len(t)), t]) for t in times]
        ys = [np.asarray(y, float) for y in values]
        Zall, yall = np.vstack(Zs), np.concatenate(ys)
        beta = np.linalg.lstsq(Zall, yall, rcond=None)[0]
        sigma2 = float(np.var(yall - Zall @ beta))
        Sigma = np.diag([sigma2, sigma2])
        N = len(yall)
        for it in range(self.max_iter):
            # E-step: conditional means and covariances of u_i
            Su, rss, uu = np.zeros((2, 2)), 0.0, []
            Ai = []
            Sinv = np.linalg.inv(Sigma)
            for Z, y in zip(Zs, ys):
                A = np.linalg.inv(Z.T @ Z / sigma2 + Sinv)            # Var(u_i | y_i)
                u = A @ Z.T @ (y - Z @ beta) / sigma2
                uu.append(u); Ai.append(A)
                Su += np.outer(u, u) + A
                r = y - Z @ beta - Z @ u
                rss += r @ r + np.trace(Z @ A @ Z.T)
            # M-step
            Sigma_new = Su / len(Zs)
            sigma2_new = rss / N
            beta_new = np.linalg.solve(Zall.T @ Zall, sum(Z.T @ (y - Z @ u) for Z, y, u in zip(Zs, ys, uu)))
            delta = max(np.max(np.abs(Sigma_new - Sigma)), abs(sigma2_new - sigma2), np.max(np.abs(beta_new - beta)))
            Sigma, sigma2, beta = Sigma_new, sigma2_new, beta_new
            if delta < self.tol:
                break
        self.beta, self.Sigma, self.sigma2, self.n_iter = beta, Sigma, float(sigma2), it + 1
        return self

    def blup(self, times, values):
        return blup(times, values, self.beta, self.Sigma, self.sigma2)

    def features(self, times, values, t_L=1.0):
        """(n, 2): predicted value at t_L and predicted slope."""
        u = self.blup(times, values)
        b0, b1 = self.beta[0] + u[:, 0], self.beta[1] + u[:, 1]
        return np.column_stack([b0 + b1 * t_L, b1])


class LME2Cox:
    """values: list over subjects of (m_i, D) arrays; times: list of (m_i,) arrays (on the history scale)."""

    def __init__(self, t_L=1.0, penalizer=0.0):
        self.t_L, self.penalizer = t_L, penalizer

    def _feats(self, times, values, z):
        F = [lme.features(times, [v[:, d] for v in values], self.t_L) for d, lme in enumerate(self.lmes)]
        X = np.hstack(F + ([z] if z is not None and z.shape[1] else []))
        return X

    def fit(self, times, values, z, y, d):
        import pandas as pd
        from lifelines import CoxPHFitter
        D = values[0].shape[1]
        self.lmes = [LME().fit(times, [v[:, k] for v in values]) for k in range(D)]
        X = self._feats(times, values, z)
        self.cols = [f"f{j}" for j in range(X.shape[1])]
        df = pd.DataFrame(X, columns=self.cols).assign(T=y, E=np.asarray(d, int))
        self.cph = CoxPHFitter(penalizer=self.penalizer).fit(df, "T", "E")
        return self

    def predict_risk(self, times, values, z):
        import pandas as pd
        X = pd.DataFrame(self._feats(times, values, z), columns=self.cols)
        return self.cph.predict_log_partial_hazard(X).values

    def predict_survival(self, times, values, z, grid):
        import pandas as pd
        X = pd.DataFrame(self._feats(times, values, z), columns=self.cols)
        return self.cph.predict_survival_function(X, times=grid).values.T
