"""Multivariate functional PCA on B-spline coefficients with the Gram metric (docs/design.md section 3.3)."""
from __future__ import annotations

import numpy as np


def _sqrtm_psd(G):
    lam, V = np.linalg.eigh(G)
    return (V * np.sqrt(np.clip(lam, 0, None))) @ V.T, (V * (1 / np.sqrt(np.clip(lam, 1e-300, None)))) @ V.T


class FPCA:
    def __init__(self, n_components: int):
        self.r = n_components

    def fit(self, c, G):
        N, D, K = c.shape
        self.D, self.K = D, K
        self.Gh, self.Gh_inv = _sqrtm_psd(G)
        Z = np.einsum("kj,ndj->ndk", self.Gh, c).reshape(N, D * K)
        self.mean_ = Z.mean(axis=0)
        cov = np.cov(Z - self.mean_, rowvar=False, bias=True)
        lam, V = np.linalg.eigh(cov)
        order = np.argsort(lam)[::-1]
        self.eigvals_, self.components_ = lam[order], V[:, order]
        self.r = min(self.r, D * K)
        return self

    def transform(self, c, r=None):
        r = self.r if r is None else r
        Z = np.einsum("kj,ndj->ndk", self.Gh, c).reshape(len(c), -1)
        return (Z - self.mean_) @ self.components_[:, :r]

    def inverse_transform(self, scores):
        r = scores.shape[1]
        Z = scores @ self.components_[:, :r].T + self.mean_
        return np.einsum("kj,ndj->ndk", self.Gh_inv, Z.reshape(len(scores), self.D, self.K))
