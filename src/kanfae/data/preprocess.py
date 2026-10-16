"""Training-only preprocessing (K2): channel standardisation, smoother lambda, basis designs, flat grid, summaries."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..functional.basis import BSplineBasis
from ..functional.smoothing import heldout_split, select_lambda, smooth
from .schema import Observed


def _require_observed(obs):
    if not isinstance(obs, Observed):
        raise TypeError(f"preprocessing/training accepts only Observed, got {type(obs).__name__}")


def hat_basis(times, n_grid=20):
    """Piecewise-linear interpolation weights onto a uniform grid on [0, 1]: (N, L, n_grid)."""
    g = np.linspace(0, 1, n_grid)
    pos = np.clip(times, 0, 1) * (n_grid - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, n_grid - 2)
    w = pos - i0
    H = np.zeros(times.shape + (n_grid,))
    np.put_along_axis(H, i0[..., None], (1 - w)[..., None], axis=-1)
    np.put_along_axis(H, (i0 + 1)[..., None], w[..., None] + np.take_along_axis(H, (i0 + 1)[..., None], -1), axis=-1)
    return H, g


def resample_to_grid(times, values, mask, n_grid=20):
    """Linear interpolation of observed points per channel onto the grid; constant extrapolation; empty -> 0."""
    g = np.linspace(0, 1, n_grid)
    N, L, D = values.shape
    out = np.zeros((N, D, n_grid))
    for i in range(N):
        for d in range(D):
            m = mask[i, :, d]
            if m.any():
                out[i, d] = np.interp(g, times[i, m], values[i, m, d])
    return out


def summary_features(times, values, mask):
    """Per channel: last observed value, mean, sd, OLS slope (0 when undefined). (N, 4D)."""
    N, L, D = values.shape
    feats = np.zeros((N, D, 4))
    for i in range(N):
        for d in range(D):
            m = mask[i, :, d]
            if not m.any():
                continue
            t, v = times[i, m], values[i, m, d]
            feats[i, d, 0] = v[-1]
            feats[i, d, 1] = v.mean()
            feats[i, d, 2] = v.std() if len(v) > 1 else 0.0
            feats[i, d, 3] = np.polyfit(t, v, 1)[0] if len(v) > 1 and np.ptp(t) > 0 else 0.0
    return feats.reshape(N, 4 * D)


@dataclass
class Prepared:
    c: np.ndarray            # (N, D, K) smoothed coefficients
    Phi: np.ndarray          # (N, L, K) basis at observation times
    values: np.ndarray       # (N, L, D) standardised values
    fit_mask: np.ndarray     # (N, L, D) points used by the smoother and the reconstruction loss
    eval_mask: np.ndarray    # (N, L, D) held-out points (reconstruction metric only)
    grid_values: np.ndarray  # (N, D, n_grid) resampled (flattened autoencoder input)
    H: np.ndarray            # (N, L, n_grid) hat-basis interpolation weights
    summary: np.ndarray      # (N, 4D [+p]) standardised summary features (+ u)
    u: np.ndarray            # (N, p) standardised baseline covariates (p may be 0)
    y: np.ndarray
    delta: np.ndarray
    ids: np.ndarray


class Preprocessor:
    """All statistics are fitted on the training Observed only (K2)."""

    def __init__(self, n_basis=12, n_grid=20, holdout_frac=0.0, lam=None, seed=0):
        self.basis = BSplineBasis(n_basis)
        self.n_grid, self.holdout_frac, self.lam_fixed, self.seed = n_grid, holdout_frac, lam, seed

    def fit(self, obs: Observed):
        _require_observed(obs)
        m = obs.mask
        D = m.shape[2]
        self.mean_ = np.array([obs.values[:, :, d][m[:, :, d]].mean() if m[:, :, d].any() else 0.0 for d in range(D)])
        self.sd_ = np.array([obs.values[:, :, d][m[:, :, d]].std() if m[:, :, d].any() else 1.0 for d in range(D)])
        self.sd_ = np.where(self.sd_ > 0, self.sd_, 1.0)
        vals = self._std(obs)
        if self.lam_fixed is None:
            self.lam_, self.lam_scores_ = select_lambda(obs.times, vals, m, obs.n_obs, self.basis, seed=self.seed)
        else:
            self.lam_, self.lam_scores_ = self.lam_fixed, {}
        if obs.u is not None and obs.u.shape[1] > 0:
            self.u_mean_, self.u_sd_ = obs.u.mean(0), np.where(obs.u.std(0) > 0, obs.u.std(0), 1.0)
        else:
            self.u_mean_ = self.u_sd_ = None
        s = summary_features(obs.times, vals, m)
        s = np.hstack([s, self._u(obs)])
        self.s_mean_, self.s_sd_ = s.mean(0), np.where(s.std(0) > 0, s.std(0), 1.0)
        return self

    def _std(self, obs):
        return np.where(obs.mask, (obs.values - self.mean_) / self.sd_, 0.0)

    def _u(self, obs):
        if self.u_mean_ is None:
            return np.zeros((len(obs), 0))
        return (obs.u - self.u_mean_) / self.u_sd_

    def transform(self, obs: Observed, holdout=False) -> Prepared:
        _require_observed(obs)
        vals = self._std(obs)
        if holdout and self.holdout_frac > 0:
            fit_mask, eval_mask = heldout_split(obs.n_obs, obs.mask, self.holdout_frac, seed=self.seed + 1)
        else:
            fit_mask, eval_mask = obs.mask, np.zeros_like(obs.mask)
        fv = np.where(fit_mask, vals, 0.0)
        c, Phi = smooth(obs.times, fv, fit_mask, self.basis, lam=self.lam_, return_design=True)
        grid = resample_to_grid(obs.times, fv, fit_mask, self.n_grid)
        H, _ = hat_basis(obs.times, self.n_grid)
        u = self._u(obs)
        s = np.hstack([summary_features(obs.times, fv, fit_mask), u])
        s = (s - self.s_mean_) / self.s_sd_
        return Prepared(c, Phi, vals, fit_mask, eval_mask, grid, H, s, u, obs.y_obs.astype(float),
                        obs.delta.astype(bool), obs.ids)
