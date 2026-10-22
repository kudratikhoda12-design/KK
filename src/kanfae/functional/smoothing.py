"""Quadrature-weighted penalised B-spline smoother (docs/design.md section 3.1, decision D20).

c_id = argmin_c  sum_l q_l (y_l - phi(t_l)^T c)^2 + lam c^T R c + lam0 ||c||^2,
q_l = length of the cell around t_l (midpoints between neighbouring observed times; first/last
cells extend to 0 and 1), so the data term approximates int (y - x)^2 dt.
"""
from __future__ import annotations

import numpy as np

from .basis import BSplineBasis


def quadrature_weights(times, mask, domain=(0.0, 1.0)):
    """times (N, L) sorted within real observations; mask (N, L, D) -> weights (N, L, D)."""
    N, L, D = mask.shape
    t = np.broadcast_to(times[:, :, None], (N, L, D))
    lo, hi = domain
    idx = np.arange(L)[None, :, None]
    # previous observed index (per n, d) via running maximum of masked indices
    prev_idx = np.where(mask, idx, -1)
    prev_idx = np.maximum.accumulate(prev_idx, axis=1)
    prev_idx = np.concatenate([np.full((N, 1, D), -1), prev_idx[:, :-1]], axis=1)
    nxt = np.where(mask, idx, L)
    nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1]
    nxt = np.concatenate([nxt[:, 1:], np.full((N, 1, D), L)], axis=1)
    tp = np.take_along_axis(t, np.clip(prev_idx, 0, L - 1), axis=1)
    tn = np.take_along_axis(t, np.clip(nxt, 0, L - 1), axis=1)
    left = np.where(prev_idx >= 0, 0.5 * (tp + t), lo)
    right = np.where(nxt < L, 0.5 * (t + tn), hi)
    return np.where(mask, right - left, 0.0)


def smooth(times, values, mask, basis: BSplineBasis, lam=1e-4, lam0=1e-6, return_design=False):
    """Coefficients c (N, D, K). Channels without observations get c = 0."""
    q = quadrature_weights(times, mask, basis.domain)
    Phi = basis(times)                                      # (N, L, K)
    A = np.einsum("nlk,nld,nlj->ndkj", Phi, q, Phi)
    b = np.einsum("nlk,nld,nld->ndk", Phi, q, values)
    reg = lam * basis.roughness() + lam0 * np.eye(basis.K)
    c = np.linalg.solve(A + reg[None, None], b[..., None])[..., 0]
    return (c, Phi) if return_design else c


def evaluate(c, Phi):
    """x_hat at observation times: c (N, D, K), Phi (N, L, K) -> (N, L, D)."""
    return np.einsum("ndk,nlk->nld", c, Phi)


def heldout_split(n_obs, mask, frac=0.15, min_points=4, seed=0):
    """Hide `frac` of the real observation times of individuals with >= min_points (all channels at a time)."""
    rng = np.random.default_rng(seed)
    N, L, D = mask.shape
    hide = np.zeros((N, L), bool)
    for i in np.nonzero(n_obs >= min_points)[0]:
        k = max(1, int(round(frac * n_obs[i])))
        hide[i, rng.choice(n_obs[i], size=k, replace=False)] = True
    fit_mask = mask & ~hide[:, :, None]
    eval_mask = mask & hide[:, :, None]
    return fit_mask, eval_mask


def individual_weighted_mse(err2, mask):
    """Mean over individuals of the mean over observed channels of the per-channel mean squared error (D19)."""
    n_id = mask.sum(axis=1)                                  # (N, D)
    per_ch = (err2 * mask).sum(axis=1) / np.maximum(n_id, 1)
    n_ch = (n_id > 0).sum(axis=1)
    keep = n_ch > 0                                          # individuals with at least one evaluated point
    per_ind = (per_ch * (n_id > 0)).sum(axis=1)[keep] / n_ch[keep]
    return float(per_ind.mean()) if keep.any() else float("nan")


def observation_weighted_mse(err2, mask):
    return float((err2 * mask).sum() / max(mask.sum(), 1))


def select_lambda(times, values, mask, n_obs, basis, grid=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1), seed=0):
    """Choose lambda by held-out-point error on the data passed in (training individuals only, K2)."""
    fit_mask, eval_mask = heldout_split(n_obs, mask, seed=seed)
    scores = {}
    for lam in grid:
        c, Phi = smooth(times, np.where(fit_mask, values, 0.0), fit_mask, basis, lam=lam, return_design=True)
        err2 = (values - evaluate(c, Phi)) ** 2
        scores[lam] = individual_weighted_mse(err2, eval_mask)
    best = min(scores, key=scores.get)
    return best, scores


def pad_ragged(times, values):
    """Lists of (m_i,) times and (m_i, D) values -> times (N, L), values (N, L, D), mask (N, L, D), n_obs (N,)."""
    N, L = len(times), max(len(t) for t in times)
    D = np.asarray(values[0]).reshape(len(times[0]), -1).shape[1]
    T, V, M = np.zeros((N, L)), np.zeros((N, L, D)), np.zeros((N, L, D), bool)
    for i, (t, v) in enumerate(zip(times, values)):
        m = len(t)
        T[i, :m] = t
        T[i, m:] = t[-1] if m else 0.0
        vv = np.asarray(v, float).reshape(m, D)
        ok = np.isfinite(vv)                                  # missing channel values (NaN) are masked out (Round 13)
        V[i, :m] = np.where(ok, vv, 0.0)
        M[i, :m] = ok
    return T, V, M, np.array([len(t) for t in times])


def curves_on_grid(times, values, grid, basis: BSplineBasis, lam):
    """Per-subject penalised B-spline smoothing of ragged data, evaluated on `grid`: (N, len(grid), D)."""
    T, V, M, _ = pad_ragged(times, values)
    c = smooth(T, V, M, basis, lam=lam)
    return np.einsum("ndk,gk->ngd", c, basis(np.asarray(grid, float)))
