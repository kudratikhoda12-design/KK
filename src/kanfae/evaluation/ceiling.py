"""Data-attainable concordance reference for the amended gate R2 (owner decision D80; definition from D75).

For each individual separately (no fitting across individuals): a, b by least squares of x1 on (1, t); A, omega by a
phase-free sinusoid fit to x2 (omega grid on [0.3, 2.5], then nonlinear least-squares refinement). The simulator's
true risk formula is applied to the estimates; C_ceiling is Harrell's C of that plug-in risk on the given split.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ..data.synthetic import SynthConfig, _eta
from .metrics import harrell_c

OMEGA_GRID = np.linspace(0.3, 2.5, 221)


def estimate_one(t, x1, x2):
    X = np.column_stack([np.ones_like(t), t])
    a, b = np.linalg.lstsq(X, x1, rcond=None)[0]
    best = None
    for om in OMEGA_GRID:
        S = np.column_stack([np.sin(2 * np.pi * om * t), np.cos(2 * np.pi * om * t)])
        coef = np.linalg.lstsq(S, x2, rcond=None)[0]
        rss = float(((x2 - S @ coef) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, om, coef)
    _, om0, (s0, c0) = best
    f = lambda p: p[0] * np.sin(2 * np.pi * p[2] * t) + p[1] * np.cos(2 * np.pi * p[2] * t) - x2  # noqa: E731
    p = least_squares(f, [s0, c0, om0], bounds=([-10, -10, 0.2], [10, 10, 3.0])).x
    return float(a), float(b), float(np.hypot(p[0], p[1])), float(p[2])


def estimate_parameters(obs, idx=None):
    idx = np.arange(len(obs)) if idx is None else np.asarray(idx)
    out = []
    for i in idx:
        m = obs.mask[i, :, 0]
        out.append(estimate_one(obs.times[i, m], obs.values[i, m, 0], obs.values[i, m, 1]))
    return np.array(out)


def plugin_eta(est, risk="V1"):
    theta = np.column_stack([est, np.zeros(len(est))])        # phase is not used by the risk formula
    return _eta(theta, None, SynthConfig(risk=risk))


def ceiling_C(obs, idx, risk="V1"):
    """C_ceiling on the split idx (observed outcomes of idx only)."""
    idx = np.asarray(idx)
    return harrell_c(obs.y_obs[idx], obs.delta[idx], -plugin_eta(estimate_parameters(obs, idx), risk))
