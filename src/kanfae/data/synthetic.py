"""Synthetic functional survival data (docs/design.md section 5)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..survival.aft import get_family
from .schema import Observed, Oracle, make_observed_outcomes

GRID = np.linspace(0.0, 1.0, 201)


@dataclass(frozen=True)
class SynthConfig:
    n: int = 1000
    risk: str = "V1"              # "V1" (latent parameters) or "V2" (functional integral)
    sampling: str = "dense"       # "dense" L ~ U{8..40}; "sparse" L ~ U{3..8}
    censoring: float = 0.3        # target censored fraction: 0, 0.3, 0.6
    family: str = "weibull"
    sigma_true: float = 0.5
    sigma_y: float = 0.1
    p_miss: float = 0.0
    seed: int = 0
    n_pilot: int = 20000
    gp_lengthscale: float = 0.3
    gp_sd: float = 0.2

    @property
    def L_range(self):
        return (8, 40) if self.sampling == "dense" else (3, 8)


def _gp_factor(cfg):
    d = GRID[:, None] - GRID[None, :]
    K = cfg.gp_sd ** 2 * np.exp(-0.5 * (d / cfg.gp_lengthscale) ** 2)
    lam, V = np.linalg.eigh(K)
    return V * np.sqrt(np.clip(lam, 0.0, None))  # g = F z,  F F^T = K


def _latent(n, rng_theta, rng_gp, cfg):
    a = rng_theta.standard_normal(n)
    b = rng_theta.standard_normal(n)
    A = rng_theta.uniform(0.5, 2.0, n)
    om = rng_theta.uniform(0.5, 2.0, n)
    ph = rng_theta.uniform(0.0, 2 * np.pi, n)
    theta = np.column_stack([a, b, A, om, ph])
    g = rng_gp.standard_normal((n, len(GRID))) @ _gp_factor(cfg).T
    return theta, g


def _curves_on_grid(theta, g):
    a, b, A, om, ph = theta.T
    x1 = a[:, None] + b[:, None] * GRID[None, :] + g
    x2 = A[:, None] * np.sin(2 * np.pi * om[:, None] * GRID[None, :] + ph[:, None])
    x3 = np.tanh(x1) + 0.5 * x2
    return np.stack([x1, x2, x3], axis=-1)  # (n, 201, 3)


def _eta(theta, g, cfg):
    a, b, A, om, _ = theta.T
    if cfg.risk == "V1":
        return 2.0 + 1.5 * np.tanh(2 * b) - 0.8 * A ** 2 + 0.5 * a * om
    if cfg.risk == "V2":
        x1 = a[:, None] + b[:, None] * GRID[None, :] + g
        return 2.0 - 0.7 * np.log1p(np.trapezoid(x1 ** 2, GRID, axis=1))
    raise ValueError(cfg.risk)


def _event_times(eta, rng_event, cfg):
    return np.exp(eta + cfg.sigma_true * get_family(cfg.family).sample(len(eta), rng_event))


def _calibrate_censoring(cfg, pilot_seq):
    """tau_admin = pilot 0.95-quantile of T*; mu solves pilot censored fraction == target."""
    r_theta, r_gp, r_event, r_cens = [np.random.default_rng(s) for s in pilot_seq.spawn(4)]
    th, g = _latent(cfg.n_pilot, r_theta, r_gp, cfg)
    T = _event_times(_eta(th, g, cfg), r_event, cfg)
    t_max = float(np.quantile(T, 0.9))
    if cfg.censoring <= 0:
        return np.inf, 0.0, t_max
    tau = float(np.quantile(T, 0.95))
    E1 = r_cens.exponential(1.0, cfg.n_pilot)

    def frac(log_mu):
        return np.mean(np.minimum(E1 / np.exp(log_mu), tau) < T)

    lo, hi = -20.0, 20.0
    if frac(lo) > cfg.censoring:
        raise ValueError("target censoring below the administrative floor")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if frac(mid) < cfg.censoring else (lo, mid)
    return tau, float(np.exp(0.5 * (lo + hi))), t_max


def _observation_process(n, rng_obs, cfg):
    lo, hi = cfg.L_range
    L = rng_obs.integers(lo, hi + 1, n)
    t = rng_obs.uniform(0.0, 1.0, (n, hi))
    real = np.arange(hi)[None, :] < L[:, None]
    t = np.where(real, t, np.inf)
    t.sort(axis=1)
    t = np.where(real, t, 0.0)
    return L, t, real


def generate(cfg: SynthConfig, chunk: int = 10000):
    """Return (Observed, Oracle). Streams: theta, gp, obs, noise, event, cens, pilot (common random numbers)."""
    ss = np.random.SeedSequence(cfg.seed)
    s_theta, s_gp, s_obs, s_noise, s_event, s_cens, s_pilot = ss.spawn(7)
    r_theta, r_gp, r_obs, r_noise, r_event, r_cens = [np.random.default_rng(s) for s in
                                                      (s_theta, s_gp, s_obs, s_noise, s_event, s_cens)]
    n = cfg.n
    theta, g = _latent(n, r_theta, r_gp, cfg)
    eta = _eta(theta, g, cfg)
    T = _event_times(eta, r_event, cfg)
    tau, mu, t_max = _calibrate_censoring(cfg, s_pilot)
    E1 = r_cens.exponential(1.0, n)
    if cfg.censoring <= 0:
        E = np.full(n, np.inf)
        C = np.full(n, np.inf)
    else:
        E = E1 / mu
        C = np.minimum(E, tau)
    y, delta = make_observed_outcomes(T, C)

    L, t, real = _observation_process(n, r_obs, cfg)
    hi = t.shape[1]
    D = 3
    values = np.zeros((n, hi, D))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        curves = _curves_on_grid(theta[s:e], g[s:e])          # (m, 201, 3)
        pos = t[s:e] * (len(GRID) - 1)
        i0 = np.clip(np.floor(pos).astype(int), 0, len(GRID) - 2)
        w = (pos - i0)[..., None]
        rows = np.arange(e - s)[:, None]
        values[s:e] = (1 - w) * curves[rows, i0] + w * curves[rows, i0 + 1]
    values = values + cfg.sigma_y * r_noise.standard_normal(values.shape)
    mask = np.repeat(real[:, :, None], D, axis=2)
    if cfg.p_miss > 0:
        mask &= r_noise.random(mask.shape) >= cfg.p_miss
    values = np.where(mask, values, 0.0)
    times = np.where(real, t, 0.0)
    ids = np.arange(n)
    obs = Observed(ids, times, values, mask, L, y, delta, None, ("x1", "x2", "x3"),
                   {"source": "synthetic", "config": asdict(cfg), "landmark": 1.0})
    ora = Oracle(ids, T, C, E, theta, eta, cfg.sigma_true, cfg.family, t_max, tau, mu)
    return obs, ora
