"""Round-12 functional-survival simulator (D113; docs/design_note_round10.md section 4).

D = 3 channels on t in [0, 1]: X_d(t) = 0.5*d*t + sum_k xi_dk psi_k(t), psi = orthonormal Fourier
(sqrt2 sin 2pi t, sqrt2 cos 2pi t, sqrt2 sin 4pi t, sqrt2 cos 4pi t), xi_dk ~ N(0, lam_k), lam = (1, .5, .25, .125).
Observations X_d(t_ij) + N(0, 0.2^2): dense (40 equispaced times) or sparse (3-8 visits at uniform times).
Scalars z1 ~ N(0,1), z2 ~ Bernoulli(.5). Channel 3 is null.
  S-Add:     eta = int phi1(X1) dt + int phi2(X2(t), t) dt + 0.3 z1 - 0.3 z2,
             phi1(x) = sin(pi x / 2), phi2(x, t) = tanh(x)(1 + t);  Weibull PH, shape 1.5
  S-Int:     S-Add's eta + 0.8 [int phi1(X1) dt][int tanh(X2) dt];  Weibull PH, shape 1.5
  S-Misspec: log T = 1 - eta/1.5 + 0.5 eps, eps ~ N(0,1) ("lognormal") or logistic ("loglogistic")
Censoring C ~ U(0, c_max), c_max calibrated once per scenario to ~30% on a fixed calibration sample (seed 99).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

GRID = np.linspace(0.0, 1.0, 50)             # model grid G
FINE = np.linspace(0.0, 1.0, 401)            # truth quadrature
LAM = np.array([1.0, 0.5, 0.25, 0.125])
D = 3
SHAPE = 1.5
SCENARIOS = ("S-Add", "S-Int", "S-Misspec-lognormal", "S-Misspec-loglogistic")


def psi(t):
    t = np.asarray(t, float)
    r2 = np.sqrt(2.0)
    return np.stack([r2 * np.sin(2 * np.pi * t), r2 * np.cos(2 * np.pi * t), r2 * np.sin(4 * np.pi * t),
                     r2 * np.cos(4 * np.pi * t)], -1)


def mu(d, t):
    return 0.5 * (d + 1) * np.asarray(t, float)           # d = 0..2  ->  0.5 * (1..3) * t


def phi1(x):
    return np.sin(np.pi * np.asarray(x) / 2.0)


def phi2(x, t):
    return np.tanh(np.asarray(x)) * (1.0 + np.asarray(t))


TRUE_SHAPES = {0: ("homogeneous", lambda x, t: phi1(x)), 1: ("bivariate", phi2), 2: ("null", lambda x, t: 0.0 * x)}


def _trap(y, t):
    return np.trapezoid(y, t, axis=-1)


@dataclass
class FSimData:
    scenario: str
    times: list                      # per subject: observation times (m_i,)
    values: list                     # per subject: (m_i, D) noisy observations
    x_true_grid: np.ndarray          # (n, G, D) true curves on GRID
    z: np.ndarray                    # (n, 2)
    y: np.ndarray
    delta: np.ndarray
    eta: np.ndarray                  # true linear predictor (PH log-hazard; AFT: the location's -1.5 multiple)
    T: np.ndarray
    C: np.ndarray
    meta: dict = field(default_factory=dict)

    def subset(self, idx):
        idx = np.asarray(idx)
        return FSimData(self.scenario, [self.times[i] for i in idx], [self.values[i] for i in idx],
                        self.x_true_grid[idx], self.z[idx], self.y[idx], self.delta[idx], self.eta[idx], self.T[idx],
                        self.C[idx], dict(self.meta))

    def survival(self, grid):
        """Oracle survival S(t | true eta) on a time grid: (n, len(grid))."""
        g = np.asarray(grid, float)
        if self.scenario.startswith("S-Misspec"):
            loc = 1.0 - self.eta / SHAPE
            w = (np.log(np.maximum(g, 1e-300))[None, :] - loc[:, None]) / 0.5
            S = norm.sf(w) if self.scenario.endswith("lognormal") else 1.0 / (1.0 + np.exp(w))
            return np.where(g[None, :] <= 0, 1.0, S)
        return np.exp(-np.exp(self.eta)[:, None] * g[None, :] ** SHAPE)


def _latent(n, rng):
    xi = rng.normal(size=(n, D, 4)) * np.sqrt(LAM)
    return xi


def _curves(xi, t):
    """xi (n, D, 4), t (T,) -> (n, T, D)."""
    P = psi(t)                                               # (T, 4)
    return np.einsum("ndk,tk->ntd", xi, P) + np.stack([mu(d, t) for d in range(D)], -1)[None]


def _eta(scenario, xi, z):
    Xf = _curves(xi, FINE)                                   # (n, F, D)
    a1 = _trap(phi1(Xf[:, :, 0]), FINE)
    a2 = _trap(phi2(Xf[:, :, 1], FINE[None, :]), FINE)
    eta = a1 + a2 + 0.3 * z[:, 0] - 0.3 * z[:, 1]
    if scenario == "S-Int":
        eta = eta + 0.8 * a1 * _trap(np.tanh(Xf[:, :, 1]), FINE)
    return eta


def _event_times(scenario, eta, rng):
    if scenario.startswith("S-Misspec"):
        eps = rng.normal(size=len(eta)) if scenario.endswith("lognormal") else rng.logistic(size=len(eta))
        return np.exp(1.0 - eta / SHAPE + 0.5 * eps)
    u = rng.random(len(eta))
    return (-np.log(u) / np.exp(eta)) ** (1.0 / SHAPE)


_CMAX = {}


def censoring_cmax(scenario, target=0.30):
    """c_max such that P(C < T) ~= target for C ~ U(0, c_max); fixed calibration sample (seed 99, n 20000)."""
    if scenario in _CMAX:
        return _CMAX[scenario]
    rng = np.random.default_rng(99)
    xi = _latent(20000, rng)
    z = np.column_stack([rng.normal(size=20000), rng.random(20000) < 0.5]).astype(float)
    T = _event_times(scenario, _eta(scenario, xi, z), rng)
    lo, hi = 1e-3, 1e3
    for _ in range(80):
        c = np.sqrt(lo * hi)
        frac = np.mean(_pcens(T, c))
        lo, hi = (c, hi) if frac > target else (lo, c)
    _CMAX[scenario] = float(np.sqrt(lo * hi))
    return _CMAX[scenario]


def _pcens(T, c):
    """P(C < T) for C ~ U(0, c): min(T, c) / c."""
    return np.minimum(T, c) / c


def generate(scenario, n=1000, seed=0, sparse=False, n_dense=40, noise=0.2):
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    rng = np.random.default_rng(seed)
    xi = _latent(n, rng)
    z = np.column_stack([rng.normal(size=n), rng.random(n) < 0.5]).astype(float)
    eta = _eta(scenario, xi, z)
    T = _event_times(scenario, eta, rng)
    C = rng.random(n) * censoring_cmax(scenario)
    y, delta = np.minimum(T, C), (T <= C)
    times, values = [], []
    for i in range(n):
        if sparse:
            m = int(rng.integers(3, 9))
            t = np.sort(rng.random(m))
        else:
            t = np.linspace(0.0, 1.0, n_dense)
        x = _curves(xi[i:i + 1], t)[0]
        times.append(t)
        values.append(x + noise * rng.normal(size=x.shape))
    return FSimData(scenario, times, values, _curves(xi, GRID), z, y, delta, eta, T, C,
                    dict(seed=seed, sparse=sparse, xi=xi, c_max=censoring_cmax(scenario)))


def true_effect(d, x, t):
    """True phi_d(x, t) for channel d (0-based) on arrays x, t (broadcast)."""
    return TRUE_SHAPES[d][1](np.asarray(x, float), np.asarray(t, float))


def rise(phi_hat, phi_true, X, t=GRID):
    """Relative ISE, density-weighted by the test curves X (n, G) at grid times t, with per-t centring (an additive
    function of t is not identified). phi_hat, phi_true: callables (x, t) -> values."""
    tt = np.broadcast_to(t[None, :], X.shape)
    a, b = np.asarray(phi_hat(X, tt), float), np.asarray(phi_true(X, tt), float)
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    den = np.mean(b ** 2)
    return float(np.mean((a - b) ** 2) / den) if den > 0 else float("nan")
