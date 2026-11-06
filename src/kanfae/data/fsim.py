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

DGP versions (D120). "v1" (Round 11, kept for the historical CK tests) is the design above. "v2" (default; Round 12
and the CK-*b tests) adds a random level xi_d0 * 1 (variance 0.5) and a random linear trend xi_dlin * sqrt(12)(t - 0.5)
(variance 0.5) and scales the Fourier variances by 7/15 so that the time-averaged marginal variance stays 1.875; then
int X_d dt and int t X_d dt vary across subjects (in v1, int X_d dt was the same for every subject: D118).
"S-Misspec-mixture" (test-only, CK-11b): log T = 1 - eta/1.5 + eps, eps the Round-2 MIS skewed two-component normal
mixture (0.7 N(-0.6, 0.5^2) + 0.3 N(1.4, 1)) standardised to SD 0.5.
Round 22 (D156, H-R22c; added scenarios only, every earlier scenario bit-identical): log T = 1 - eta/1.5 + eps with
eps of mean 0 and SD 0.5, under which both the Weibull and the log-normal AFT are misspecified:
  "S-Misspec-bimodal"    symmetric bimodal 0.5 N(-1.1, 0.3^2) + 0.5 N(1.1, 0.3^2), standardised (SD sqrt(1.30))
  "S-Misspec-studentt3"  Student t, 3 degrees of freedom, standardised (SD sqrt 3)
  "S-Misspec-logistic05" standard logistic, standardised (SD pi / sqrt 3); unlike "S-Misspec-loglogistic" (0.5 x
                         standard logistic, SD 0.907; Round 12), its SD is 0.5
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

GRID = np.linspace(0.0, 1.0, 50)             # model grid G
FINE = np.linspace(0.0, 1.0, 401)            # truth quadrature
LAM = np.array([1.0, 0.5, 0.25, 0.125])
LAM_V2 = np.r_[0.5, 0.5, LAM * 7.0 / 15.0]               # level, linear trend, Fourier (D120)
D = 3
SHAPE = 1.5
SCENARIOS = ("S-Add", "S-Int", "S-Misspec-lognormal", "S-Misspec-loglogistic", "S-Misspec-mixture", "S-Last",
             "S-Misspec-bimodal", "S-Misspec-studentt3", "S-Misspec-logistic05")
R22_LAWS = {"S-Misspec-bimodal": float(np.sqrt(0.09 + 1.21)), "S-Misspec-studentt3": float(np.sqrt(3.0)),
            "S-Misspec-logistic05": float(np.pi / np.sqrt(3.0))}         # SD of the unstandardised error (D156)
# S-Last (D133, Round 15): the hazard depends on the channels' current values at the landmark t = 1 only,
# eta = sin(pi X_1(1) / 2) + 1.5 tanh(X_2(1)) + 0.3 z_1 - 0.3 z_2 (channel 3 null); Weibull PH, shape 1.5. With sparse
# visits X(1) itself is never observed, so the last observed value is a noisy, lagged proxy and the trajectory carries
# extra information about X(1). Used for b2s_res_cox; the other scenarios are unchanged.
DGPS = ("v1", "v2")
_MIS_MU = 0.7 * -0.6 + 0.3 * 1.4
_MIS_SD = float(np.sqrt(0.7 * (0.25 + 0.36) + 0.3 * (1.0 + 1.96) - _MIS_MU ** 2))


def psi(t):
    t = np.asarray(t, float)
    r2 = np.sqrt(2.0)
    return np.stack([r2 * np.sin(2 * np.pi * t), r2 * np.cos(2 * np.pi * t), r2 * np.sin(4 * np.pi * t),
                     r2 * np.cos(4 * np.pi * t)], -1)


def basis_v2(t):
    """(T, 6): 1, sqrt(12)(t - 0.5), then the four Fourier functions."""
    t = np.asarray(t, float)
    return np.concatenate([np.ones(t.shape + (1,)), (np.sqrt(12.0) * (t - 0.5))[..., None], psi(t)], -1)


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
            if self.scenario in R22_LAWS:                 # eps = 0.5 e / sd_e, so e = sd_e * w (D156)
                e = R22_LAWS[self.scenario] * w
                if self.scenario == "S-Misspec-bimodal":
                    S = 0.5 * norm.sf((e + 1.1) / 0.3) + 0.5 * norm.sf((e - 1.1) / 0.3)
                elif self.scenario == "S-Misspec-studentt3":
                    S = student_t.sf(e, 3)
                else:
                    S = 1.0 / (1.0 + np.exp(e))
                return np.where(g[None, :] <= 0, 1.0, S)
            if self.scenario.endswith("lognormal"):
                S = norm.sf(w)
            elif self.scenario.endswith("loglogistic"):
                S = 1.0 / (1.0 + np.exp(w))
            else:                                         # mixture: eps = 0.5 (e - mu) / sd, so e = mu + sd * w
                e = _MIS_MU + _MIS_SD * w
                S = 0.7 * norm.sf((e + 0.6) / 0.5) + 0.3 * norm.sf(e - 1.4)
            return np.where(g[None, :] <= 0, 1.0, S)
        return np.exp(-np.exp(self.eta)[:, None] * g[None, :] ** SHAPE)


def _latent(n, rng, dgp="v1"):
    if dgp == "v1":
        return rng.normal(size=(n, D, 4)) * np.sqrt(LAM)
    return rng.normal(size=(n, D, 6)) * np.sqrt(LAM_V2)


def _curves(xi, t):
    """xi (n, D, 4) [v1] or (n, D, 6) [v2], t (T,) -> (n, T, D)."""
    P = psi(t) if xi.shape[2] == 4 else basis_v2(t)
    return np.einsum("ndk,tk->ntd", xi, P) + np.stack([mu(d, t) for d in range(D)], -1)[None]


def _eta(scenario, xi, z):
    Xf = _curves(xi, FINE)                                   # (n, F, D)
    a1 = _trap(phi1(Xf[:, :, 0]), FINE)
    a2 = _trap(phi2(Xf[:, :, 1], FINE[None, :]), FINE)
    if scenario == "S-Last":
        x1 = _curves(xi, np.array([1.0]))[:, 0, :]                       # (n, D) at the landmark
        return np.sin(np.pi * x1[:, 0] / 2.0) + 1.5 * np.tanh(x1[:, 1]) + 0.3 * z[:, 0] - 0.3 * z[:, 1]
    eta = a1 + a2 + 0.3 * z[:, 0] - 0.3 * z[:, 1]
    if scenario == "S-Int":
        eta = eta + 0.8 * a1 * _trap(np.tanh(Xf[:, :, 1]), FINE)
    return eta


def _event_times(scenario, eta, rng):
    if scenario == "S-Misspec-mixture":
        n = len(eta)
        comp = rng.random(n) < 0.7
        e = np.where(comp, rng.normal(-0.6, 0.5, n), rng.normal(1.4, 1.0, n))
        return np.exp(1.0 - eta / SHAPE + 0.5 * (e - _MIS_MU) / _MIS_SD)
    if scenario in R22_LAWS:
        n = len(eta)
        if scenario == "S-Misspec-bimodal":
            e = np.where(rng.random(n) < 0.5, -1.1, 1.1) + 0.3 * rng.normal(size=n)
        elif scenario == "S-Misspec-studentt3":
            e = rng.standard_t(3, size=n)
        else:
            e = rng.logistic(size=n)
        return np.exp(1.0 - eta / SHAPE + 0.5 * e / R22_LAWS[scenario])
    if scenario.startswith("S-Misspec"):
        eps = rng.normal(size=len(eta)) if scenario.endswith("lognormal") else rng.logistic(size=len(eta))
        return np.exp(1.0 - eta / SHAPE + 0.5 * eps)
    u = rng.random(len(eta))
    return (-np.log(u) / np.exp(eta)) ** (1.0 / SHAPE)


_CMAX = {}


def censoring_cmax(scenario, target=0.30, dgp="v1"):
    """c_max such that P(C < T) ~= target for C ~ U(0, c_max); fixed calibration sample (seed 99, n 20000)."""
    if (scenario, dgp) in _CMAX:
        return _CMAX[(scenario, dgp)]
    rng = np.random.default_rng(99)
    xi = _latent(20000, rng, dgp)
    z = np.column_stack([rng.normal(size=20000), rng.random(20000) < 0.5]).astype(float)
    T = _event_times(scenario, _eta(scenario, xi, z), rng)
    lo, hi = 1e-3, 1e3
    for _ in range(80):
        c = np.sqrt(lo * hi)
        frac = np.mean(_pcens(T, c))
        lo, hi = (c, hi) if frac > target else (lo, c)
    _CMAX[(scenario, dgp)] = float(np.sqrt(lo * hi))
    return _CMAX[(scenario, dgp)]


def _pcens(T, c):
    """P(C < T) for C ~ U(0, c): min(T, c) / c."""
    return np.minimum(T, c) / c


def generate(scenario, n=1000, seed=0, sparse=False, n_dense=40, noise=0.2, dgp="v2"):
    if scenario not in SCENARIOS or dgp not in DGPS:
        raise ValueError((scenario, dgp))
    rng = np.random.default_rng(seed)
    xi = _latent(n, rng, dgp)
    z = np.column_stack([rng.normal(size=n), rng.random(n) < 0.5]).astype(float)
    eta = _eta(scenario, xi, z)
    T = _event_times(scenario, eta, rng)
    C = rng.random(n) * censoring_cmax(scenario, dgp=dgp)
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
                    dict(seed=seed, sparse=sparse, xi=xi, c_max=censoring_cmax(scenario, dgp=dgp), dgp=dgp))


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


def rise_1d(phi_hat, phi_true, X, t=GRID):
    """Homogeneous estimate and truth (D120): RISE over x at the pooled test values X (n, G), one global centring."""
    tt = np.broadcast_to(t[None, :], X.shape)
    a, b = np.asarray(phi_hat(X, tt), float).ravel(), np.asarray(phi_true(X, tt), float).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(np.mean((a - b) ** 2) / np.mean(b ** 2))


ID_TOL = 3e-8          # D120 (inside the gap of the ratio spectrum on development seed 902): a direction is unidentified if Var_i(int f(X_i(t), t) dt) <= ID_TOL * mean-square(f)
ID_BASIS = (10, 10)    # tensor cubic B-spline basis in (x, t) used to represent candidate null directions


def unidentified_directions(X, t=GRID, tol=ID_TOL, basis=ID_BASIS):
    """Orthonormal (in the mean over the n*G evaluation points) vectors spanning the per-t-centred functions
    f(x, t) = sum_jk v_jk B_j(x) B_k(t) whose functional int f(X_i(t), t) dt has (relative) zero variance across the
    curves X (n, G). Returns (F (n*G, r), ratios of all directions, sorted ascending)."""
    from ..functional.basis import BSplineBasis
    n, G = X.shape
    bx = BSplineBasis(basis[0], 3, (float(X.min()), float(X.max())))
    bt = BSplineBasis(basis[1], 3, (float(t[0]), float(t[-1])))
    Bx = bx(np.clip(X, X.min(), X.max()))                        # (n, G, nx)
    Bt = bt(t)                                                   # (G, nt)
    M = np.einsum("ngj,gk->ngjk", Bx, Bt).reshape(n, G, -1)      # (n, G, P)
    w = np.full(G, t[1] - t[0]); w[[0, -1]] *= 0.5
    W = np.einsum("ngp,g->np", M, w)
    Wc = W - W.mean(0)
    Mc = (M - M.mean(0, keepdims=True)).reshape(n * G, -1)       # per-t centring
    A = Wc.T @ Wc / n
    B = Mc.T @ Mc / (n * G)
    lam, Q = np.linalg.eigh(B)
    keep = lam > 1e-12 * lam.max()
    T_ = Q[:, keep] / np.sqrt(lam[keep])                         # whitening: v = T_ u has unit mean square
    r, U = np.linalg.eigh(T_.T @ A @ T_)
    F = Mc @ (T_ @ U[:, r <= tol])
    return F, np.sort(r)


def rise_identifiable(phi_hat, phi_true, X, t=GRID, tol=ID_TOL, basis=ID_BASIS):
    """D120 primary shape metric: RISE after per-t centring and after projecting the error and the truth orthogonal
    to the unidentified directions of the empirical trajectory operator on the test curves X (n, G)."""
    n, G = X.shape
    tt = np.broadcast_to(t[None, :], X.shape)
    a, b = np.asarray(phi_hat(X, tt), float), np.asarray(phi_true(X, tt), float)
    a = (a - a.mean(0, keepdims=True)).ravel()
    b = (b - b.mean(0, keepdims=True)).ravel()
    F, _ = unidentified_directions(X, t, tol, basis)
    N = n * G
    e = a - b
    e = e - F @ (F.T @ e) / N
    b = b - F @ (F.T @ b) / N
    return float(np.mean(e ** 2) / np.mean(b ** 2))
