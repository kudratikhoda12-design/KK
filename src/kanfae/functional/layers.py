"""Functional linear layer h_j = b_j + sum_d int beta_jd(t) x_d(t) dt = b_j + sum_d c_d^T J b_jd."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .basis import BSplineBasis


class FunctionalLinear(nn.Module):
    def __init__(self, n_channels: int, n_features: int, basis_x: BSplineBasis, basis_beta: BSplineBasis, seed=None):
        super().__init__()
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        Kb = basis_beta.K
        self.weight = nn.Parameter(torch.randn(n_features, n_channels, Kb, generator=gen) / np.sqrt(n_channels))
        self.bias = nn.Parameter(torch.zeros(n_features))
        self.register_buffer("J", torch.tensor(basis_x.cross_gram(basis_beta), dtype=torch.get_default_dtype()))
        self.register_buffer("R_beta", torch.tensor(basis_beta.roughness(), dtype=torch.get_default_dtype()))

    def forward(self, c):                       # c: (B, D, K)
        return torch.einsum("bdk,kl,hdl->bh", c, self.J, self.weight) + self.bias

    def roughness_penalty(self):
        return torch.einsum("hdl,lm,hdm->", self.weight, self.R_beta, self.weight)


class NonlinearFunctional(nn.Module):
    """Nonlinear functional layer (revision 3, D38):

        h_j = b_j + sum_d sum_m int beta_jdm(t) g_dm(x_d(t)) dt,   g_d(x) = [x, MLP_d(x)],  MLP_d: 1 -> width -> M-1.

    x_d(t) = sum_k c_dk phi_k(t) is evaluated at Gauss-Legendre nodes (8 per sub-interval of the union of both
    bases' break points), so the identity part reproduces FunctionalLinear exactly (test F6a).
    """

    def __init__(self, n_channels, n_features, basis_x: BSplineBasis, basis_beta: BSplineBasis, n_maps=5, width=16,
                 seed=None, identity_only=False):
        super().__init__()
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        brk = np.union1d(basis_x.breaks, basis_beta.breaks)
        z, wq = np.polynomial.legendre.leggauss(8)
        tq = np.concatenate([0.5 * (b - a) * z + 0.5 * (a + b) for a, b in zip(brk[:-1], brk[1:])])
        wq = np.concatenate([0.5 * (b - a) * wq for a, b in zip(brk[:-1], brk[1:])])
        dt = torch.get_default_dtype()
        self.register_buffer("Phi_q", torch.tensor(basis_x(tq), dtype=dt))                      # (Q, K)
        self.register_buffer("Psi_w", torch.tensor(basis_beta(tq) * wq[:, None], dtype=dt))     # (Q, Kb)
        self.register_buffer("R_beta", torch.tensor(basis_beta.roughness(), dtype=dt))
        self.D, self.M, self.identity_only = n_channels, (1 if identity_only else n_maps), identity_only
        Kb = basis_beta.K
        self.weight = nn.Parameter(torch.randn(n_features, n_channels, self.M, Kb, generator=gen)
                                   / np.sqrt(n_channels * self.M))
        self.bias = nn.Parameter(torch.zeros(n_features))
        if not identity_only:
            self.W1 = nn.Parameter(torch.randn(n_channels, width, generator=gen))
            self.b1 = nn.Parameter(torch.randn(n_channels, width, generator=gen))
            self.W2 = nn.Parameter(torch.randn(n_channels, width, n_maps - 1, generator=gen) / np.sqrt(width))
            self.b2 = nn.Parameter(torch.zeros(n_channels, n_maps - 1))

    def curves(self, c):                                       # (B, D, K) -> (B, D, Q)
        return torch.einsum("bdk,qk->bdq", c, self.Phi_q)

    def feature_maps(self, c):                                 # (B, D, Q, M)
        x = self.curves(c)
        if self.identity_only:
            return x.unsqueeze(-1)
        hid = torch.nn.functional.silu(x.unsqueeze(-1) * self.W1[None, :, None, :] + self.b1[None, :, None, :])
        learned = torch.einsum("bdqw,dwm->bdqm", hid, self.W2) + self.b2[None, :, None, :]
        return torch.cat([x.unsqueeze(-1), learned], dim=-1)

    def forward(self, c):
        return torch.einsum("bdqm,ql,hdml->bh", self.feature_maps(c), self.Psi_w, self.weight) + self.bias

    def roughness_penalty(self):
        return torch.einsum("hdml,lk,hdmk->", self.weight, self.R_beta, self.weight)


def _union_gauss_legendre(basis_x: BSplineBasis, basis_beta: BSplineBasis, n_nodes=8):
    brk = np.union1d(basis_x.breaks, basis_beta.breaks)
    z, w = np.polynomial.legendre.leggauss(n_nodes)
    tq = np.concatenate([0.5 * (b - a) * z + 0.5 * (a + b) for a, b in zip(brk[:-1], brk[1:])])
    wq = np.concatenate([0.5 * (b - a) * w for a, b in zip(brk[:-1], brk[1:])])
    return tq, wq


class EnergyFunctional(nn.Module):
    """Linear + functional-energy front end (2026-10-17, candidate A2 / D41):

        h_j = b_j + sum_d int beta_jd(t) x_d(t) dt + sum_d int gamma_jd(t) x_d(t)^2 dt,
        int gamma(t) x(t)^2 dt = sum_l q_l c^T T_l c,   T_l[k,k'] = int psi_l phi_k phi_k'  (exact Gauss-Legendre).
    The energy term can express phase-invariant amplitude functionals that linear functionals cannot."""

    def __init__(self, n_channels, n_features, basis_x, basis_beta, seed=None):
        super().__init__()
        self.lin = FunctionalLinear(n_channels, n_features, basis_x, basis_beta, seed=seed)
        gen = torch.Generator().manual_seed(None if seed is None else seed + 7) if seed is not None else None
        tq, wq = _union_gauss_legendre(basis_x, basis_beta)
        Phi, Psi = basis_x(tq), basis_beta(tq)
        T = np.einsum("q,ql,qk,qm->lkm", wq, Psi, Phi, Phi)               # (Kb, K, K)
        self.register_buffer("T", torch.tensor(T, dtype=torch.get_default_dtype()))
        self.q = nn.Parameter(torch.randn(n_features, n_channels, basis_beta.K, generator=gen) / np.sqrt(n_channels))

    def energies(self, c):                                              # (B, D, Kb)
        return torch.einsum("bdk,lkm,bdm->bdl", c, self.T, c)

    def forward(self, c):
        return self.lin(c) + torch.einsum("bdl,hdl->bh", self.energies(c), self.q)

    def roughness_penalty(self):
        return self.lin.roughness_penalty() + torch.einsum("hdl,lm,hdm->", self.q, self.lin.R_beta, self.q)


class TensorSplineFunctional(nn.Module):
    """Functional tensor-spline front end (2026-10-17, candidate A3 / D41), the deferred int F(x(t), t) dt variant:

        h_j = b_j + sum_d int beta_jd(t) x_d(t) dt + sum_d sum_m int beta_jdm(t) B_m(x_d(t)) dt,
    B_m: cubic B-splines on a uniform grid of G intervals over [x_lo, x_hi] (standardised scale), i.e. a
    tensor-product spline surface F_jd(x, t). The feature integrals contain no trainable parameters."""

    def __init__(self, n_channels, n_features, basis_x, basis_beta, grid=5, x_range=(-3.0, 3.0), seed=None):
        super().__init__()
        self.lin = FunctionalLinear(n_channels, n_features, basis_x, basis_beta, seed=seed)
        tq, wq = _union_gauss_legendre(basis_x, basis_beta)
        dt = torch.get_default_dtype()
        self.register_buffer("Phi_q", torch.tensor(basis_x(tq), dtype=dt))
        self.register_buffer("Psi_w", torch.tensor(basis_beta(tq) * wq[:, None], dtype=dt))
        h = (x_range[1] - x_range[0]) / grid
        self.register_buffer("knots", torch.arange(-3, grid + 4, dtype=dt) * h + x_range[0])
        nb = grid + 3
        gen = torch.Generator().manual_seed(seed + 11) if seed is not None else None
        self.w = nn.Parameter(torch.randn(n_features, n_channels, nb, basis_beta.K, generator=gen)
                              / np.sqrt(n_channels * nb))

    def _xbasis(self, x):
        t, xe = self.knots, x.unsqueeze(-1)
        B = ((xe >= t[:-1]) & (xe < t[1:])).to(x.dtype)
        for p in range(1, 4):
            B = (xe - t[:-(p + 1)]) / (t[p:-1] - t[:-(p + 1)]) * B[..., :-1] + \
                (t[p + 1:] - xe) / (t[p + 1:] - t[1:-p]) * B[..., 1:]
        return B                                                           # (..., nb)

    def features(self, c):                                                 # (B, D, nb, Kb)
        # The features contain no trainable parameters; for a fixed (non-differentiated) input they are computed
        # once and cached (efficiency fix, 2026-10-17). Inputs that require gradients are never cached.
        # Fix 2026-10-17 (D53): the cache now holds a reference to the input and a hit requires the SAME tensor
        # object at the same version. The first version keyed on (data_ptr, shape, version) without keeping the input
        # alive, so a freed validation tensor's address could be reused by a same-shape test tensor and return the
        # validation features (replication seed 110: test C 0.49).
        cacheable = not c.requires_grad
        cache = self.__dict__.setdefault("_feat_cache", {})
        key = id(c)
        if cacheable and key in cache:
            src, ver, out = cache[key]
            if src is c and ver == c._version:
                return out
        x = torch.einsum("bdk,qk->bdq", c, self.Phi_q)
        out = torch.einsum("bdqm,ql->bdml", self._xbasis(x), self.Psi_w)
        if cacheable:
            if len(cache) > 8:
                cache.clear()
            cache[key] = (c, c._version, out)
        return out

    def forward(self, c):
        return self.lin(c) + torch.einsum("bdml,hdml->bh", self.features(c), self.w)

    def roughness_penalty(self):
        return self.lin.roughness_penalty() + torch.einsum("hdml,lk,hdmk->", self.w, self.lin.R_beta, self.w)


class DerivEnergySpline(TensorSplineFunctional):
    """Tensor-spline front + derivative energy (D46 "tsd"): adds sum_d int gamma'_jd(t) x_d'(t)^2 dt, computed exactly
    as sum_l q_jdl c_d^T T'_l c_d with T'_l = int psi_l phi' phi'^T (Gauss-Legendre, exact for these polynomials)."""

    def __init__(self, n_channels, n_features, basis_x, basis_beta, grid=5, x_range=(-3.0, 3.0), seed=None):
        super().__init__(n_channels, n_features, basis_x, basis_beta, grid, x_range, seed)
        tq, wq = _union_gauss_legendre(basis_x, basis_beta)
        dPhi, Psi = basis_x(tq, deriv=1), basis_beta(tq)
        self.register_buffer("Td", torch.tensor(np.einsum("q,ql,qk,qm->lkm", wq, Psi, dPhi, dPhi),
                                                dtype=torch.get_default_dtype()))
        gen = torch.Generator().manual_seed(seed + 13) if seed is not None else None
        self.qd = nn.Parameter(torch.randn(n_features, n_channels, basis_beta.K, generator=gen) / np.sqrt(n_channels) * 0.1)

    def deriv_energies(self, c):                                           # (B, D, Kb)
        return torch.einsum("bdk,lkm,bdm->bdl", c, self.Td, c)

    def forward(self, c):
        return super().forward(c) + torch.einsum("bdl,hdl->bh", self.deriv_energies(c), self.qd)

    def roughness_penalty(self):
        return super().roughness_penalty() + torch.einsum("hdl,lm,hdm->", self.qd, self.lin.R_beta, self.qd)


def _projector_quadform(Phi, psi, w, rel_tol=1e-12):
    """M with c^T M c = ||P_S x||^2 for x = Phi c and S = span(columns of psi), L2 metric with weights w.
    Uses an eigen-decomposition of the Gram matrix of psi (pseudo-inverse below rel_tol), so a nearly collinear
    subspace (low frequency in a short window) is handled stably."""
    G = psi.T @ (psi * w[:, None])
    B = psi.T @ (Phi * w[:, None])                                         # (k, K): <psi_i, phi_k>
    lam, V = np.linalg.eigh(G)
    keep = lam > rel_tol * lam.max()
    Ginv = (V[:, keep] / lam[keep]) @ V[:, keep].T
    return B.T @ Ginv @ B


class FrequencyAwareFunctional(TensorSplineFunctional):
    """Frequency-aware functional front end (revision 4, "fa"; D57/D61; docs/design.md section 3.2c).

    h = TensorSpline(c) + W standardise(S(c)) + w0, where per channel S stacks
      R_d(f_j) = (||P_{1,t,sin 2 pi f_j t,cos 2 pi f_j t} x_d||^2 - ||P_{1,t} x_d||^2) / (||x_d - P_{1,t} x_d||^2 + eps)
                 on a log-spaced grid f_j in f_range (phase-invariant least-squares spectrum; per window),
      e0 = log(eps + ||x - P_{1,t} x||^2), e1 = log(eps + int x'^2), e2 = log(eps + int x''^2),
      L_d(f_j) = -log(1 - R_d(f_j) + delta), delta = 1e-4 (log-gap: separates the exact peak R = 1 from the
                 low-frequency plateau R ~ 0.98-1 that a <= 2-cycle window produces below the true frequency),
      f_hat_d = sum_j f_j softmax_j(kappa_d L_d(f_j)) (full window; kappa_d = exp(rho_d) learnable, kappa_0 = 4),
      optionally log(eps + <x, sin>^2 + <x, cos>^2) (phase-dependent Fourier power; search variant "fourier").
    Every quadratic form is exact up to Gauss-Legendre quadrature (64 nodes per interval). Standardisation is a
    non-affine BatchNorm with a cumulative average (full-batch training: exactly the training mean / variance)."""

    EPS = 1e-8
    DELTA = 1e-4

    def __init__(self, n_channels, n_features, basis_x, basis_beta, grid=5, x_range=(-3.0, 3.0), seed=None,
                 n_freq=33, f_range=(0.25, 4.0), fourier=False, windows=False, kappa0=4.0):
        super().__init__(n_channels, n_features, basis_x, basis_beta, grid, x_range, seed)
        dt = torch.get_default_dtype()
        self.freqs_np = np.geomspace(f_range[0], f_range[1], n_freq)
        self.fourier, self.windows = fourier, windows
        z, wz = np.polynomial.legendre.leggauss(64)
        wins = [(0.0, 1.0)] + ([(0.0, 0.5), (0.5, 1.0)] if windows else [])
        M, M0, N = [], [], []
        for lo, hi in wins:
            brk = np.union1d(basis_x.breaks, [lo, hi])
            brk = brk[(brk >= lo) & (brk <= hi)]
            tq = np.concatenate([0.5 * (b - a) * z + 0.5 * (a + b) for a, b in zip(brk[:-1], brk[1:])])
            wq = np.concatenate([0.5 * (b - a) * wz for a, b in zip(brk[:-1], brk[1:])])
            Phi = basis_x(tq)
            base = np.column_stack([np.ones_like(tq), tq])
            M0.append(_projector_quadform(Phi, base, wq))
            N.append((Phi * wq[:, None]).T @ Phi)
            M.append(np.stack([_projector_quadform(Phi, np.column_stack([base, np.sin(2 * np.pi * f * tq),
                                                                         np.cos(2 * np.pi * f * tq)]), wq)
                               for f in self.freqs_np]))
            if lo == 0.0 and hi == 1.0:
                S = np.stack([np.sin(2 * np.pi * f * tq) for f in self.freqs_np])
                C = np.stack([np.cos(2 * np.pi * f * tq) for f in self.freqs_np])
                self.register_buffer("Fs", torch.tensor((S * wq) @ Phi, dtype=dt))           # (J, K)
                self.register_buffer("Fc", torch.tensor((C * wq) @ Phi, dtype=dt))
                d1, d2 = basis_x(tq, deriv=1), basis_x(tq, deriv=2)
                self.register_buffer("Om1", torch.tensor((d1 * wq[:, None]).T @ d1, dtype=dt))
                self.register_buffer("Om2", torch.tensor((d2 * wq[:, None]).T @ d2, dtype=dt))
        self.register_buffer("Mf", torch.tensor(np.stack(M), dtype=dt))                     # (nw, J, K, K)
        self.register_buffer("M0", torch.tensor(np.stack(M0), dtype=dt))                    # (nw, K, K)
        self.register_buffer("N", torch.tensor(np.stack(N), dtype=dt))                      # (nw, K, K)
        self.register_buffer("freqs", torch.tensor(self.freqs_np, dtype=dt))
        nw, J = len(wins), n_freq
        self.n_spec = n_channels * (2 * nw * J + 3 + (J if fourier else 0) + 1)
        self.rho = nn.Parameter(torch.full((n_channels,), float(np.log(kappa0))))
        self.norm = nn.BatchNorm1d(self.n_spec, affine=False, momentum=None)
        gen = torch.Generator().manual_seed(seed + 17) if seed is not None else None
        self.spec = nn.Linear(self.n_spec, n_features)
        with torch.no_grad():
            self.spec.weight.copy_(torch.randn(n_features, self.n_spec, generator=gen) / np.sqrt(self.n_spec))
            self.spec.bias.zero_()

    def spectrum(self, c):
        """R (B, D, nw, J), energies (B, D, 3), log Fourier power (B, D, J) or None. No trainable parameters."""
        cacheable = not c.requires_grad
        cache = self.__dict__.setdefault("_spec_cache", {})          # identity-checked (D53)
        key = id(c)
        if cacheable and key in cache:
            src, ver, out = cache[key]
            if src is c and ver == c._version:
                return out
        e_f = torch.einsum("bdk,wjkl,bdl->bdwj", c, self.Mf, c)
        e_0 = torch.einsum("bdk,wkl,bdl->bdw", c, self.M0, c)
        tot = torch.einsum("bdk,wkl,bdl->bdw", c, self.N, c)
        R = (e_f - e_0.unsqueeze(-1)) / (tot - e_0 + self.EPS).unsqueeze(-1)
        en = torch.stack([torch.log(self.EPS + (tot - e_0)[..., 0]),
                          torch.log(self.EPS + torch.einsum("bdk,kl,bdl->bd", c, self.Om1, c)),
                          torch.log(self.EPS + torch.einsum("bdk,kl,bdl->bd", c, self.Om2, c))], -1)
        fp = (torch.log(self.EPS + torch.einsum("bdk,jk->bdj", c, self.Fs) ** 2
                        + torch.einsum("bdk,jk->bdj", c, self.Fc) ** 2) if self.fourier else None)
        out = (R, en, fp)
        if cacheable:
            if len(cache) > 8:
                cache.clear()
            cache[key] = (c, c._version, out)
        return out

    def log_gap(self, R):
        return -torch.log(1.0 - R + self.DELTA)

    def soft_frequency(self, R, rho=None):                            # (B, D, nw, J) -> (B, D)
        kappa = torch.exp(self.rho if rho is None else rho)[None, :, None]
        return (torch.softmax(kappa * self.log_gap(R[:, :, 0, :]), -1) * self.freqs).sum(-1)

    def spectral_vector(self, c):
        R, en, fp = self.spectrum(c)
        parts = [R.flatten(2), self.log_gap(R).flatten(2), en] + ([fp] if fp is not None else []) + \
            [self.soft_frequency(R).unsqueeze(-1)]
        return torch.cat(parts, -1).flatten(1)                        # (B, n_spec)

    def forward(self, c):
        return super().forward(c) + self.spec(self.norm(self.spectral_vector(c)))
