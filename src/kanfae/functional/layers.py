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
