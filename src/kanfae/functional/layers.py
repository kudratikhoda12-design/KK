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
