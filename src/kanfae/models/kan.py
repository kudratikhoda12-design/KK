"""In-house B-spline Kolmogorov-Arnold layer (docs/design.md section 3.5, decision D1).

out_q = sum_p [ w^b_pq SiLU(x_p) + w^s_pq sum_g a_pqg B_g(x_p) ] + bias_q
B_g: degree-k B-splines on a uniform grid of G intervals on grid_range, extended by k knots on each side.
Outside the extended grid the spline term is exactly zero.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class KANLayer(nn.Module):
    def __init__(self, n_in, n_out, grid_size=5, spline_order=3, grid_range=(-2.0, 2.0), seed=None):
        super().__init__()
        self.n_in, self.n_out, self.G, self.k = n_in, n_out, grid_size, spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        knots = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.get_default_dtype()) * h + grid_range[0]
        self.register_buffer("knots", knots)                       # G + 2k + 1 knots -> G + k basis functions
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        nb = grid_size + spline_order
        self.coef = nn.Parameter(0.1 * torch.randn(n_in, n_out, nb, generator=gen) / (n_in ** 0.5))
        self.w_base = nn.Parameter(torch.randn(n_in, n_out, generator=gen) / (n_in ** 0.5))
        self.w_spline = nn.Parameter(torch.ones(n_in, n_out))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def basis(self, x):
        """x (B, n_in) -> (B, n_in, G + k) B-spline values (Cox-de Boor)."""
        t = self.knots
        xe = x.unsqueeze(-1)
        B = ((xe >= t[:-1]) & (xe < t[1:])).to(x.dtype)
        for p in range(1, self.k + 1):
            left = (xe - t[: -(p + 1)]) / (t[p:-1] - t[: -(p + 1)]) * B[..., :-1]
            right = (t[p + 1:] - xe) / (t[p + 1:] - t[1:-p]) * B[..., 1:]
            B = left + right
        return B

    def spline_term(self, x):
        return torch.einsum("bik,iok->bo", self.basis(x), self.coef * self.w_spline.unsqueeze(-1))

    def forward(self, x):
        return F.silu(x) @ self.w_base + self.spline_term(x) + self.bias

    @staticmethod
    def n_params(n_in, n_out, G, k):
        return n_in * n_out * (G + k + 2) + n_out


class KANBlock(nn.Module):
    """LayerNorm (no affine) followed by a stack of KAN layers, e.g. dims = [16, 8, 4]."""

    def __init__(self, dims, grid_size=5, spline_order=3, seed=None):
        super().__init__()
        self.norm = nn.LayerNorm(dims[0], elementwise_affine=False)
        self.layers = nn.ModuleList([KANLayer(a, b, grid_size, spline_order,
                                              seed=None if seed is None else seed + i)
                                     for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))])

    def forward(self, x):
        x = self.norm(x)
        for layer in self.layers:
            x = layer(x)
        return x
