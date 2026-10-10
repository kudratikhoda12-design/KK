"""Kolmogorov-Arnold Network layers with learnable B-spline activation functions.

A KAN layer maps x in R^{n_in} to y in R^{n_out} with
        y_j = sum_i phi_{j,i}(x_i)
where every edge (i -> j) carries its own learnable univariate function

        phi(x) = w_b * silu(x) + w_s * sum_c c_c B_c(x)

B_c are cubic B-spline basis functions on a uniform grid (Cox-de Boor recursion);
c_c, w_b, w_s are trainable. (Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid: int = 5, k: int = 3,
                 grid_range=(-3.0, 3.0), noise_scale: float = 0.1):
        super().__init__()
        self.in_dim, self.out_dim, self.grid_size, self.k = in_dim, out_dim, grid, k
        self.grid_range = grid_range
        h = (grid_range[1] - grid_range[0]) / grid
        # extended knot vector: G + 2k + 1 knots -> G + k basis functions per input
        knots = torch.arange(-k, grid + k + 1, dtype=torch.float32) * h + grid_range[0]
        self.register_buffer("knots", knots.expand(in_dim, -1).contiguous())
        self.coef = nn.Parameter(noise_scale * torch.randn(out_dim, in_dim, grid + k) / grid)
        self.base_w = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.base_w, a=5 ** 0.5)
        self.spline_w = nn.Parameter(torch.ones(out_dim, in_dim))
        self.last_phi = None                     # cached edge activations (for regularisation)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Cox-de Boor recursion. x: (N, in) -> bases: (N, in, G + k)."""
        t = self.knots
        x = x.unsqueeze(-1)
        B = ((x >= t[:, :-1]) & (x < t[:, 1:])).to(x.dtype)          # order-0 (piecewise constant)
        for p in range(1, self.k + 1):
            left = (x - t[:, : -(p + 1)]) / (t[:, p:-1] - t[:, : -(p + 1)]) * B[..., :-1]
            right = (t[:, p + 1:] - x) / (t[:, p + 1:] - t[:, 1:-p]) * B[..., 1:]
            B = left + right
        return B

    def edge_activations(self, x: torch.Tensor) -> torch.Tensor:
        """phi_{j,i}(x_i) for every sample and edge: (N, out, in)."""
        x = x.clamp(*self.grid_range)            # flat extrapolation outside the grid
        spline = torch.einsum("nic,oic->noi", self.b_splines(x), self.coef)
        return self.base_w * F.silu(x).unsqueeze(1) + self.spline_w * spline

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.edge_activations(x)
        self.last_phi = phi
        return phi.sum(-1)

    def regularization(self, lamb_l1: float, lamb_entropy: float = 0.0) -> torch.Tensor:
        """Sparsity regulariser from the KAN paper: L1 norm of each edge's mean |phi|
        (+ optional entropy term pushing the network towards few active edges)."""
        a = self.last_phi.abs().mean(0)                                # (out, in)
        reg = lamb_l1 * a.sum()
        if lamb_entropy > 0:
            p = a / (a.sum() + 1e-8)
            reg = reg + lamb_entropy * -(p * torch.log(p + 1e-8)).sum()
        return reg

    @torch.no_grad()
    def edge_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Std of each edge's activation over the data: how much hazard it moves. (out, in)"""
        return self.edge_activations(x).std(0)


class KAN(nn.Module):
    """Stack of KAN layers, e.g. width=[d, 1] (additive) or [d, 4, 1]."""

    def __init__(self, width, grid=5, k=3):
        super().__init__()
        self.width = width
        self.layers = nn.ModuleList(KANLayer(a, b, grid=grid, k=k) for a, b in zip(width[:-1], width[1:]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization(self, lamb_l1, lamb_coef=0.0, lamb_entropy=0.0):
        reg = sum(l.regularization(lamb_l1, lamb_entropy) for l in self.layers)
        if lamb_coef > 0:
            reg = reg + lamb_coef * sum(l.coef.abs().mean() for l in self.layers)
        return reg
