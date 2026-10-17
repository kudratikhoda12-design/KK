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
        self.register_buffer("mask", torch.ones(n_in, n_out))             # edge pruning mask (1 = active)
        self.symbolic = {}                                                # (i, o) -> callable replacing the edge

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

    def edge_activations(self, x):
        """phi_io(x_i) = w_base[i,o] SiLU(x_i) + w_spline[i,o] sum_g coef[i,o,g] B_g(x_i), masked: (B, n_in, n_out).
        Edges with a symbolic replacement use it instead (symbolic substitution)."""
        phi = F.silu(x).unsqueeze(-1) * self.w_base + torch.einsum("bik,iok->bio", self.basis(x),
                                                                   self.coef * self.w_spline.unsqueeze(-1))
        if self.symbolic:
            phi = phi.clone()
            for (i, o), f in self.symbolic.items():
                phi[:, i, o] = f(x[:, i])
        return phi * self.mask

    def forward(self, x):
        if not self.symbolic and bool((self.mask == 1).all()):
            return F.silu(x) @ self.w_base + self.spline_term(x) + self.bias
        return self.edge_activations(x).sum(1) + self.bias

    def regularization(self, x, lam_ent=2.0, lam_coef=1.0):
        """CoxKAN Eqs. 5-8 / KAN-AFT L_R for one layer: |Phi|_1 + lam_ent S(Phi) + lam_coef |C|_1, with
        |phi|_1 = mean_s |phi(x_s)|, S = -sum p log p (p = |phi|_1 / |Phi|_1), |c|_1 = mean_g |c_g| per edge."""
        l1 = self.edge_activations(x).abs().mean(0)                        # (n_in, n_out)
        tot = l1.sum()
        p = l1 / (tot + 1e-12)
        ent = -(p * torch.log(p + 1e-12)).sum()
        coef_l1 = (self.coef.abs().mean(-1) * self.mask).sum()
        return tot + lam_ent * ent + lam_coef * coef_l1, {"l1": l1.detach(), "entropy": float(ent),
                                                         "coef_l1": float(coef_l1)}

    @torch.no_grad()
    def prune_edges(self, x, theta=1e-2):
        """KAN-AFT / CoxKAN edge pruning: deactivate edges whose L1 norm (mean |phi| over x) is below theta."""
        l1 = self.edge_activations(x).abs().mean(0)
        newly = (l1 < theta) & (self.mask > 0)
        self.mask[l1 < theta] = 0.0
        return int(newly.sum())

    @staticmethod
    def n_params(n_in, n_out, G, k):
        return n_in * n_out * (G + k + 2) + n_out


class KANBlock(nn.Module):
    """Per-feature BatchNorm (no affine; D30) followed by a stack of KAN layers, e.g. dims = [16, 4].

    A per-sample LayerNorm (the original design) would discard each sample's feature mean and scale,
    i.e. 2 of r = 4 latent degrees of freedom at the decoder input; BatchNorm keeps that information."""

    def __init__(self, dims, grid_size=5, spline_order=3, seed=None):
        super().__init__()
        self.norm = nn.BatchNorm1d(dims[0], affine=False)
        self.layers = nn.ModuleList([KANLayer(a, b, grid_size, spline_order,
                                              seed=None if seed is None else seed + i)
                                     for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))])

    def forward(self, x):
        x = self.norm(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def layer_inputs(self, x):
        """Inputs to each KAN layer (after normalisation), for regularisation, pruning and symbolic fitting."""
        x = self.norm(x)
        ins = []
        for layer in self.layers:
            ins.append(x)
            x = layer(x)
        return ins

    def regularization(self, x, lam_ent=2.0, lam_coef=1.0):
        return sum(l.regularization(xi, lam_ent, lam_coef)[0] for l, xi in zip(self.layers, self.layer_inputs(x)))

    @torch.no_grad()
    def prune(self, x, theta=1e-2):
        """Edge pruning in every layer, then KAN-AFT node pruning for hidden nodes: a hidden node k is removed when
        min(max_j |phi_0,j,k|_1, max_o |phi_1,k,o|_1) < theta."""
        ins = self.layer_inputs(x)
        n_edges = sum(l.prune_edges(xi, theta) for l, xi in zip(self.layers, ins))
        n_nodes = 0
        for a, b, xa, xb in zip(self.layers[:-1], self.layers[1:], ins[:-1], ins[1:]):
            I = (a.edge_activations(xa).abs().mean(0)).max(0).values          # per hidden node (n_out of a)
            O = (b.edge_activations(xb).abs().mean(0)).max(1).values          # per hidden node (n_in of b)
            dead = torch.minimum(I, O) < theta
            a.mask[:, dead] = 0.0
            b.mask[dead, :] = 0.0
            n_nodes += int(dead.sum())
        return {"edges_pruned": n_edges, "nodes_pruned": n_nodes}
