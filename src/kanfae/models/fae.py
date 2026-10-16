"""Functional autoencoders (MLP / KAN / linear blocks), flattened autoencoder and AFT heads (design.md 3.4, 4)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..functional.basis import BSplineBasis
from ..functional.layers import FunctionalLinear
from .kan import KANBlock


def mlp(dims):
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def make_block(kind, n_in, n_out, hidden=32, kan_grid=5, kan_hidden=None, seed=None):
    """'mlp': [n_in, hidden, n_out]; 'kan': single KAN layer [n_in, n_out] (D30) or [n_in, kan_hidden, n_out];
    'linear': a single linear map."""
    if kind == "mlp":
        return mlp([n_in, hidden, n_out])
    if kind == "kan":
        dims = [n_in, n_out] if kan_hidden is None else [n_in, kan_hidden, n_out]
        return KANBlock(dims, grid_size=kan_grid, seed=seed)
    if kind == "linear":
        return nn.Linear(n_in, n_out)
    raise ValueError(kind)


def n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _kan_block_params(n_in, n_out, hidden, G, k):
    dims = [n_in, n_out] if hidden is None else [n_in, hidden, n_out]
    return sum(a * b * (G + k + 2) + b for a, b in zip(dims[:-1], dims[1:]))


def mlp_block_params(n_in, n_out, hidden):
    return n_in * hidden + hidden + hidden * n_out + n_out


def kan_match(D, K, H=16, r=4, hidden=32, G=5, k=3, max_hidden=16, tol=0.10):
    """Choose KAN encoder/decoder hidden widths (None = single KAN layer) to match the MLP block parameter
    count (D31). Among candidates within +-tol, prefer (a) fewest hidden widths below r (bottlenecks),
    (b) the widest narrowest bottleneck, (c) fewest hidden layers, (d) smallest |deviation|. If none is within tol, take the smallest |deviation|.
    Returns (enc_hidden, dec_hidden, info)."""
    target = mlp_block_params(H, r, hidden) + mlp_block_params(r, D * K, hidden)
    opts = [None] + list(range(1, max_hidden + 1))
    cands = []
    for he in opts:
        for hd in opts:
            tot = _kan_block_params(H, r, he, G, k) + _kan_block_params(r, D * K, hd, G, k)
            dev = (tot - target) / target
            narrow_w = [h for h in (he, hd) if h is not None and h < r]
            layers = sum(h is not None for h in (he, hd))
            # (out_of_tol, n_bottlenecks, -narrowest bottleneck width, n_hidden_layers, |dev|)
            cands.append((abs(dev) > tol, len(narrow_w), -min(narrow_w, default=r), layers, abs(dev),
                          he, hd, tot, dev))
    within = [c for c in cands if not c[0]]
    best = min(within, key=lambda c: c[1:5]) if within else min(cands, key=lambda c: c[4])
    _, narrow, _, layers, _, he, hd, tot, dev = best
    return he, hd, {"mlp_block_params": target, "kan_block_params": tot, "rel_dev": dev,
                    "n_bottleneck": narrow, "n_hidden_layers": layers}


def kan2_match(D, K, H=16, r=4, hidden=32, G=5, k=3, max_hidden=32):
    """Depth-2 KAN ablation: one common hidden width for encoder and decoder, minimising |deviation|."""
    target = mlp_block_params(H, r, hidden) + mlp_block_params(r, D * K, hidden)
    w = min(range(1, max_hidden + 1), key=lambda w: abs(_kan_block_params(H, r, w, G, k)
                                                       + _kan_block_params(r, D * K, w, G, k) - target))
    tot = _kan_block_params(H, r, w, G, k) + _kan_block_params(r, D * K, w, G, k)
    return w, {"mlp_block_params": target, "kan_block_params": tot, "rel_dev": (tot - target) / target}


class AFTHead(nn.Module):
    def __init__(self, n_in, kind="linear", hidden=16, kan_grid=5):
        super().__init__()
        self.net = make_block(kind, n_in, 1, hidden=hidden, kan_grid=kan_grid) if kind != "linear" else nn.Linear(n_in, 1)
        self.rho = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x).squeeze(-1), F.softplus(self.rho) + 1e-3


class FunctionalAE(nn.Module):
    def __init__(self, D, basis: BSplineBasis, n_beta=8, H=16, r=4, block="mlp", hidden=32, kan_grid=5,
                 kan_hidden=(None, None), head="linear", p_u=0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.D, self.K = D, basis.K
        self.flin = FunctionalLinear(D, H, basis, BSplineBasis(n_beta), seed=seed)
        self.encoder = make_block(block, H, r, hidden, kan_grid, kan_hidden[0], seed=seed)
        self.decoder = make_block(block, r, D * basis.K, hidden, kan_grid, kan_hidden[1], seed=seed + 100)
        self.head = AFTHead(r + p_u, head)

    def encode(self, c):
        return self.encoder(self.flin(c))

    def forward(self, c, u):
        z = self.encode(c)
        c_hat = self.decoder(z).view(-1, self.D, self.K)
        eta, sigma = self.head(torch.cat([z, u], dim=1))
        return z, c_hat, eta, sigma

    def reconstruct_at(self, c_hat, Phi):           # (B, D, K), (B, L, K) -> (B, L, D)
        return torch.einsum("bdk,blk->bld", c_hat, Phi)

    def roughness_penalty(self):
        return self.flin.roughness_penalty()

    def block_params(self):
        return n_params(self.encoder) + n_params(self.decoder)


class FlatAE(nn.Module):
    def __init__(self, D, n_grid=20, r=4, hidden=32, head="linear", p_u=0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.D, self.G = D, n_grid
        self.encoder = mlp([D * n_grid, hidden, r])
        self.decoder = mlp([r, hidden, D * n_grid])
        self.head = AFTHead(r + p_u, head)

    def encode(self, grid):
        return self.encoder(grid.reshape(len(grid), -1))

    def forward(self, grid, u):
        z = self.encode(grid)
        g_hat = self.decoder(z).view(-1, self.D, self.G)
        eta, sigma = self.head(torch.cat([z, u], dim=1))
        return z, g_hat, eta, sigma

    def reconstruct_at(self, g_hat, H):             # (B, D, G), (B, L, G) -> (B, L, D)
        return torch.einsum("bdg,blg->bld", g_hat, H)

    def roughness_penalty(self):
        return torch.zeros(())

    def block_params(self):
        return n_params(self.encoder) + n_params(self.decoder)
