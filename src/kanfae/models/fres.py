"""Landmark-anchored residual functional CoxKAN, b2s_res_cox (Round 15, D133).

Each channel is split into its last observed value before the landmark and the trajectory residual:
  eta = sum_d psi_d(X_d,last) + sum_k psi_k(z_k) + sum_d gate_d * sum_m int g_dm(X_d(t) - X_d,last) h_dm(t) dt,
where psi are static KAN edges (the landmark part, like static_coxkan on last values + z) and the second sum is a B2-S
functional layer on the residual curves. This is the existing FKANNet "b2s" kind applied to transformed inputs
(xg -> xg - last, z -> [last, z]), so the training, ensembling, pruning and symbolic machinery are unchanged.

Validation gate (gate_d in {0, 1}): on the tuned seed-0 member, the residual channels are pruned as wholes by their
integrated score (D120 channel score, residual layer only) at theta_rel in THETAS_D120 plus "all off" (the pure
landmark model), each followed by the same 200-epoch masked fine-tune; the parsimonious D120 rule picks the largest
theta (all off counts as the largest) whose validation loss is within `tol` (0.01, mean NLPL per event) of the
minimum. The chosen residual mask is applied to every ensemble member, each then fine-tuned for 200 epochs with the
mask fixed. Residual trajectory effects therefore enter only where validation loss supports them.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from ..survival.cox import cox_nll
from .fkan import THETAS_D120, FKANEnsemble, FKANSurv

ALL_OFF = float("inf")


def residual_pack(pk, residual=True):
    """{xg, last, z, y, d} -> the b2s inputs {xg: xg - last (residual=True; the full curve xg otherwise),
    z: [last, z], y, d}. residual=False is the "anchored" variant (D133 dry run): last-value edges plus a gated B2-S
    on the full trajectory, which nests both b2s_cox and the pure landmark model."""
    z = pk["z"] if pk["z"] is not None and pk["z"].shape[1] else np.zeros((len(pk["last"]), 0))
    xg = pk["xg"] - pk["last"][:, None, :] if residual else pk["xg"]
    return dict(xg=xg, z=np.hstack([pk["last"], z]), y=pk["y"], d=pk["d"])


def _gate(m, theta, train, epochs):
    m = copy.deepcopy(m)
    if theta == ALL_OFF:
        m.net.g.mask[:] = 0.0
    elif theta > 0:
        m.prune(theta, "integrated", only=("g",))
    m.fine_tune(epochs)
    m._finalise(train)
    return m


class ResFKAN:
    """b2s_res_cox: FKANEnsemble (D112 protocol) on residual inputs, then the validation gate."""

    def __init__(self, thetas=THETAS_D120, tol=0.01, fine_tune_epochs=200, lrs=(1e-3, 3e-3), wds=(1e-4, 1e-3),
                 seeds=(0, 1, 2), residual=True, gate="split", cv_folds=5, **kw):
        self.thetas, self.tol, self.ft, self.residual = tuple(thetas) + (ALL_OFF,), tol, fine_tune_epochs, residual
        self.gate, self.cv_folds, self.kw = gate, cv_folds, kw
        self.ens_kw = dict(lrs=lrs, wds=wds, seeds=seeds, kind="b2s", loss="cox", **kw)

    def fit(self, train, val):
        tr, va = residual_pack(train, self.residual), residual_pack(val, self.residual)
        self.D = train["last"].shape[1]
        ens = FKANEnsemble(**self.ens_kw).fit(tr, va)
        self.lr, self.wd, self.tuning = ens.lr, ens.wd, ens.tuning
        cand, self.gate_grid = [], []
        for th in self.thetas:
            m = _gate(ens.members[0], th, tr, self.ft)
            cand.append(m)
            self.gate_grid.append(dict(theta=th if th != ALL_OFF else "all_off", val_loss=float(m.val_loss),
                                       residual_on=[int((m.net.g.mask[d] > 0).any()) for d in range(self.D)]))
        score = "val_loss"
        if self.gate == "cv":                                       # D134: cross-validated gate
            cvs = self._cv_scores(tr, va, ens)
            for g, s in zip(self.gate_grid, cvs):
                g["cv_loss"] = s
            score = "cv_loss"
        vmin = min(g[score] for g in self.gate_grid)
        ok = [i for i, g in enumerate(self.gate_grid) if g[score] <= vmin + self.tol]
        k = max(ok, key=lambda i: self.thetas[i])
        self.gate_grid[k]["chosen"] = True
        mask = cand[k].net.g.mask.clone()
        self.residual_on = self.gate_grid[k]["residual_on"]
        self.members = [cand[k]]
        for m in ens.members[1:]:
            m = copy.deepcopy(m)
            m.net.g.mask[:] = mask
            m.fine_tune(self.ft)
            m._finalise(tr)
            self.members.append(m)
        self._tr = None
        return self

    def _cv_scores(self, tr, va, ens):
        """D134 CV gate: pooled training + validation subjects, `cv_folds` folds (seed 0, stratified on the event);
        per fold the seed-0 model at the tuned (lr, wd) is trained on the other folds for the tuned model's best epoch
        count (no early stopping), each gate setting is fine-tuned `ft` epochs, and the held-out fold's NLPL per event
        is recorded. Returns the fold-mean NLPL per gate setting (order of self.thetas)."""
        from sklearn.model_selection import StratifiedKFold
        pool = {k: np.concatenate([tr[k], va[k]]) for k in ("xg", "z", "y", "d")}
        best_ep = int(np.argmin(ens.members[0].history)) + 1
        folds = StratifiedKFold(self.cv_folds, shuffle=True, random_state=0).split(pool["y"], pool["d"])
        sc = np.zeros(len(self.thetas))
        for a, b in folds:
            ftr = {k: v[a] for k, v in pool.items()}
            m = FKANSurv(**{**self.kw, "kind": "b2s", "loss": "cox", "lr": self.lr, "wd": self.wd, "seed": 0,
                            "max_epochs": best_ep, "patience": best_ep, "early_stopping": False}).fit(ftr, None)
            for j, th in enumerate(self.thetas):
                mm = _gate(m, th, ftr, self.ft)
                eta = torch.tensor(mm.eta(pool["xg"][b], pool["z"][b]))
                sc[j] += float(cox_nll(eta, torch.tensor(pool["y"][b]), torch.tensor(pool["d"][b]), "efron")) / self.cv_folds
        return sc.tolist()

    # prediction on {xg, last, z}
    def _inp(self, xg, last, z):
        p = residual_pack(dict(xg=xg, last=last, z=z, y=None, d=None), self.residual)
        return p["xg"], p["z"]

    def eta(self, xg, last, z):
        X, Z = self._inp(xg, last, z)
        return np.mean([m.eta(X, Z) for m in self.members], 0)

    def predict_risk(self, xg, last, z):
        return self.eta(xg, last, z)

    def predict_survival(self, xg, last, z, times):
        X, Z = self._inp(xg, last, z)
        return np.mean([m.predict_survival(X, Z, times) for m in self.members], 0)

    def last_value_effect(self, d, x):
        """Member-mean static edge psi_d(x) of the last value of channel d, centred, on the original scale."""
        import torch
        out = []
        for m in self.members:
            inp = torch.zeros((len(x), m.net.stat.n_in))
            inp[:, d] = torch.tensor((np.asarray(x, float) - m.zm[d]) / m.zs[d])
            with torch.no_grad():
                out.append(m.net.stat.edge_activations(inp)[:, d, 0].numpy())
        e = np.mean(out, 0)
        return e - e.mean()

    def residual_effect(self, d, r, t):
        """Member-mean residual surface sum_m g_dm(r) h_dm(t) of channel d (r = X_d(t) - X_d,last)."""
        return np.mean([m.effect(d, r, t) for m in self.members], 0)
