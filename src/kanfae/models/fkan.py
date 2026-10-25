"""Functional KAN survival models (Round 11, D112; docs/design_note_round10.md sections 1-2).

Linear predictor eta = (functional part) + sum_k psi_k(z_k), every psi/phi/g/h a univariate B-spline KAN edge
(`KANLayer`), inputs standardised with training statistics (no BatchNorm, so edges are readable in original units).
  kind "static"  : eta = KAN(z)  (1 layer [p -> 1], or [p -> hidden -> 1] with hidden > 0)   -- B1, static CoxKAN
  kind "b2h"     : eta = sum_d int phi_d(x_d(t)) dt + static                                 -- B2-H
  kind "b2s"     : eta = sum_d sum_m int g_dm(x_d(t)) h_dm(t) dt + static (rank M)          -- B2-S
  kind "b2deep"  : s_dm = int g_dm(x_d(t)) h_dm(t) dt;  eta = KAN[D*M + p -> hidden -> 1](s, z)  -- B2-deep
Curves enter as reconstructed values on the 50-point grid t in [0, 1] (trapezoid quadrature).
Losses: Cox (exact Efron/Breslow partial likelihood, optional strata; Breslow baseline) or AFT (log T = mu + eta +
sigma eps, Weibull; optional non-parametric residual calibration of the error distribution).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..survival.aft import aft_nll
from ..survival.cox import breslow_baseline, cox_nll, cox_survival
from .kan import KANLayer

GRID = np.linspace(0.0, 1.0, 50)


def _trap_w(grid):
    w = np.full(len(grid), grid[1] - grid[0])
    w[[0, -1]] *= 0.5
    return w


class FKANNet(nn.Module):
    def __init__(self, kind, D, p, M=2, hidden=0, grid_size=5, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.kind, self.D, self.p, self.M, self.hidden = kind, D, p, M, hidden
        self.register_buffer("tgrid", torch.tensor(GRID))
        self.register_buffer("tw", torch.tensor(_trap_w(GRID)))
        rng = (-3.0, 3.0)
        if kind == "b2h":
            self.phi = KANLayer(D, 1, grid_size, grid_range=rng, seed=seed)
        if kind in ("b2s", "b2deep"):
            self.g = KANLayer(D, M, grid_size, grid_range=rng, seed=seed)
            self.h = KANLayer(1, D * M, grid_size, grid_range=(-0.05, 1.05), seed=seed + 1)
        if kind == "b2deep":
            self.deep1 = KANLayer(D * M + p, hidden or 4, grid_size, grid_range=rng, seed=seed + 2)
            self.deep2 = KANLayer(hidden or 4, 1, grid_size, grid_range=rng, seed=seed + 3)
        elif kind == "static" and hidden > 0:
            self.deep1 = KANLayer(p, hidden, grid_size, grid_range=rng, seed=seed + 2)
            self.deep2 = KANLayer(hidden, 1, grid_size, grid_range=rng, seed=seed + 3)
        elif p > 0:
            self.stat = KANLayer(p, 1, grid_size, grid_range=rng, seed=seed + 4)

    # ---- contributions -------------------------------------------------------------------------------------------
    def channel_terms(self, xg):
        """Per-subject functional contributions before summation: b2h (n, D); b2s/b2deep (n, D, M)."""
        n, G, D = xg.shape
        flat = xg.reshape(n * G, D)
        if self.kind == "b2h":
            a = self.phi.edge_activations(flat)[:, :, 0].reshape(n, G, D)
            return torch.einsum("ngd,g->nd", a, self.tw)
        g = self.g.edge_activations(flat).reshape(n, G, D, self.M)
        h = self.h.edge_activations(self.tgrid[:, None])[:, 0, :].reshape(G, D, self.M)
        return torch.einsum("ngdm,gdm,g->ndm", g, h, self.tw)

    def forward(self, xg, z):
        n = z.shape[0] if z is not None else xg.shape[0]
        if self.kind == "static":
            if self.hidden > 0:
                return self.deep2(self.deep1(z))[:, 0]
            return self.stat(z)[:, 0] if self.p > 0 else torch.zeros(n, dtype=self.tw.dtype)
        terms = self.channel_terms(xg)
        if self.kind == "b2deep":
            s = terms.reshape(n, -1)
            return self.deep2(self.deep1(torch.cat([s, z], 1) if self.p else s))[:, 0]
        eta = terms.reshape(n, -1).sum(1)
        return eta + (self.stat(z)[:, 0] if self.p > 0 else 0.0)

    def layers_with_inputs(self, xg, z):
        """(name, layer, inputs) for regularisation and pruning (inputs as the layer sees them)."""
        out = []
        n = z.shape[0] if z is not None else xg.shape[0]
        if self.kind in ("b2h", "b2s", "b2deep"):
            flat = xg.reshape(-1, self.D)
            out.append(("phi" if self.kind == "b2h" else "g", self.phi if self.kind == "b2h" else self.g, flat))
        if self.kind == "b2deep":
            s = self.channel_terms(xg).reshape(n, -1)
            x1 = torch.cat([s, z], 1) if self.p else s
            out += [("deep1", self.deep1, x1), ("deep2", self.deep2, self.deep1(x1))]
        elif self.kind == "static" and self.hidden > 0:
            out += [("deep1", self.deep1, z), ("deep2", self.deep2, self.deep1(z))]
        elif self.p > 0:
            out.append(("stat", self.stat, z))
        return out


class FKANSurv:
    """Fit/predict wrapper: standardisation, Cox/AFT loss, training with early stopping, pruning, symbolic."""

    def __init__(self, kind="b2s", loss="cox", ties="efron", M=2, hidden=0, lr=3e-3, wd=1e-4, lam=0.0,
                 max_epochs=3000, patience=100, residual_calibration=False, seed=0, family="weibull", edge="spline",
                 early_stopping=True):
        self.cfg = dict(kind=kind, loss=loss, ties=ties, M=M, hidden=hidden, lr=lr, wd=wd, lam=lam,
                        max_epochs=max_epochs, patience=patience, residual_calibration=residual_calibration,
                        seed=seed, family=family, edge=edge, early_stopping=early_stopping)

    # ---- data ----------------------------------------------------------------------------------------------------
    def _std_fit(self, xg, z):
        self.xm = float(np.mean(xg)) if xg is not None else 0.0
        self.xs = float(np.std(xg)) if xg is not None else 1.0
        if xg is not None:                        # per-channel pooled over grid and subjects
            self.xm = xg.mean(axis=(0, 1))
            self.xs = np.where(xg.std(axis=(0, 1)) > 0, xg.std(axis=(0, 1)), 1.0)
        self.zm = z.mean(0) if z is not None and z.shape[1] else np.zeros(0)
        self.zs = np.where(z.std(0) > 0, z.std(0), 1.0) if z is not None and z.shape[1] else np.ones(0)

    def _tensors(self, xg, z):
        X = torch.tensor((xg - self.xm) / self.xs) if xg is not None else None
        Z = torch.tensor((z - self.zm) / self.zs) if z is not None and z.shape[1] else torch.zeros((len(xg), 0))
        return X, Z

    def _loss(self, eta, y, d, strata=None):
        if self.cfg["loss"] == "cox":
            return cox_nll(eta, y, d, self.cfg["ties"], strata=strata)
        sigma = nn.functional.softplus(self.log_sigma) + 1e-3
        return aft_nll(y, d, self.mu + eta, sigma, self.cfg["family"])

    def _reg(self, X, Z):
        if self.cfg["lam"] <= 0:
            return 0.0
        return self.cfg["lam"] * sum(l.regularization(x)[0] for _, l, x in self.net.layers_with_inputs(X, Z))

    # ---- training ------------------------------------------------------------------------------------------------
    def fit(self, train, val, strata_tr=None, strata_va=None):
        """train/val: dicts with keys xg (n, G, D) or None, z (n, p), y, d. val None (with early_stopping False):
        train on the whole training set for max_epochs; the monitored loss is then the training loss."""
        torch.set_default_dtype(torch.float64)
        c = self.cfg
        if val is None:
            if c["early_stopping"]:
                raise ValueError("early stopping needs a validation split")
            val, strata_va = train, strata_tr
        self._std_fit(train["xg"], train["z"])
        Xtr, Ztr = self._tensors(train["xg"], train["z"])
        Xva, Zva = self._tensors(val["xg"], val["z"])
        D = 0 if train["xg"] is None else train["xg"].shape[2]
        self.net = FKANNet(c["kind"], D, Ztr.shape[1], c["M"], c["hidden"], seed=c["seed"])
        if c["edge"] == "linear":                  # static 1-layer CoxKAN whose edges are linear in original units
            for k in range(self.net.stat.n_in):
                self.net.stat.symbolic[(k, 0)] = SymbolicEdge("x", 0.0, 0.0, 1.0, 0.0, self.zm[k], self.zs[k])
            self._sym_params = [e.p for e in self.net.stat.symbolic.values()]
        self.mu = nn.Parameter(torch.tensor(float(np.log(np.mean(train["y"])))))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self.data = dict(Xtr=Xtr, Ztr=Ztr, Xva=Xva, Zva=Zva, ytr=torch.tensor(train["y"]), dtr=torch.tensor(train["d"]),
                         yva=torch.tensor(val["y"]), dva=torch.tensor(val["d"]), str=strata_tr, sva=strata_va)
        self.history = self._train(c["max_epochs"], c["patience"])
        self.train_ = train
        self._finalise(train)
        return self

    def refit_symbolic(self, max_iter=500, anchor=None, mu=0.0, safe=False):
        """CoxKAN-style refit after substitution: all remaining parameters (including the substituted formulas'
        affine parameters) by L-BFGS on the unpenalised training loss, then the baseline is re-estimated.
        D133 (Round 15) option: with `anchor` (the pruned model before substitution) and mu > 0, the objective adds
        mu * sum over anchored terms of mean((term - term_anchor)^2) / var(term_anchor) on the training inputs
        (functional distillation, see anchor_terms); val_loss stays the unpenalised validation loss.
        safe (D138, Round 16; default False): if the refit leaves any parameter or the objective non-finite, or does not
        lower the training objective it minimises, every parameter is restored to its pre-refit value (the guarded
        substituted model); self.refit_fallback records whether this happened."""
        dd = self.data
        params = self._params()
        if safe:
            snap = [q.detach().clone() for q in params]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, tolerance_grad=1e-10, tolerance_change=1e-14,
                                history_size=50, line_search_fn="strong_wolfe")
        ref = None
        if anchor is not None and mu > 0:
            with torch.no_grad():
                ref = [(r.detach().clone(), max(float(r.var()), 1e-8)) for r in anchor_terms(anchor.net, dd["Xtr"], dd["Ztr"])]

        def objective():
            loss = self._loss(self.net(dd["Xtr"], dd["Ztr"]), dd["ytr"], dd["dtr"], dd["str"])
            if ref is not None:
                cur = anchor_terms(self.net, dd["Xtr"], dd["Ztr"])
                loss = loss + mu * sum(((c - r) ** 2).mean() / v for c, (r, v) in zip(cur, ref))
            return loss

        def closure():
            opt.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        if safe:
            with torch.no_grad():
                obj0 = float(objective())                 # objective before the refit
        opt.step(closure)
        loss = closure()
        self.refit_fallback = False
        if safe:
            ok = np.isfinite(float(loss.detach())) and all(bool(torch.isfinite(q).all()) for q in params)
            if not ok or not float(loss.detach()) < obj0:
                with torch.no_grad():
                    for q, s in zip(params, snap):
                        q.copy_(s)
                self.refit_fallback = True
                loss = closure()
        self.refit_grad_norm = float(torch.sqrt(sum((q.grad ** 2).sum() for q in params if q.grad is not None)))
        self.refit_train_loss = float(self._loss(self.net(dd["Xtr"], dd["Ztr"]), dd["ytr"], dd["dtr"], dd["str"]).detach())
        self.refit_objective = float(loss.detach())
        with torch.no_grad():
            self.val_loss = float(self._loss(self.net(dd["Xva"], dd["Zva"]), dd["yva"], dd["dva"], dd["sva"]))
        self._finalise(self.train_)
        return self

    def _params(self):
        return (list(self.net.parameters()) + list(getattr(self, "_sym_params", []))
                + ([self.mu, self.log_sigma] if self.cfg["loss"] == "aft" else []))

    def _train(self, max_epochs, patience):
        c, dd = self.cfg, self.data
        opt = torch.optim.Adam(self._params(), lr=c["lr"], weight_decay=c["wd"])
        best, best_state, bad, hist = np.inf, None, 0, []
        for ep in range(max_epochs):
            self.net.train()
            opt.zero_grad()
            loss = self._loss(self.net(dd["Xtr"], dd["Ztr"]), dd["ytr"], dd["dtr"], dd["str"]) + self._reg(dd["Xtr"], dd["Ztr"])
            loss.backward()
            opt.step()
            with torch.no_grad():
                vl = float(self._loss(self.net(dd["Xva"], dd["Zva"]), dd["yva"], dd["dva"], dd["sva"]))
            hist.append(vl)
            if not np.isfinite(vl):
                break
            if not c["early_stopping"]:
                best = vl
                continue
            if vl < best - 1e-7:
                best, bad = vl, 0
                best_state = (copy.deepcopy(self.net.state_dict()), float(self.mu.detach()), float(self.log_sigma.detach()),
                              [q.detach().clone() for q in getattr(self, "_sym_params", [])])
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None and c["early_stopping"]:
            self.net.load_state_dict(best_state[0])
            with torch.no_grad():
                self.mu.fill_(best_state[1]); self.log_sigma.fill_(best_state[2])
                for q, v in zip(getattr(self, "_sym_params", []), best_state[3]):
                    q.copy_(v)
        self.val_loss = best
        return hist

    def _finalise(self, train):
        with torch.no_grad():
            eta_tr = self.net(self.data["Xtr"], self.data["Ztr"]).numpy()
        if self.cfg["loss"] == "cox":
            self.baseline = breslow_baseline(eta_tr, train["y"], train["d"],
                                             None if self.data["str"] is None else np.asarray(self.data["str"]))
        else:
            self.sigma = float(nn.functional.softplus(self.log_sigma.detach()) + 1e-3)
            if self.cfg["residual_calibration"]:
                from ..evaluation.metrics import km_curve
                r = (np.log(train["y"]) - float(self.mu) - eta_tr) / self.sigma
                self.res_grid = np.sort(np.unique(r))
                self.res_S = km_curve(r - r.min() + 1.0, train["d"], self.res_grid - r.min() + 1.0)

    # ---- prediction ----------------------------------------------------------------------------------------------
    def eta(self, xg, z):
        X, Z = self._tensors(xg, z)
        self.net.eval()
        with torch.no_grad():
            return self.net(X, Z).numpy()

    def predict_risk(self, xg, z):
        e = self.eta(xg, z)
        return e if self.cfg["loss"] == "cox" else -e

    def predict_survival(self, xg, z, times, strata=None):
        e = self.eta(xg, z)
        times = np.asarray(times, float)
        if self.cfg["loss"] == "cox":
            if strata is None:
                return cox_survival(e, self.baseline[None], times)
            return np.vstack([cox_survival(e[i:i + 1], self.baseline[s], times) for i, s in enumerate(strata)])
        w = (np.log(np.maximum(times, 1e-300))[None, :] - float(self.mu) - e[:, None]) / self.sigma
        if self.cfg["residual_calibration"]:
            idx = np.searchsorted(self.res_grid, w, side="right") - 1
            S = np.where(idx >= 0, self.res_S[np.clip(idx, 0, None)], 1.0)
        else:
            S = np.exp(-np.exp(w))                                # Weibull: S = exp(-exp(w))
        return np.where(times[None, :] <= 0, 1.0, S)

    # ---- shapes --------------------------------------------------------------------------------------------------
    def effect(self, d, x, t):
        """Learned phi_d(x, t) on the original x scale (b2h: phi_d(x); b2s: sum_m g_dm(x) h_dm(t)). For AFT the
        PH-equivalent shape -phi/sigma is returned (Weibull AFT <=> PH with eta_PH = -eta_AFT/sigma)."""
        x = np.asarray(x, float)
        t = np.broadcast_to(np.asarray(t, float), x.shape)
        xs = torch.tensor(((x - self.xm[d]) / self.xs[d]).ravel())
        with torch.no_grad():
            if self.net.kind == "b2h":
                inp = torch.zeros((len(xs), self.net.D)); inp[:, d] = xs
                out = self.net.phi.edge_activations(inp)[:, d, 0].numpy()
            else:
                inp = torch.zeros((len(xs), self.net.D)); inp[:, d] = xs
                g = self.net.g.edge_activations(inp)[:, d, :]                         # (N, M)
                h = self.net.h.edge_activations(torch.tensor(t.ravel())[:, None])[:, 0, :]
                h = h.reshape(len(xs), self.net.D, self.net.M)[:, d, :]
                out = (g * h).sum(1).numpy()
        out = out.reshape(x.shape)
        return out if self.cfg["loss"] == "cox" else -out / self.sigma

    def selected_channels(self):
        if self.net.kind == "b2h":
            return [int(self.net.phi.mask[d, 0] > 0) for d in range(self.net.D)]
        if self.net.kind in ("b2s", "b2deep"):
            return [int((self.net.g.mask[d] > 0).any()) for d in range(self.net.D)]
        first = self.net.deep1 if self.net.hidden > 0 else self.net.stat          # static: selected inputs
        return [int((first.mask[k] > 0).any()) for k in range(first.n_in)]

    # ---- scale-aware pruning -------------------------------------------------------------------------------------
    @torch.no_grad()
    def edge_scores(self):
        """Centred, layer-normalised scores. Functional layers: score of channel term (d[, m]) = mean_i |c_id - mean|
        of the per-subject contribution to eta; other layers: mean |phi_e - mean phi_e| over the layer's inputs."""
        X, Z = self.data["Xtr"], self.data["Ztr"]
        out = {}
        for name, layer, inp in self.net.layers_with_inputs(X, Z):
            if name in ("phi", "g"):
                c = self.net.channel_terms(X)
                c = c[:, :, None] if c.dim() == 2 else c                  # (n, D, M)
                s = (c - c.mean(0)).abs().mean(0)                          # (D, M)
            else:
                a = layer.edge_activations(inp)
                s = (a - a.mean(0)).abs().mean(0)                          # (in, out)
            out[name] = (layer, s / max(float(s.max()), 1e-12))
        return out

    @torch.no_grad()
    def channel_scores(self):
        """D120: S_d = mean_i |eta_id - mean eta_d| of each channel's integrated functional effect
        eta_id = int phi_d(X_id(t), t) dt (summed over the rank-M terms), normalised by max_d S_d."""
        c = self.net.channel_terms(self.data["Xtr"])
        c = c.sum(2) if c.dim() == 3 else c                          # (n, D)
        s = (c - c.mean(0)).abs().mean(0)
        return s / max(float(s.max()), 1e-12)

    @torch.no_grad()
    def prune(self, theta_rel, channel_score="edge", only=None):
        """channel_score "edge" (Round 11): every edge/term against theta_rel with layer-normalised scores.
        "integrated" (D120): functional channels as wholes by channel_scores(); other layers as in "edge".
        only (D133): restrict pruning to the named layers (e.g. ("g",) for the residual channels of b2s_res)."""
        n_pruned = 0
        for name, (layer, s) in self.edge_scores().items():
            if only is not None and name not in only:
                continue
            if channel_score == "integrated" and name in ("phi", "g"):
                cs = self.channel_scores()
                sel = (cs[:, None] < theta_rel).expand_as(layer.mask) & (layer.mask > 0)
            else:
                sel = (s < theta_rel) & (layer.mask > 0)
            layer.mask[sel] = 0.0
            n_pruned += int(sel.sum())
        return n_pruned

    def fine_tune(self, epochs=200):
        self.history_ft = self._train(epochs, epochs)
        return self


LAMS_D112, THETAS_D112 = (0.0, 1e-4, 1e-3, 1e-2), (0.0, 0.01, 0.02, 0.05, 0.10)
LAMS_D120, THETAS_D120 = (0.0, 1e-4, 1e-3, 1e-2, 3e-2), (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25)


def select_lambda_theta(make, train, val, lams=LAMS_D120, thetas=THETAS_D120, fine_tune_epochs=200, strata_tr=None,
                        strata_va=None, rule="parsimonious", tol=0.01, channel_score="integrated"):
    """Train at each lambda, prune at each theta_rel, fine-tune with the mask fixed (every theta, including 0).
    rule "min" (D112): smallest validation loss (ties: larger theta, then larger lambda), with lams/thetas D112 and
    channel_score "edge". rule "parsimonious" (D120, default): among grid points whose validation loss is within `tol`
    of the minimum, the largest theta, then the largest lambda. make(lam) -> unfitted FKANSurv.
    Returns (selected model, records)."""
    recs, models = [], []
    for lam in lams:
        base = make(lam).fit(train, val, strata_tr, strata_va)
        for th in thetas:
            m = copy.deepcopy(base)
            n_pr = m.prune(th, channel_score) if th > 0 else 0
            m.fine_tune(fine_tune_epochs)
            m._finalise(train)
            recs.append(dict(lam=lam, theta=th, val_loss=float(m.val_loss), pruned=n_pr, selected=m.selected_channels()))
            models.append(m)
    if rule == "min":
        k = min(range(len(recs)), key=lambda i: (round(recs[i]["val_loss"], 10), -recs[i]["theta"], -recs[i]["lam"]))
    else:
        vmin = min(r["val_loss"] for r in recs)
        ok = [i for i, r in enumerate(recs) if r["val_loss"] <= vmin + tol]
        k = max(ok, key=lambda i: (recs[i]["theta"], recs[i]["lam"]))
    recs[k]["chosen"] = True
    return models[k], recs


class FKANEnsemble:
    """D112 training protocol: lr {1e-3, 3e-3} x weight decay {1e-4, 1e-3} tuned on validation loss (seed 0), then a
    3-seed ensemble (seeds 0-2) at the selected setting. eta and shapes are member means (valid because eta is additive
    in the edge functions; shapes are centred per t when compared, so member constants do not matter)."""

    def __init__(self, lrs=(1e-3, 3e-3), wds=(1e-4, 1e-3), seeds=(0, 1, 2), **kw):
        self.lrs, self.wds, self.seeds, self.kw = lrs, wds, seeds, kw

    def fit(self, train, val, strata_tr=None, strata_va=None):
        self.tuning = []
        best = None
        for lr in self.lrs:
            for wd in self.wds:
                m = FKANSurv(**{**self.kw, "lr": lr, "wd": wd, "seed": self.seeds[0]}).fit(train, val, strata_tr, strata_va)
                self.tuning.append(dict(lr=lr, wd=wd, val_loss=float(m.val_loss), epochs=len(m.history)))
                if best is None or m.val_loss < best[0]:
                    best = (m.val_loss, lr, wd, m)
        _, self.lr, self.wd, m0 = best
        self.members = [m0] + [FKANSurv(**{**self.kw, "lr": self.lr, "wd": self.wd, "seed": s}).fit(
            train, val, strata_tr, strata_va) for s in self.seeds[1:]]
        return self

    def eta(self, xg, z):
        return np.mean([m.eta(xg, z) for m in self.members], 0)

    def predict_risk(self, xg, z):
        return np.mean([m.predict_risk(xg, z) for m in self.members], 0)

    def predict_survival(self, xg, z, times, strata=None):
        return np.mean([m.predict_survival(xg, z, times, strata) for m in self.members], 0)

    def effect(self, d, x, t):
        return np.mean([m.effect(d, x, t) for m in self.members], 0)


# ---- symbolic extraction (stages 1-3) and substitution ----------------------------------------------------------------
_TORCH_F = {
    "x": lambda u: u, "x^2": lambda u: u ** 2, "x^3": lambda u: u ** 3, "x^4": lambda u: u ** 4, "exp": torch.exp,
    "log": torch.log, "sqrt": torch.sqrt, "1/x": lambda u: 1.0 / u, "sin": torch.sin, "cos": torch.cos,
    "tanh": torch.tanh, "sigmoid": torch.sigmoid, "arctan": torch.arctan, "abs": torch.abs,
    "gaussian": lambda u: torch.exp(-u ** 2), "softplus": nn.functional.softplus,
}


class SymbolicEdge(nn.Module):
    """c * f(a * x_orig + b) + d with x_orig = x_layer * scale + offset; (a, b, c, d) trainable (affine refit).
    guard (D133): (sign, floor, tau); the argument u = a x + b is replaced by the smooth bound
    sign * (floor + tau * softplus((sign * u - floor) / tau)), i.e. sign * u is kept above floor. For log, sqrt and 1/x
    (sign = that of u on the training range, floor = half the smallest |u| there, tau = floor / 10) this differs from u
    by at most 5e-6 * floor on the training range and can never leave the formula's domain; for exp (sign -1,
    floor = -(largest training u + 10), tau 1) it caps u smoothly 10 above its training range (relative change of
    exp(u) at most 5e-5 there), so exp cannot overflow. guard None = the Round-11 edge (default)."""

    def __init__(self, name, a, b, c, d, offset=0.0, scale=1.0, guard=None):
        super().__init__()
        self.name, self.offset, self.scale, self.guard = name, float(offset), float(scale), guard
        self.p = nn.Parameter(torch.tensor([a, b, c, d], dtype=torch.float64))

    def forward(self, x):
        u = self.p[0] * (x * self.scale + self.offset) + self.p[1]
        if self.guard is not None:
            sg, fl, tau = self.guard
            u = sg * (fl + tau * nn.functional.softplus((sg * u - fl) / tau))
        return self.p[2] * _TORCH_F[self.name](u) + self.p[3]


GUARDED = ("log", "sqrt", "1/x", "exp")


def domain_guard(name, a, b, x_train):
    """(sign, floor, tau) for a restricted-domain (log, sqrt, 1/x) or overflow-prone (exp) formula fitted on x_train
    (original units), or None if f needs no guard or the training arguments do not have one strict sign (then the
    formula is left as fitted)."""
    if name not in GUARDED:
        return None
    u = a * np.asarray(x_train, float) + b
    if name == "exp":
        return (-1.0, -(float(np.max(u)) + 10.0), 1.0)
    sg = 1.0 if name in ("log", "sqrt") else float(np.sign(np.median(u)))
    if sg == 0 or np.any(sg * u <= 0):
        return None
    fl = 0.5 * float(np.min(sg * u))
    return (sg, fl, fl / 10.0)


def anchor_terms(net, X, Z):
    """Distillation targets (D133): the functional surfaces phi_d(X_ig, t_g) at every training (subject, grid time) --
    b2h: phi_d(x); b2s/b2deep: sum_m g_dm(x) h_dm(t) -- one (n*G,) tensor per channel, then every static/deep edge's
    activation on its training inputs, one tensor per edge."""
    out = []
    n = X.shape[0] if X is not None else Z.shape[0]
    if net.kind in ("b2h", "b2s", "b2deep"):
        G = X.shape[1]
        flat = X.reshape(-1, net.D)
        if net.kind == "b2h":
            a = net.phi.edge_activations(flat)[:, :, 0]                          # (nG, D)
            out += [a[:, d] for d in range(net.D)]
        else:
            g = net.g.edge_activations(flat).reshape(n, G, net.D, net.M)
            h = net.h.edge_activations(net.tgrid[:, None])[:, 0, :].reshape(G, net.D, net.M)
            s = (g * h[None]).sum(3)                                              # (n, G, D)
            out += [s[:, :, d].reshape(-1) for d in range(net.D)]
    for name, layer, inp in net.layers_with_inputs(X, Z):
        if name in ("phi", "g"):
            continue
        a = layer.edge_activations(inp)
        out += [a[:, i, o] for i in range(layer.n_in) for o in range(layer.n_out)]
    return out


def _edge_inputs(model):
    """(layer name, layer, inputs as the layer sees them, per-input (offset, scale) to original units)."""
    net, X, Z = model.net, model.data["Xtr"], model.data["Ztr"]
    out = []
    for name, layer, inp in net.layers_with_inputs(X, Z):
        if name in ("phi", "g"):
            os_ = [(float(model.xm[d]), float(model.xs[d])) for d in range(net.D)]
        elif name == "stat" or (name == "deep1" and net.kind == "static"):
            os_ = [(float(m), float(s)) for m, s in zip(model.zm, model.zs)]
        elif name == "deep1":                                   # b2deep: functional summaries (identity) then z
            os_ = [(0.0, 1.0)] * (net.D * net.M) + [(float(m), float(s)) for m, s in zip(model.zm, model.zs)]
        else:
            os_ = [(0.0, 1.0)] * layer.n_in
        out.append((name, layer, inp, os_))
    if net.kind in ("b2s", "b2deep"):
        out.append(("h", net.h, net.tgrid[:, None], [(0.0, 1.0)]))
    return out


@torch.no_grad()
def symbolic_edges(model, use_pysr=False, pysr_timeout=60, max_pysr=10, min_share=0.05, seed=0, max_abs_arg=None):
    """Fit stages 1-2 (and, if use_pysr, stage 3 on edges with >= min_share of the layer's total centred score, at
    most max_pysr per model) to every active edge, in original input units. Returns a list of records."""
    from ..symbolic.fit import fit_symbolic
    recs, n_pysr = [], 0
    for name, layer, inp, os_ in _edge_inputs(model):
        a = layer.edge_activations(inp)                                   # (N, in, out)
        sc = (a - a.mean(0)).abs().mean(0)
        share = sc / max(float(sc.sum()), 1e-12)
        order = sorted(((float(share[i, o]), i, o) for i in range(layer.n_in) for o in range(layer.n_out)), reverse=True)
        for sh, i, o in order:
            if layer.mask[i, o] == 0 or (i, o) in layer.symbolic:
                continue
            off, scl = os_[i]
            xo = inp[:, i].numpy() * scl + off
            yv = a[:, i, o].numpy()
            if len(np.unique(xo)) > 2000:                                  # thin very large input sets
                sel = np.random.default_rng(seed).choice(len(xo), 2000, replace=False)
                xo, yv = xo[sel], yv[sel]

            def func(xg, layer=layer, i=i, o=o, off=off, scl=scl):
                full = torch.zeros((len(xg), layer.n_in))
                full[:, i] = torch.tensor((np.asarray(xg) - off) / scl)
                return layer.edge_activations(full)[:, i, o].numpy()

            pysr_ok = use_pysr and sh >= min_share and n_pysr < max_pysr
            res = fit_symbolic(xo, yv, func, seed=seed, use_pysr=pysr_ok, pysr_budget={"timeout": pysr_timeout},
                               max_abs_arg=max_abs_arg)
            n_pysr += int(pysr_ok and res.stage in ("pysr", "none"))
            recs.append(dict(layer=name, i=i, o=o, share=sh, offset=off, scale=scl, result=res))
    return recs


def substitute_symbolic(model, recs, stages=("linear", "library"), guard=False):
    """Replace every accepted edge of the given stages by its formula (trainable affine parameters).
    guard (D133): restricted-domain library formulas get a domain guard computed on the edge's training inputs."""
    ei = {name: (layer, inp, os_) for name, layer, inp, os_ in _edge_inputs(model)}
    layers = {name: v[0] for name, v in ei.items()}
    n = 0
    for r in recs:
        res = r["result"]
        if not res.accepted or res.stage not in stages:
            continue
        if res.stage == "linear":
            e = SymbolicEdge("x", res.params["a"], res.params["b"], 1.0, 0.0, r["offset"], r["scale"])
        else:
            p = res.params
            gd = None
            if guard:
                _, inp, _ = ei[r["layer"]]
                xo = inp[:, r["i"]].detach().numpy() * r["scale"] + r["offset"]
                gd = domain_guard(p["f"], p["a"], p["b"], xo)
            e = SymbolicEdge(p["f"], p["a"], p["b"], p["c"], p["d"], r["offset"], r["scale"], guard=gd)
        layers[r["layer"]].symbolic[(r["i"], r["o"])] = e
        n += 1
    model._sym_params = [e.p for layer in layers.values() for e in layer.symbolic.values() if isinstance(e, SymbolicEdge)]
    return n


def refit_distill(sub, anchor, mus=(1e-2, 1e-1, 1.0), max_iter=200):
    """D133 symbolic_distill: refit copies of the substituted model `sub` (substituted with guard=True) with the
    functional-distillation anchor at each mu, choose mu by the unpenalised validation loss (a non-finite fit is never
    chosen). Returns (chosen model or None, records)."""
    recs, best = [], None
    for mu in mus:
        m = copy.deepcopy(sub)
        m.refit_symbolic(max_iter=max_iter, anchor=anchor, mu=mu)
        ok = bool(np.isfinite(m.val_loss) and np.isfinite(m.refit_train_loss))
        recs.append(dict(mu=mu, val_loss=float(m.val_loss), train_loss=float(m.refit_train_loss), finite=ok,
                         grad_norm=float(m.refit_grad_norm)))
        if ok and (best is None or m.val_loss < best[0] - 1e-12):
            best = (m.val_loss, mu, m)
    if best is not None:
        [r for r in recs if r["mu"] == best[1]][0]["chosen"] = True
    return (best[2] if best else None), recs
