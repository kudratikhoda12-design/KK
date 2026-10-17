"""Scalar-covariate KAN-AFT (Reference A: Jose, Francis & Kattumannil, arXiv:2512.20305v2) with four censoring losses.

eta(z) = KAN(z): additive [p -> 1] (n_hidden = 0) or [p -> n_hidden -> 1].
loss = "parametric": censoring-aware Weibull AFT likelihood (our reference implementation; sigma estimated);
       "bj" | "ipcw" | "transform": censoring-adjusted least squares (KAN-AFT Appendix A; decision D42), with survival
       curves from the KM of censored residuals.
Regularisation (CoxKAN Eqs. 5-8 / KAN-AFT §3.2.2): lam * sum_l (|Phi_l|_1 + lam_ent S(Phi_l) + lam_coef |C_l|_1).
Pruning: edges (and hidden nodes) with L1 norm < theta (KAN-AFT: theta = 1e-2).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..survival.aft import aft_nll, aft_survival
from ..survival.censored_ls import (bj_impute_logscale, censoring_G, ipcw_weights, residual_survival,
                                    transform_targets)
from ..symbolic.fit import fit_symbolic
from .kan import KANLayer


class KANNet(nn.Module):
    def __init__(self, p, n_hidden=0, grid=5, k=3, seed=0):
        super().__init__()
        dims = [p, 1] if n_hidden == 0 else [p, n_hidden, 1]
        self.layers = nn.ModuleList([KANLayer(a, b, grid, k, seed=seed + i) for i, (a, b) in
                                     enumerate(zip(dims[:-1], dims[1:]))])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x.squeeze(-1)

    def inputs(self, x):
        ins = []
        for l in self.layers:
            ins.append(x); x = l(x)
        return ins

    def regularization(self, x, lam_ent, lam_coef):
        return sum(l.regularization(xi, lam_ent, lam_coef)[0] for l, xi in zip(self.layers, self.inputs(x)))


class KANAFT:
    def __init__(self, loss="parametric", n_hidden=0, grid=5, lam=1e-3, lam_ent=2.0, lam_coef=1.0, lr=1e-2,
                 max_epochs=2000, patience=100, family="weibull", bj_rounds=10, seed=0):
        self.__dict__.update(dict(loss=loss, n_hidden=n_hidden, grid=grid, lam=lam, lam_ent=lam_ent, lam_coef=lam_coef,
                                  lr=lr, max_epochs=max_epochs, patience=patience, family=family, bj_rounds=bj_rounds,
                                  seed=seed))

    def _std(self, X):
        return torch.tensor((np.asarray(X, float) - self.mu) / self.sd)

    def _train(self, Xt, target, w, Xv, tv, wv):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best, state, bad = np.inf, None, 0
        for ep in range(self.max_epochs):
            self.net.train(); opt.zero_grad()
            loss = self._loss(Xt, target, w) + self.lam * self.net.regularization(Xt, self.lam_ent, self.lam_coef)
            loss.backward(); opt.step()
            with torch.no_grad():
                v = float(self._loss(Xv, tv, wv))
            if v < best - 1e-6:
                best, state, bad = v, copy.deepcopy(self.net.state_dict()), 0
                self.best_rho = float(self.rho) if self.loss == "parametric" else None
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(state)
        return ep + 1

    def _loss(self, X, target, w):
        eta = self.net(X)
        if self.loss == "parametric":
            y, d = target
            return aft_nll(y, d, eta, torch.nn.functional.softplus(self.rho) + 1e-3, self.family)
        return (w * (target - eta) ** 2).sum() / w.sum().clamp_min(1e-12)

    def fit(self, Xtr, ytr, dtr, Xva, yva, dva):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(self.seed)
        Xtr, Xva = np.asarray(Xtr, float), np.asarray(Xva, float)
        self.mu, self.sd = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1.0)
        self.scale = float(np.median(ytr))
        Xt, Xv = self._std(Xtr), self._std(Xva)
        self.net = KANNet(Xtr.shape[1], self.n_hidden, self.grid, seed=self.seed)
        yt, yv = np.asarray(ytr, float) / self.scale, np.asarray(yva, float) / self.scale
        dt, dv = np.asarray(dtr, bool), np.asarray(dva, bool)
        self.info = {}
        if self.loss == "parametric":
            self.rho = nn.Parameter(torch.tensor(0.0))
            self.net.register_parameter("rho", self.rho)
            ep = self._train(Xt, (torch.tensor(yt), torch.tensor(dt)), None, Xv, (torch.tensor(yv), torch.tensor(dv)), None)
            self.sigma = float(torch.nn.functional.softplus(torch.tensor(self.best_rho)) + 1e-3)
            self.info["epochs"] = ep
        else:
            G = censoring_G(yt, dt)
            ones_t, ones_v = torch.ones(len(yt)), torch.ones(len(yv))
            if self.loss == "ipcw":
                tgt_t, w_t = torch.tensor(np.log(yt)), torch.tensor(ipcw_weights(yt, dt, G))
                tgt_v, w_v = torch.tensor(np.log(yv)), torch.tensor(ipcw_weights(yv, dv, G))
                self.info["epochs"] = self._train(Xt, tgt_t, w_t, Xv, tgt_v, w_v)
            elif self.loss == "transform":
                Tt, a = transform_targets(yt, dt, G)
                Tv, _ = transform_targets(yv, dv, G)
                Tv = np.maximum(Tv, 1e-6)
                self.info.update(alpha=a, epochs=self._train(Xt, torch.tensor(np.log(Tt)), ones_t, Xv,
                                                             torch.tensor(np.log(Tv)), ones_v))
            elif self.loss == "bj":
                eta_t, eta_v = np.zeros(len(yt)), np.zeros(len(yv))
                hist = []
                for k in range(self.bj_rounds):
                    tgt_t = torch.tensor(bj_impute_logscale(np.log(yt), dt, eta_t))
                    tgt_v = torch.tensor(bj_impute_logscale(np.log(yv), dv, eta_v))
                    self._train(Xt, tgt_t, ones_t, Xv, tgt_v, ones_v)
                    with torch.no_grad():
                        new_t, eta_v = self.net(Xt).numpy(), self.net(Xv).numpy()
                    hist.append(float(np.mean(np.abs(new_t - eta_t))))
                    eta_t = new_t
                    if hist[-1] < 1e-3:
                        break
                self.info.update(bj_rounds=k + 1, bj_converged=hist[-1] < 1e-3, bj_last_change=hist[-1])
            with torch.no_grad():
                self.resid_S = residual_survival(np.log(yt), dt, self.net(Xt).numpy())
        self.X_train_std = Xt
        return self

    def eta(self, X):
        with torch.no_grad():
            return self.net(self._std(X)).numpy() + np.log(self.scale)

    def predict_survival(self, X, times):
        times = np.asarray(times, float)
        if self.loss == "parametric":
            return aft_survival(torch.tensor(times), torch.tensor(self.eta(X)), self.sigma, self.family).numpy()
        return self.resid_S(times / self.scale, self.eta(X) - np.log(self.scale))

    def predict_risk(self, X):
        return -self.eta(X)

    def prune(self, theta=1e-2):
        info = {"edges_pruned": 0, "nodes_pruned": 0}
        ins = self.net.inputs(self.X_train_std)
        for l, xi in zip(self.net.layers, ins):
            info["edges_pruned"] += l.prune_edges(xi, theta)
        if len(self.net.layers) == 2:
            a, b = self.net.layers
            with torch.no_grad():
                I = a.edge_activations(ins[0]).abs().mean(0).max(0).values
                O = b.edge_activations(ins[1]).abs().mean(0).max(1).values
                dead = torch.minimum(I, O) < theta
                a.mask[:, dead] = 0.0; b.mask[dead, :] = 0.0
                info["nodes_pruned"] = int(dead.sum())
        return info

    def symbolic(self, use_pysr=True, pysr_budget=None, substitute=True):
        """Fit every active edge; substitute accepted formulas (others keep their spline). Returns a list of records."""
        recs = []
        ins = self.net.inputs(self.X_train_std)
        for li, (layer, xin) in enumerate(zip(self.net.layers, ins)):
            for i in range(layer.n_in):
                for o in range(layer.n_out):
                    if layer.mask[i, o] == 0:
                        continue
                    xs = xin[:, i].detach().numpy()

                    def func(z, layer=layer, i=i, o=o):
                        Z = torch.zeros(len(z), layer.n_in)
                        Z[:, i] = torch.tensor(np.asarray(z, float))
                        with torch.no_grad():
                            return layer.edge_activations(Z)[:, i, o].numpy()
                    ys = func(xs)
                    res = fit_symbolic(xs, ys, func, use_pysr=use_pysr, pysr_budget=pysr_budget)
                    rec = dict(layer=li, edge=(i, o), input_mean=float(self.mu[i]) if li == 0 else None,
                               input_sd=float(self.sd[i]) if li == 0 else None, **res.to_dict())
                    recs.append(rec)
                    if substitute and res.accepted:
                        layer.symbolic[(i, o)] = _torch_formula(res)
        return recs


def _torch_formula(res):
    import sympy
    if res.stage == "linear":
        a, b = res.params["a"], res.params["b"]
        return lambda x: a * x + b
    if res.stage == "library":
        from ..symbolic.fit import LIBRARY
        f = {n: g for n, g, _, _ in LIBRARY}[res.params["f"]]
        a, b, c, d = (res.params[k] for k in "abcd")
        return lambda x: torch.tensor(c * f(a * x.numpy() + b) + d)
    expr = sympy.sympify(res.params["sympy"])
    fn = sympy.lambdify(sympy.Symbol("x0"), expr, "numpy")
    return lambda x: torch.tensor(np.broadcast_to(np.asarray(fn(x.numpy()), float), x.shape).copy())
