"""Survival models sharing one interface: fit(train, val) / predict_risk(X).

* CoxPH     - classical linear Cox proportional hazards (lifelines)
* DeepSurv  - MLP log-risk trained with the Cox partial likelihood (Katzman et al. 2018)
* CoxKAN    - KAN log-risk trained with the Cox partial likelihood (Knottenbelt et al. 2024)
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines import CoxPHFitter

from .kan import KAN
from .metrics import harrell_c


# ---------------------------------------------------------------------------
# Cox partial likelihood (Breslow handling of tied event times)
# ---------------------------------------------------------------------------
class CoxLoss:
    """Negative log partial likelihood

        -1/N_events * sum_{i: e_i=1} [ r_i - log sum_{j: t_j >= t_i} exp(r_j) ]

    Pre-computes the descending time order and, for ties, the index of the last
    member of each tie group so the risk set contains *all* subjects with t_j >= t_i.
    """

    def __init__(self, t: np.ndarray, e: np.ndarray):
        t = torch.as_tensor(t, dtype=torch.float32)
        self.order = torch.argsort(t, descending=True)
        t_sorted = t[self.order]
        _, inverse, counts = torch.unique_consecutive(t_sorted, return_inverse=True, return_counts=True)
        self.tie_last = (torch.cumsum(counts, 0) - 1)[inverse]
        self.e = torch.as_tensor(e, dtype=torch.float32)[self.order]

    def __call__(self, risk: torch.Tensor) -> torch.Tensor:
        r = risk.view(-1)[self.order]
        log_risk_set = torch.logcumsumexp(r, 0)[self.tie_last]
        return -((r - log_risk_set) * self.e).sum() / self.e.sum()


def _t(X):
    return torch.as_tensor(np.asarray(X, dtype=np.float32))


class TorchCoxModel:
    """Full-batch Adam training of any torch log-risk network with early stopping on
    validation C-index."""

    name = "torch"

    def __init__(self, lr=1e-3, weight_decay=0.0, epochs=400, patience=60, eval_every=5, seed=42):
        self.lr, self.wd, self.epochs, self.patience, self.eval_every, self.seed = lr, weight_decay, epochs, patience, eval_every, seed
        self.history = []

    def build(self, d):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def penalty(self):
        return 0.0

    def fit(self, train, val):
        torch.manual_seed(self.seed)
        self.net = self.build(train["X"].shape[1])
        Xtr, Xva = _t(train["X"]), _t(val["X"])
        loss_fn, val_loss_fn = CoxLoss(train["t"], train["e"]), CoxLoss(val["t"], val["e"])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, bad = -np.inf, None, 0
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            opt.zero_grad()
            loss = loss_fn(self.net(Xtr))
            (loss + self.penalty()).backward()
            opt.step()
            if epoch % self.eval_every == 0:
                self.net.eval()
                with torch.no_grad():
                    r_va = self.net(Xva)
                    c = harrell_c(val["t"], val["e"], r_va.numpy().ravel())
                    vl = float(val_loss_fn(r_va))
                self.history.append(dict(epoch=epoch, train_loss=float(loss.detach()), val_loss=vl, val_c=c))
                if c > best + 1e-4:
                    best, best_state, bad = c, copy.deepcopy(self.net.state_dict()), 0
                else:
                    bad += self.eval_every
                    if bad >= self.patience:
                        break
        self.net.load_state_dict(best_state)
        self.best_val_c = best
        return self

    @torch.no_grad()
    def predict_risk(self, X) -> np.ndarray:
        self.net.eval()
        return self.net(_t(X)).numpy().ravel()


class DeepSurv(TorchCoxModel):
    name = "DeepSurv"

    def __init__(self, hidden=(64, 32), dropout=0.2, **kw):
        super().__init__(**kw)
        self.hidden, self.dropout = hidden, dropout

    def build(self, d):
        layers, prev = [], d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SELU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1, bias=False))          # Cox model has no intercept
        return nn.Sequential(*layers)


class CoxKAN(TorchCoxModel):
    """log h(t|x) = log h0(t) + KAN(x).  With width [d, 1] the log-hazard is a sum of
    learned univariate shape functions phi_i(x_i): a fully transparent additive model."""

    name = "CoxKAN"

    def __init__(self, width=(None, 1), grid=5, k=3, lamb_l1=1e-4, lamb_coef=1e-4, lamb_entropy=0.0, **kw):
        super().__init__(**kw)
        self.width, self.grid, self.k = list(width), grid, k
        self.lamb_l1, self.lamb_coef, self.lamb_entropy = lamb_l1, lamb_coef, lamb_entropy

    def build(self, d):
        self.width[0] = d
        return KAN(self.width, grid=self.grid, k=self.k)

    def penalty(self):
        return self.net.regularization(self.lamb_l1, self.lamb_coef, self.lamb_entropy)

    @torch.no_grad()
    def shape_functions(self, X, feature: int, n=200, lo=-3.0, hi=3.0):
        """For an additive [d,1] CoxKAN: phi_i on a grid (other features are irrelevant)."""
        layer = self.net.layers[0]
        xs = torch.linspace(lo, hi, n)
        Z = torch.zeros(n, layer.in_dim)
        Z[:, feature] = xs
        return xs.numpy(), layer.edge_activations(Z)[:, 0, feature].numpy()

    @torch.no_grad()
    def feature_importance(self, X) -> np.ndarray:
        """Std over the data of each input's contribution to log-hazard (first layer,
        summed over outgoing edges)."""
        return self.net.layers[0].edge_importance(_t(X)).sum(0).numpy()


class CoxPH:
    name = "CoxPH"

    def __init__(self, penalizer=0.01):
        self.penalizer = penalizer

    def fit(self, train, val=None):
        df = train["X"].copy()
        df["T"], df["E"] = train["t"], train["e"]
        self.cph = CoxPHFitter(penalizer=self.penalizer).fit(df, "T", "E")
        return self

    def predict_risk(self, X) -> np.ndarray:
        return self.cph.predict_log_partial_hazard(pd.DataFrame(X)).to_numpy().ravel()

    def hazard_ratios(self) -> pd.DataFrame:
        s = self.cph.summary
        return pd.DataFrame({
            "coef": s["coef"], "HR": s["exp(coef)"],
            "HR_lower95": s["exp(coef) lower 95%"], "HR_upper95": s["exp(coef) upper 95%"],
            "p": s["p"],
        })
