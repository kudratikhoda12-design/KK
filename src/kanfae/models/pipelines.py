"""Uniform interface for all compared models (docs/experiment_protocol.md section 2).

Every model: fit(train: Observed, val: Observed, seed) -> self; predict_survival(obs, times) -> (N, T);
predict_risk(obs) -> (N,) (higher = higher risk); attributes n_params, fit_seconds, info.
Only `Observed` is accepted (criterion M5); all preprocessing is fitted on `train` only (K2).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch

from ..data.preprocess import Preprocessor
from ..data.schema import Observed
from ..functional.fpca import FPCA
from ..survival.aft import aft_nll, aft_survival
from ..survival.linear_aft import fit_linear_aft, predict_eta, predict_survival
from .fae import FlatAE, FunctionalAE, kan2_match, kan_match, n_params

# Gate repair 2 (2026-10-16): H 16 -> 32 and MLP hidden 32 -> 64; train C ~= test C ~= 0.79-0.80 at the original
# size showed under-fitting (reports/logs/diag_R2_capacity.log). KAN blocks are re-matched automatically (D31).
DEFAULTS = dict(lr=3e-3, wd=1e-4, alpha=1.0, strategy="S2", r=4, H=32, hidden=64, kan_grid=5, kan_hidden=None,
                head="linear", family="weibull", n_basis=12, n_beta=8, lam_beta=1e-3,
                # Gate repair 1 (2026-10-16): the original budget (400 epochs, patience 30) was always exhausted
                # in R2 attempt 1 (reports/logs/diag_R2_budget.log); full-batch Adam needs a larger budget.
                max_epochs=3000, patience=100, naive=False, holdout_frac=0.15)


def _check(obs):
    if not isinstance(obs, Observed):
        raise TypeError(f"fit/predict accept only Observed, got {type(obs).__name__}")


def _ind_weighted_rec(xhat, values, mask):
    m = mask.to(xhat.dtype)
    n_id = m.sum(1)                                              # (B, D)
    per_ch = ((xhat - values) ** 2 * m).sum(1) / n_id.clamp_min(1)
    valid = (n_id > 0).to(xhat.dtype)
    nv = valid.sum(1)
    per_ind = (per_ch * valid).sum(1) / nv.clamp_min(1)
    keep = nv > 0
    return per_ind[keep].mean()


class BaseModel:
    name = "base"

    def __init__(self, **cfg):
        self.cfg = {**DEFAULTS, **cfg}
        self.info = {}

    def _prep_fit(self, train, seed):
        _check(train)
        self.pre = Preprocessor(n_basis=self.cfg["n_basis"], holdout_frac=self.cfg["holdout_frac"], seed=seed).fit(train)
        self.tscale = float(np.median(train.y_obs))

    def prepare(self, obs, holdout=False):
        _check(obs)
        return self.pre.transform(obs, holdout=holdout)


class NeuralAFT(BaseModel):
    """kind: fae_mlp | fae_kan | fae_kan2 | fae_linear | flat_ae."""

    def __init__(self, kind="fae_mlp", **cfg):
        super().__init__(**cfg)
        self.kind = self.name = kind

    def _tensors(self, P):
        flat = self.kind == "flat_ae"
        f = lambda a, dt=torch.float64: torch.tensor(np.asarray(a), dtype=dt)
        return dict(inp=f(P.grid_values if flat else P.c), design=f(P.H if flat else P.Phi), values=f(P.values),
                    mask=f(P.fit_mask, torch.bool), u=f(P.u), y=f(P.y / self.tscale), d=f(P.delta, torch.bool))

    def _build(self, D, p_u, seed):
        c = self.cfg
        if self.kind == "flat_ae":
            return FlatAE(D, self.pre.n_grid, c["r"], c["hidden"], c["head"], p_u, seed)
        block = {"fae_mlp": "mlp", "fae_kan": "kan", "fae_kan2": "kan", "fae_linear": "linear"}[self.kind]
        kh = (None, None)
        if self.kind == "fae_kan":           # parameter matching to the MLP blocks (D31)
            he, hd, self.match_info = kan_match(D, self.pre.basis.K, c["H"], c["r"], c["hidden"], c["kan_grid"])
            kh = (he, hd)
        elif self.kind == "fae_kan2":        # depth-2 ablation (D30), common hidden width matched to the MLP
            w, self.match_info = kan2_match(D, self.pre.basis.K, c["H"], c["r"], c["hidden"], c["kan_grid"])
            kh = (c["kan_hidden"] or w, c["kan_hidden"] or w)
        return FunctionalAE(D, self.pre.basis, c["n_beta"], c["H"], c["r"], block, c["hidden"], c["kan_grid"],
                            kh, c["head"], p_u, seed)

    def _objective(self, T, parts):
        z, out, eta, sigma = self.model(T["inp"], T["u"])
        d = torch.ones_like(T["d"]) if self.cfg["naive"] else T["d"]
        nll = aft_nll(T["y"], d, eta, sigma, self.cfg["family"])
        rec = _ind_weighted_rec(self.model.reconstruct_at(out, T["design"]), T["values"], T["mask"])
        pen = self.cfg["lam_beta"] * self.model.roughness_penalty()
        a = self.cfg["alpha"]
        return {"nll": nll + pen, "rec": rec + pen, "joint": nll + a * rec + pen}[parts], (nll, rec)

    def _run(self, Ttr, Tva, parts, params):
        opt = torch.optim.Adam(params, lr=self.cfg["lr"], weight_decay=self.cfg["wd"])
        best, best_state, bad, hist = np.inf, None, 0, []
        for ep in range(self.cfg["max_epochs"]):
            self.model.train()
            opt.zero_grad()
            loss, _ = self._objective(Ttr, parts)
            loss.backward()
            opt.step()
            self.model.eval()
            with torch.no_grad():
                vl, (vn, vr) = self._objective(Tva, parts)
            vl = float(vl)
            hist.append((ep, float(loss.detach()), vl))
            if not np.isfinite(vl):
                break
            if vl < best - 1e-6:
                best, best_state, bad = vl, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"parts": parts, "epochs": len(hist), "best_val": best, "diverged": not np.isfinite(best)}

    def fit(self, train, val, seed=0):
        t0 = time.time()
        torch.set_default_dtype(torch.float64)
        self._prep_fit(train, seed)
        Ptr, Pva = self.prepare(train), self.prepare(val)
        Ttr, Tva = self._tensors(Ptr), self._tensors(Pva)
        self.model = self._build(Ptr.values.shape[2], Ptr.u.shape[1], seed)
        s = self.cfg["strategy"]
        runs = []
        if s in ("S1", "S3"):
            ae = [p for n, p in self.model.named_parameters() if not n.startswith("head.")]
            runs.append(self._run(Ttr, Tva, "rec", ae))
            if s == "S1":
                for n, p in self.model.named_parameters():
                    p.requires_grad_(n.startswith("head."))
                runs.append(self._run(Ttr, Tva, "nll", list(self.model.head.parameters())))
                for p in self.model.parameters():
                    p.requires_grad_(True)
        if s in ("S2", "S3"):
            parts = "joint" if self.cfg["alpha"] > 0 else "nll"
            runs.append(self._run(Ttr, Tva, parts, list(self.model.parameters())))
        self.info = {"runs": runs, "lambda": self.pre.lam_, "block_params": self.model.block_params()}
        self.n_params = n_params(self.model)
        self.fit_seconds = time.time() - t0
        return self

    def _forward(self, P):
        T = self._tensors(P)
        self.model.eval()
        with torch.no_grad():
            z, out, eta, sigma = self.model(T["inp"], T["u"])
            xhat = self.model.reconstruct_at(out, T["design"])
        return z.numpy(), eta.numpy(), float(sigma), xhat.numpy()

    def predict_survival(self, obs, times, P=None):
        P = P or self.prepare(obs)
        _, eta, sigma, _ = self._forward(P)
        t = np.asarray(times, float) / self.tscale
        return aft_survival(torch.tensor(t), torch.tensor(eta), sigma, self.cfg["family"]).numpy()

    def predict_risk(self, obs, P=None):
        P = P or self.prepare(obs)
        return -self._forward(P)[1]

    def latent(self, obs, P=None):
        P = P or self.prepare(obs)
        return self._forward(P)[0]

    def reconstruct(self, obs, P=None):
        P = P or self.prepare(obs)
        return self._forward(P)[3], P


class _LinearAFTOnFeatures(BaseModel):
    def _features(self, P):
        raise NotImplementedError

    def fit(self, train, val=None, seed=0):
        t0 = time.time()
        self._prep_fit(train, seed)
        P = self.prepare(train)
        X = self._fit_features(P)
        self.fit_ = fit_linear_aft(X, P.y, P.delta, self.cfg["family"], l2=self.cfg.get("l2", 1e-3), compute_se=False)
        self.info = {"converged": self.fit_.converged, "sigma": float(np.exp(self.fit_.log_sigma))}
        self.n_params = X.shape[1] + 2
        self.fit_seconds = time.time() - t0
        return self

    def predict_survival(self, obs, times, P=None):
        return predict_survival(self.fit_, self._features(P or self.prepare(obs)), times)

    def predict_risk(self, obs, P=None):
        return -predict_eta(self.fit_, self._features(P or self.prepare(obs)))


class _FPCAMixin:
    def _fit_features(self, P):
        self.fpca = FPCA(self.cfg["n_components"]).fit(P.c, self.pre.basis.gram())
        S = self.fpca.transform(P.c)
        self.s_mean, self.s_sd = S.mean(0), np.where(S.std(0) > 0, S.std(0), 1.0)
        return self._features(P)

    def _features(self, P):
        S = (self.fpca.transform(P.c) - self.s_mean) / self.s_sd
        return np.hstack([S, P.u])


class FPCAAFT(_FPCAMixin, _LinearAFTOnFeatures):
    name = "fpca_aft"


class SummaryAFT(_LinearAFTOnFeatures):
    name = "sum_aft"

    def _fit_features(self, P):
        return P.summary

    def _features(self, P):
        return P.summary


def _keep_mask(X):
    """Columns with non-zero training variance (fix 2026-10-16: a training-constant summary column made lifelines'
    internal standardisation divide 0/0 -> NaN on C-MAPSS Option A splits 1-4)."""
    return np.asarray(X).std(axis=0) > 1e-12


class _CoxOnFeatures(BaseModel):
    def fit(self, train, val=None, seed=0):
        from lifelines import CoxPHFitter
        t0 = time.time()
        self._prep_fit(train, seed)
        P = self.prepare(train)
        X = self._fit_features(P)
        self.keep = _keep_mask(X)
        X = X[:, self.keep]
        self.info = {"dropped_constant_columns": int((~self.keep).sum())}
        self.cols = [f"f{j}" for j in range(X.shape[1])]
        df = pd.DataFrame(X, columns=self.cols)
        df["T"], df["E"] = P.y, P.delta.astype(int)
        self.cph = CoxPHFitter(penalizer=self.cfg.get("penalizer", 0.01)).fit(df, "T", "E")
        self.n_params = X.shape[1]
        self.fit_seconds = time.time() - t0
        return self

    def predict_survival(self, obs, times, P=None):
        X = self._features(P or self.prepare(obs))[:, self.keep]
        sf = self.cph.predict_survival_function(pd.DataFrame(X, columns=self.cols), times=np.asarray(times, float))
        return sf.to_numpy().T

    def predict_risk(self, obs, P=None):
        X = self._features(P or self.prepare(obs))[:, self.keep]
        return self.cph.predict_log_partial_hazard(pd.DataFrame(X, columns=self.cols)).to_numpy()


class FPCACox(_FPCAMixin, _CoxOnFeatures):
    name = "fpca_cox"


class SummaryCox(_CoxOnFeatures):
    name = "sum_cox"

    def _fit_features(self, P):
        return P.summary

    def _features(self, P):
        return P.summary


class SummaryRSF(BaseModel):
    name = "sum_rsf"

    def fit(self, train, val=None, seed=0):
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        t0 = time.time()
        self._prep_fit(train, seed)
        P = self.prepare(train)
        self.rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=self.cfg.get("min_samples_leaf", 10),
                                        max_features="sqrt", n_jobs=4, random_state=seed)
        self.rsf.fit(P.summary, Surv.from_arrays(P.delta, P.y))
        self.n_params = int(sum(e.tree_.node_count for e in self.rsf.estimators_))   # total tree nodes
        self.fit_seconds = time.time() - t0
        return self

    def predict_survival(self, obs, times, P=None):
        P = P or self.prepare(obs)
        S = self.rsf.predict_survival_function(P.summary, return_array=True)
        ut = self.rsf.unique_times_
        idx = np.searchsorted(ut, np.asarray(times, float), side="right") - 1
        return np.where(idx[None, :] >= 0, S[:, np.clip(idx, 0, None)], 1.0)

    def predict_risk(self, obs, P=None):
        return self.rsf.predict((P or self.prepare(obs)).summary)


def make_model(name, **cfg):
    if name in ("fae_mlp", "fae_kan", "fae_kan2", "fae_linear", "flat_ae"):
        return NeuralAFT(name, **cfg)
    return {"fpca_aft": FPCAAFT, "fpca_cox": FPCACox, "sum_aft": SummaryAFT, "sum_cox": SummaryCox,
            "sum_rsf": SummaryRSF}[name](**cfg)
