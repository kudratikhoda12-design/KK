"""Round 12 (L35): the pre-registered simulation study (D113, as amended by D120, D123 and D126).

fsim scenarios (DGP v2, n 1,000, split_groups 60/20/20 stratified on the event indicator, split seed = data seed):
  S-Add     seeds 3000-3019    S-Int  3100-3119    S-Sparse (S-Add, 3-8 visits) 3200-3219
  S-Misspec 3300-3319: each seed s generates both sub-scenarios from independent streams,
            lognormal: SeedSequence([s, 0]), loglogistic: SeedSequence([s, 1])
(S-Int/P0 continuity runs in scripts/run_round12_p0.py.)

Front ends. "bspline": per-subject penalised cubic B-spline smoothing (12 functions; lambda by held-out-point CV on
the training subjects) evaluated on the 50-point grid. "pace" (S-Sparse only): PACE per channel fitted on the training
subjects, conditional-expectation reconstructions and scores for everyone.
Models (one record per model; every KAN model is the D112 ensemble: lr {1e-3, 3e-3} x wd {1e-4, 1e-3} tuned on
validation loss at seed 0, then seeds 0-2; the member with seed 0 is also reported as the single-seed model):
  b2s_cox, b2s_aft, b2h_cox, b2h_aft (functional KAN), b1_fpca1_cox/aft (FPCA scores + z -> additive KAN),
  b1_sum_cox/aft (summaries + z -> additive KAN), static_coxkan (last values + z), afcm_v3 (AFCM_D123, primary
  comparator), afcm_v0 (Round-12a AFCM; S-Add sensitivity), fpca_cox, sum_cox (lifelines), sum_rsf (RSF, 200 trees,
  min leaf 10, seed 0); S-Int adds b1_fpca2_cox (KAN [p -> 4 -> 1]) and b2deep_cox; S-Misspec runs b2s_cox, b2s_aft,
  b2s_aft_rc (the b2s_aft ensemble with residual calibration of every member) and afcm_v3; S-Sparse runs the
  functional and FPCA models on both front ends; flat_ae (S-Sparse, C4F flat autoencoder, protocol 2.1 tuning,
  3-seed ensemble) and b2s_cox_quantile (S-Add; B2 input sensitivity, x~ = F_t(X(t))) are phase-2 cells.
Pruning and symbolic (S-Add, and S-Sparse with the B-spline front end): b2s_cox at the ensemble's tuned (lr, wd), seed 0,
through select_lambda_theta (D120 defaults); selection recall/FDR; stages 1-2 on every active edge and PySR (60 s) on
edges with >= 5% of their layer's centred score, at most 10; accepted linear/library formulas substituted and refitted
(L-BFGS, 200 iterations); shape and survival metrics before pruning (the seed-0 member), after pruning, after
substitution before the refit (information) and after the refit. PySR-accepted edges keep their spline in the substituted model (no PySR substitution path).
Metrics on the test split: survival-curve error against the oracle on [0, q90(training times)] (101 points),
marginal-IPCW IBS (ibs_time_grid), Harrell C, Uno C (tau = q90 of training times), D-calibration p (10 bins), and,
for functional and FPCA models, identifiable RISE (primary; B2-H channel 1: 1-D RISE) and raw RISE of phi_1 and phi_2
at the true test curves. B1/fpca shapes are beta-type: phi_d(x, t) = x * sum_k b_dk psi_dk(t) with b_dk the
least-squares slope of the fitted score effect. S-Misspec shapes are compared on the eta scale of the DGP
(log T = 1 - eta/1.5 + 0.5 eps): AFT estimates as -1.5 x the AFT edge (= 1.5 sigma x effect()), Cox estimates as is.
Records: $R12_OUT (jsonl, one line per model x replicate; failures recorded with tracebacks). Resumable.
"""
import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kanfae.data import fsim  # noqa: E402
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.evaluation import metrics as M  # noqa: E402
from kanfae.functional.basis import BSplineBasis  # noqa: E402
from kanfae.functional.pace import PACE  # noqa: E402
from kanfae.functional.smoothing import curves_on_grid, pad_ragged, select_lambda  # noqa: E402
from kanfae.models.afcm import AFCM_D123, LAM_GRID, LAM_T_GRID, fit_afcm  # noqa: E402
from kanfae.models.fbaselines import GridFPCA, LinearCox, last_values, summaries  # noqa: E402
from kanfae.models.fkan import FKANEnsemble, FKANSurv, select_lambda_theta, substitute_symbolic, symbolic_edges  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
G = fsim.GRID
N = 1000
SEED_BASE = {"S-Add": 3000, "S-Int": 3100, "S-Sparse": 3200, "S-Misspec": 3300}
OUT = Path(os.environ.get("R12_OUT", ROOT / "reports/round12/dev/results_dev.jsonl"))


def sub_seed(seed, k):
    return int(np.random.SeedSequence([seed, k]).generate_state(1)[0])


# ------------------------------------------------------------------------------------------------------------ data
class Ctx:
    """One replicate: data, split, front ends and features (built lazily)."""

    def __init__(self, scen, rep, sub=None, dev_seed=None):
        self.scen, self.rep, self.sub = scen, rep, sub
        self.seed = dev_seed if dev_seed is not None else SEED_BASE[scen] + rep
        if scen == "S-Misspec":
            self.data_seed = sub_seed(self.seed, 0 if sub == "lognormal" else 1)
            self.dat = fsim.generate(f"S-Misspec-{sub}", N, self.data_seed)
        else:
            self.data_seed = self.seed
            self.dat = fsim.generate("S-Int" if scen == "S-Int" else "S-Add", N, self.seed, sparse=scen == "S-Sparse")
        self.tr, self.va, self.te = split_groups(np.arange(N), self.seed, strata=self.dat.delta)
        d = self.dat
        self.y_tr, self.d_tr, self.y_te, self.d_te = d.y[self.tr], d.delta[self.tr], d.y[self.te], d.delta[self.te]
        self.t_max = float(np.quantile(self.y_tr, 0.9))
        self.curve_grid = np.linspace(0.0, self.t_max, 101)
        self.S_true = d.subset(self.te).survival(self.curve_grid)
        self.X_te = d.x_true_grid[self.te]
        self._fe, self._feat = {}, {}

    def front(self, name):
        if name not in self._fe:
            d = self.dat
            if name == "bspline":
                basis = BSplineBasis(12, 3, (0.0, 1.0))
                T, V, Mk, n_obs = pad_ragged([d.times[i] for i in self.tr], [d.values[i] for i in self.tr])
                lam, _ = select_lambda(T, V, Mk, n_obs, basis, seed=0)
                self._fe[name] = dict(xg=curves_on_grid(d.times, d.values, G, basis, lam), lam=lam)
            elif name == "pace":
                xg, scores, phis, info = np.zeros((N, len(G), 3)), [], [], []
                for c in range(3):
                    p = PACE(seed=0).fit([d.times[i] for i in self.tr], [d.values[i][:, c] for i in self.tr])
                    vals = [v[:, c] for v in d.values]
                    sc = p.scores(d.times, vals)
                    xg[:, :, c] = p.mu[None, :] + sc @ p.phi.T
                    scores.append(sc); phis.append(p.phi); info.append(dict(K=p.K, sigma2=p.sigma2, h_mu=p.h_mu, h_cov=p.h_cov))
                self._fe[name] = dict(xg=xg, scores=np.hstack(scores), phis=phis, info=info)
        return self._fe[name]

    def fpca(self, front):
        """(features (n, sum K), eigenfunctions per channel, K per channel)."""
        key = ("fpca", front)
        if key not in self._feat:
            fe = self.front(front)
            if front == "pace":
                self._feat[key] = (fe["scores"], fe["phis"], [p.shape[1] for p in fe["phis"]])
            else:
                g = GridFPCA().fit(fe["xg"][self.tr])
                self._feat[key] = (g.transform(fe["xg"]), g.phi, g.K)
        return self._feat[key]

    def summ(self):
        if "summ" not in self._feat:
            self._feat["summ"] = summaries(self.dat.times, self.dat.values)
        return self._feat["summ"]

    def lastv(self):
        if "last" not in self._feat:
            self._feat["last"] = last_values(self.dat.times, self.dat.values)
        return self._feat["last"]

    def pack(self, idx, xg=None, F=None):
        z = self.dat.z[idx] if F is None else np.hstack([F[idx], self.dat.z[idx]])
        return dict(xg=None if xg is None else xg[idx], z=z, y=self.dat.y[idx], d=self.dat.delta[idx].astype(float))


# ------------------------------------------------------------------------------------------------------- metrics
def survival_metrics(ctx, S_fn, risk):
    S = S_fn(ctx.curve_grid)
    tg = M.ibs_time_grid(ctx.y_te, ctx.y_tr)
    Sy = np.diag(S_fn(ctx.y_te))
    return dict(curve_error=M.survival_curve_error(S, ctx.S_true, ctx.curve_grid),
                ibs=float(M.ibs_marginal(S_fn(tg), tg, ctx.y_te, ctx.d_te, ctx.y_tr, ctx.d_tr)[0]),
                harrell_c=float(M.harrell_c(ctx.y_te, ctx.d_te, risk)),
                uno_c=float(M.uno_c(ctx.y_tr, ctx.d_tr, ctx.y_te, ctx.d_te, risk, ctx.t_max)),
                dcal_p=float(M.d_calibration(Sy, ctx.d_te)[1]))


def shape_metrics(ctx, effect, homog_ch1=False):
    out, prim, raw = {}, [], []
    for d in (0, 1):
        fh = lambda x, t, d=d: effect(d, x, t)  # noqa: E731
        ft = lambda x, t, d=d: fsim.true_effect(d, x, t)  # noqa: E731
        X = ctx.X_te[:, :, d]
        rid, rr = fsim.rise_identifiable(fh, ft, X), fsim.rise(fh, ft, X)
        p = fsim.rise_1d(fh, ft, X) if (homog_ch1 and d == 0) else rid
        out[f"phi{d + 1}"] = dict(rise_primary=p, rise_id=rid, rise_raw=rr)
        prim.append(p); raw.append(rr)
    out["mean_primary"], out["mean_raw"] = float(np.mean(prim)), float(np.mean(raw))
    out["mean_id"] = float(np.mean([out["phi1"]["rise_id"], out["phi2"]["rise_id"]]))
    return out


def scale_misspec(ctx, model, d, x, t):
    """S-Misspec: eta-scale shape. AFT members: 1.5 * sigma * effect(); Cox as is."""
    if ctx.scen != "S-Misspec":
        return model.effect(d, x, t)
    if model.cfg["loss"] == "aft":
        return 1.5 * model.sigma * model.effect(d, x, t)
    return model.effect(d, x, t)


def ens_effect(ctx, ens):
    return lambda d, x, t: np.mean([scale_misspec(ctx, m, d, x, t) for m in ens.members], 0)


def beta_effect(phis, slopes):
    """phi_d(x, t) = x * sum_k b_dk psi_dk(t) at grid times (nearest grid point)."""
    def eff(d, x, t):
        x = np.asarray(x, float)
        g = np.clip(np.rint(np.broadcast_to(t, x.shape) * (len(G) - 1)).astype(int), 0, len(G) - 1)
        return x * (phis[d] @ slopes[d])[g]
    return eff


def split_slopes(beta, K):
    out, k0 = [], 0
    for k in K:
        out.append(np.asarray(beta[k0:k0 + k], float)); k0 += k
    return out


def kan_feature_slopes(member, F_tr, n_feat, ctx):
    """Least-squares slope of each additive edge psi_k against the original-unit feature (AFT: -psi/sigma; S-Misspec
    AFT: -1.5 psi)."""
    with torch.no_grad():
        a = member.net.stat.edge_activations(member.data["Ztr"]).numpy()[:, :n_feat, 0]
    sl = np.array([np.polyfit(F_tr[:, k], a[:, k], 1)[0] for k in range(n_feat)])
    if member.cfg["loss"] == "aft":
        sl = -1.5 * sl if ctx.scen == "S-Misspec" else -sl / member.sigma
    return sl


# -------------------------------------------------------------------------------------------------------- models
def run_ensemble(ctx, rec, kind, loss, xg=None, F=None, hidden=0, shapes="functional", fpca=None):
    tr, va, te = ctx.pack(ctx.tr, xg, F), ctx.pack(ctx.va, xg, F), ctx.pack(ctx.te, xg, F)
    e = FKANEnsemble(kind=kind, loss=loss, hidden=hidden).fit(tr, va)
    rec.update(lr=e.lr, wd=e.wd, tuning=e.tuning, epochs=[len(m.history) for m in e.members])
    out = {"ensemble": e}

    def evals(models, tag):
        S_fn = lambda tt: np.mean([m.predict_survival(te["xg"], te["z"], tt) for m in models], 0)  # noqa: E731
        risk = np.mean([m.predict_risk(te["xg"], te["z"]) for m in models], 0)
        res = survival_metrics(ctx, S_fn, risk)
        if shapes == "functional" and ctx.scen != "S-Int":
            eff = lambda d, x, t: np.mean([scale_misspec(ctx, m, d, x, t) for m in models], 0)  # noqa: E731
            res["shape"] = shape_metrics(ctx, eff, homog_ch1=(kind == "b2h"))
        elif shapes == "fpca" and ctx.scen != "S-Int":
            Ftr, phis, K = fpca
            sl = np.mean([kan_feature_slopes(m, Ftr, sum(K), ctx) for m in models], 0)
            res["shape"] = shape_metrics(ctx, beta_effect(phis, split_slopes(sl, K)))
        rec[tag] = res

    evals(e.members, "metrics")
    evals(e.members[:1], "metrics_seed0")
    return out


def run_afcm(ctx, rec, xg, spec):
    m, recs = fit_afcm(ctx.pack(ctx.tr, xg), ctx.pack(ctx.va, xg), **spec)
    te = ctx.pack(ctx.te, xg)
    res = survival_metrics(ctx, lambda tt: m.predict_survival(te["xg"], te["z"], tt), m.predict_risk(te["xg"], te["z"]))
    if ctx.scen != "S-Int":
        res["shape"] = shape_metrics(ctx, m.effect)
    rec.update(metrics=res, lam_x=np.asarray(m.lam_x).tolist(), lam_t=np.asarray(m.lam_t).tolist(),
               val_nll=min(r["val_nll"] for r in recs))


def run_linear_cox(ctx, rec, F, fpca=None):
    X = np.hstack([F, ctx.dat.z])
    m = LinearCox().fit(X[ctx.tr], ctx.y_tr, ctx.d_tr)
    Xte = X[ctx.te]
    res = survival_metrics(ctx, lambda tt: m.predict_survival(Xte, tt), m.predict_risk(Xte))
    if fpca is not None and ctx.scen != "S-Int":
        _, phis, K = fpca
        res["shape"] = shape_metrics(ctx, beta_effect(phis, split_slopes(m.cph.params_.values[:sum(K)], K)))
    rec.update(metrics=res, coef=m.cph.params_.values.tolist())


def run_rsf(ctx, rec, F):
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    X = np.hstack([F, ctx.dat.z])
    rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=10, max_features="sqrt", n_jobs=1, random_state=0)
    rsf.fit(X[ctx.tr], Surv.from_arrays(ctx.d_tr.astype(bool), ctx.y_tr))
    Sf = rsf.predict_survival_function(X[ctx.te], return_array=True)
    ut = rsf.unique_times_

    def S_fn(tt):
        idx = np.searchsorted(ut, np.asarray(tt, float), side="right") - 1
        return np.where(idx[None, :] >= 0, Sf[:, np.clip(idx, 0, None)], 1.0)

    rec.update(metrics=survival_metrics(ctx, S_fn, rsf.predict(X[ctx.te])))


def run_flat_ae(ctx, rec):
    from kanfae.data.schema import Observed
    from kanfae.models.pipelines import NeuralAFT
    from run_neural_synthetic import FRONT_KEYS, NN_GRID
    d = ctx.dat
    T, V, Mk, n_obs = pad_ragged(d.times, d.values)
    obs = Observed(np.arange(N), T, V, Mk, n_obs, d.y, d.delta.astype(bool), d.z)
    sel = json.loads((ROOT / "reports/learnability_selected_fa4f.json").read_text())
    cfg = {k: v for k, v in sel["config"].items() if k not in ("lr", "wd") + FRONT_KEYS}
    o_tr, o_va, o_te = obs.subset(ctx.tr), obs.subset(ctx.va), obs.subset(ctx.te)
    tun = []
    for g in NN_GRID:
        m = NeuralAFT("flat_ae", **{**cfg, **g}).fit(o_tr, o_va, seed=0)
        tv = M.ibs_time_grid(o_va.y_obs, o_tr.y_obs)
        tun.append((float(M.ibs_marginal(m.predict_survival(o_va, tv), tv, o_va.y_obs, o_va.delta, o_tr.y_obs,
                                         o_tr.delta)[0]), g, m))
    best = min((t for t in tun if np.isfinite(t[0])), key=lambda t: t[0])
    members = [best[2]] + [NeuralAFT("flat_ae", **{**cfg, **best[1]}).fit(o_tr, o_va, seed=s) for s in (1, 2)]
    S_fn = lambda tt: np.mean([m.predict_survival(o_te, tt) for m in members], 0)  # noqa: E731
    risk = np.mean([m.predict_risk(o_te) for m in members], 0)
    rec.update(metrics=survival_metrics(ctx, S_fn, risk), tuning=[dict(val_ibs=t[0], **t[1]) for t in tun], best=best[1])


def run_quantile_b2s(ctx, rec):
    """B2 input sensitivity (D113): x~ = F_t(X(t)) with the training empirical CDF per grid time; shapes mapped back."""
    xg = ctx.front("bspline")["xg"]
    qs = np.sort(xg[ctx.tr], axis=0)
    qp = (np.arange(1, len(ctx.tr) + 1) - 0.5) / len(ctx.tr)
    xq = np.stack([np.stack([np.interp(xg[:, g, c], qs[:, g, c], qp) for g in range(len(G))], 1) for c in range(3)], 2)
    out = run_ensemble(ctx, rec, "b2s", "cox", xg=xq, shapes=None)
    e = out["ensemble"]

    def eff(d, x, t):
        x = np.asarray(x, float)
        g = np.clip(np.rint(np.broadcast_to(t, x.shape) * (len(G) - 1)).astype(int), 0, len(G) - 1)
        xt = np.empty(x.shape)
        for gi in np.unique(g):
            mm = g == gi
            xt[mm] = np.interp(x[mm], qs[:, gi, d], qp)
        return np.mean([m.effect(d, xt, t) for m in e.members], 0)

    rec["metrics"]["shape"] = shape_metrics(ctx, eff)


def run_prune_symbolic(ctx, rec, xg, lr, wd, use_pysr=True):
    tr, va, te = ctx.pack(ctx.tr, xg), ctx.pack(ctx.va, xg), ctx.pack(ctx.te, xg)
    rec.update(lr=lr, wd=wd)
    sel, grid = select_lambda_theta(lambda l: FKANSurv(kind="b2s", loss="cox", lam=l, lr=lr, wd=wd, seed=0), tr, va)
    s = sel.selected_channels()
    chosen = [g for g in grid if g.get("chosen")][0]
    rec.update(selected=s, recall=(s[0] + s[1]) / 2, fdr=s[2] / max(sum(s), 1), chosen_lam=chosen["lam"],
               chosen_theta=chosen["theta"], grid=grid)

    def ev(m):
        r = survival_metrics(ctx, lambda tt: m.predict_survival(te["xg"], te["z"], tt), m.predict_risk(te["xg"], te["z"]))
        r["shape"] = shape_metrics(ctx, m.effect)
        return r

    rec["after_pruning"] = ev(sel)
    t0 = time.time()
    recs = symbolic_edges(sel, use_pysr=use_pysr, pysr_timeout=60, max_pysr=10, min_share=0.05)
    acc = [r for r in recs if r["result"].accepted]
    rec["symbolic"] = dict(n_edges=len(recs), n_accepted=len(acc), acceptance_rate=len(acc) / max(len(recs), 1),
                           mean_r2_val_accepted=float(np.mean([r["result"].r2_val for r in acc])) if acc else None,
                           stages={st: sum(r["result"].stage == st for r in recs) for st in ("linear", "library", "pysr", "none")},
                           edges=[dict(layer=r["layer"], i=r["i"], o=r["o"], share=r["share"], stage=r["result"].stage,
                                       accepted=r["result"].accepted, formula=r["result"].formula,
                                       r2_val=r["result"].r2_val) for r in recs],
                           seconds=round(time.time() - t0, 1))
    sub = copy.deepcopy(sel)
    n_sub = substitute_symbolic(sub, recs, stages=("linear", "library"))
    if n_sub:
        sub._finalise(sub.train_)
        rec["after_substitution_before_refit"] = ev(sub)
        sub.refit_symbolic(max_iter=200)
    rec["symbolic"]["n_substituted"] = n_sub
    rec["after_substitution"] = ev(sub)


# --------------------------------------------------------------------------------------------------------- plans
def plan(scen, phase):
    """Ordered (model, front) cells per replicate. H-R12 cells first."""
    if scen == "S-Misspec":
        return [("b2s_cox", "bspline"), ("b2s_aft", "bspline"), ("afcm_v3", "bspline")]
    if phase == "phase2":
        return {"S-Sparse": [("flat_ae", "raw")], "S-Add": [("b2s_cox_quantile", "bspline")]}.get(scen, [])
    fronts = ("bspline", "pace") if scen == "S-Sparse" else ("bspline",)
    cells = [("b2s_cox", "bspline"), ("afcm_v3", "bspline")]
    if scen == "S-Add":
        cells.append(("afcm_v0", "bspline"))
    for f in fronts:
        cells += [(m, f) for m in ("b2s_cox", "afcm_v3") if (m, f) not in cells]
        cells += [(m, f) for m in ("b2s_aft", "b2h_cox", "b2h_aft", "b1_fpca1_cox", "b1_fpca1_aft", "fpca_cox")]
    cells += [(m, "raw") for m in ("b1_sum_cox", "b1_sum_aft", "static_coxkan", "sum_cox", "sum_rsf")]
    if scen == "S-Int":
        cells += [("b1_fpca2_cox", "bspline"), ("b2deep_cox", "bspline")]
    if scen in ("S-Add", "S-Sparse"):
        cells.append(("prune_symbolic", "bspline"))
    return cells


def run_cell(ctx, model, front, ens_cache):
    rec = dict(round="R12", scenario=ctx.scen, sub=ctx.sub, replicate=ctx.rep, seed=ctx.seed, data_seed=ctx.data_seed,
               model=model, front=front, n_train=len(ctx.tr), n_test=len(ctx.te),
               censored_frac=float(1 - ctx.dat.delta.mean()))
    t0 = time.time()
    try:
        xg = ctx.front(front)["xg"] if front in ("bspline", "pace") else None
        if front in ("bspline", "pace"):
            rec["front_info"] = {k: v for k, v in ctx.front(front).items() if k in ("lam", "info")}
        if model in ("b2s_cox", "b2s_aft", "b2h_cox", "b2h_aft", "b2deep_cox"):
            kind, loss = model.split("_")
            out = run_ensemble(ctx, rec, kind, loss, xg=xg, hidden=4 if kind == "b2deep" else 0,
                               shapes=None if kind == "b2deep" else "functional")
            ens_cache[(model, front)] = (out["ensemble"].lr, out["ensemble"].wd)
            if model == "b2s_aft" and ctx.scen == "S-Misspec":                       # residual-calibrated twin
                rc = {k: v for k, v in rec.items() if k != "metrics_seed0"}
                rc["model"] = "b2s_aft_rc"
                e = out["ensemble"]
                for m in e.members:
                    m.cfg["residual_calibration"] = True
                    m._finalise(m.train_)
                te = ctx.pack(ctx.te, xg)
                rc["metrics"] = survival_metrics(ctx, lambda tt: np.mean([m.predict_survival(te["xg"], te["z"], tt)
                                                                        for m in e.members], 0),
                                                 np.mean([m.predict_risk(te["xg"], te["z"]) for m in e.members], 0))
                rc["metrics"]["shape"] = rec["metrics"].get("shape")
                rc.update(status="ok", seconds=0.0, note="same fitted ensemble as b2s_aft, residual calibration")
                _append(rc)
        elif model in ("b1_fpca1_cox", "b1_fpca1_aft", "b1_fpca2_cox"):
            fp = ctx.fpca(front)
            run_ensemble(ctx, rec, "static", model.split("_")[-1], F=fp[0], hidden=4 if "fpca2" in model else 0,
                         shapes="fpca" if "fpca1" in model else None, fpca=(fp[0][ctx.tr], fp[1], fp[2]))
            rec["K"] = list(map(int, fp[2]))
        elif model in ("b1_sum_cox", "b1_sum_aft"):
            run_ensemble(ctx, rec, "static", model.split("_")[-1], F=ctx.summ(), shapes=None)
        elif model == "static_coxkan":
            run_ensemble(ctx, rec, "static", "cox", F=ctx.lastv(), shapes=None)
        elif model == "afcm_v3":
            run_afcm(ctx, rec, xg, AFCM_D123)
        elif model == "afcm_v0":
            run_afcm(ctx, rec, xg, dict(lams_x=LAM_GRID, lams_t=LAM_T_GRID))
        elif model == "fpca_cox":
            fp = ctx.fpca(front)
            run_linear_cox(ctx, rec, fp[0], fpca=fp)
        elif model == "sum_cox":
            run_linear_cox(ctx, rec, ctx.summ())
        elif model == "sum_rsf":
            run_rsf(ctx, rec, ctx.summ())
        elif model == "flat_ae":
            run_flat_ae(ctx, rec)
        elif model == "b2s_cox_quantile":
            run_quantile_b2s(ctx, rec)
        elif model == "prune_symbolic":
            lw = ens_cache.get(("b2s_cox", front)) or tuned_lr_wd(ctx, front)
            if lw is None:
                raise RuntimeError("no b2s_cox record for this replicate (tuned lr/wd needed)")
            run_prune_symbolic(ctx, rec, xg, *lw)
        else:
            raise ValueError(model)
        rec["status"] = "ok"
    except Exception as exc:                                         # failed runs are recorded, never dropped
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])
    rec["seconds"] = round(time.time() - t0, 1)
    _append(rec)
    return rec


def _append(rec):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def tuned_lr_wd(ctx, front):
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            r = json.loads(l)
            if (r.get("status") == "ok" and r["model"] == "b2s_cox" and r["front"] == front and r["scenario"] == ctx.scen
                    and r["replicate"] == ctx.rep and r["seed"] == ctx.seed):
                return r["lr"], r["wd"]
    return None


def done():
    if not OUT.exists():
        return set()
    out = set()
    for l in OUT.read_text().splitlines():
        r = json.loads(l)
        if r.get("status") == "ok":
            out.add((r["scenario"], r.get("sub"), r["replicate"], r["model"], r["front"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--sub", default=None)
    ap.add_argument("--reps", default="0-19")
    ap.add_argument("--phase", default="core")
    ap.add_argument("--dev-seed", type=int, default=None, help="development dry-run: use this data seed (rep must be 0)")
    ap.add_argument("--deadline-utc", default=None, help="ISO time; phase-2 cells are not started after it (D113 drop)")
    a = ap.parse_args()
    lo, hi = map(int, a.reps.split("-"))
    if a.dev_seed is None:
        assert "dev" not in OUT.name, "registered seeds need a non-development R12_OUT"
    else:
        assert "dev" in OUT.name and lo == hi == 0, "development seeds write only to a development file"
    have = done()
    for rep in range(lo, hi + 1):
        ctx = Ctx(a.scenario, rep, a.sub, a.dev_seed)
        ens_cache = {}
        for model, front in plan(a.scenario, a.phase):
            if (a.scenario, a.sub, rep, model, front) in have:
                continue
            if a.deadline_utc and a.phase == "phase2" and time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) > a.deadline_utc:
                _append(dict(round="R12", scenario=a.scenario, sub=a.sub, replicate=rep, seed=ctx.seed, model=model,
                             front=front, status="dropped_budget", note="D113 drop order: 12 h budget reached"))
                continue
            r = run_cell(ctx, model, front, ens_cache)
            print(f"[{a.scenario}{'/' + a.sub if a.sub else ''} rep {rep} {model}/{front}] {r['status']} {r['seconds']}s",
                  flush=True)


if __name__ == "__main__":
    main()
