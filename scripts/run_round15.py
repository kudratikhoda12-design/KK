"""Round 15 (D133): Phase-2 methods -- mechanics dry run on development seeds only (no formal seed is used here).

Jobs (python scripts/run_round15.py --job <name>; records appended to $R15_OUT, one JSON line per job):
  distill_<scen>_<seed>   symbolic_distill on fsim S-Add (dense) or S-Sparse (3-8 visits), DGP v2, n 1,000, the
                          Round-12 Ctx (split_groups 60/20/20 on delta, B-spline front end). b2s_cox ensemble (D112
                          protocol) for the tuned (lr, wd); select_lambda_theta (D120) on the seed-0 model; symbolic
                          stages 1-2 on every active edge (no PySR: PySR-accepted edges are never substituted, so the
                          substituted model is the same); then, from the same substituted formulas:
                            sub_noguard      substituted, no refit (the D128 primary)
                            refit_old        unconstrained L-BFGS refit (Round-12 procedure, 200 iterations)
                            sub_guard        domain-guarded substitution, no refit
                            refit_guard      guarded, unconstrained refit (mu = 0)
                            distill_mu<mu>   guarded + functional-distillation refit at mu in {1e-2, 1e-1, 1}
                            distill          the mu chosen by unpenalised validation NLPL (the method)
                          each with survival metrics against the oracle and identifiable RISE (phi_1, phi_2).
  res_<scen>_<seed>       b2s_res_cox against b2s_cox, static_coxkan (last values + z) and landmark_cox (linear Cox on
                          last values + z) on fsim S-Last or S-Sparse (sparse, n 1,000); last observed raw values.
  res_dev_real_<seed>     the same models on the Round-13 real-data harness with synthetic aids-shaped development
                          records (run_round13._dev_longdata(seed); RealCtx split 0) -- mechanics only.
  afcmlaml_<seed>         D134 check C2: V3 basis/penalties with LAML-selected lambdas (AIC-type rule for information)
  (R15_GATE=cv)           D134 check G: the anchored/residual model with the cross-validated gate
  afcmpars_<seed>         follow-up: V3 with the D120 parsimonious lambda rule (tol 0.01), same data as afcm_<seed>
  (R15_ANCHORED=1)        follow-up: res jobs fit the "anchored" variant (full curve + last value) instead
  afcm_<seed>             the CK-6c procedure (S-Add, DGP v2, n 2,000) with AFCM V3 (single fit), afcm_ens (3-resample
                          bootstrap-averaged V3, resample seed 0) and V3 with per-channel lambda_t.
Development seeds (verified unused before this round): 961-966, plus 914 (the D123 AFCM development seed) and 22
(the CK-6c seed: a known failure case, used only to see whether the variants change it; it is not a formal seed here).
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
sys.path.insert(0, str(ROOT / "tests"))
import run_round12 as R12  # noqa: E402
from kanfae.data import fsim  # noqa: E402
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.models.afcm import AFCM_D123, fit_afcm, fit_afcm_bagged, fit_afcm_laml  # noqa: E402
from kanfae.models.fbaselines import LinearCox, impute_train_mean, last_values  # noqa: E402
from kanfae.models.fkan import (FKANEnsemble, refit_distill, select_lambda_theta, substitute_symbolic,  # noqa: E402
                                symbolic_edges)
from kanfae.models.fres import ResFKAN  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
OUT = Path(os.environ.get("R15_OUT", ROOT / "reports/round15/dev/results_dev.jsonl"))
MUS = (1e-2, 1e-1, 1.0)
ANCHORED = os.environ.get("R15_ANCHORED") == "1"      # dry-run follow-up: the full-curve "anchored" variant
GATE = os.environ.get("R15_GATE", "split")              # D134 development check G: "cv" = cross-validated gate


class Ctx15(R12.Ctx):
    """Round-12 replicate context, extended with S-Last (sparse, 3-8 visits)."""

    def __init__(self, scen, seed):
        if scen != "S-Last":
            super().__init__(scen, 0, dev_seed=seed)
            return
        self.scen, self.rep, self.sub, self.seed, self.data_seed = scen, 0, None, seed, seed
        self.dat = fsim.generate("S-Last", R12.N, seed, sparse=True)
        self.tr, self.va, self.te = split_groups(np.arange(R12.N), seed, strata=self.dat.delta)
        d = self.dat
        self.y_tr, self.d_tr, self.y_te, self.d_te = d.y[self.tr], d.delta[self.tr], d.y[self.te], d.delta[self.te]
        self.t_max = float(np.quantile(self.y_tr, 0.9))
        self.curve_grid = np.linspace(0.0, self.t_max, 101)
        self.S_true = d.subset(self.te).survival(self.curve_grid)
        self.X_te = d.x_true_grid[self.te]
        self._fe, self._feat = {}, {}

    def last(self):
        return impute_train_mean(last_values(self.dat.times, self.dat.values), self.tr)


def ev(ctx, m, te, shape=True):
    e = m.eta(te["xg"], te["z"])
    if not np.all(np.isfinite(e)):
        return dict(status="nonfinite", n_nonfinite_eta=int(np.sum(~np.isfinite(e))))
    r = R12.survival_metrics(ctx, lambda tt: m.predict_survival(te["xg"], te["z"], tt), m.predict_risk(te["xg"], te["z"]))
    if shape and ctx.scen in ("S-Add", "S-Sparse"):
        r["shape"] = R12.shape_metrics(ctx, m.effect)
    r["val_loss"] = float(m.val_loss)
    return r


def job_distill(scen, seed):
    ctx = Ctx15(scen, seed)
    xg = ctx.front("bspline")["xg"]
    tr, va, te = ctx.pack(ctx.tr, xg), ctx.pack(ctx.va, xg), ctx.pack(ctx.te, xg)
    rec = dict(job="distill", scenario=scen, seed=seed)
    ens = FKANEnsemble(kind="b2s", loss="cox").fit(tr, va)
    rec.update(lr=ens.lr, wd=ens.wd)
    t0 = time.time()
    sel, grid = select_lambda_theta(lambda l: R12.FKANSurv(kind="b2s", loss="cox", lam=l, lr=ens.lr, wd=ens.wd, seed=0), tr, va)
    s = sel.selected_channels()
    rec.update(selected=s, recall=(s[0] + s[1]) / 2, fdr=s[2] / max(sum(s), 1), grid=grid, select_seconds=round(time.time() - t0))
    rec["after_pruning"] = ev(ctx, sel, te)
    recs = symbolic_edges(sel, use_pysr=False)
    acc = [r for r in recs if r["result"].accepted]
    rec["symbolic"] = dict(n_edges=len(recs), n_accepted=len(acc), formulas=[
        dict(layer=r["layer"], i=r["i"], o=r["o"], stage=r["result"].stage, formula=r["result"].formula,
             r2_val=r["result"].r2_val) for r in acc])
    out = {}
    sub0 = copy.deepcopy(sel)
    rec["symbolic"]["n_substituted"] = substitute_symbolic(sub0, recs)
    sub0._finalise(sub0.train_)
    out["sub_noguard"] = ev(ctx, sub0, te)
    m = copy.deepcopy(sub0).refit_symbolic(max_iter=200)
    out["refit_old"] = ev(ctx, m, te)
    subg = copy.deepcopy(sel)
    substitute_symbolic(subg, recs, guard=True)
    subg._finalise(subg.train_)
    rec["symbolic"]["n_guarded"] = sum(1 for lay in (subg.net.g, subg.net.h, getattr(subg.net, "stat", None)) if lay is not None
                                       for e in lay.symbolic.values() if getattr(e, "guard", None) is not None)
    out["sub_guard"] = ev(ctx, subg, te)
    m = copy.deepcopy(subg).refit_symbolic(max_iter=200)
    out["refit_guard"] = ev(ctx, m, te)
    for mu in MUS:
        m = copy.deepcopy(subg).refit_symbolic(max_iter=200, anchor=sel, mu=mu)
        out[f"distill_mu{mu:g}"] = ev(ctx, m, te)
    best, drecs = refit_distill(subg, sel, MUS, max_iter=200)
    rec["distill_grid"] = drecs
    out["distill"] = ev(ctx, best, te) if best is not None else dict(status="no_finite_refit")
    rec["variants"] = out
    rec["seconds"] = round(time.time() - t0)
    return rec


def job_res(scen, seed):
    ctx = Ctx15(scen, seed)
    xg, last = ctx.front("bspline")["xg"], ctx.last()
    return _res_models(ctx, xg, last, ctx.dat.z, lambda S, r: R12.survival_metrics(ctx, S, r),
                       dict(job="res", scenario=scen, seed=seed))


def _res_models(ctx, xg, last, z, metric, rec):
    def pk(idx, F=None):
        return dict(xg=xg[idx], z=z[idx] if F is None else np.hstack([F[idx], z[idx]]), y=ctx_y(ctx)[idx],
                    d=ctx_d(ctx)[idx].astype(float))
    tr, va, te = pk(ctx.tr), pk(ctx.va), pk(ctx.te)
    out = {}
    t0 = time.time()
    e = FKANEnsemble(kind="b2s", loss="cox").fit(tr, va)
    out["b2s_cox"] = metric(lambda tt: e.predict_survival(te["xg"], te["z"], tt), e.predict_risk(te["xg"], te["z"]))
    trL, vaL, teL = pk(ctx.tr, last), pk(ctx.va, last), pk(ctx.te, last)
    s = FKANEnsemble(kind="static", loss="cox").fit(dict(trL, xg=None), dict(vaL, xg=None))
    out["static_coxkan"] = metric(lambda tt: s.predict_survival(None, teL["z"], tt), s.predict_risk(None, teL["z"]))
    X = np.hstack([last, z])
    lc = LinearCox().fit(X[ctx.tr], ctx_y(ctx)[ctx.tr], ctx_d(ctx)[ctx.tr])
    out["landmark_cox"] = metric(lambda tt: lc.predict_survival(X[ctx.te], tt), lc.predict_risk(X[ctx.te]))
    t1 = time.time()
    rtr, rva = dict(tr, last=last[ctx.tr]), dict(va, last=last[ctx.va])
    r = ResFKAN(residual=ANCHORED is False, gate=GATE).fit(rtr, rva)
    lt, zt = last[ctx.te], z[ctx.te]
    out[("b2s_anchor_cox" if ANCHORED else "b2s_res_cox") + ("_cvgate" if GATE == "cv" else "")] = metric(lambda tt: r.predict_survival(te["xg"], lt, zt, tt), r.predict_risk(te["xg"], lt, zt))
    rec.update(models=out, res_gate=r.gate_grid, res_on=r.residual_on, res_lr=r.lr, res_wd=r.wd,
               seconds_baselines=round(t1 - t0), seconds_res=round(time.time() - t1))
    return rec


def ctx_y(ctx):
    return ctx.dat.y if hasattr(ctx, "dat") else ctx.y


def ctx_d(ctx):
    return ctx.dat.delta if hasattr(ctx, "dat") else ctx.d


def job_res_dev_real(seed):
    import run_round13 as R13
    R13.DEV["dev_aids"] = lambda: R13._dev_longdata(seed)
    ctx = R13.RealCtx("dev_aids", 0)
    xg = ctx.front("bspline")["xg"]
    return _res_models(ctx, xg, ctx.lastv(), ctx.lm.z, lambda S, r: R13.real_metrics(ctx, S, r),
                       dict(job="res_dev_real", scenario="dev_aids", seed=seed))


def job_afcm(seed):
    import test_round11 as R11
    from test_round12_ck import shapes
    dat = fsim.generate("S-Add", n=2000, seed=seed)
    tr, va, te = R11._split(2000, seed)
    xg, lam = R11._smoothed(dat, tr)
    ptr, pva = R11._pack(dat, xg, tr), R11._pack(dat, xg, va)
    X = dat.x_true_grid[te]
    rec = dict(job="afcm", seed=seed, n=2000, smoothing_lambda=lam)
    t0 = time.time()
    m, _ = fit_afcm(ptr, pva, **AFCM_D123)
    rec["v3"] = dict(shapes=shapes(m, X), lam_x=m.lam_x, lam_t=m.lam_t, seconds=round(time.time() - t0))
    t0 = time.time()
    m, recs = fit_afcm_bagged(ptr, pva, B=3, seed=0, **AFCM_D123)
    rec["v3_bag3"] = dict(shapes=shapes(m, X), resamples=recs, seconds=round(time.time() - t0))
    t0 = time.time()
    m, _ = fit_afcm(ptr, pva, per_channel="t", **AFCM_D123)
    rec["v3_pc_t"] = dict(shapes=shapes(m, X), lam_x=m.lam_x, lam_t=list(np.broadcast_to(m.lam_t, 3)),
                          seconds=round(time.time() - t0))
    return rec


def job_afcmpars(seed):
    """Dry-run follow-up: V3 with the D120 parsimonious rule (tol 0.01) instead of the minimum validation NLPL."""
    import test_round11 as R11
    from test_round12_ck import shapes
    dat = fsim.generate("S-Add", n=2000, seed=seed)
    tr, va, te = R11._split(2000, seed)
    xg, lam = R11._smoothed(dat, tr)
    t0 = time.time()
    m, recs = fit_afcm(R11._pack(dat, xg, tr), R11._pack(dat, xg, va), rule="parsimonious", tol=0.01, **AFCM_D123)
    vmin = min(r["val_nll"] for r in recs)
    return dict(job="afcmpars", seed=seed, v3_pars=dict(shapes=shapes(m, dat.x_true_grid[te]), lam_x=m.lam_x, lam_t=m.lam_t,
                n_within_tol=sum(r["val_nll"] <= vmin + 0.01 for r in recs), seconds=round(time.time() - t0)))


def job_afcmlaml(seed):
    """D134 development check C2: V3's basis and penalties, lambdas chosen by LAML on the training data (primary);
    the AIC-type rule is reported for information."""
    import test_round11 as R11
    from test_round12_ck import shapes
    dat = fsim.generate("S-Add", n=2000, seed=seed)
    tr, va, te = R11._split(2000, seed)
    xg, lam = R11._smoothed(dat, tr)
    ptr = R11._pack(dat, xg, tr)
    kw = {k: v for k, v in AFCM_D123.items()}
    rec = dict(job="afcmlaml", seed=seed, n=2000, smoothing_lambda=lam)
    for crit in ("laml", "aic"):
        t0 = time.time()
        m, recs = fit_afcm_laml(ptr, criterion=crit, **kw)
        ch = [r for r in recs if r.get("chosen")][0]
        rec[crit] = dict(shapes=shapes(m, dat.x_true_grid[te]), lam_x=ch["lam_x"], lam_t=ch["lam_t"], edf=ch["edf"],
                         grid=recs, seconds=round(time.time() - t0))
    s = rec["laml"]["shapes"]
    rec["laml_meets_ck6c_criterion"] = bool(s["phi1"]["rise_id"] < 0.15 and s["phi2"]["rise_id"] < 0.10)
    return rec


def run(job):
    parts = job.split("_")
    t0 = time.time()
    try:
        if parts[0] == "distill":
            rec = job_distill(parts[1], int(parts[2]))
        elif job.startswith("res_dev_real"):
            rec = job_res_dev_real(int(parts[-1]))
        elif parts[0] == "res":
            rec = job_res(parts[1], int(parts[2]))
        elif parts[0] == "afcmlaml":
            rec = job_afcmlaml(int(parts[1]))
        elif parts[0] == "afcmpars":
            rec = job_afcmpars(int(parts[1]))
        elif parts[0] == "afcm":
            rec = job_afcm(int(parts[1]))
        else:
            raise ValueError(job)
        rec["status"] = "ok"
    except Exception as e:                                               # every failure is recorded
        rec = dict(job=job, status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    rec.update(job_name=job, wall_seconds=round(time.time() - t0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    print(job, rec["status"], rec["wall_seconds"], "s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", required=True)
    for j in ap.parse_args().job:
        run(j)
