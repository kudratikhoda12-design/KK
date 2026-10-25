"""Round 15 formal run (D133 as locked by D134/D135). Fresh seeds; every job recorded (failures with tracebacks).

Jobs (fixed order; scripts/run_round15_lanes.py pulls them from one queue):
  ck6d_24            only if the C2 development check passed (D134): the CK-6c procedure (S-Add, DGP v2, n 2,000,
                     seed 24) with the LAML-selected AFCM; criterion identifiable RISE phi_1 < 0.15 and phi_2 < 0.10.
  add_<s>            S-Add, s in 3500-3519 (dense, n 1,000): B2-S Cox ensemble (D112) with shapes; the symbolic_distill
                     pipeline on its tuned (lr, wd) (select_lambda_theta D120 on the seed-0 model; stages 1-2; models
                     after pruning, substituted without refit, old unconstrained refit, guarded substitution, guarded
                     distillation refit at every mu in MUS and the mu chosen by unpenalised validation NLPL); and, if
                     C2 passed, AFCM V3 (validation NLPL) and the LAML AFCM (for H-R15c, used only if CK-6d passes).
  sparse_<s>         S-Sparse, s in 3600-3619 (3-8 visits): the same B2-S ensemble and symbolic_distill pipeline, then
                     family B: static CoxKAN (last values + z), landmark Cox (linear, last values + z), b2s_res_cox
                     (residual, secondary) and b2s_anchor_cox (primary), both with the single-split validation gate.
  last_<s>           S-Last, s in 3700-3719 (3-8 visits): B2-S Cox ensemble, static CoxKAN, landmark Cox, b2s_res_cox,
                     b2s_anchor_cox.
  real_<ds>_<k>      exploratory (D134): ds in pbc_E2, pbc_E1, prothro, aids; Round-13 split k in 0-4: b2s_anchor_cox,
                     b2s_res_cox and, if development check G passed, b2s_anchor_cox_cvgate; for k = 0 also landmark_cox,
                     to check that the Round-13 records reproduce before they are used as paired comparators.
Metrics: as Round 12 (simulation; survival-curve error vs the oracle, IBS, Harrell/Uno C, D-cal p, identifiable RISE
for B2 models) and Round 13 (real data). Records: $R15_OUT.
"""
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
import run_round15 as D15  # noqa: E402
from kanfae.models.afcm import AFCM_D123, fit_afcm, fit_afcm_laml  # noqa: E402
from kanfae.models.fbaselines import LinearCox  # noqa: E402
from kanfae.models.fkan import FKANEnsemble, FKANSurv, refit_distill, select_lambda_theta, substitute_symbolic, symbolic_edges  # noqa: E402
from kanfae.models.fres import ResFKAN  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
OUT = Path(os.environ.get("R15_OUT", ROOT / "reports/round15/results_formal.jsonl"))
MUS = (1e-2, 1e-1, 1.0, 10.0)                                      # D134: 10 added
FLAGS = json.loads((ROOT / "reports/round15/dev_check_flags.json").read_text())   # {"C2_pass": bool, "G_pass": bool}
SEEDS = {"add": range(3500, 3520), "sparse": range(3600, 3620), "last": range(3700, 3720)}
REAL = ("pbc_E2", "pbc_E1", "prothro", "aids")


def jobs():
    J = ["ck6d_24"] if FLAGS["C2_pass"] else []
    for r in range(20):
        J += [f"add_{SEEDS['add'][r]}", f"sparse_{SEEDS['sparse'][r]}", f"last_{SEEDS['last'][r]}"]
    J += [f"real_{ds}_{k}" for k in range(5) for ds in REAL]
    return J


def ev(ctx, m, te, shape=True):
    return D15.ev(ctx, m, te, shape)


def ens_metrics(ctx, e, te, shape=True):
    r = R12.survival_metrics(ctx, lambda tt: e.predict_survival(te["xg"], te["z"], tt), e.predict_risk(te["xg"], te["z"]))
    if shape and ctx.scen in ("S-Add", "S-Sparse"):
        r["shape"] = R12.shape_metrics(ctx, e.effect)
    return r


def distill_pipeline(ctx, tr, va, te, lr, wd):
    rec = {}
    t0 = time.time()
    sel, grid = select_lambda_theta(lambda l: FKANSurv(kind="b2s", loss="cox", lam=l, lr=lr, wd=wd, seed=0), tr, va)
    s = sel.selected_channels()
    rec.update(selected=s, recall=(s[0] + s[1]) / 2, fdr=s[2] / max(sum(s), 1), grid=grid)
    out = {"after_pruning": ev(ctx, sel, te)}
    recs = symbolic_edges(sel, use_pysr=False)
    acc = [r for r in recs if r["result"].accepted]
    rec["symbolic"] = dict(n_edges=len(recs), n_accepted=len(acc), formulas=[
        dict(layer=r["layer"], i=r["i"], o=r["o"], stage=r["result"].stage, formula=r["result"].formula,
             r2_val=r["result"].r2_val) for r in acc])
    sub0 = copy.deepcopy(sel)
    rec["symbolic"]["n_substituted"] = substitute_symbolic(sub0, recs)
    sub0._finalise(sub0.train_)
    out["sub_noguard"] = ev(ctx, sub0, te)
    out["refit_old"] = ev(ctx, copy.deepcopy(sub0).refit_symbolic(max_iter=200), te)
    subg = copy.deepcopy(sel)
    substitute_symbolic(subg, recs, guard=True)
    subg._finalise(subg.train_)
    rec["symbolic"]["n_guarded"] = sum(1 for lay in (subg.net.g, subg.net.h, getattr(subg.net, "stat", None)) if lay is not None
                                       for e in lay.symbolic.values() if getattr(e, "guard", None) is not None)
    out["sub_guard"] = ev(ctx, subg, te)
    for mu in MUS:
        out[f"distill_mu{mu:g}"] = ev(ctx, copy.deepcopy(subg).refit_symbolic(max_iter=200, anchor=sel, mu=mu), te)
    best, drecs = refit_distill(subg, sel, MUS, max_iter=200)
    rec["distill_grid"] = drecs
    out["distill"] = ev(ctx, best, te) if best is not None else dict(status="no_finite_refit")
    rec["variants"], rec["distill_seconds"] = out, round(time.time() - t0)
    return rec


def family_b(ctx, xg, tr, va, te, rec):
    last, z = ctx.last(), ctx.dat.z
    y, d = ctx.dat.y, ctx.dat.delta.astype(float)

    def pk(idx, F):
        return dict(xg=None, z=np.hstack([F[idx], z[idx]]), y=y[idx], d=d[idx])
    s = FKANEnsemble(kind="static", loss="cox").fit(pk(ctx.tr, last), pk(ctx.va, last))
    teL = pk(ctx.te, last)
    out = {"static_coxkan": R12.survival_metrics(ctx, lambda tt: s.predict_survival(None, teL["z"], tt), s.predict_risk(None, teL["z"]))}
    X = np.hstack([last, z])
    lc = LinearCox().fit(X[ctx.tr], y[ctx.tr], d[ctx.tr])
    out["landmark_cox"] = R12.survival_metrics(ctx, lambda tt: lc.predict_survival(X[ctx.te], tt), lc.predict_risk(X[ctx.te]))
    gates = {}
    for name, residual in (("b2s_anchor_cox", False), ("b2s_res_cox", True)):
        t0 = time.time()
        r = ResFKAN(residual=residual).fit(dict(tr, last=last[ctx.tr]), dict(va, last=last[ctx.va]))
        lt, zt = last[ctx.te], z[ctx.te]
        out[name] = R12.survival_metrics(ctx, lambda tt, r=r: r.predict_survival(te["xg"], lt, zt, tt), r.predict_risk(te["xg"], lt, zt))
        gates[name] = dict(gate_grid=r.gate_grid, residual_on=r.residual_on, lr=r.lr, wd=r.wd, seconds=round(time.time() - t0))
    rec["family_b"], rec["gates"] = out, gates


def job_sim(kind, seed):
    scen = {"add": "S-Add", "sparse": "S-Sparse", "last": "S-Last"}[kind]
    ctx = D15.Ctx15(scen, seed)
    xg = ctx.front("bspline")["xg"]
    tr, va, te = ctx.pack(ctx.tr, xg), ctx.pack(ctx.va, xg), ctx.pack(ctx.te, xg)
    rec = dict(job=kind, scenario=scen, seed=seed, n_train=len(ctx.tr), n_test=len(ctx.te),
               censored_frac=float(1 - ctx.dat.delta.mean()))
    t0 = time.time()
    e = FKANEnsemble(kind="b2s", loss="cox").fit(tr, va)
    rec["b2s_cox"] = dict(ens_metrics(ctx, e, te), lr=e.lr, wd=e.wd, seed0=ev(ctx, e.members[0], te))
    rec["b2s_seconds"] = round(time.time() - t0)
    if kind in ("add", "sparse"):
        rec.update(distill_pipeline(ctx, tr, va, te, e.lr, e.wd))
    if kind in ("sparse", "last"):
        family_b(ctx, xg, tr, va, te, rec)
    if kind == "add" and FLAGS["C2_pass"]:
        for name, fit in (("afcm_v3", lambda: fit_afcm(tr, va, **AFCM_D123)[0]),
                          ("afcm_laml", lambda: fit_afcm_laml(tr, **AFCM_D123)[0])):
            m = fit()
            rec[name] = dict(R12.survival_metrics(ctx, lambda tt, m=m: m.predict_survival(te["xg"], te["z"], tt),
                                                  m.predict_risk(te["xg"], te["z"])),
                             shape=R12.shape_metrics(ctx, m.effect), lam_x=m.lam_x, lam_t=m.lam_t)
    return rec


def job_ck6d(seed):
    import test_round11 as R11
    from test_round12_ck import shapes
    from kanfae.data import fsim
    dat = fsim.generate("S-Add", n=2000, seed=seed)
    tr, va, te = R11._split(2000, seed)
    xg, lam = R11._smoothed(dat, tr)
    m, recs = fit_afcm_laml(R11._pack(dat, xg, tr), **AFCM_D123)
    s = shapes(m, dat.x_true_grid[te])
    return dict(job="ck6d", seed=seed, shapes=s, lam_x=m.lam_x, lam_t=m.lam_t, grid=recs, smoothing_lambda=lam,
                threshold_phi1=0.15, threshold_phi2=0.10,
                passed=bool(s["phi1"]["rise_id"] < 0.15 and s["phi2"]["rise_id"] < 0.10))


def job_real(ds, k):
    import run_round13 as R13
    ctx = R13.RealCtx(ds, k)
    xg, last, z = ctx.front("bspline")["xg"], ctx.lastv(), ctx.lm.z
    y, d = ctx.y, ctx.d.astype(float)
    pk = lambda idx: dict(xg=xg[idx], z=z[idx], y=y[idx], d=d[idx], last=last[idx])  # noqa: E731
    rec = dict(job="real", dataset=ds, split=k, channels=list(ctx.lm.channels))
    variants = [("b2s_anchor_cox", False, "split"), ("b2s_res_cox", True, "split")]
    if FLAGS["G_pass"]:
        variants.append(("b2s_anchor_cox_cvgate", False, "cv"))
    out, gates = {}, {}
    for name, residual, gate in variants:
        t0 = time.time()
        r = ResFKAN(residual=residual, gate=gate).fit(pk(ctx.tr), pk(ctx.va))
        te = pk(ctx.te)
        out[name] = R13.real_metrics(ctx, lambda tt, r=r: r.predict_survival(te["xg"], te["last"], te["z"], tt),
                                     r.predict_risk(te["xg"], te["last"], te["z"]))
        gr = np.percentile(np.concatenate([last[ctx.tr], last[ctx.va]]), [5, 95], axis=0)
        gates[name] = dict(gate_grid=r.gate_grid, residual_on=r.residual_on, lr=r.lr, wd=r.wd, seconds=round(time.time() - t0),
                           last_value_effects={c: r.last_value_effect(j, np.linspace(gr[0, j], gr[1, j], 50)).tolist()
                                               for j, c in enumerate(ctx.lm.channels)},
                           last_value_grid={c: np.linspace(gr[0, j], gr[1, j], 50).tolist() for j, c in enumerate(ctx.lm.channels)})
    if k == 0:
        X = np.hstack([last, z])
        m = LinearCox().fit(X[ctx.tr], ctx.y_tr, ctx.d_tr)
        out["landmark_cox_repro"] = R13.lifelines_metrics(ctx, m, X)
    rec["models"], rec["gates"] = out, gates
    return rec


def done():
    seen = set()
    for f in OUT.parent.glob("results_lane*.jsonl"):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("status") == "ok":
                    seen.add(r["job_name"])
    return seen


def run_job(job):
    parts = job.split("_")
    t0 = time.time()
    try:
        if parts[0] == "ck6d":
            rec = job_ck6d(int(parts[1]))
        elif parts[0] == "real":
            rec = job_real("_".join(parts[1:-1]), int(parts[-1]))
        else:
            rec = job_sim(parts[0], int(parts[1]))
        rec["status"] = "ok"
    except Exception as e:
        rec = dict(job=job, status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    rec.update(job_name=job, wall_seconds=round(time.time() - t0), flags=FLAGS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    return rec
