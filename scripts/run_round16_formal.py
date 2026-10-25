"""Round 16 (D138): fixed-mu NaN-safe symbolic distillation and the non-cliff gate. Every job recorded.

Jobs (fixed order; scripts/run_round16_lanes.py pulls them from one queue):
  add_<s>     S-Add, s in 3800-3819: B2-S Cox ensemble (D112) with shapes; the symbolic pipeline on its tuned (lr, wd):
              select_lambda_theta (D120) on the seed-0 model; stages 1-2 (no PySR) twice -- unrestricted (as Round 15:
              no-refit and old unconstrained refit, for continuity) and with max_abs_arg 50 (Round 16) -- then the
              Round-16 model: guarded substitution of the max_abs_arg-50 formulas and ONE distillation refit at
              mu = 10, safe = True, 200 L-BFGS iterations.
  sparse_<s>  S-Sparse, s in 3900-3919: the same, then family B: static CoxKAN (last values + z), landmark Cox,
              b2s_anchor_cox with the non-cliff gate (primary) and with the D133 gate (secondary).
  last_<s>    S-Last, s in 4000-4019: B2-S Cox ensemble, static CoxKAN, landmark Cox, both anchored variants.
  dev_add_<s> development check (not a formal job): the add_ pipeline on an already-used seed (S-Add 3515, the Round-15
              failure), recorded to reports/round16/dev/.
Metrics as Round 15 (survival-curve error vs the oracle, IBS, Harrell/Uno C, D-cal p; identifiable RISE). $R16_OUT.
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
import run_round12 as R12  # noqa: E402
import run_round15 as D15  # noqa: E402
from kanfae.models.fbaselines import LinearCox  # noqa: E402
from kanfae.models.fkan import FKANEnsemble, FKANSurv, select_lambda_theta, substitute_symbolic, symbolic_edges  # noqa: E402
from kanfae.models.fres import ResFKAN  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
OUT = Path(os.environ.get("R16_OUT", ROOT / "reports/round16/results_formal.jsonl"))
MU, MAX_ABS_ARG, ITERS = 10.0, 50.0, 200
SEEDS = {"add": range(3800, 3820), "sparse": range(3900, 3920), "last": range(4000, 4020)}


def jobs():
    J = []
    for r in range(20):
        J += [f"add_{SEEDS['add'][r]}", f"sparse_{SEEDS['sparse'][r]}", f"last_{SEEDS['last'][r]}"]
    return J


def ev(ctx, m, te):
    return D15.ev(ctx, m, te)


def n_guarded(m):
    return sum(1 for lay in (m.net.g, m.net.h, getattr(m.net, "stat", None)) if lay is not None
               for e in lay.symbolic.values() if getattr(e, "guard", None) is not None)


def formulas(recs):
    return [dict(layer=r["layer"], i=r["i"], o=r["o"], stage=r["result"].stage, formula=r["result"].formula,
                 r2_val=r["result"].r2_val) for r in recs if r["result"].accepted]


def symbolic_pipeline(ctx, tr, va, te, lr, wd):
    rec = {}
    t0 = time.time()
    sel, grid = select_lambda_theta(lambda l: FKANSurv(kind="b2s", loss="cox", lam=l, lr=lr, wd=wd, seed=0), tr, va)
    s = sel.selected_channels()
    rec.update(selected=s, recall=(s[0] + s[1]) / 2, fdr=s[2] / max(sum(s), 1), grid=grid)
    out = {"after_pruning": ev(ctx, sel, te)}
    recs0 = symbolic_edges(sel, use_pysr=False)                                  # Round-15 (unrestricted) formulas
    sub0 = copy.deepcopy(sel)
    n0 = substitute_symbolic(sub0, recs0)
    sub0._finalise(sub0.train_)
    out["sub_noguard"] = ev(ctx, sub0, te)
    out["refit_old"] = ev(ctx, copy.deepcopy(sub0).refit_symbolic(max_iter=ITERS), te)
    recs = symbolic_edges(sel, use_pysr=False, max_abs_arg=MAX_ABS_ARG)          # Round 16
    subg = copy.deepcopy(sel)
    n1 = substitute_symbolic(subg, recs, guard=True)
    subg._finalise(subg.train_)
    out["sub_guard_r16"] = ev(ctx, subg, te)
    m = copy.deepcopy(subg).refit_symbolic(max_iter=ITERS, anchor=sel, mu=MU, safe=True)
    out["distill_fixed"] = ev(ctx, m, te)
    rec["symbolic"] = dict(n_edges=len(recs), n_accepted_r15=sum(r["result"].accepted for r in recs0),
                           n_accepted_r16=sum(r["result"].accepted for r in recs), n_substituted_r15=n0,
                           n_substituted_r16=n1, n_guarded=n_guarded(subg), refit_fallback=bool(m.refit_fallback),
                           refit_train_loss=m.refit_train_loss, formulas_r16=formulas(recs),
                           formulas_r15_rejected=[f for f in formulas(recs0) if f["formula"] not in
                                                  {g["formula"] for g in formulas(recs)}])
    rec["variants"], rec["symbolic_seconds"] = out, round(time.time() - t0)
    return rec


def family_b(ctx, tr, va, te, rec):
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
    for name, rule in (("b2s_anchor_cox_noncliff", "noncliff"), ("b2s_anchor_cox_d133", "parsimonious")):
        t0 = time.time()
        r = ResFKAN(residual=False, gate_rule=rule).fit(dict(tr, last=last[ctx.tr]), dict(va, last=last[ctx.va]))
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
    r = R12.survival_metrics(ctx, lambda tt: e.predict_survival(te["xg"], te["z"], tt), e.predict_risk(te["xg"], te["z"]))
    if scen in ("S-Add", "S-Sparse"):
        r["shape"] = R12.shape_metrics(ctx, e.effect)
    rec["b2s_cox"] = dict(r, lr=e.lr, wd=e.wd)
    rec["b2s_seconds"] = round(time.time() - t0)
    if kind in ("add", "sparse"):
        rec.update(symbolic_pipeline(ctx, tr, va, te, e.lr, e.wd))
    if kind in ("sparse", "last"):
        family_b(ctx, tr, va, te, rec)
    return rec


def run_job(job):
    parts = job.split("_")
    t0 = time.time()
    try:
        if parts[0] == "dev":
            rec = job_sim(parts[1], int(parts[2]))
            rec["development_check"] = True
        else:
            rec = job_sim(parts[0], int(parts[1]))
        rec["status"] = "ok"
    except Exception as e:
        rec = dict(job=job, status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    rec.update(job_name=job, wall_seconds=round(time.time() - t0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    return rec


def done():
    seen = set()
    for f in OUT.parent.glob("results_lane*.jsonl"):
        for l in f.read_text().splitlines():
            if l.strip() and json.loads(l).get("status") == "ok":
                seen.add(json.loads(l)["job_name"])
    return seen


if __name__ == "__main__":
    for j in sys.argv[1:]:
        r = run_job(j)
        print(j, r["status"], r["wall_seconds"], r.get("error", ""))
