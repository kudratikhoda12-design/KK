"""Round 17 (D141): two-way gated landmark anchoring (Cox and log-normal KAN-AFT) and fixed-mu symbolic distillation for
KAN-AFT. One harness for the development check and the formal run; every job is recorded, failures included.

Jobs:
  devsparse_<s>  development, already-used Round-16 seeds S-Sparse 3900-3919 (3-8 visits)
  devlast_<s>    development, already-used Round-16 seeds S-Last 4000-4019 (3-8 visits)
  devmisspec_<s> development seeds 961-969, S-Misspec-lognormal (dense; log T = 1 - eta/1.5 + 0.5 eps, eps ~ N(0,1))
  last_<s>       formal, S-Last 4100-4119
  sparse_<s>     formal, S-Sparse 4200-4219
  misspec_<s>    formal, S-Misspec-lognormal 4300-4319
Models per job (B-spline front end, D112 ensembles, n 1,000, split 60/20/20 on delta):
  last / sparse (dev and formal): b2s_cox (comparator), b2s_anchor_cox_twoway (ResFKAN residual=False,
    gate_last_value=True, the comparator as the pure-trajectory candidate), b2s_aft (log-normal KAN-AFT ensemble,
    comparator), b2s_anchor_aft_twoway (the same with the log-normal AFT loss), landmark_cox (last values + z).
  misspec: b2s_aft, the AFT symbolic pipeline (select_lambda_theta D120 on the seed-0 model at the ensemble's tuned
    lr/wd; stages 1-2 without PySR with max_abs_arg 50; guarded substitution; one distillation refit at mu = 10,
    safe = True, 200 L-BFGS iterations; also an unanchored refit of the same substitution for context),
    b2s_anchor_aft_twoway.
For every two-way model the record also holds the anchored alternative (the anchored state of step (1), whatever step
(2) decided) and both validation losses, so the development analysis can evaluate any last_tol from one fit.
last_tol: 0 in development mode; the D141 values in formal mode. Output: $R17_OUT.
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
from kanfae.models.fbaselines import LinearCox, impute_train_mean, last_values  # noqa: E402
from kanfae.models.fkan import FKANEnsemble, FKANSurv, select_lambda_theta, substitute_symbolic, symbolic_edges  # noqa: E402
from kanfae.models.fres import ResFKAN  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
OUT = Path(os.environ.get("R17_OUT", ROOT / "reports/round17/dev/results_dev.jsonl"))
MU, MAX_ABS_ARG, ITERS = 10.0, 50.0, 200
FAMILY = "lognormal"
LAST_TOL = {"dev": {"cox": 0.0, "aft": 0.0}, "formal": {"cox": 0.01, "aft": 0.04}}    # formal values: D141 (devplan §4)
DEV = {"devsparse": range(3900, 3920), "devlast": range(4000, 4020), "devmisspec": range(961, 970)}
FORMAL = {"last": range(4100, 4120), "sparse": range(4200, 4220), "misspec": range(4300, 4320)}
SCEN = {"sparse": "S-Sparse", "last": "S-Last", "misspec": "S-Misspec"}


def jobs(mode="formal"):
    S = DEV if mode == "dev" else FORMAL
    keys = list(S)
    n = max(len(S[k]) for k in keys)
    return [f"{k}_{S[k][i]}" for i in range(n) for k in keys if i < len(S[k])]


class Ctx17(D15.Ctx15):
    """Round-15 context, plus S-Misspec-lognormal with last values."""

    def __init__(self, scen, seed):
        if scen == "S-Misspec":
            R12.Ctx.__init__(self, "S-Misspec", 0, sub="lognormal", dev_seed=seed)
        else:
            super().__init__(scen, seed)

    def last(self):
        return impute_train_mean(last_values(self.dat.times, self.dat.values), self.tr)


def shape(ctx, models):
    return R12.shape_metrics(ctx, lambda d, x, t: np.mean([R12.scale_misspec(ctx, m, d, x, t) for m in models], 0))


def ev_single(ctx, m, te, with_shape):
    e = m.eta(te["xg"], te["z"])
    if not np.all(np.isfinite(e)):
        return dict(status="nonfinite", n_nonfinite_eta=int(np.sum(~np.isfinite(e))))
    r = R12.survival_metrics(ctx, lambda tt: m.predict_survival(te["xg"], te["z"], tt), m.predict_risk(te["xg"], te["z"]))
    if with_shape:
        r["shape"] = shape(ctx, [m])
    r["val_loss"] = float(m.val_loss)
    return r


def formulas(recs):
    return [dict(layer=r["layer"], i=r["i"], o=r["o"], stage=r["result"].stage, formula=r["result"].formula,
                 r2_val=r["result"].r2_val) for r in recs if r["result"].accepted]


def aft_symbolic(ctx, tr, va, te, lr, wd):
    """AFT symbolic_distill_fixed (D141 = the D138 pipeline with the log-normal AFT loss)."""
    rec, t0 = {}, time.time()
    sel, grid = select_lambda_theta(lambda l: FKANSurv(kind="b2s", loss="aft", family=FAMILY, lam=l, lr=lr, wd=wd, seed=0),
                                    tr, va)
    s = sel.selected_channels()
    rec.update(selected=s, recall=(s[0] + s[1]) / 2, fdr=s[2] / max(sum(s), 1), grid=grid)
    out = {"after_pruning": ev_single(ctx, sel, te, True)}
    recs = symbolic_edges(sel, use_pysr=False, max_abs_arg=MAX_ABS_ARG)
    recs0 = symbolic_edges(sel, use_pysr=False)
    sub = copy.deepcopy(sel)
    n1 = substitute_symbolic(sub, recs, guard=True)
    sub._finalise(sub.train_)
    out["sub_guard"] = ev_single(ctx, sub, te, True)
    out["refit_plain"] = ev_single(ctx, copy.deepcopy(sub).refit_symbolic(max_iter=ITERS), te, True)
    m = copy.deepcopy(sub).refit_symbolic(max_iter=ITERS, anchor=sel, mu=MU, safe=True)
    out["distill_fixed"] = ev_single(ctx, m, te, True)
    rec["symbolic"] = dict(n_edges=len(recs), n_accepted=sum(r["result"].accepted for r in recs), n_substituted=n1,
                           n_guarded=sum(1 for lay in (m.net.g, m.net.h, getattr(m.net, "stat", None)) if lay is not None
                                         for e in lay.symbolic.values() if getattr(e, "guard", None) is not None),
                           refit_fallback=bool(m.refit_fallback), refit_train_loss=m.refit_train_loss,
                           formulas=formulas(recs),
                           rejected=[f for f in formulas(recs0) if f["formula"] not in {g["formula"] for g in formulas(recs)}])
    rec["variants"], rec["symbolic_seconds"] = out, round(time.time() - t0)
    return rec


def twoway(ctx, loss, tr, va, te, last, ref, last_tol):
    t0 = time.time()
    fam = dict(family=FAMILY) if loss == "aft" else {}
    r = ResFKAN(residual=False, gate_last_value=True, last_tol=last_tol, loss=loss, **fam).fit(
        dict(tr, last=last[ctx.tr]), dict(va, last=last[ctx.va]), ref=ref)
    lt, zt = last[ctx.te], ctx.dat.z[ctx.te]

    def met(model):
        return R12.survival_metrics(ctx, lambda tt: model.predict_survival(te["xg"], lt, zt, tt),
                                    model.predict_risk(te["xg"], lt, zt))
    out = met(r)
    alt = r.anchored_alternative()
    out_alt = out if not r.reduced else met(alt)
    info = dict(reduced=r.reduced, ref_val_loss=r.ref_loss, anchored_val_loss=r.anchored_choice["val_loss"],
                anchored_choice=r.anchored_choice, traj_on=r.residual_on, last_on=r.last_on, gate_grid=r.gate_grid,
                n_fine_tunes=r.n_fine_tunes, lr=r.lr, wd=r.wd, last_tol=last_tol, seconds=round(time.time() - t0))
    return out, out_alt, info


def job(kind, seed, mode):
    ctx = Ctx17(SCEN[kind], seed)
    xg = ctx.front("bspline")["xg"]
    tr, va, te = ctx.pack(ctx.tr, xg), ctx.pack(ctx.va, xg), ctx.pack(ctx.te, xg)
    last = ctx.last()
    tol = LAST_TOL[mode]
    if any(v is None for v in tol.values()):
        raise RuntimeError("formal last_tol values are fixed in D141 before any formal seed is used")
    rec = dict(job=kind, scenario=SCEN[kind] + ("-lognormal" if kind == "misspec" else ""), seed=seed, mode=mode,
               n_train=len(ctx.tr), n_test=len(ctx.te), censored_frac=float(1 - ctx.dat.delta.mean()))
    models, alts, gates = {}, {}, {}
    losses = ("aft",) if kind == "misspec" else ("cox", "aft")
    for loss in losses:
        t0 = time.time()
        fam = dict(family=FAMILY) if loss == "aft" else {}
        e = FKANEnsemble(kind="b2s", loss=loss, **fam).fit(tr, va)
        name = f"b2s_{loss}"
        models[name] = R12.survival_metrics(ctx, lambda tt: e.predict_survival(te["xg"], te["z"], tt),
                                            e.predict_risk(te["xg"], te["z"]))
        if kind == "misspec":
            models[name]["shape"] = shape(ctx, e.members)
        models[name].update(lr=e.lr, wd=e.wd, val_loss_seed0=float(e.members[0].val_loss), seconds=round(time.time() - t0))
        if kind == "misspec":
            rec.update(aft_symbolic(ctx, tr, va, te, e.lr, e.wd))
        a, alt, info = twoway(ctx, loss, tr, va, te, last, e, tol[loss])
        models[f"b2s_anchor_{loss}_twoway"], alts[f"b2s_anchor_{loss}_twoway"], gates[f"b2s_anchor_{loss}_twoway"] = a, alt, info
    if kind != "misspec":
        X = np.hstack([last, ctx.dat.z])
        y, d = ctx.dat.y, ctx.dat.delta.astype(float)
        lc = LinearCox().fit(X[ctx.tr], y[ctx.tr], d[ctx.tr])
        models["landmark_cox"] = R12.survival_metrics(ctx, lambda tt: lc.predict_survival(X[ctx.te], tt),
                                                      lc.predict_risk(X[ctx.te]))
    rec.update(models=models, anchored_alternative=alts, gates=gates)
    return rec


def run_job(name, mode):
    k, s = name.rsplit("_", 1)
    kind = k.replace("dev", "")
    t0 = time.time()
    try:
        rec = job(kind, int(s), mode)
        rec["status"] = "ok"
    except Exception as e:
        rec = dict(job=kind, seed=int(s), mode=mode, status="failed", error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc())
    rec.update(job_name=name, wall_seconds=round(time.time() - t0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    return rec


def done(mode):
    seen = set()
    for f in OUT.parent.glob("results_lane*.jsonl"):
        for l in f.read_text().splitlines():
            if l.strip() and json.loads(l).get("status") == "ok":
                seen.add(json.loads(l)["job_name"])
    return seen


if __name__ == "__main__":
    mode = os.environ.get("R17_MODE", "dev")
    for j in sys.argv[1:]:
        r = run_job(j, mode)
        print(j, r["status"], r["wall_seconds"], r.get("error", ""))
