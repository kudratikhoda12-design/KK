"""Round 20 (D150): NP-KAN-AFT (semiparametric functional KAN AFT, src/kanfae/models/fkan_surv.py) and the two-half
cross-confirmed anchor (CrossConfirmRefitAnchor, src/kanfae/models/fres.py). One harness for the two development
stages and the formal run; every job is recorded, failures included. Plan and rules: docs/round20_devplan.md.

Validation split as Rounds 18-19 (scripts/run_round18.split_val): h1 = val_select, h2 = val_confirm.
Modes ($R20_MODE) and jobs:
  devA     devsparse_<s> 4500-4519, devmisspec_<s> 4600-4619 (already-used Round-18 seeds): full-validation B2-S
           ensembles for the NP objective choice and the H-R20a go check:
             sparse:  b2s_cox_fullval, b2s_npaft_bj_fullval, b2s_npaft_rank_fullval
             misspec: the same + b2s_weibull_fullval
  devB     devlast_<s> 4400-4419, devsparse_<s>, devmisspec_<s>: cross-confirmation at margin 0 (every margin is
           evaluated offline from the recorded held-out losses), per family (cox; lnaft = log-normal KAN-AFT;
           npaft = NP-KAN-AFT with the objective chosen after devA):
             last: cox, lnaft, npaft (+ landmark_cox);  sparse: cox (+ landmark_cox);  misspec: lnaft, npaft
  formal   last_<s> 4700-4719, sparse_<s> 4800-4819, misspec_<s> 4900-4919: the devB families with the D150 margins
           (if GO-b), plus b2s_cox_fullval and b2s_weibull_fullval on misspec and b2s_npaft_fullval on sparse (if GO-a).
Per cross-confirmed family <f>: b2s_<f>_h1, b2s_<f>_h2 (B2-S ensembles on each half), b2s_<f>_fullval (the primary
comparator), b2s_anchor_<f>_cross (the estimator); alternatives: "refit" (the full-validation refit, whatever the
decision) and "h1_anchored" (the h1-direction anchored ensemble before the refit). Output: $R20_OUT.
"""
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
import run_round17 as R17  # noqa: E402
import run_round18 as R18  # noqa: E402
from kanfae.models.fbaselines import LinearCox  # noqa: E402
from kanfae.models.fkan import FKANEnsemble  # noqa: E402
from kanfae.models.fkan_surv import NPKANAFT  # noqa: E402
from kanfae.models.fres import CrossConfirmRefitAnchor  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
OUT = Path(os.environ.get("R20_OUT", ROOT / "reports/round20/dev/results_dev.jsonl"))
SETTINGS = {"devA": dict(np_objective=None, margin=None, go_a=True, go_b=False),
            "devB": dict(np_objective="rank_km", margin={"cox": 0.0, "lnaft": 0.0, "npaft": 0.0}, go_a=False, go_b=True),
            "formal": dict(np_objective=None, margin=None, go_a=None, go_b=None)}      # devB objective, formal: D150
DEV_A = {"devsparse": range(4500, 4520), "devmisspec": range(4600, 4620)}
DEV_B = {"devlast": range(4400, 4420), "devsparse": range(4500, 4520), "devmisspec": range(4600, 4620)}
FORMAL = {"last": range(4700, 4720), "sparse": range(4800, 4820), "misspec": range(4900, 4920)}
FAMILIES = {"last": ("cox", "lnaft", "npaft"), "sparse": ("cox",), "misspec": ("lnaft", "npaft")}
RELATIVE = {"cox": False, "lnaft": False, "npaft": True}          # margin units (D150): NP-AFT relative to |loss|
SCEN = R17.SCEN


def jobs(mode):
    S = {"devA": DEV_A, "devB": DEV_B}.get(mode, FORMAL)
    return [f"{k}_{S[k][i]}" for i in range(20) for k in S]


def family_kw(f, np_objective=None):
    if f == "cox":
        return dict(loss="cox")
    if f == "lnaft":
        return dict(loss="aft", family="lognormal")
    if f == "weibull":
        return dict(loss="aft", family="weibull")
    if np_objective is None:
        raise RuntimeError("the NP-KAN-AFT objective is chosen after development stage A (D150)")
    return dict(loss="aft", family="np_aft", np_objective=np_objective)


def ens(f, tr, va, np_objective=None):
    return FKANEnsemble(kind="b2s", **family_kw(f, np_objective)).fit(tr, va)


def job(kind, seed, mode):
    st = SETTINGS[mode]
    if mode == "formal" and (st["go_a"] is None or st["go_b"] is None):
        raise RuntimeError("formal settings are fixed in D150 before any formal seed is used")
    ctx = R17.Ctx17(SCEN[kind], seed)
    xg = ctx.front("bspline")["xg"]
    last = ctx.last()
    h1, h2 = R18.split_val(ctx.va, ctx.dat.delta, seed)
    tr, va, p1, p2, te = (ctx.pack(i, xg) for i in (ctx.tr, ctx.va, h1, h2, ctx.te))
    rec = dict(job=kind, scenario=SCEN[kind] + ("-lognormal" if kind == "misspec" else ""), seed=seed, mode=mode,
               n_train=len(ctx.tr), n_h1=len(h1), n_h2=len(h2), n_test=len(ctx.te),
               censored_frac=float(1 - ctx.dat.delta.mean()))
    lt, zt = last[ctx.te], ctx.dat.z[ctx.te]
    models, alts, gates = {}, {}, {}

    def met_ens(e):
        return R12.survival_metrics(ctx, lambda tt: e.predict_survival(te["xg"], te["z"], tt), e.predict_risk(te["xg"], te["z"]))

    def timed(name, fit):
        t0 = time.time()
        e = fit()
        models[name] = met_ens(e)
        models[name].update(lr=e.lr, wd=e.wd, seconds=round(time.time() - t0))
        return e

    if mode == "devA" or (mode == "formal" and st["go_a"]):          # full-validation B2-S comparisons (H-R20a)
        names = []
        if mode == "devA":
            names = [("b2s_cox_fullval", lambda: ens("cox", tr, va)),
                     ("b2s_npaft_bj_fullval", lambda: NPKANAFT("bj_km").fit(tr, va)),
                     ("b2s_npaft_rank_fullval", lambda: NPKANAFT("rank_km").fit(tr, va))]
            if kind == "misspec":
                names.append(("b2s_weibull_fullval", lambda: ens("weibull", tr, va)))
        elif kind == "misspec":
            names = [("b2s_cox_fullval", lambda: ens("cox", tr, va)), ("b2s_weibull_fullval", lambda: ens("weibull", tr, va))]
            if not st["go_b"]:
                names.append(("b2s_npaft_fullval", lambda: ens("npaft", tr, va, st["np_objective"])))
        elif kind == "sparse":
            names = [("b2s_npaft_fullval", lambda: ens("npaft", tr, va, st["np_objective"]))]
            if not st["go_b"]:
                names.append(("b2s_cox_fullval", lambda: ens("cox", tr, va)))
        for name, fit in names:
            timed(name, fit)

    if mode == "devB" or (mode == "formal" and st["go_b"]):           # cross-confirmed anchors (H-R20b)
        for f in FAMILIES[kind]:
            refs = {h: timed(f"b2s_{f}_{h}", lambda v=v: ens(f, tr, v, st["np_objective"]))
                    for h, v in (("h1", p1), ("h2", p2), ("fullval", va))}
            t0 = time.time()
            kw = family_kw(f, st["np_objective"])
            m = CrossConfirmRefitAnchor(margin=st["margin"][f], relative_margin=RELATIVE[f], **kw).fit(
                dict(tr, last=last[ctx.tr]), dict(p1, last=last[h1]), dict(p2, last=last[h2]),
                refs["h1"], refs["h2"], refs["fullval"])

            def met(members=None, model=m):
                return R12.survival_metrics(ctx, lambda tt: model.predict_survival(te["xg"], lt, zt, tt, members),
                                            model.predict_risk(te["xg"], lt, zt, members))
            name = f"b2s_anchor_{f}_cross"
            models[name] = met()
            alt_h1 = m._alts[0][1]
            alts[name] = dict(refit=models[name] if not m.reduced else met(m.refit_members()),
                              h1_anchored=R12.survival_metrics(ctx, lambda tt: alt_h1.predict_survival(te["xg"], lt, zt, tt),
                                                               alt_h1.predict_risk(te["xg"], lt, zt)))
            gates[name] = dict(confirmed=bool(m.confirmed), margin=st["margin"][f], relative_margin=RELATIVE[f],
                               dirs=m.dirs, traj_on_h1=alt_h1.residual_on, last_on_h1=alt_h1.last_on,
                               seconds=round(time.time() - t0))
        if kind != "misspec":
            X = np.hstack([last, ctx.dat.z])
            y, d = ctx.dat.y, ctx.dat.delta.astype(float)
            lc = LinearCox().fit(X[ctx.tr], y[ctx.tr], d[ctx.tr])
            models["landmark_cox"] = R12.survival_metrics(ctx, lambda tt: lc.predict_survival(X[ctx.te], tt),
                                                          lc.predict_risk(X[ctx.te]))
    rec.update(models=models, alternatives=alts, gates=gates, np_objective=st["np_objective"])
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
    mode = os.environ.get("R20_MODE", "devA")
    for j in sys.argv[1:]:
        r = run_job(j, mode)
        print(j, r["status"], r["wall_seconds"], r.get("error", ""))
