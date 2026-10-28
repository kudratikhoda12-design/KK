"""Round 18 (D144): landmark anchoring decided on a confirmation split. One harness for the development check and the
formal run; every job is recorded, failures included.

Validation split: each replicate's validation subjects (20% of n, as before) are split once, stratified on the event
indicator, into val_select (half) and val_confirm (half), by a generator seeded with (replicate seed, 18). val_select is
used for everything except the final anchoring decision: (lr, wd) tuning and early stopping of the comparator AND of
the anchored ensemble, the candidate fine-tunes and the step-(1) choice. val_confirm is used only for the one-vs-one
decision (scripts' ResFKAN(confirm=...), src/kanfae/models/fres.py).
Jobs:
  devlast_<s> / devsparse_<s> / devmisspec_<s>  development, already-used Round-17 seeds 4100-4119 / 4200-4219 /
                                                4300-4319 (treated as development data for Round 18)
  last_<s> / sparse_<s> / misspec_<s>           formal, 4400-4419 (S-Last), 4500-4519 (S-Sparse),
                                                4600-4619 (S-Misspec-lognormal)
Models (B-spline front end, D112 ensembles, n 1,000, split 60/20/20 on delta):
  last / sparse: Cox and log-normal KAN-AFT; misspec: log-normal KAN-AFT only. Per family:
    b2s_<loss>            comparator, tuned and early-stopped on val_select
    b2s_anchor_<loss>_confirm  two-way gate (Round 17 candidates and step (1) on val_select), step (2) on val_confirm
    b2s_<loss>_fullval    the comparator trained as in Rounds 12-17 on the whole validation set (exploratory: the
                          price of setting val_confirm aside)
  landmark_cox (last values + z; training data only) on S-Last and S-Sparse.
For every confirm model the record holds the anchored alternative and both confirmation losses, so any margin can be
evaluated from one fit. Margin: 0 in development mode; the D144 values in formal mode. Output: $R18_OUT.
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
from kanfae.models.fbaselines import LinearCox  # noqa: E402
from kanfae.models.fkan import FKANEnsemble  # noqa: E402
from kanfae.models.fres import ResFKAN  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
OUT = Path(os.environ.get("R18_OUT", ROOT / "reports/round18/dev/results_dev.jsonl"))
FAMILY = "lognormal"
LAST_TOL = {"dev": {"cox": 0.0, "aft": 0.0}, "formal": {"cox": None, "aft": None}}    # formal values: D144
DEV = {"devlast": range(4100, 4120), "devsparse": range(4200, 4220), "devmisspec": range(4300, 4320)}
FORMAL = {"last": range(4400, 4420), "sparse": range(4500, 4520), "misspec": range(4600, 4620)}
SCEN = R17.SCEN


def jobs(mode="formal"):
    S = DEV if mode == "dev" else FORMAL
    keys = list(S)
    return [f"{k}_{S[k][i]}" for i in range(20) for k in keys]


def split_val(va, delta, seed):
    """val -> (val_select, val_confirm): half of each event stratum to val_confirm (generator seeded (seed, 18))."""
    rng = np.random.default_rng([seed, 18])
    sel, conf = [], []
    for v in (0, 1):
        idx = rng.permutation(va[delta[va] == v])
        h = len(idx) // 2
        conf += idx[:h].tolist()
        sel += idx[h:].tolist()
    return np.sort(np.array(sel)), np.sort(np.array(conf))


def job(kind, seed, mode):
    ctx = R17.Ctx17(SCEN[kind], seed)
    xg = ctx.front("bspline")["xg"]
    last = ctx.last()
    va_sel, va_conf = split_val(ctx.va, ctx.dat.delta, seed)
    tr, va, vs, te = ctx.pack(ctx.tr, xg), ctx.pack(ctx.va, xg), ctx.pack(va_sel, xg), ctx.pack(ctx.te, xg)
    conf = dict(ctx.pack(va_conf, xg), last=last[va_conf])
    tol = LAST_TOL[mode]
    if any(v is None for v in tol.values()):
        raise RuntimeError("formal margins are fixed in D144 before any formal seed is used")
    rec = dict(job=kind, scenario=SCEN[kind] + ("-lognormal" if kind == "misspec" else ""), seed=seed, mode=mode,
               n_train=len(ctx.tr), n_val_select=len(va_sel), n_val_confirm=len(va_conf),
               events_val_select=int(ctx.dat.delta[va_sel].sum()), events_val_confirm=int(ctx.dat.delta[va_conf].sum()),
               n_test=len(ctx.te), censored_frac=float(1 - ctx.dat.delta.mean()))
    lt, zt = last[ctx.te], ctx.dat.z[ctx.te]
    models, alts, gates = {}, {}, {}
    for loss in (("aft",) if kind == "misspec" else ("cox", "aft")):
        fam = dict(family=FAMILY) if loss == "aft" else {}
        for name, val in ((f"b2s_{loss}", vs), (f"b2s_{loss}_fullval", va)):
            t0 = time.time()
            e = FKANEnsemble(kind="b2s", loss=loss, **fam).fit(tr, val)
            models[name] = R12.survival_metrics(ctx, lambda tt, e=e: e.predict_survival(te["xg"], te["z"], tt),
                                                e.predict_risk(te["xg"], te["z"]))
            models[name].update(lr=e.lr, wd=e.wd, seconds=round(time.time() - t0))
            if name == f"b2s_{loss}":
                ref = e
        t0 = time.time()
        r = ResFKAN(residual=False, gate_last_value=True, last_tol=tol[loss], loss=loss, **fam).fit(
            dict(tr, last=last[ctx.tr]), dict(vs, last=last[va_sel]), ref=ref, confirm=conf)

        def met(model):
            return R12.survival_metrics(ctx, lambda tt: model.predict_survival(te["xg"], lt, zt, tt),
                                        model.predict_risk(te["xg"], lt, zt))
        m = f"b2s_anchor_{loss}_confirm"
        models[m] = met(r)
        alts[m] = models[m] if not r.reduced else met(r.anchored_alternative())
        gates[m] = dict(reduced=r.reduced, confirm_loss_ref=r.confirm_loss_ref,
                        confirm_loss_anchored=r.confirm_loss_anchored, select_loss_ref=r.ref_loss,
                        select_loss_anchored=r.anchored_choice["val_loss"], anchored_choice=r.anchored_choice,
                        traj_on=r.residual_on, last_on=r.last_on, gate_grid=r.gate_grid, n_fine_tunes=r.n_fine_tunes,
                        lr=r.lr, wd=r.wd, last_tol=tol[loss], seconds=round(time.time() - t0))
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
    mode = os.environ.get("R18_MODE", "dev")
    for j in sys.argv[1:]:
        r = run_job(j, mode)
        print(j, r["status"], r["wall_seconds"], r.get("error", ""))
