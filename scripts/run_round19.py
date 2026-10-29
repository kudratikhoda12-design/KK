"""Round 19 (D147): the Round-18 confirmation-split decision with a full-validation estimator, tested against the usual
full-validation B2-S. One harness for the development check and the formal run; every job is recorded, failures included.

Validation split as Round 18 (scripts/run_round18.split_val: half of each event stratum to val_confirm, generator
seeded (seed, 18)).
Jobs:
  devlast_<s> / devsparse_<s> / devmisspec_<s>  development, already-used Round-18 seeds 4400-4419 / 4500-4519 /
                                                4600-4619 (treated as development data for Round 19)
  last_<s> / sparse_<s> / misspec_<s>           formal, 4700-4719 (S-Last), 4800-4819 (S-Sparse),
                                                4900-4919 (S-Misspec-lognormal)
Families: S-Last Cox and log-normal AFT; S-Sparse Cox; S-Misspec log-normal AFT. Per family:
  b2s_<loss>_fullval   PRIMARY comparator: B2-S tuned and early-stopped on the whole validation set (Rounds 12-17)
  b2s_<loss>_select    B2-S on val_select only (takes part in the decision; secondary comparator, = Round 18's b2s_<loss>)
  b2s_anchor_<loss>_refit   ConfirmRefitAnchor (src/kanfae/models/fres.py): decision on val_confirm; not anchored ->
                       b2s_<loss>_fullval exactly; anchored -> the chosen state refitted on the whole validation set
  b2s_anchor_<loss>_confirm_r18  the Round-18 estimator from the same fit (secondary)
  landmark_cox (S-Last, S-Sparse)
Development mode: margin 0; both refit variants ("retrain", "finetune") and the Round-18 anchored model are evaluated
for the step-(1) state in every replicate, with both confirmation losses, so every (variant, margin) can be evaluated
offline. Formal mode: the D147 variant and margins. Output: $R19_OUT.
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
from kanfae.models.fres import ConfirmRefitAnchor  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
OUT = Path(os.environ.get("R19_OUT", ROOT / "reports/round19/dev/results_dev.jsonl"))
FAMILY = "lognormal"
SETTINGS = {"dev": dict(margin={"cox": 0.0, "aft": 0.0}, refit="retrain"),
            "formal": dict(margin={"cox": None, "aft": None}, refit=None)}          # formal values: D147
DEV = {"devlast": range(4400, 4420), "devsparse": range(4500, 4520), "devmisspec": range(4600, 4620)}
FORMAL = {"last": range(4700, 4720), "sparse": range(4800, 4820), "misspec": range(4900, 4920)}
LOSSES = {"last": ("cox", "aft"), "sparse": ("cox",), "misspec": ("aft",)}
SCEN = R17.SCEN


def jobs(mode="formal"):
    S = DEV if mode == "dev" else FORMAL
    return [f"{k}_{S[k][i]}" for i in range(20) for k in S]


def job(kind, seed, mode):
    ctx = R17.Ctx17(SCEN[kind], seed)
    xg = ctx.front("bspline")["xg"]
    last = ctx.last()
    va_sel, va_conf = R18.split_val(ctx.va, ctx.dat.delta, seed)
    tr, va, vs, te = ctx.pack(ctx.tr, xg), ctx.pack(ctx.va, xg), ctx.pack(va_sel, xg), ctx.pack(ctx.te, xg)
    st = SETTINGS[mode]
    if st["refit"] is None or any(v is None for v in st["margin"].values()):
        raise RuntimeError("formal refit variant and margins are fixed in D147 before any formal seed is used")
    rec = dict(job=kind, scenario=SCEN[kind] + ("-lognormal" if kind == "misspec" else ""), seed=seed, mode=mode,
               n_train=len(ctx.tr), n_val_select=len(va_sel), n_val_confirm=len(va_conf), n_test=len(ctx.te),
               censored_frac=float(1 - ctx.dat.delta.mean()))
    lt, zt = last[ctx.te], ctx.dat.z[ctx.te]
    models, alts, gates = {}, {}, {}
    for loss in LOSSES[kind]:
        fam = dict(family=FAMILY) if loss == "aft" else {}
        refs = {}
        for name, val in ((f"b2s_{loss}_select", vs), (f"b2s_{loss}_fullval", va)):
            t0 = time.time()
            e = FKANEnsemble(kind="b2s", loss=loss, **fam).fit(tr, val)
            refs[name] = e
            models[name] = R12.survival_metrics(ctx, lambda tt, e=e: e.predict_survival(te["xg"], te["z"], tt),
                                                e.predict_risk(te["xg"], te["z"]))
            models[name].update(lr=e.lr, wd=e.wd, seconds=round(time.time() - t0))
        t0 = time.time()
        m = ConfirmRefitAnchor(loss=loss, margin=st["margin"][loss], refit=st["refit"], **fam).fit(
            dict(tr, last=last[ctx.tr]), dict(vs, last=last[va_sel]), dict(ctx.pack(va_conf, xg), last=last[va_conf]),
            refs[f"b2s_{loss}_select"], refs[f"b2s_{loss}_fullval"])

        def met(members=None, model=m):
            return R12.survival_metrics(ctx, lambda tt: model.predict_survival(te["xg"], lt, zt, tt, members),
                                        model.predict_risk(te["xg"], lt, zt, members))

        def met_gate(g):
            return R12.survival_metrics(ctx, lambda tt: g.predict_survival(te["xg"], lt, zt, tt), g.predict_risk(te["xg"], lt, zt))
        name = f"b2s_anchor_{loss}_refit"
        models[name] = met()
        models[f"b2s_anchor_{loss}_confirm_r18"] = met_gate(m.gate)
        g = m.gate
        alt = {"r18": met_gate(g.anchored_alternative())}
        for how in (("retrain", "finetune") if mode == "dev" else (st["refit"],)):
            alt[how] = models[name] if (not m.reduced and how == st["refit"]) else met(m.refit_members(how))
        alts[name] = alt
        gates[name] = dict(reduced=m.reduced, refit=st["refit"], margin=st["margin"][loss],
                           confirm_loss_ref=g.confirm_loss_ref, confirm_loss_anchored=g.confirm_loss_anchored,
                           select_loss_ref=g.ref_loss, select_loss_anchored=g.anchored_choice["val_loss"],
                           anchored_choice=g.anchored_choice, traj_on=g.residual_on, last_on=g.last_on,
                           n_fine_tunes=g.n_fine_tunes, seconds=round(time.time() - t0))
    if kind != "misspec":
        X = np.hstack([last, ctx.dat.z])
        y, d = ctx.dat.y, ctx.dat.delta.astype(float)
        lc = LinearCox().fit(X[ctx.tr], y[ctx.tr], d[ctx.tr])
        models["landmark_cox"] = R12.survival_metrics(ctx, lambda tt: lc.predict_survival(X[ctx.te], tt),
                                                      lc.predict_risk(X[ctx.te]))
    rec.update(models=models, alternatives=alts, gates=gates)
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
    mode = os.environ.get("R19_MODE", "dev")
    for j in sys.argv[1:]:
        r = run_job(j, mode)
        print(j, r["status"], r["wall_seconds"], r.get("error", ""))
