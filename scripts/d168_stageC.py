"""D168 Stage C / D170: the confirmatory paired comparison B2-S Cox (three-member ensemble) against the frozen mgcv
reference AFCM. Runs only after the Stage-B gate passed (reports/d168/stageB/result.json, verdict PASS) and after the
D170 pre-registration is committed.

Replicate s in 6100..6209 (110): the Round-12 design exactly (scripts/run_round12.Ctx with seed s: S-Add, DGP v2,
n 1,000, data seed s, split_groups 60/20/20 stratified on the event with split seed s, B-spline front end).
Identical data for both methods: the replicate's training/validation/test arrays (smoothed curves on the grid, z, y,
delta, indices, true test curves) are written once to reports/d168/stageC/data/rep_<s>.npz (git-ignored; SHA-256 in
the record), read back, checked equal to the arrays each method uses, and both methods are fitted from them.
  b2s_cox    scripts/run_round12.run_ensemble(ctx, "b2s", "cox") (the Round-12 / H-R12 code path, unchanged)
  reference  kanfae.models.afcm_mgcv.AFCMReference on the training subjects (REML; validation subjects not used)
Metrics for both: identifiable and raw RISE of phi_1, phi_2 and their mean (primary: mean identifiable RISE), and
survival-curve error, IBS, Harrell C, Uno C, D-calibration p (scripts/run_round12.survival_metrics).
A method's replicate counts as FAILED if it raises, returns non-finite predictions or surfaces, or (reference) does not
report "full convergence" of mgcv's REML optimisation. Every record is kept; nothing is refitted.
Usage: python scripts/d168_stageC.py --lane k --lanes 4   (resumable: seeds with a final record are skipped)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
os.environ.setdefault("R12_OUT", str(ROOT / "reports/d168/stageC/unused_r12.jsonl"))
SEEDS = list(range(6100, 6210))
BASE = ROOT / "reports/d168/stageC"
DATA = BASE / "data"
PREREG = ROOT / "docs/d170_stageC_preregistration.md"
FROZEN = ("scripts/afcm_mgcv.R", "src/kanfae/models/afcm_mgcv.py", "scripts/d168_stageC.py",
          "scripts/analyze_d168_stageC.py")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def guards():
    b = BASE.parent / "stageB/result.json"
    if not b.exists() or json.loads(b.read_text()).get("verdict") != "PASS":
        raise SystemExit("Stage B has not passed; Stage C is not run (D168).")
    rel = [str(PREREG.relative_to(ROOT)), *FROZEN]
    tracked = all(subprocess.run(["git", "ls-files", "--error-unmatch", f], cwd=ROOT, capture_output=True).returncode == 0
                  for f in rel)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *rel], cwd=ROOT, capture_output=True, text=True).stdout
    if not tracked or dirty.strip():
        raise SystemExit("D170 pre-registration or a frozen file is not committed; Stage C not run.")
    text = PREREG.read_text()
    for f in FROZEN[:2]:
        if sha_file(ROOT / f) not in text:
            raise SystemExit(f"{f} differs from the hash registered in D170; Stage C not run.")


def done():
    seen = set()
    for f in BASE.glob("results_lane*.jsonl"):
        for l in f.read_text().splitlines():
            if l.strip():
                seen.add(json.loads(l)["seed"])
    return seen


def export(ctx, xg, s):
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / f"rep_{s}.npz"
    d = ctx.dat
    np.savez_compressed(p, xg=xg, z=d.z, y=d.y, delta=d.delta, tr=ctx.tr, va=ctx.va, te=ctx.te, x_te=ctx.X_te)
    return p


def load_check(p, ctx, xg):
    a = np.load(p)
    d = ctx.dat
    same = all(np.array_equal(a[k], v) for k, v in (("xg", xg), ("z", d.z), ("y", d.y), ("delta", d.delta), ("tr", ctx.tr),
                                                   ("va", ctx.va), ("te", ctx.te), ("x_te", ctx.X_te)))
    if not same:
        raise RuntimeError("exported data differ from the arrays in use")
    return a


def finite_metrics(m):
    return all(np.isfinite(v) for k, v in m.items() if k != "shape" and isinstance(v, (int, float)))


def run_seed(s):
    import run_round12 as R12
    from kanfae.models.afcm_mgcv import AFCMReference
    rec = dict(stage="C", decision="D170", seed=s, design="Round-12 S-Add, n 1000, data and split seed s")
    t_all = time.time()
    try:
        ctx = R12.Ctx("S-Add", 0, None, dev_seed=s)
        xg = ctx.front("bspline")["xg"]
        p = export(ctx, xg, s)
        a = load_check(p, ctx, xg)
        rec.update(data_sha256=sha_file(p), n_train=len(ctx.tr), n_val=len(ctx.va), n_test=len(ctx.te),
                   events_train=int(ctx.d_tr.sum()), tied_event_times_train=int(ctx.d_tr.sum() -
                                                                                len(np.unique(ctx.y_tr[ctx.d_tr.astype(bool)]))))
        xa = a["xg"]
        pk = lambda idx: dict(xg=xa[idx], z=a["z"][idx], y=a["y"][idx], d=a["delta"][idx].astype(float))  # noqa: E731
        # B2-S Cox ensemble (Round-12 code path; its packs are the ctx arrays, verified identical to the export above)
        t0 = time.time()
        b = dict(status="ok")
        try:
            r = {}
            R12.run_ensemble(ctx, r, "b2s", "cox", xg=xa, shapes="functional")
            b.update(metrics=r["metrics"], lr=r["lr"], wd=r["wd"], epochs=r["epochs"])
            if not (finite_metrics(r["metrics"]) and np.isfinite(r["metrics"]["shape"]["mean_id"])):
                b["status"] = "failed"; b["error"] = "non-finite metrics"
        except Exception as e:
            b.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
        b["seconds"] = round(time.time() - t0, 1)
        rec["b2s_cox"] = b
        # mgcv reference AFCM on the same exported training data
        t0 = time.time()
        c = dict(status="ok")
        try:
            ref = AFCMReference().fit(pk(a["tr"]), pk(a["te"]))
            c["info"] = ref.info
            if not ref.ok:
                c.update(status="failed", error=ref.info.get("error"))
            elif not ref.finite:
                c.update(status="failed", error="non-finite surfaces or predictions")
            elif ref.info.get("outer_conv") != "full convergence":
                c.update(status="failed", error=f"not converged: {ref.info.get('outer_conv')}")
            if ref.ok:
                m = R12.survival_metrics(ctx, ref.predict_survival_test, ref.predict_risk_test())
                m["shape"] = R12.shape_metrics(ctx, ref.effect)
                c["metrics"] = m
                if c["status"] == "ok" and not (finite_metrics(m) and np.isfinite(m["shape"]["mean_id"])):
                    c.update(status="failed", error="non-finite metrics")
        except Exception as e:
            c.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
        c["seconds"] = round(time.time() - t0, 1)
        rec["reference"] = c
        rec["status"] = "ok"
    except Exception as e:
        rec.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
    rec["seconds_total"] = round(time.time() - t_all, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", type=int, required=True)
    ap.add_argument("--lanes", type=int, default=4)
    a = ap.parse_args()
    guards()
    out = BASE / f"results_lane{a.lane}.jsonl"
    have = done()
    for s in SEEDS[a.lane::a.lanes]:
        if s in have:
            continue
        rec = run_seed(s)
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        print(f"[lane {a.lane}] {s} {rec['status']} b2s={rec.get('b2s_cox', {}).get('status')} "
              f"ref={rec.get('reference', {}).get('status')} {rec['seconds_total']}s", flush=True)


if __name__ == "__main__":
    main()
