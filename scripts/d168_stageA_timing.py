"""D168 Stage A, step A8: run time of one Stage-C replicate on the Stage-C design, on DEVELOPMENT seed 11 only.

Design as Round 12 / H-R12 (scripts/run_round12.Ctx with dev_seed=11: S-Add, DGP v2, n 1,000, split_groups 60/20/20
stratified on the event, B-spline front end): the B2-S Cox three-member ensemble exactly as in Round 12
(run_round12.run_ensemble) and the mgcv reference AFCM on the same exported training curves. Writes one record to
reports/d168/stageA/timing.jsonl. Not a formal seed; its RISE values are not used for any decision.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
os.environ.setdefault("R12_OUT", str(ROOT / "reports/d168/stageA/unused_r12_dev.jsonl"))
import run_round12 as R12  # noqa: E402
from kanfae.models.afcm_mgcv import AFCMReference  # noqa: E402

OUT = ROOT / "reports/d168/stageA/timing.jsonl"


def main():
    t_all = time.time()
    ctx = R12.Ctx("S-Add", 0, None, dev_seed=11)
    t0 = time.time()
    xg = ctx.front("bspline")["xg"]
    t_front = time.time() - t0
    rec = dict(dataset="stageC_design_dev11", seed=11, n=R12.N, n_train=len(ctx.tr), n_test=len(ctx.te),
               role="development timing only")
    t0 = time.time()
    r = {}
    R12.run_ensemble(ctx, r, "b2s", "cox", xg=xg, shapes="functional")
    rec["b2s_cox"] = dict(seconds=round(time.time() - t0, 1), mean_id=r["metrics"]["shape"]["mean_id"])
    t0 = time.time()
    ref = AFCMReference().fit(ctx.pack(ctx.tr, xg), ctx.pack(ctx.te, xg))
    sm = R12.shape_metrics(ctx, ref.effect) if ref.ok else None
    rec["reference"] = dict(seconds=round(time.time() - t0, 1), status=ref.info.get("status"),
                            outer_conv=ref.info.get("outer_conv"), mean_id=sm["mean_id"] if sm else None)
    rec["front_seconds"] = round(t_front, 1)
    rec["seconds_total"] = round(time.time() - t_all, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(json.dumps(rec, default=float))


if __name__ == "__main__":
    main()
