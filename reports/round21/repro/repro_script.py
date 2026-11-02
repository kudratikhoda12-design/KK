import json, sys, time
sys.path.insert(0, "/home/user/KK/src"); sys.path.insert(0, "/home/user/KK/scripts")
import os
os.environ["R13_OUT"] = sys.argv[1] + "/unused.jsonl"
import run_round13 as R13
ref = {}
for i in range(1, 5):
    for l in open(f"/home/user/KK/reports/round13/results_lane{i}.jsonl"):
        r = json.loads(l)
        if r.get("job") == "single" and r.get("dataset") == "pbc_E2" and r.get("split") == 0 and r.get("model") in ("landmark_cox", "b2s_cox"):
            ref[r["model"]] = r
out = {}
for model in ("landmark_cox", "b2s_cox"):
    ctx = R13.RealCtx("pbc_E2", 0)
    rec = dict(model=model); t0 = time.time()
    R13.run_single_cell(ctx, model, rec)
    keys = sorted(rec["metrics"])
    diffs = {k: abs(rec["metrics"][k] - ref[model]["metrics"][k]) for k in keys}
    out[model] = dict(new=rec["metrics"], r13=ref[model]["metrics"], absdiff=diffs, max_absdiff=max(diffs.values()),
                      lr_wd=(rec.get("lr"), rec.get("wd"), ref[model].get("lr"), ref[model].get("wd")), seconds=round(time.time()-t0))
    print(model, out[model]["max_absdiff"], out[model]["seconds"], flush=True)
json.dump(out, open(sys.argv[1] + "/repro_pbcE2_s0.json", "w"), indent=1)
