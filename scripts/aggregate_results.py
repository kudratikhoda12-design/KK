"""Aggregate reports/experiment_results.jsonl into reports/experiment_results.csv and summary tables.

Stochastic models: metrics are averaged over seeds within a replicate/split first, then mean/SD over
replicates/splits. Every record (tuning, final, failed, diverged) is kept in experiment_results.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
recs = [json.loads(l) for l in (ROOT / "reports/experiment_results.jsonl").read_text().splitlines() if l.strip()]
df = pd.DataFrame(recs)
df.drop(columns=[c for c in ("traceback",) if c in df], errors="ignore").to_csv(ROOT / "reports/experiment_results.csv", index=False)


# Superseded records (D37): for re-run (analysis, model) pairs only the re-run's records enter the summaries;
# all records remain in experiment_results.csv as history.
SUPERSEDED = {("cmapss_A", "sum_cox"): "D37", ("cmapss_A_artificial", "sum_cox"): "D37"}
# D62: Option B re-run as code version v3 (adds engine-cluster bootstrap CIs; same models, splits, tuning).
SUPERSEDED.update({("cmapss_B", m): "v3" for m in ("fpca_aft", "fpca_cox", "sum_aft", "sum_cox", "sum_rsf", "km_null")})
if "code_version" not in df:
    df["code_version"] = np.nan
df["used_in_summary"] = True
for (an, mo), ver in SUPERSEDED.items():
    sel = (df.analysis == an) & (df.model == mo)
    for ds in df.loc[sel, "dataset"].unique():          # per dataset: only once that dataset has the re-run
        sd = sel & (df.dataset == ds)
        if (sd & (df.code_version == ver)).any():
            df.loc[sd & (df.code_version != ver), "used_in_summary"] = False
df.drop(columns=[c for c in ("traceback",) if c in df], errors="ignore").to_csv(ROOT / "reports/experiment_results.csv", index=False)
df["record_set"] = np.where(df.used_in_summary, "current", "superseded (re-run: D37 / D62)")
status = df.groupby(["analysis", "model", "record_set", "phase", "status"]).size().rename("n").reset_index()
status.to_csv(OUT / "run_status_counts.csv", index=False)
fin = df[(df.phase == "final") & df.used_in_summary].copy()
metric_cols = [c for c in fin.columns if c not in ("run_id", "dataset", "analysis", "setting", "replicate", "model",
                                                     "config", "seed", "phase", "status", "error", "data_kind",
                                                     "fit_info", "split_seed", "data_seed", "used_in_summary", "code_version", "record_set")
               and pd.api.types.is_numeric_dtype(fin[c])]
ok = fin[fin.status == "ok"]
per_rep = ok.groupby(["dataset", "analysis", "setting", "model", "replicate"])[metric_cols].mean().reset_index()
agg = per_rep.groupby(["dataset", "analysis", "setting", "model"])[metric_cols].agg(["mean", "std", "count"])
agg.columns = [f"{a}__{b}" for a, b in agg.columns]
agg = agg.reset_index()
agg.to_csv(OUT / "summary_by_model.csv", index=False)
per_rep.to_csv(OUT / "per_replicate.csv", index=False)

# selected configurations
sel = fin[fin.seed == 0].groupby(["analysis", "setting", "model"])["config"].agg(lambda s: dict(s.value_counts()))
sel.reset_index().to_csv(OUT / "selected_configs.csv", index=False)
n_bad = (df.status != "ok").sum()
print(f"{len(df)} records; {len(fin)} final; {n_bad} not ok; tables in {OUT}")

# Round-1 snapshot (owner reporting fix, round 5): the round-1 report part must show round-1 records only, i.e. without
# the C-MAPSS FD003 runs (v2) and the Option-B re-runs (v3), with only the round-1 supersession rule (D37).
r1 = df[(df.dataset != "cmapss_FD003") & ~df.code_version.isin(["v2", "v3"])].copy()
r1["used_in_summary"] = True
for (an, mo), ver in {("cmapss_A", "sum_cox"): "D37", ("cmapss_A_artificial", "sum_cox"): "D37"}.items():
    sel = (r1.analysis == an) & (r1.model == mo)
    if (sel & (r1.code_version == ver)).any():
        r1.loc[sel & (r1.code_version != ver), "used_in_summary"] = False
r1["record_set"] = np.where(r1.used_in_summary, "current", "superseded (D37)")
r1.groupby(["analysis", "model", "record_set", "phase", "status"]).size().rename("n").reset_index().to_csv(
    OUT / "r1_run_status_counts.csv", index=False)
f1 = r1[(r1.phase == "final") & r1.used_in_summary & (r1.status == "ok")]
pr1 = f1.groupby(["dataset", "analysis", "setting", "model", "replicate"])[metric_cols].mean().reset_index()
a1 = pr1.groupby(["dataset", "analysis", "setting", "model"])[metric_cols].agg(["mean", "std", "count"])
a1.columns = [f"{a}__{b}" for a, b in a1.columns]
a1.reset_index().to_csv(OUT / "r1_summary_by_model.csv", index=False)
r1f = r1[(r1.phase == "final") & r1.used_in_summary]
r1f[r1f.seed == 0].groupby(["analysis", "setting", "model"])["config"].agg(lambda s: dict(s.value_counts())) \
    .reset_index().to_csv(OUT / "r1_selected_configs.csv", index=False)       # round 6: R1.9 from the snapshot
print(f"round-1 snapshot: {len(r1)} records ({int((r1.status != 'ok').sum())} not ok)")
