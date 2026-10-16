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
if "code_version" not in df:
    df["code_version"] = np.nan
df["used_in_summary"] = True
for (an, mo), ver in SUPERSEDED.items():
    sel = (df.analysis == an) & (df.model == mo)
    if (sel & (df.code_version == ver)).any():
        df.loc[sel & (df.code_version != ver), "used_in_summary"] = False
df.drop(columns=[c for c in ("traceback",) if c in df], errors="ignore").to_csv(ROOT / "reports/experiment_results.csv", index=False)
df["record_set"] = np.where(df.used_in_summary, "current", "superseded (D37)")
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
