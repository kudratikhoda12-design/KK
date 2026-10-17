"""Render reports/final_research_prototype_report.md from reports/report_template.md.

Placeholders: {{T:name}} -> a Markdown table built from saved results; {{V:key}} -> a single value.
Every number in tables and placeholders comes from reports/tables/*.csv, reports/test_results*.csv,
reports/logs/*.json(l) or logs; nothing is typed by hand. Unknown placeholders raise an error.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
S = pd.read_csv(REP / "tables/summary_by_model.csv")
STATUS = pd.read_csv(REP / "tables/run_status_counts.csv")
TL = pd.read_csv(REP / "test_results_latest.csv").fillna("")
TA = pd.read_csv(REP / "test_results.csv").fillna("")
LABEL = {"fpca_aft": "FPCA + AFT", "fpca_cox": "FPCA + Cox", "sum_aft": "Summary + AFT", "sum_cox": "Summary + Cox",
         "sum_rsf": "Summary + RSF", "km_null": "KM null (reference)", "oracle_true": "True model (reference)"}
ORDER = list(LABEL)


def ms(r, m, d=3):
    mu, sd = r.get(f"{m}__mean", np.nan), r.get(f"{m}__std", np.nan)
    if pd.isna(mu):
        return "–"
    return f"{mu:.{d}f} ± {sd:.{d}f}" if not pd.isna(sd) else f"{mu:.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def model_table(ds, setting, cols):
    sub = S[(S.dataset == ds) & (S.setting == setting)]
    rows = []
    for m in ORDER:
        r = sub[sub.model == m]
        if r.empty:
            continue
        r = r.iloc[0].to_dict()
        n = int(r.get(f"{cols[0][1]}__count", 0))
        rows.append([LABEL[m]] + [ms(r, c, d) for _, c, d in cols] + [n])
    return md_table(["Model"] + [h for h, _, _ in cols] + ["n"], rows)


SYN_COLS = [("Curve error ↓", "curve_error", 3), ("IBS ↓", "ibs", 3), ("Uno C ↑", "uno_c", 3),
            ("Harrell C ↑", "harrell_c", 3), ("D-cal p", "dcal_p", 2)]


def criteria_table():
    rows = []
    for crit, g in TL.groupby("criterion", sort=False):
        if not crit:
            continue
        att = TA[TA.criterion == crit]
        outcomes = g.outcome.tolist()
        st = "FAIL" if "failed" in outcomes else ("PASS" if all(o == "passed" for o in outcomes) else
                                                   ("PASS (+skip)" if "passed" in outcomes else "SKIPPED"))
        n_fail_hist = int((att.outcome == "failed").sum())
        rows.append([g.stage.iloc[0], crit, "/".join(sorted(set(g.kind))), len(g), st, n_fail_hist])
    rows.sort(key=lambda r: ("ABCDEFG".index(r[0]) if r[0] in "ABCDEFG" else 9, r[1]))
    return md_table(["Stage", "Criterion", "Type", "Test cases (latest)", "Latest status", "Failed attempts (history)"], rows)


def failures_table():
    f = TA[TA.outcome == "failed"]
    rows = [[r.run_id, r.criterion, r.nodeid.split("::")[-1][:60], r.message.replace("|", "/").replace("\n", " ")[:150]]
            for r in f.itertuples()]
    return md_table(["Test run", "Criterion", "Test", "Error (truncated)"], rows)


def run_counts_table():
    runs = TA.groupby("run_id", sort=False).outcome.value_counts().unstack(fill_value=0)
    rows = [[k, int(r.get("passed", 0)), int(r.get("failed", 0)), int(r.get("skipped", 0))] for k, r in runs.iterrows()]
    return md_table(["pytest run (KANFAE_RUN_ID)", "passed", "failed", "skipped"], rows)


def status_table():
    p = STATUS.pivot_table(index=["analysis", "model", "record_set"], columns="status", values="n", aggfunc="sum",
                           fill_value=0)
    rows = [[a, LABEL.get(m, m), rs] + [int(r.get(c, 0)) for c in ("ok", "diverged", "failed")]
            for (a, m, rs), r in p.iterrows()]
    return md_table(["Analysis", "Model", "Record set", "ok", "diverged", "failed"], rows)


def runtime_table():
    rows = []
    for ds in ["synthetic", "cmapss_FD001", "pbc"]:
        for m in ORDER[:5]:
            sub = S[(S.dataset == ds) & (S.model == m)]
            if sub.empty:
                continue
            rows.append([ds, LABEL[m], f"{sub['fit_seconds__mean'].mean():.2f}", f"{sub['n_params__mean'].mean():.0f}"])
    return md_table(["Dataset", "Model", "Mean fit time per final run (s)", "Mean parameter count*"], rows)


def cmapss_table(analysis):
    cols = [("IBS ↓", "ibs", 3), ("Harrell C ↑", "harrell_c", 3), ("RUL RMSE internal ↓", "rul_test_rmse", 1),
            ("RUL MAE internal ↓", "rul_test_mae", 1), ("RUL RMSE official ↓", "rul_official_rmse", 1),
            ("n official", "rul_official_n", 0)]
    return model_table("cmapss_FD001", analysis, cols)


def cmapss_bands():
    sub = S[(S.dataset == "cmapss_FD001") & (S.setting == "cmapss_B")]
    rows = []
    for m in ORDER:
        r = sub[sub.model == m]
        if r.empty:
            continue
        r = r.iloc[0].to_dict()
        rows.append([LABEL[m]] + [ms(r, f"band_{b}_rmse", 1) for b in ("30_100", "100_200", "200_999")]
                    + [ms(r, f"band_{b}_harrell_c", 3) for b in ("30_100", "100_200", "200_999")]
                    + [ms(r, "rul_test_rmse_engine_weighted", 1)])
    return md_table(["Model", "RMSE s∈[30,100)", "RMSE s∈[100,200)", "RMSE s≥200", "C s∈[30,100)", "C s∈[100,200)",
                     "C s≥200", "Engine-weighted RMSE"], rows)


def pbc_table(analysis):
    cols = [("IBS (marginal KM) ↓", "ibs", 3), ("IBS (conditional IPCW) ↓", "ibs_conditional_ipcw", 3),
            ("Uno C ↑", "uno_c", 3), ("Harrell C ↑", "harrell_c", 3), ("D-cal p", "dcal_p", 2)]
    return model_table("pbc", analysis, cols)


def recon_table():
    sub = S[(S.dataset == "synthetic") & (S.model == "fpca_aft")]
    rows = []
    for st in ["P0", "P-c0", "P-c60", "P-sparse", "P-V2", "P-lognormal"]:
        r = sub[sub.setting == st]
        if r.empty:
            continue
        r = r.iloc[0].to_dict()
        rows.append([st, ms(r, "rec_rmse_ind_smoother"), ms(r, "rec_rmse_ind_fpca"), ms(r, "rec_rmse_obs_smoother"),
                     ms(r, "rec_rmse_obs_fpca")])
    return md_table(["Setting", "Smoother, individual-weighted", "FPCA, individual-weighted", "Smoother, observation-weighted",
                     "FPCA, observation-weighted"], rows)


def selected_table():
    sel = pd.read_csv(REP / "tables/selected_configs.csv")
    def clean(c):
        c = re.sub(r"np\.int64\((\d+)\)", r"\1", str(c))
        items = re.findall(r"'(\{[^']*\})': (\d+)", c)
        return "; ".join(f"{cfg} ×{n}" for cfg, n in items) if items else c
    rows = [[r.setting if r.analysis == "synthetic_baselines" else r.analysis, LABEL.get(r.model, r.model), clean(r.config)]
            for r in sel.itertuples() if r.model in ORDER[:5]]
    return md_table(["Analysis/setting", "Model", "Selected configuration: count over replicates/splits"], rows)


def visits_table():
    df = pd.DataFrame([json.loads(l) for l in (REP / "experiment_results.jsonl").read_text().splitlines()])
    d = df[df.phase == "descriptive"]
    rows = [[r.analysis, int(r.n_cohort), int(r.events), f"{r.visits_median:.0f} ({int(r.visits_min)}–{int(r.visits_max)})",
             f"{100 * r.frac_le2_visits:.0f}%", f"{r.hr_per_visit:.3f} ({r.hr_lo:.3f}–{r.hr_hi:.3f})", f"{r.p:.3f}"]
            for r in d.itertuples()]
    return md_table(["Endpoint", "Cohort", "Events", "Visits ≤ 730 d: median (range)", "≤ 2 visits",
                     "HR per visit (95% CI)", "p"], rows)


def env_table():
    lock = dict(l.split("==") for l in (ROOT / "requirements-lock.txt").read_text().split() if "==" in l)
    keys = ["torch", "numpy", "scipy", "pandas", "scikit-learn", "lifelines", "scikit-survival", "matplotlib", "rdata",
            "pytest", "hypothesis"]
    return md_table(["Package", "Version (lock file)"], [[k, lock.get(k, "?")] for k in keys])


TABLES = {
    "criteria": criteria_table, "failures": failures_table, "run_counts": run_counts_table, "status": status_table,
    "runtime": runtime_table, "recon": recon_table, "selected": selected_table, "visits": visits_table,
    "env": env_table, "cmapss_A": lambda: cmapss_table("cmapss_A"), "cmapss_B": lambda: cmapss_table("cmapss_B"),
    "cmapss_A_art": lambda: cmapss_table("cmapss_A_artificial"), "cmapss_bands": cmapss_bands,
    "pbc_E1": lambda: pbc_table("pbc_E1"), "pbc_E2": lambda: pbc_table("pbc_E2"),
}
for st in ["P0", "P-c0", "P-c60", "P-sparse", "P-V2", "P-lognormal"]:
    TABLES[f"syn_{st}"] = (lambda st=st: model_table("synthetic", st, SYN_COLS))


def gate_vals():
    v = {}
    for a in (1, 2, 3):
        txt = (REP / f"logs/stageD_gate_attempt{a}.log").read_text()
        m = re.search(r"R2: C = ([0-9.]+), oracle C = ([0-9.]+); probe R2 \(a,b,A,omega\) = \[([^\]]+)\], mean ([0-9.]+)", txt)
        v[f"R2C_{a}"], v["R2_oracle"], v[f"R2probe_{a}"], v[f"R2probe_mean_{a}"] = m.group(1), m.group(2), m.group(3), m.group(4)
        m1 = re.search(r"R1: C = ([0-9.]+), oracle C = ([0-9.]+), sigma = ([0-9.]+)", txt)
        v["R1C"], v["R1sigma"] = m1.group(1), m1.group(3)
        m3 = re.search(r"R3: survival-curve error aware \[([^\]]+)\] naive \[([^\]]+)\]", txt)
        v[f"R3aware_{a}"], v[f"R3naive_{a}"] = m3.group(1), m3.group(2)
        v[f"R3dC_{a}"] = re.search(r"\|C - C_R2\| = ([0-9.]+)", txt).group(1)
    v["R2_threshold"] = f"{float(v['R2_oracle']) - 0.03:.4f}"
    return v


def values():
    v = gate_vals()
    T1 = TA[TA.run_id == "final_full_suite"]              # round-1 text quotes ITS final full-suite run (round 3)
    T1 = T1 if len(T1) else TL
    v["n_tests_latest"] = str(len(T1))
    for o in ("passed", "failed", "skipped"):
        v[f"latest_{o}"] = str(int((T1.outcome == o).sum()))
    v["n_test_runs"] = str(TA.run_id.nunique())
    v["n_test_records"] = str(len(TA))
    df = pd.read_csv(REP / "experiment_results.csv", low_memory=False)
    v["n_exp_records"] = str(len(df))
    v["n_exp_not_ok"] = str(int((df.status != "ok").sum()))
    v["git_head"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                   text=True).stdout.strip()
    cm = json.loads((REP / "logs/cmapss_inspection.json").read_text())["FD001"]
    v["cm_life_min"], v["cm_life_med"], v["cm_life_max"] = (str(cm["train_life"][k]) for k in ("min", "median", "max"))
    v["cm_tlen_min"], v["cm_tlen_max"] = str(cm["test_length"]["min"]), str(cm["test_length"]["max"])
    v["cm_const"] = ", ".join(cm["near_constant_sd_lt_1em6"])
    return v


def main():
    tpl = (REP / "report_template.md").read_text()
    vals = values()

    def rep_t(m):
        k = m.group(1)
        if k not in TABLES:
            raise KeyError(f"unknown table {k}")
        return TABLES[k]()

    def rep_v(m):
        k = m.group(1)
        if k.startswith("S|"):                      # S|setting|model|metric[|stat]  -> value from summary table
            parts = k.split("|")
            _, st, mo, me = parts[:4]
            stat = parts[4] if len(parts) > 4 else "mean"
            r = S[(S.setting == st) & (S.model == mo)]
            if r.empty or f"{me}__{stat}" not in r:
                raise KeyError(f"no summary value for {k}")
            v = float(r.iloc[0][f"{me}__{stat}"])
            d = 1 if ("rmse" in me or "mae" in me or me.endswith("_n") or "not_reached" in me) else (0 if stat == "count" else 3)
            return f"{v:.{d}f}"
        if k not in vals:
            raise KeyError(f"unknown value {k}")
        return vals[k]

    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", rep_t, tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-|.]+)\}\}", rep_v, out)
    (REP / "final_research_prototype_report.md").write_text(out)
    (REP / "report_values.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(out.splitlines()), "lines")


if __name__ == "__main__":
    main()
