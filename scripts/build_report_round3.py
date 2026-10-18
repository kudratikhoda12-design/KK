"""Assemble the final report, revision 3: round-3 part (reports/report_template_round3.md), then the round-2 and
round-1 reports (historical records, regenerated from the same records; headings renumbered R2.x / R1.x).

Placeholders: {{T:name}} table, {{V:key}} value, {{F:path}} embedded document. Unknown placeholders raise an error.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table, f = R1.md_table, R2.f
GATE_RE = D.GATE_RE


def _log(name):
    return (REP / "logs" / name).read_text()


# ---------------------------------------------------------------- revision 4
def diag_freq():
    d = json.loads((REP / "diag_frequency_information.json").read_text())
    rows = []
    for view in ("truth", "smoothed_K12"):
        for k, lab in (("fourier_power", "Fixed Fourier power"), ("deriv_ratio", "Derivative-energy ratio (A4-type)"),
                       ("quadrature_projection", "Quadrature-pair projection spectrum (new)")):
            r = d[view][k]
            rows.append([view, lab, f(r["linear_probe_R2_heldout"], 3), f(r.get("estimate_corr"), 3),
                         f(r["error_variance_explained_by_phase"], 3)])
    return md_table(["Curves", "Feature", "Linear probe R² for ω (held out)", "corr(estimate, ω)",
                     "Share of estimate error explained by phase"], rows)


def fq_tests():
    TA, TL = R1.TA, R1.TL
    rows = []
    for _, r in TL.iterrows():
        if not str(r.criterion).startswith("FQ"):
            continue
        h = TA[TA.nodeid == r.nodeid]
        rows.append([r.criterion, r.nodeid.split("::")[-1][:70], r.kind, len(h), int((h.outcome == "failed").sum()),
                     r.outcome.upper()])
    return md_table(["Criterion", "Test", "Kind", "Attempts", "Failed attempts", "Latest"], rows)


def r4_search():
    rows = []
    for r in D.jsonl("learnability_search_fa.jsonl"):
        c = r["config"]
        cfg = ", ".join(f"{k}={v}" for k, v in c.items() if k in ("front", "r", "alpha") or k.startswith("fa_"))
        rows.append([r["step"], r["kind"], cfg, r["seed"], f(r["train_C"]), f(r["val_C"]),
                     f(r["train_C"] - r["val_C"]), f(r["val_C"] - r["val_oracle_C"]),
                     " / ".join(f(p, 2) for p in r["val_probe"]), f(r["val_probe_mean"], 3), r["epochs"][0]])
    return md_table(["Step", "Model", "Configuration (other settings = A3 selection)", "Seed", "Train C", "Val C",
                     "Train − val C", "Val C − oracle", "Val probe a / b / A / ω", "Probe mean", "Epochs"], rows)


def r4_gate():
    log = _log("revised_gate_fa.log")
    r1 = re.search(r"R1: C = ([0-9.]+), oracle C = ([0-9.]+), sigma = ([0-9.]+)", log)
    m = GATE_RE.search(log)
    r3 = re.search(r"R3: survival-curve error aware \[([^\]]+)\] naive \[([^\]]+)\]", log)
    dc = re.search(r"\|C - C_R2\| = ([0-9.]+)", log).group(1)
    v = json.loads((REP / "revised_gate_verdict.json").read_text())
    C1, o1, s1 = float(r1.group(1)), float(r1.group(2)), float(r1.group(3))
    C2, o2 = float(m.group(1)), float(m.group(2))
    rows = [["R1 (oracle-feature head; representation not involved)", f"C {C1:.4f} ≥ {o1 - 0.02:.4f}; |σ̂ − 0.5| = "
             f"{abs(s1 - 0.5):.4f} ≤ 0.05", "PASS" if v["gate"]["R1"] else "FAIL"],
            ["R2 (fae_mlp end to end)", f"C {C2:.4f} vs ≥ {o2 - 0.03:.4f}; probe a / b / A / ω = "
             f"{' / '.join(m.group(3).split())}, mean {m.group(4)} vs ≥ 0.8", "PASS" if v["gate"]["R2"] else "FAIL"],
            ["R3 (censoring awareness)", f"aware [{r3.group(1)}] vs naive [{r3.group(2)}]; |C − C_R2| = {dc} ≤ 0.02",
             "PASS" if v["gate"]["R3"] else "FAIL"]]
    return md_table(["Criterion (seed 41, fresh)", "Values vs unchanged thresholds", "Outcome"], rows)


def r4_confirm():
    rows = []
    rs = D.jsonl("learnability_confirmation_fae_mlp_fa.jsonl")
    for r in rs:
        rows.append([r["data_seed"], f(r["test_C"]), f(r["threshold"]), f(r["test_C"] - r["threshold"]),
                     " / ".join(f(p, 3) for p in r["probe"]), f(r["probe_mean"], 3), "PASS" if r["pass"] else "FAIL"])
    rows.append(["**Total**", "", "", "", "", "", f"**{sum(r['pass'] for r in rs)}/{len(rs)}**"])
    return md_table(["Fresh seed", "Test C", "Threshold", "C − threshold", "Probe a / b / A / ω", "Probe mean", "Pass"],
                    rows)


def diag_fa():
    rows = []
    for r in D.jsonl("diag_fa_overfitting.jsonl"):
        rows.append([r["variant"], r["kind"], f(r["train_C"]), f(r["val_C"]), f(r["train_C"] - r["val_C"]),
                     f(r["val_C"] - r["val_oracle_C"]), " / ".join(f(p, 2) for p in r["val_probe"]),
                     f(np.mean(r["val_probe"]), 3), r["epochs"][0]])
    return md_table(["Diagnostic variant", "Model", "Train C", "Val C", "Train − val", "Val C − oracle",
                     "Val probe a / b / A / ω", "Probe mean", "Epochs"], rows)


def diag_r1():
    rows = []
    for r in D.jsonl("diag_r1_seeds.jsonl"):
        rows.append([r["seed"], f(r["C"]), f(r["oracle_C"]), f(r["sigma"]), f(r.get("sigma_profile_train")), r["epochs"],
                     f(r["eta_fit_R2_train"], 3), "yes" if r["pass_C"] else "no", "yes" if r["pass_sigma"] else "no",
                     "yes" if abs(r.get("sigma_profile_train", np.nan) - 0.5) <= 0.05 else "no"])
    return md_table(["Data seed", "C", "Oracle C", "σ̂ at early stop (truth 0.5)", "Profile σ̂ (η fixed)", "Epochs",
                     "η fit R² (train)", "C criterion", "σ criterion (as registered)",
                     "Profile σ̂ within ±0.05 (information only)"], rows)


def gate_overview():
    rows = []
    for r in D.gate_history():
        rows.append([r["label"], "21", f(r["C"]), f(r["threshold"]), f(r["probe_mean"], 3), f(r["probe"][3], 3), "FAIL"])
    v = json.loads((REP / "revised_gate_verdict.json").read_text())
    rows.append(["Revision 4: frequency-aware front (D61)", "41 (fresh)", f(v["gate_R2_C"]), f(v["gate_R2_threshold"]),
                 f(v["gate_probe_mean"], 3), f(v["gate_probe"][3], 3), "PASS" if v["gate"]["R2"] else "FAIL"])
    return md_table(["Design", "Gate seed", "R2 test C", "Threshold", "Probe mean", "ω probe", "R2"], rows)


# ---------------------------------------------------------------- PBC (E2 primary) and C-MAPSS (B primary)
def pbc_kan(endpoint):
    rows = []
    for r in D.kanaft_pbc():
        if r["endpoint"] != endpoint:
            continue
        li, k = r["linear"]["test"], r["kan"]["test"]
        rows.append([r["split"], r["events_test"], f(li["ibs"]), f(k["ibs"]), f(r["kan_symbolic_test"]["ibs"]),
                     f(li["ibs_conditional"]), f(k["ibs_conditional"]), f(li["uno_c"], 3), f(k["uno_c"], 3)])
    return md_table(["Split", "Test events", "IBS linear AFT", "IBS KAN-AFT", "IBS KAN formulas", "Cond. IBS linear",
                     "Cond. IBS KAN", "Uno C linear", "Uno C KAN"], rows)


CM_COLS = [("RUL RMSE ↓", "rul_test_rmse", 1), ("Engine-weighted RMSE ↓", "rul_test_rmse_engine_weighted", 1),
           ("RUL MAE ↓", "rul_test_mae", 1), ("Harrell C ↑", "rul_test_harrell_c", 3),
           ("Official-test RMSE ↓", "rul_official_rmse", 1)]


def cm_primary(ds):
    return R1.model_table(ds, "cmapss_B", CM_COLS)


def _per_rep():
    pr = pd.read_csv(REP / "tables/per_replicate.csv")
    return pr[(pr.analysis == "cmapss_B")]


def cm_cluster(ds):
    pr = _per_rep()
    pr = pr[pr.dataset == ds]
    rows = []
    for m in R1.ORDER:
        sub = pr[pr.model == m].sort_values("replicate")
        for _, r in sub.iterrows():
            if "rul_test_rmse_cb_lo" not in r or pd.isna(r.get("rul_test_rmse_cb_lo")):
                continue
            rows.append([R1.LABEL[m], int(r.replicate), int(r.get("rul_test_n_engines", np.nan)),
                         f"{r.rul_test_rmse:.1f} [{r.rul_test_rmse_cb_lo:.1f}, {r.rul_test_rmse_cb_hi:.1f}]",
                         f"{r.rul_test_mae:.1f} [{r.rul_test_mae_cb_lo:.1f}, {r.rul_test_mae_cb_hi:.1f}]",
                         f"{r.rul_test_harrell_c:.3f} [{r.rul_test_harrell_c_cb_lo:.3f}, {r.rul_test_harrell_c_cb_hi:.3f}]"])
    return md_table(["Model", "Split", "Test engines", "RMSE [engine-cluster 95% CI]", "MAE [CI]", "Harrell C [CI]"], rows)


def cm_bands(ds):
    pr = _per_rep()
    pr = pr[pr.dataset == ds]
    rows = []
    for m in R1.ORDER:
        sub = pr[pr.model == m]
        if sub.empty:
            continue
        cells = [R1.LABEL[m]]
        for b in ("30_100", "100_200", "200_999"):
            if f"band_{b}_rmse" not in sub or sub[f"band_{b}_rmse"].isna().all():
                cells.append("–")
                continue
            cells.append(f"{sub[f'band_{b}_rmse'].mean():.1f} (splits {sub[f'band_{b}_rmse'].min():.1f}–"
                         f"{sub[f'band_{b}_rmse'].max():.1f}); CI width {np.mean(sub[f'band_{b}_rmse_cb_hi'] - sub[f'band_{b}_rmse_cb_lo']):.1f}"
                         if f"band_{b}_rmse_cb_hi" in sub else f"{sub[f'band_{b}_rmse'].mean():.1f}")
        rows.append(cells)
    return md_table(["Model", "RMSE, landmark s ∈ [30, 100)", "s ∈ [100, 200)", "s ≥ 200"], rows)


def decisions_r3():
    lines = (ROOT / "docs/decisions.md").read_text().splitlines()
    rows = [l for l in lines if re.match(r"\| D(5[7-9]|[6-9]\d) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


TABLES = {"diag_freq": diag_freq, "fq_tests": fq_tests, "r4_search": r4_search, "r4_gate": r4_gate,
          "r4_confirm": r4_confirm, "diag_fa": diag_fa, "diag_r1": diag_r1, "gate_overview": gate_overview,
          "pbc_E2": lambda: R1.pbc_table("pbc_E2"), "pbc_E1": lambda: R1.pbc_table("pbc_E1"),
          "pbc_kan_E2": lambda: pbc_kan("E2"), "pbc_kan_E1": lambda: pbc_kan("E1"),
          "pbc_kan_paired": R2.pbc_kan_paired,
          "cm_B_FD001": lambda: cm_primary("cmapss_FD001"), "cm_B_FD003": lambda: cm_primary("cmapss_FD003"),
          "cm_cluster_FD001": lambda: cm_cluster("cmapss_FD001"), "cm_cluster_FD003": lambda: cm_cluster("cmapss_FD003"),
          "cm_bands_FD001": lambda: cm_bands("cmapss_FD001"), "cm_bands_FD003": lambda: cm_bands("cmapss_FD003"),
          "cm_A_FD001": lambda: R1.cmapss_table("cmapss_A"), "decisions_r3": decisions_r3,
          "round_tests": R2.round2_tests, "test_runs": R2.test_run_history}


def values():
    v = json.loads((REP / "revised_gate_verdict.json").read_text())
    out = {"verdict": v["verdict"], "r4_C": f(v["gate_R2_C"]), "r4_thr": f(v["gate_R2_threshold"]),
           "r4_probe_mean": f(v["gate_probe_mean"], 3), "r4_omega": f(v["gate_probe"][3], 3),
           "r4_conf": f"{v['confirmations_passed']}/{v['confirmations_total']}"}
    TL = R1.TA[R1.TA.run_id == "final_full_suite_round3"]       # round-3 text quotes ITS final suite run (round 4)
    TL = TL if len(TL) else R1.TL
    out["n_tests_latest"] = str(len(TL))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[f"latest_{o}"] = str(int((TL.outcome == o).sum()))
    out["n_test_records"], out["n_test_runs"] = str(len(R1.TA)), str(R1.TA.run_id.nunique())
    sel = json.loads((REP / "learnability_selected_fa.json").read_text())
    out["r4_sel_id"], out["r4_sel"] = sel["config_id"], ", ".join(f"{k}={v}" for k, v in sel["config"].items())
    out["git_head"] = R2.values()["git_head"]
    return out


def renumber(txt, prefix, part_re):
    for pat, rep in part_re:
        txt = re.sub(pat, rep, txt, flags=re.M)
    return txt


def main():
    vals = values()
    tpl = (REP / "report_template_round3.md").read_text()
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    R2.main()                                       # writes round 2 + round 1 into final_research_prototype_report.md
    prev = (REP / "final_research_prototype_report.md").read_text()
    prev = re.sub(r"^## 0\.", "## R2-0.", prev, flags=re.M)
    prev = re.sub(r"^### 0\.(\d)", r"### R2-0.\1", prev, flags=re.M)
    prev = re.sub(r"^## I\.(\d+) ", r"## R2.\1 ", prev, flags=re.M)
    prev = re.sub(r"^### I\.(\d+)\.(\d+) ", r"### R2.\1.\2 ", prev, flags=re.M)
    prev = re.sub(r"^## II\.(\d+) ", r"## R1.\1 ", prev, flags=re.M)
    prev = re.sub(r"^### II\.(\d+)\.(\d+) ", r"### R1.\1.\2 ", prev, flags=re.M)
    prev = re.sub(r"^#### II\.(\d+)\.(\d+)\.(\d+) ", r"#### R1.\1.\2.\3 ", prev, flags=re.M)
    prev = prev.replace("# Part II: round-1 report (2026-10-16), historical record",
                        "# Part C: round-1 report (2026-10-16), historical record")
    full = out + "\n\n# Part B: round-2 report (2026-10-17), historical record\n\n" + prev
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round3.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
