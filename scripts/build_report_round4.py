"""Assemble the final report, revision 4: round-4 part (reports/report_template_round4.md), then the round-3, round-2
and round-1 reports as historical records (Parts B, C, D; sections R3.x, R2.x, R1.x)."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round3 as R3  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table, f = R1.md_table, R2.f


def r4b_search():
    rows = []
    for r in D.jsonl("learnability_search_fa4b.jsonl"):
        c = r["config"]
        cfg = ", ".join(f"{k}={v}" for k, v in c.items() if k in ("r", "es_target", "head_dropout", "head_wd", "wd"))
        rows.append([r["step"], r["kind"], cfg or "base", r["seed"], f(r["train_C"]), f(r["val_C"]),
                     f(r["train_C"] - r["val_C"]), f(r["val_C"] - r["val_oracle_C"]),
                     " / ".join(f(p, 2) for p in r["val_probe"]), f(r["val_probe_mean"], 3), r["epochs"][0]])
    return md_table(["Step", "Model", "Change from 4b base", "Seed", "Train C", "Val C", "Train − val", "Val C − oracle",
                     "Val probe a / b / A / ω", "Probe mean", "Epochs"], rows)


def r4b_gate():
    log = (REP / "logs" / "revised_gate_fa4b.log").read_text()
    v = json.loads((REP / "revised_gate_verdict_fa4b.json").read_text())
    r1 = re.search(r"R1: C = ([0-9.]+), oracle C = ([0-9.]+), sigma = ([0-9.]+)", log)
    r1a = re.search(r"R1-A: C = ([0-9.]+), oracle C = ([0-9.]+), profile sigma \(train, eta fixed\) = ([0-9.]+)", log)
    m = D.GATE_RE.search(log)
    r3 = re.search(r"R3: survival-curve error aware \[([^\]]+)\] naive \[([^\]]+)\]", log)
    dc = re.search(r"\|C - C_R2\| = ([0-9.]+)", log).group(1)
    ok = lambda k: "PASS" if v["gate"].get(k) else "FAIL"  # noqa: E731
    rows = [["R1 (original, early-stopped σ̂; reported, not in the 4b rule)",
             f"C {float(r1.group(1)):.4f} ≥ {float(r1.group(2)) - 0.02:.4f}; σ̂ {float(r1.group(3)):.4f} (|σ̂ − 0.5| ≤ 0.05)", ok("R1")],
            ["R1-A (amended D68: profile σ̂, training data, η fixed)",
             f"C {float(r1a.group(1)):.4f}; profile σ̂ {float(r1a.group(3)):.4f} (|σ̂ − 0.5| ≤ 0.05)", ok("R1A")],
            ["R2", f"C {float(m.group(1)):.4f} vs ≥ {float(m.group(2)) - 0.03:.4f}; probe a / b / A / ω = "
             f"{' / '.join(m.group(3).split())}, mean {m.group(4)} vs ≥ 0.8", ok("R2")],
            ["R3", f"aware [{r3.group(1)}] vs naive [{r3.group(2)}]; |C − C_R2| = {dc} ≤ 0.02", ok("R3")]]
    return md_table(["Criterion (fresh seed 61; thresholds unchanged)", "Values", "Outcome"], rows)


def r4b_confirm():
    rs = D.jsonl("learnability_confirmation_fae_mlp_fa4b.jsonl")
    rows = [[r["data_seed"], f(r["test_C"]), f(r["threshold"]), f(r["test_C"] - r["threshold"]),
             " / ".join(f(p, 3) for p in r["probe"]), f(r["probe_mean"], 3), "yes" if r["pass_probe"] else "no",
             "PASS" if r["pass"] else "FAIL"] for r in rs]
    rows.append(["**Total**", "", "", "", "", "", "", f"**{sum(r['pass'] for r in rs)}/{len(rs)}**"])
    return md_table(["Fresh seed", "Test C", "Threshold", "C − threshold", "Probe a / b / A / ω", "Probe mean",
                     "Probe criterion", "R2"], rows)


def gate_overview():
    base = R3.gate_overview().splitlines()
    v = json.loads((REP / "revised_gate_verdict_fa4b.json").read_text())
    base.append(f"| Revision 4b: compact front + head weight decay (D71) | 61 (fresh) | {f(v['gate_R2_C'])} | "
                f"{f(v['gate_R2_threshold'])} | {f(v['gate_probe_mean'], 3)} | {f(v['gate_probe'][3], 3)} | "
                f"{'PASS' if v['gate'].get('R2') else 'FAIL'} |")
    return "\n".join(base)


def curves_table():
    d = json.loads((REP / "diag_overfit_curves.json").read_text())
    rows = []
    for name, r in d.items():
        cs = [x for x in r["curve"] if "val_C" in x]
        b = max(cs, key=lambda x: x["val_C"])
        rows.append([name, r["epoch_min_val_objective"], r["epoch_min_val_nll"], f(r["val_nll_min"]),
                     f(r["val_nll_at_obj_stop"]), f"{b['val_C']:.4f} (epoch {b['ep']})",
                     " / ".join(f"{a:.2f}/{c:.2f}" for a, c in r["probe_train_vs_val"].values()),
                     f(r.get("train_spectral_path_rms"), 2), f(r["train_A3_path_rms"], 2)])
    return md_table(["Configuration (seed 21, full 3000 epochs)", "Epoch: min val objective", "Epoch: min val NLL",
                     "Min val NLL", "Val NLL at objective minimum", "Best val C (epoch)",
                     "Probe R² train/val (a, b, A, ω)", "Spectral path RMS", "A3 path RMS"], rows)


def r1a_side():
    rows = []
    for r in D.jsonl("r1a_fresh_seeds.jsonl"):
        rows.append([r["seed"], f(r["C"]), f(r["oracle_C"] - 0.02), f(r["sigma"]), "pass" if r["pass_sigma"] else "FAIL",
                     f(r["sigma_profile_train"]), "pass" if abs(r["sigma_profile_train"] - 0.5) <= 0.05 else "FAIL",
                     r["epochs"]])
    rs = D.jsonl("r1a_fresh_seeds.jsonl")
    rows.append(["**Fresh seeds 601–610**", "", "", "", f"**{sum(r['pass_sigma'] for r in rs)}/10 pass**", "",
                 f"**{sum(abs(r['sigma_profile_train'] - 0.5) <= 0.05 for r in rs)}/10 pass**", ""])
    return md_table(["Seed", "C", "C threshold", "Original: early-stopped σ̂", "Original check",
                     "R1-A: profile σ̂ (train, η fixed)", "R1-A check", "Stopping epoch"], rows)


def latent_ceiling():
    d = json.loads((REP / "diag_4b_ceiling.json").read_text())
    rows = [[k, f(v["model_val_C"]), f(v["linear_head_refit_val_C"]), f(v["mlp_head_val_C"]),
             f(v["plugin_true_function_val_C"])] for k, v in d.items() if isinstance(v, dict)]
    return md_table(["Frozen latent of", "Model as trained", "Linear AFT head refit", "MLP AFT head",
                     "True risk function on decoded parameters"], rows) + \
        f"\n\nValidation oracle C {f(d['oracle_val_C'])}; R2 threshold {f(d['oracle_val_C'] - 0.03)}."


def data_ceiling():
    d = json.loads((REP / "diag_data_ceiling.json").read_text())
    rows = [["Oracle (true η)", f(d["oracle_val_C"]), "–"],
            ["R2 threshold (oracle − 0.03)", f(d["R2_threshold_val"]), "–"],
            ["True risk function on least-squares estimates from the raw observations", f(d["parametric_plugin_val_C"]),
             " / ".join(f(x, 3) for x in d["parametric_param_R2_val"])],
            ["True risk function on gradient-boosting estimates (inputs: coefficients, spectral, least-squares)",
             f(d["flexible_plugin_val_C"]), " / ".join(f(x, 3) for x in d["flexible_param_R2_val"])],
            ["Gradient boosting trained directly on the true η", f(d["flexible_direct_eta_val_C"]), "–"]]
    return md_table(["Seed 21 validation", "C", "Parameter R² a / b / A / ω"], rows)


def data_ceiling_seeds():
    rs = D.jsonl("diag_data_ceiling_seeds.jsonl")
    rows = [[r["seed"], f(r["oracle_C"]), f(r["threshold"]), f(r["parametric_plugin_C"]), f"{r['gap_to_threshold']:+.4f}",
             f(r["only_this_parameter_estimated_C"]["b"]), f(r["only_this_parameter_estimated_C"]["a"]),
             f(r["only_this_parameter_estimated_C"]["A"]), f(r["only_this_parameter_estimated_C"]["omega"])] for r in rs]
    g = np.array([r["gap_to_threshold"] for r in rs])
    rows.append(["**Summary**", "", "", "", f"mean {g.mean():+.4f}; {int((g >= 0).sum())}/10 ≥ 0", "", "", "", ""])
    return md_table(["Fresh seed (all 2000)", "Oracle C", "R2 threshold", "Parametric plug-in C", "Plug-in − threshold",
                     "Only b estimated", "Only a estimated", "Only A estimated", "Only ω estimated"], rows)


def hr_tests():
    TA, TL = R1.TA, R1.TL
    rows = []
    for _, r in TL.iterrows():
        if not (str(r.criterion).startswith(("HR", "FQ", "R1-A")) or "fa4b" in r.nodeid):
            continue
        h = TA[TA.nodeid == r.nodeid]
        rows.append([r.criterion, r.nodeid.split("::")[0].split("/")[-1], r.nodeid.split("::")[-1][:60], len(h),
                     int((h.outcome == "failed").sum()), r.outcome.upper()])
    return md_table(["Criterion", "File", "Test", "Attempts", "Failed attempts", "Latest"], rows)


def decisions_r4():
    lines = (ROOT / "docs/decisions.md").read_text().splitlines()
    rows = [l for l in lines if re.match(r"\| D(6[7-9]|7[0-8]) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    TA = R1.TA
    rows = []
    for run in ("final_full_suite", "final_full_suite_round2_v3", "final_full_suite_round3", "final_full_suite_round4", "final_full_suite_round4_rerun"):
        g = TA[TA.run_id == run]
        if len(g):
            rows.append([run, int((g.outcome == "passed").sum()), int((g.outcome == "failed").sum()),
                         int((g.outcome == "xfailed").sum()), int((g.outcome == "skipped").sum()), len(g),
                         "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                   for n in g[g.outcome == "failed"].nodeid)])
    return md_table(["Full-suite run", "Passed", "Failed", "Expected failures (historical, strict)", "Skipped", "Total",
                     "Failed tests"], rows)


TABLES = {"r4b_search": r4b_search, "r4b_gate": r4b_gate, "r4b_confirm": r4b_confirm, "gate_overview": gate_overview,
          "curves": curves_table, "r1a_side": r1a_side, "latent_ceiling": latent_ceiling, "data_ceiling": data_ceiling,
          "data_ceiling_seeds": data_ceiling_seeds, "hr_tests": hr_tests, "decisions_r4": decisions_r4,
          "suite_table": suite_table, "pbc_E2": lambda: R1.pbc_table("pbc_E2"), "pbc_E1": lambda: R1.pbc_table("pbc_E1"),
          "test_runs": R2.test_run_history}


def values():
    v = json.loads((REP / "revised_gate_verdict_fa4b.json").read_text())
    sel = json.loads((REP / "learnability_selected_fa4b.json").read_text())
    out = {"verdict": v["verdict"], "C": f(v["gate_R2_C"]), "thr": f(v["gate_R2_threshold"]),
           "probe": f(v["gate_probe_mean"], 3), "omega": f(v["gate_probe"][3], 3),
           "conf": f"{v['confirmations_passed']}/{v['confirmations_total']}", "sel_id": sel["config_id"],
           "git_head": R2.values()["git_head"]}
    TL = R1.TA[R1.TA.run_id == "final_full_suite_round4_rerun"]
    TL = TL if len(TL) else R1.TL
    out["n_tests"] = str(len(TL))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((TL.outcome == o).sum()))
    rs = D.jsonl("diag_data_ceiling_seeds.jsonl")
    out["ceiling_below"] = str(sum(r["gap_to_threshold"] < 0 for r in rs))
    return out


def main():
    R1.use_snapshot("final_full_suite_round4_rerun")           # round-4 part: test records up to round 4
    vals = values()
    tpl = (REP / "report_template_round4.md").read_text()
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    R3.main()
    prev = (REP / "final_research_prototype_report.md").read_text()
    prev = re.sub(r"^## 0\. Round-3 summary", "## R3-0. Round-3 summary", prev, flags=re.M)
    prev = re.sub(r"^### 0\.(\d) ", r"### R3-0.\1 ", prev, flags=re.M)
    prev = prev.replace("# Part B: round-2 report (2026-10-17), historical record",
                        "# Part C: round-2 report (2026-10-17), historical record")
    prev = prev.replace("# Part C: round-1 report (2026-10-16), historical record",
                        "# Part D: round-1 report (2026-10-16), historical record")
    prev = prev.replace("# Part B and Part C: historical records", "# Parts C and D: historical records (round 2, round 1)")
    prev = prev.replace("Part B is the round-2 report, and Part C, nested at its end, is the round-1 report.",
                        "In this revision, Part C is the round-2 report and Part D, nested at its end, is the round-1 report.")
    full = out + "\n\n# Part B: round-3 report (2026-10-17), historical record\n\n" + prev
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round4.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
