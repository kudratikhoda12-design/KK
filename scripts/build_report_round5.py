"""Assemble the final report, revision 5: round-5 part (reports/report_template_round5.md), then the round-4, 3, 2 and 1
reports as historical records (Parts B–E; sections R4.x, R3.x, R2.x, R1.x), each rendered from ITS round's snapshot of
the test records (owner reporting fix, round 5)."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round4 as R4  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table, f = R1.md_table, R2.f
FINAL5 = "final_full_suite_round5"


def det_table():
    rows = []
    for l in (REP / "logs/determinism_check_4b.log").read_text().splitlines():
        if l.startswith("{"):
            r = json.loads(l)
            rows.append([r["threads"], f(r["val_C"]), r["epochs"][0], f(r["best_val"][0])])
    return md_table(["Threads", "Validation C (seed 21, Revision-4b selection, training seed 0)", "Epochs",
                     "Best validation objective"], rows)


def step1_search():
    base = dict(front="fa", fa_compact=True, fa_nfreq=17, r=12, alpha=1.0, optimizer="adam", lr=3e-3, wd=1e-4,
                head_wd=1e-2, es_target="objective", max_epochs=3000, patience=100, H=32, hidden=64)
    rows = []
    for r in D.jsonl("learnability_search_fa4c.jsonl"):
        ch = ", ".join(f"{k}={v}" for k, v in r["config"].items() if v != base.get(k)) or "4b base"
        rows.append([r["step"], r["kind"], ch, r["seed"], f(r["train_C"]), f(r["val_C"]),
                     " / ".join(f(p, 2) for p in r["val_probe"]), f(r["val_probe_mean"], 3), r["jumps"],
                     f(r["max_train_nll_after300"], 2), r["epoch_min_val_nll"], r["stop_epoch"],
                     f"{r['delivered_val_nll']:.4f} / {r['min_val_nll']:.4f}"])
    return md_table(["Step", "Model", "Change from 4b base", "Seed", "Train C", "Val C", "Val probe a / b / A / ω",
                     "Probe mean", "NLL jumps > 0.2 (after epoch 300)", "Max train NLL (after 300)",
                     "Epoch: min val NLL", "Stop epoch", "Delivered / min val NLL"], rows)


def step1_accept():
    s = json.loads((REP / "learnability_selected_fa4c.json").read_text())
    rows = [[p["seed"], p["jumps"], "yes" if p["jumps"] == 0 else "no", f"{p['delivered_val_nll']:.4f}",
             f"{p['min_val_nll']:.4f}", "yes" if p["delivered_val_nll"] <= p["min_val_nll"] + 0.01 else "no",
             f(p["val_C"]), f(p["val_probe_mean"], 3)] for p in s["step1"]["per_seed"]]
    rows.append(["**Step 1**", "", f"**{'all' if s['step1']['jumps_zero_all_seeds'] else 'not all'}**", "", "",
                 f"**{'all' if s['step1']['stopping_conflict_resolved_all_seeds'] else 'not all'}**", "",
                 f"**{s['step1']['STEP1']}**"])
    return md_table(["Training seed (data seed 21)", "Jumps", "Zero jumps", "Delivered val NLL", "Min val NLL",
                     "Within 0.01", "Val C", "Val probe mean"], rows)


def stab_diag():
    d = json.loads((REP / "diag_4c_stability.json").read_text())
    rows = [[k, v["epochs"], v["jumps"], f(v["max_train_nll_after300"], 2), f(v["val_C"])] for k, v in d["full_budget"].items()]
    t = md_table(["Full budget, no early stopping (D77-comparable)", "Epochs", "Training-NLL jumps > 0.2",
                  "Max training NLL after epoch 300", "Val C (delivered state)"], rows)
    rows2 = [[r["threads"], f(r["val_C"]), r["epochs"], r["jumps"]] for r in d["thread_spread"]]
    return t + "\n\n" + md_table(["Threads (4c selection)", "Val C", "Epochs", "Jumps"], rows2)


def gate_4c():
    log = (REP / "logs/revised_gate_fa4c.log").read_text()
    v = json.loads((REP / "revised_gate_verdict_fa4c.json").read_text())
    r1 = re.search(r"R1: C = ([0-9.]+), oracle C = ([0-9.]+), sigma = ([0-9.]+)", log)
    r1a = re.search(r"R1-A: C = ([0-9.]+), oracle C = ([0-9.]+), profile sigma \(train, eta fixed\) = ([0-9.]+)", log)
    m = D.GATE_RE.search(log)
    am = re.search(r"R2-amended: C_ceiling = ([0-9.]+); threshold = min\(([0-9.]+), ([0-9.]+)\) = ([0-9.]+); C = ([0-9.]+); "
                   r"probe mean ([0-9.]+) \(>= 0.78\), omega ([-0-9.]+)", log)
    r3 = re.search(r"R3: survival-curve error aware \[([^\]]+)\] naive \[([^\]]+)\]", log)
    dc = re.search(r"\|C - C_R2\| = ([0-9.]+)", log).group(1)
    C, thr, pm, om = float(am.group(5)), float(am.group(4)), float(am.group(6)), float(am.group(7))
    ok = lambda k: "PASS" if v["gate"].get(k) else "FAIL"  # noqa: E731
    rows = [["R1, original (early-stopped σ̂; reported, not gating)", f"C {float(r1.group(1)):.4f}; σ̂ {float(r1.group(3)):.4f}", ok("R1")],
            ["R1-A (gating; profile σ̂ on training data, η fixed)", f"C {float(r1a.group(1)):.4f} ≥ {float(r1a.group(2)) - 0.02:.4f}; "
             f"profile σ̂ {float(r1a.group(3)):.4f}", ok("R1A")],
            ["R2 (amended), concordance part", f"C {C:.4f} ≥ min(C_oracle − 0.03 = {float(am.group(2)):.4f}, C_ceiling − 0.015 = "
             f"{float(am.group(3)):.4f}) = {thr:.4f} (C_ceiling {float(am.group(1)):.4f})", "PASS" if C >= thr else "FAIL"],
            ["R2 (amended), probe part", f"mean {pm:.3f} (≥ 0.78); ω {om:.3f} (≥ 0.80); a / b / A / ω = "
             f"{' / '.join(m.group(3).split())}", "PASS" if pm >= 0.78 and om >= 0.80 else "FAIL"],
            ["R2 overall", "both parts required", ok("R2")],
            ["R3 (unchanged)", f"aware [{r3.group(1)}] vs naive [{r3.group(2)}]; |C − C_R2| = {dc} ≤ 0.02", ok("R3")]]
    return md_table(["Criterion (fresh seed 81)", "Values", "Outcome"], rows)


def confirm_4c():
    rs = D.jsonl("learnability_confirmation_fae_mlp_fa4c.jsonl")
    rows = [[r["data_seed"], f(r["test_C"]), f(r["oracle_C"]), f(r["C_ceiling"]), f(r["threshold"]),
             "PASS" if r["pass_C"] else "FAIL", " / ".join(f(p, 3) for p in r["probe"]), f(r["probe_mean"], 3),
             "PASS" if r["pass_probe"] else "FAIL", "PASS" if r["pass"] else "FAIL"] for r in rs]
    rows.append(["**Total**", "", "", "", "", f"{sum(r['pass_C'] for r in rs)}/5", "", "", f"{sum(r['pass_probe'] for r in rs)}/5",
                 f"**{sum(r['pass'] for r in rs)}/5**"])
    return md_table(["Fresh seed", "Test C", "Oracle C", "C_ceiling", "Threshold", "C part", "Probe a / b / A / ω",
                     "Probe mean", "Probe part", "R2"], rows)


def gate_overview():
    t = R4.gate_overview().splitlines()
    v = json.loads((REP / "revised_gate_verdict_fa4c.json").read_text())
    t.append(f"| Revision 4c: balance + clipping + NLL stopping (D81; **amended criteria** D82) | 81 (fresh) | "
             f"{f(v['gate_R2_C'])} | {f(v['gate_R2_threshold'])} (amended) | {f(v['gate_probe_mean'], 3)} | "
             f"{f(v['gate_probe'][3], 3)} | {'PASS' if v['gate'].get('R2') else 'FAIL'} (C part PASS; probe part FAIL) |")
    return "\n".join(t)


def seed135():
    rows = []
    for l in (REP / "logs/diag_seed135_trainval.log").read_text().splitlines():
        m = re.match(r"seed (\d+): stop epoch (\d+) \(min-val-NLL epoch (\d+)\); train C ([0-9.]+); val C ([0-9.]+) "
                     r"\(oracle ([0-9.]+)\); val probe \[([^\]]+)\]", l)
        if m:
            rows.append([m.group(1), m.group(2), m.group(4), m.group(5), m.group(6), " / ".join(m.group(7).split())])
    return md_table(["Seed (train/validation only)", "Stop epoch (= min val NLL)", "Train C", "Val C", "Val oracle C",
                     "Val probe a / b / A / ω"], rows)


def new_tests():
    TA, TL = R1.TA, R1.TL
    rows = []
    for _, r in TL.iterrows():
        if not (str(r.criterion).startswith(("RS", "CEIL")) or "fa4c" in r.nodeid):
            continue
        h = TA[TA.nodeid == r.nodeid]
        rows.append([r.criterion, r.nodeid.split("::")[0].split("/")[-1], r.nodeid.split("::")[-1][:60], len(h),
                     int((h.outcome == "failed").sum()), r.outcome.upper()])
    return md_table(["Criterion", "File", "Test", "Attempts", "Failed attempts", "Latest"], rows)


def suite_table():
    rows = []
    for run in ("final_full_suite", "final_full_suite_round2_v3", "final_full_suite_round3", "final_full_suite_round4",
                "final_full_suite_round4_rerun", FINAL5):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid)])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total",
                     "Failed tests"], rows)


def decisions_r5():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D(79|[89]\d) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


TABLES = {"det": det_table, "step1_search": step1_search, "step1_accept": step1_accept, "stab": stab_diag,
          "gate_4c": gate_4c, "confirm_4c": confirm_4c, "gate_overview": gate_overview, "seed135": seed135,
          "new_tests": new_tests, "suite_table": suite_table, "decisions_r5": decisions_r5,
          "test_runs": R2.test_run_history, "pbc_E2": lambda: R1.pbc_table("pbc_E2"),
          "pbc_E1": lambda: R1.pbc_table("pbc_E1")}


def values():
    v = json.loads((REP / "revised_gate_verdict_fa4c.json").read_text())
    s = json.loads((REP / "learnability_selected_fa4c.json").read_text())
    out = {"verdict": v["verdict"], "C": f(v["gate_R2_C"]), "thr": f(v["gate_R2_threshold"]), "ceil": f(v["gate_C_ceiling"]),
           "probe": f(v["gate_probe_mean"], 3), "omega": f(v["gate_probe"][3], 3),
           "conf": f"{v['confirmations_passed']}/{v['confirmations_total']}", "step1": s["step1"]["STEP1"],
           "sel_id": s["config_id"], "git_head": R2.values()["git_head"]}
    g = R1.TA[R1.TA.run_id == FINAL5]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def main():
    R1.use_snapshot(FINAL5)
    vals = values()
    tpl = (REP / "report_template_round5.md").read_text()
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    R4.main()
    prev = (REP / "final_research_prototype_report.md").read_text()
    prev = re.sub(r"^## 0\. Round-4 summary", "## R4-0. Round-4 summary", prev, flags=re.M)
    prev = re.sub(r"^### 0\.(\d) ", r"### R4-0.\1 ", prev, flags=re.M)
    for a, b in (("# Part D: round-1 report", "# Part E: round-1 report"), ("# Part C: round-2 report", "# Part D: round-2 report"),
                 ("# Part B: round-3 report", "# Part C: round-3 report"), ("# Parts B–D: historical records",
                                                                             "# Parts C–E: historical records (round 3, 2, 1)"),
                 ("# Parts C and D: historical records (round 2, round 1)", "# Parts D and E: historical records (round 2, round 1)"),
                 ("In this revision, Part C is the round-2 report and Part D, nested at its end, is the round-1 report.",
                  "In this revision, Part D is the round-2 report and Part E, nested at its end, is the round-1 report.")):
        prev = prev.replace(a, b)
    for a, b in (("Part A is round 4", "Part B is round 4"),
                 ("Parts B–D are the round-3, round-2 and round-1 reports", "Parts C–E are the round-3, round-2 and round-1 reports"),
                 ("Part A is round 3", "Part C is round 3"),
                 ("Parts B and C are the round-2 and round-1 reports", "Parts D and E are the round-2 and round-1 reports"),
                 ("Part I covers round 2", "Part D (sections R2.x) covers round 2"),
                 ("Part II is the round-1 report", "Part E (sections R1.x) is the round-1 report"),
                 ("where the two disagree, Part I supersedes it", "where they disagree, the later round supersedes it"),
                 ("Part B (round 3), Part C (round 2) and Part D (round 1)", "Part C (round 3), Part D (round 2) and Part E (round 1)"),
                 ("Part B (round 3), Part C (round 2) and Part D (round 1)", "Part C (round 3), Part D (round 2) and Part E (round 1)")):
        prev = prev.replace(a, b)
    prev = re.sub(r"Part [A-E], (§R\d)", r"\1", prev)            # cross-references: section IDs are unique
    full = out + "\n\n# Part B: round-4 report (2026-10-18), historical record\n\n" + prev
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round5.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
