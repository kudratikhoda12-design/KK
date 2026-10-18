"""Assemble the final report, revision 6: round-6 part (reports/report_template_round6.md), then the round-5..1 reports
as historical records (Parts B–F), each rendered from its own round's snapshot of the test records."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round5 as R5  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table, f = R1.md_table, R2.f
FINAL6 = "final_full_suite_round6_rerun"


def step1_candidates():
    s = json.loads((REP / "learnability_selected_fa4d.json").read_text())
    rows = [[f"cand{r['index']}", r["config"]["es_mode"], r["config"]["min_epochs"], r["config"]["lr"], f(r["mean_val_C"]),
             f(r["mean_probe"], 3), f(r["min_seed_probe"], 3), f(r["min_seed_omega"], 3), r["max_jumps"],
             "**yes**" if r["eligible"] else "no"] for r in s["candidates"]]
    return md_table(["Candidate", "Stopping mode", "Burn-in", "lr", "Mean val C (seeds 0–2)", "Mean probe (≥ 0.80)",
                     "Min seed probe (≥ 0.78)", "Min seed ω (≥ 0.80)", "Max jumps (= 0)", "Eligible"], rows) + \
        f"\n\n**Step 1: {s['STEP1']}.**"


def step1_seeds():
    rows = []
    for r in D.jsonl("learnability_search_fa4d.jsonl"):
        c = r["config"]
        rows.append([r["step"], r["kind"], c["es_mode"], c["min_epochs"], c["lr"], r["seed"], f(r["val_C"]),
                     " / ".join(f(p, 3) for p in r["val_probe"]), f(r["val_probe_mean"], 3), r["jumps"],
                     r["objective_stop_epoch"], r["risk_state_epoch"] if r["risk_state_epoch"] is not None else r["refit_stop_epoch"]])
    return md_table(["Candidate", "Model", "Stopping mode", "Burn-in", "lr", "Training seed", "Val C",
                     "Val probe a / b / A / ω", "Probe mean", "Jumps", "Objective stop epoch (latent)",
                     "Risk-state / refit epoch"], rows)


def new_tests():
    TA, TL = R1.TA, R1.TL
    rows = []
    for _, r in TL.iterrows():
        if not (str(r.criterion).startswith("BI") or "fa4d" in r.nodeid):
            continue
        h = TA[TA.nodeid == r.nodeid]
        rows.append([r.criterion, r.nodeid.split("::")[0].split("/")[-1], r.nodeid.split("::")[-1][:60], len(h),
                     int((h.outcome == "failed").sum()), r.outcome.upper()])
    return md_table(["Criterion", "File", "Test", "Attempts", "Failed attempts", "Latest"], rows)


def gate_overview():
    t = R5.gate_overview().splitlines()
    t.append("| Revision 4d: two-state stopping + burn-in, multi-seed selection (D87/D88) | 92 (fresh; not generated) | – | – "
             "| – | – | **not evaluated: Step 1 failed (D89)** |")
    return "\n".join(t)


def suite_table():
    rows = []
    for run in ("final_full_suite", "final_full_suite_round2_v3", "final_full_suite_round3", "final_full_suite_round4_rerun",
                "final_full_suite_round5", "final_full_suite_round6", FINAL6):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total",
                     "Failed tests"], rows)


def decisions_r6():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D(8[6-9]|9\d) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


TABLES = {"step1_candidates": step1_candidates, "step1_seeds": step1_seeds, "new_tests": new_tests,
          "gate_overview": gate_overview, "suite_table": suite_table, "decisions_r6": decisions_r6,
          "test_runs": R2.test_run_history, "pbc_E2": lambda: R1.pbc_table("pbc_E2"), "pbc_E1": lambda: R1.pbc_table("pbc_E1")}


def values():
    s = json.loads((REP / "learnability_selected_fa4d.json").read_text())
    out = {"step1": s["STEP1"], "git_head": R2.values()["git_head"]}
    g = R1.TA[R1.TA.run_id == FINAL6]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


PHRASES = {"Part A is round 5": "Part B is round 5",
           "Parts B–E are the round-4, 3, 2 and 1 reports": "Parts C–F are the round-4, 3, 2 and 1 reports",
           "Part B is round 4": "Part C is round 4",
           "Parts C–E are the round-3, round-2 and round-1 reports": "Parts D–F are the round-3, round-2 and round-1 reports",
           "Part C is round 3": "Part D is round 3",
           "Parts D and E are the round-2 and round-1 reports": "Parts E and F are the round-2 and round-1 reports",
           "Part D (sections R2.x) covers round 2": "Part E (sections R2.x) covers round 2",
           "Part E (sections R1.x) is the round-1 report": "Part F (sections R1.x) is the round-1 report",
           "Part B (round 4), Part C (round 3), Part D (round 2) and Part E (round 1)":
               "Part C (round 4), Part D (round 3), Part E (round 2) and Part F (round 1)",
           "Part C (round 3), Part D (round 2) and Part E (round 1)": "Part D (round 3), Part E (round 2) and Part F (round 1)",
           "In this revision, Part D is the round-2 report and Part E, nested at its end, is the round-1 report.":
               "In this revision, Part E is the round-2 report and Part F, nested at its end, is the round-1 report.",
           "# Parts B–E: historical records": "# Parts C–F: historical records",
           "# Parts C–E: historical records (round 3, 2, 1)": "# Parts D–F: historical records (round 3, 2, 1)",
           "# Parts D and E: historical records (round 2, round 1)": "# Parts E and F: historical records (round 2, round 1)"}


def main():
    R1.use_snapshot(FINAL6)
    vals = values()
    tpl = (REP / "report_template_round6.md").read_text()
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    R5.main()
    prev = (REP / "final_research_prototype_report.md").read_text()
    prev = re.sub(r"^## 0\. Round-5 summary", "## R5-0. Round-5 summary", prev, flags=re.M)
    prev = re.sub(r"^### 0\.(\d) ", r"### R5-0.\1 ", prev, flags=re.M)
    prev = re.sub(r"^# Part ([B-E]):", lambda m: f"# Part {chr(ord(m.group(1)) + 1)}:", prev, flags=re.M)   # one pass
    pat = re.compile("|".join(re.escape(k) for k in sorted(PHRASES, key=len, reverse=True)))
    prev = pat.sub(lambda m: PHRASES[m.group(0)], prev)                                                    # one pass
    prev = re.sub(r"Part [A-F], (§R\d)", r"\1", prev)
    full = out + "\n\n# Part B: round-5 report (2026-10-18), historical record\n\n" + prev
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round6.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
