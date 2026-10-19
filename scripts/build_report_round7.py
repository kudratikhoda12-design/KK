"""Assemble the final report, revision 7: round-7 part (reports/report_template_round7.md), then the round-6..1 reports
as historical records (Parts B–G), each rendered from its own round's snapshot of the test records."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round6 as R6  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table, f = R1.md_table, R2.f
FINAL7 = "final_full_suite_round7"


def step1_candidates():
    s = json.loads((REP / "learnability_selected_fa4e.json").read_text())
    yn = lambda b: "**yes**" if b else "no"  # noqa: E731
    rows = [[f"cand{r['index']}" + (" (= 4d cand1)" if r["index"] == 0 else ""), r["config"]["r"], r["config"]["patience"],
             r["config"]["alpha"], f(r["mean_val_C"]), f(r["mean_probe"], 3), f(r["min_seed_probe"], 3),
             f(r["min_seed_omega"], 3), r["max_jumps"], yn(r["tier1"]), yn(r["tier2"])] for r in s["candidates"]]
    return md_table(["Candidate", "r", "Patience", "α", "Mean val C (seeds 0–2)", "Mean probe (≥ 0.80)",
                     "Min seed probe (T1 ≥ 0.78; T2 ≥ 0.775)", "Min seed ω (≥ 0.80)", "Max jumps (= 0)", "Tier 1",
                     "Tier 2"], rows) + \
        f"\n\n**Step 1: {s['STEP1']}**: selected cand{s['selected_index']} (`{s['config_id']}`) via **Tier {s['tier']}**."


def step1_seeds():
    rows = []
    for r in D.jsonl("learnability_search_fa4e.jsonl"):
        c = r["config"]
        rows.append([r["step"], r["kind"], c["r"], c["patience"], c["alpha"], r["seed"], f(r["val_C"]),
                     " / ".join(f(p, 3) for p in r["val_probe"]), f(r["val_probe_mean"], 3), r["jumps"],
                     r["joint_epochs"], r["objective_stop_epoch"], r["risk_state_epoch"]])
    return md_table(["Candidate", "Model", "r", "Patience", "α", "Training seed", "Val C", "Val probe a / b / A / ω",
                     "Probe mean", "Jumps", "Epochs run", "Objective-minimum epoch (latent)", "Risk-state epoch"], rows)


def repro():
    d = {(r["config_id"], r["seed"]): r for r in D.jsonl("learnability_search_fa4d.jsonl")}
    rows = []
    for r in D.jsonl("learnability_search_fa4e.jsonl"):
        o = d.get((r["config_id"], r["seed"]))
        if o:
            same = o["val_C"] == r["val_C"] and o["val_probe"] == r["val_probe"] and o["delivered_val_nll"] == r["delivered_val_nll"]
            rows.append([r["seed"], f(o["val_C"], 6), f(r["val_C"], 6), f(o["val_probe_mean"], 6), f(r["val_probe_mean"], 6),
                         "**yes**" if same else "NO"])
    return md_table(["Training seed", "4d cand1 val C (round 6)", "4e cand0 val C (round 7)", "4d probe mean",
                     "4e probe mean", "Bit-identical (C, probes, delivered NLL)"], rows)


def gate_4e():
    log = (REP / "logs/revised_gate_fa4e.log").read_text()
    v = json.loads((REP / "revised_gate_verdict_fa4e.json").read_text())
    r1 = re.search(r"R1: C = ([0-9.]+), oracle C = ([0-9.]+), sigma = ([0-9.]+)", log)
    r1a = re.search(r"R1-A: C = ([0-9.]+), oracle C = ([0-9.]+), profile sigma \(train, eta fixed\) = ([0-9.]+)", log)
    m = D.GATE_RE.search(log)
    am = re.search(r"R2-amended: C_ceiling = ([0-9.]+); threshold = min\(([0-9.]+), ([0-9.]+)\) = ([0-9.]+); C = ([0-9.]+); "
                   r"probe mean ([0-9.]+) \(>= 0.75\), omega ([-0-9.]+)", log)
    m15 = re.search(r"0.015 margin \(reported only\): threshold ([0-9.]+); pass (\w+)", log)
    three = re.search(r"4e \(mean >= 0.75, omega >= 0.75\) (\w+); D82 \(mean >= 0.78, omega >= 0.80\) (\w+); "
                      r"original \(mean >= 0.80\) (\w+)", log)
    r3 = re.search(r"R3: survival-curve error aware \[([^\]]+)\] naive \[([^\]]+)\] ; C \(aware, seed 0\) = ([0-9.]+)", log)
    dc = re.search(r"\|C - C_R2\| = ([0-9.]+)", log).group(1)
    C, thr, pm, om = float(am.group(5)), float(am.group(4)), float(am.group(6)), float(am.group(7))
    ok = lambda k: "PASS" if v["gate"].get(k) else "**FAIL**"  # noqa: E731
    pf = lambda s: "pass" if s == "True" else "fail"  # noqa: E731
    rows = [["R1, original (early-stopped σ̂; reported, not gating)", f"C {float(r1.group(1)):.4f}; σ̂ {float(r1.group(3)):.4f}",
             ok("R1")],
            ["R1-A (gating; profile σ̂ on training data, η fixed)", f"C {float(r1a.group(1)):.4f} ≥ {float(r1a.group(2)) - 0.02:.4f}; "
             f"profile σ̂ {float(r1a.group(3)):.4f}", ok("R1A")],
            ["R2, concordance part", f"C {C:.4f} vs min(C_oracle − 0.03 = {float(am.group(2)):.4f}, C_ceiling − 0.020 = "
             f"{float(am.group(3)):.4f}) = {thr:.4f} (C_ceiling {float(am.group(1)):.4f}); short by {thr - C:.4f}. "
             f"With margin 0.015 (reported): threshold {float(m15.group(1)):.4f}, {pf(m15.group(2))}",
             "PASS" if C >= thr else "**FAIL**"],
            ["R2, probe part (D91: mean ≥ 0.75 and ω ≥ 0.75)", f"mean {pm:.3f}; ω {om:.3f}; a / b / A / ω = "
             f"{' / '.join(m.group(3).split())}. Under D82 (0.78/0.80): {pf(three.group(2))}; under the original "
             f"(mean ≥ 0.80): {pf(three.group(3))}", "PASS" if pm >= 0.75 and om >= 0.75 else "**FAIL**"],
            ["R2 overall", "both parts required", ok("R2")],
            ["R3 (unchanged)", f"error aware [{r3.group(1)}] < naive [{r3.group(2)}] (pass); C under censoring "
             f"{float(r3.group(3)):.4f}; \\|C − C_R2\\| = {dc} > 0.02", ok("R3")]]
    return md_table(["Criterion (fresh seed 92)", "Values", "Outcome"], rows)


def confirm_4e():
    rs = D.jsonl("learnability_confirmation_fae_mlp_fa4e.jsonl")
    pf = lambda b: "PASS" if b else "**FAIL**"  # noqa: E731
    rows = [[r["data_seed"], f(r["test_C"]), f(r["oracle_C"]), f(r["C_ceiling"]), f(r["threshold"]),
             f(r["test_C"] - r["C_ceiling"]), pf(r["pass_C"]), " / ".join(f(p, 3) for p in r["probe"]),
             f(r["probe_mean"], 3), pf(r["pass_probe_075_075"]), pf(r["pass_probe_078_080"]),
             pf(r["pass_probe_original_080"]), pf(r["pass"])] for r in rs]
    rows.append(["**Total**", "", "", "", "", "", f"{sum(r['pass_C'] for r in rs)}/5", "", "",
                 f"{sum(r['pass_probe_075_075'] for r in rs)}/5", f"{sum(r['pass_probe_078_080'] for r in rs)}/5",
                 f"{sum(r['pass_probe_original_080'] for r in rs)}/5", f"**{sum(r['pass'] for r in rs)}/5**"])
    return md_table(["Fresh seed", "Test C", "Oracle C", "C_ceiling", "Threshold (margin 0.020)", "C − C_ceiling", "C part",
                     "Probe a / b / A / ω", "Probe mean", "Probe 0.75/0.75 (gating)", "Probe 0.78/0.80",
                     "Probe original 0.80", "R2"], rows)


def diag92():
    d = json.loads((REP / "diag_4e_seed92.json").read_text())
    rows = [[r["training_seed"], f(r["val_C"]), f(r["val_C_minus_ceiling"]), " / ".join(f(p, 3) for p in r["val_probe"]),
             f(r["val_probe_mean"], 3), r["objective_stop_epoch"], r["risk_state_epoch"]] for r in d["rows"]]
    return md_table(["Training seed", "Val C", "Val C − C_ceiling", "Val probe a / b / A / ω", "Probe mean",
                     "Objective-minimum epoch", "Risk-state epoch"], rows) + \
        (f"\n\nSeed-92 validation split: oracle C {f(d['val_oracle_C'])}, C_ceiling {f(d['val_C_ceiling'])}; "
         f"validation C mean {f(d['val_C_mean'])}, SD {f(d['val_C_sd'])}, range {f(d['val_C_range'])}.")


def retro():
    """Historical gate seeds and confirmations re-scored under the three probe definitions (information only)."""
    pf = lambda b: "pass" if b else "fail"  # noqa: E731
    rows = []
    for name, vf, cf in (("Revision 4 (fa)", "revised_gate_verdict.json", "learnability_confirmation_fae_mlp_fa.jsonl"),
                         ("Revision 4b", "revised_gate_verdict_fa4b.json", "learnability_confirmation_fae_mlp_fa4b.jsonl"),
                         ("Revision 4c", "revised_gate_verdict_fa4c.json", "learnability_confirmation_fae_mlp_fa4c.jsonl"),
                         ("Revision 4e", "revised_gate_verdict_fa4e.json", "learnability_confirmation_fae_mlp_fa4e.jsonl")):
        v = json.loads((REP / vf).read_text())
        pts = [(f"gate {v['gate_seed']}", v["gate_probe_mean"], v["gate_probe"][3])] + \
              [(str(r["data_seed"]), r["probe_mean"], r["probe"][3]) for r in D.jsonl(cf)]
        for lab, pm, om in pts:
            rows.append([name, lab, f(pm, 3), f(om, 3), pf(pm >= 0.80), pf(pm >= 0.78 and om >= 0.80),
                         pf(pm >= 0.75 and om >= 0.75)])
        rows.append([f"**{name}: verdict as recorded**", "", "", "", "", "", v["verdict"]])
    return md_table(["Revision", "Seed", "Test probe mean", "Test ω", "Original (mean ≥ 0.80)", "D82 (0.78 / 0.80)",
                     "D91 (0.75 / 0.75)"], rows)


def gate_overview():
    t = R6.gate_overview().splitlines()
    v = json.loads((REP / "revised_gate_verdict_fa4e.json").read_text())
    t.append(f"| Revision 4e: r 16, two-checkpoint, burn-in 300, lr 1e-3 (D92; criteria D93) | 92 (fresh) | "
             f"{f(v['gate_R2_C'])} | {f(v['gate_R2_threshold'])} (C_ceiling margin 0.020) | {f(v['gate_probe_mean'], 3)} | "
             f"{f(v['gate_probe'][3], 3)} | **FAIL** (C part FAIL by 0.0022; probe part PASS); R3 FAIL; confirmations "
             f"{v['confirmations_passed']}/5; **{v['verdict']}** |")
    return "\n".join(t)


def new_tests():
    TA, TL = R1.TA, R1.TL
    rows = []
    for _, r in TL.iterrows():
        if "fa4e" not in r.nodeid:
            continue
        h = TA[TA.nodeid == r.nodeid]
        rows.append([r.criterion or "–", r.nodeid.split("::")[-1][:60], len(h), int((h.outcome == "failed").sum()),
                     r.outcome.upper()])
    return md_table(["Criterion", "Test (tests/test_learnability_fa4e.py)", "Attempts", "Failed attempts", "Latest"], rows)


def suite_table():
    rows = []
    for run in ("final_full_suite", "final_full_suite_round2_v3", "final_full_suite_round3", "final_full_suite_round4_rerun",
                "final_full_suite_round5", "final_full_suite_round6", "final_full_suite_round6_rerun", FINAL7):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total",
                     "Failed tests"], rows)


def gate_repro():
    """The 4e gate inside the final suite vs the registered gate run (D79/D86 determinism)."""
    a = (REP / "logs/revised_gate_fa4e.log").read_text()
    p = REP / f"logs/{FINAL7}.log"
    if not p.exists():
        return "_The final suite has not run yet._"
    b = p.read_text()
    i = b.index("tests/test_learnability_fa4e.py::")                    # only the 4e section (other gate files print
    m = re.search(r"^tests/(?!test_learnability_fa4e)", b[i:], flags=re.M)  # the same kind of lines)
    b = b[i:i + m.start()] if m else b[i:]
    pat =[r"R1: C = [0-9.]+, oracle C = [0-9.]+, sigma = [0-9.]+", r"R1-A: C = [^\n]+", r"R2: C = [^;]+; probe R2 [^;]+",
           r"R2-amended: [^\n]+", r"R3: survival-curve error [^\n]+", r"\|C - C_R2\| = [0-9.]+"]
    rows = []
    for q in pat:
        x, y = re.search(q, a), re.search(q, b)
        cell = x.group(0)[:90].replace("|", "\\|") if x else "–"            # pipes escaped for pandoc pipe tables
        rows.append([cell, "**identical**" if x and y and x.group(0) == y.group(0) else "DIFFERENT"])
    return md_table(["Printed gate value (registered run)", "Final suite"], rows)


def decisions_r7():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D(9[1-8]) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


TABLES = {"step1_candidates": step1_candidates, "step1_seeds": step1_seeds, "repro": repro, "gate_4e": gate_4e,
          "confirm_4e": confirm_4e, "diag92": diag92, "retro": retro, "gate_overview": gate_overview,
          "new_tests": new_tests, "suite_table": suite_table, "gate_repro": gate_repro, "decisions_r7": decisions_r7,
          "test_runs": R2.test_run_history, "pbc_E2": lambda: R1.pbc_table("pbc_E2"), "pbc_E1": lambda: R1.pbc_table("pbc_E1")}


def values():
    v = json.loads((REP / "revised_gate_verdict_fa4e.json").read_text())
    s = json.loads((REP / "learnability_selected_fa4e.json").read_text())
    out = {"step1": s["STEP1"], "tier": str(s["tier"]), "verdict": v["verdict"], "C": f(v["gate_R2_C"]),
           "thr": f(v["gate_R2_threshold"]), "ceil": f(v["gate_C_ceiling"]), "probe": f(v["gate_probe_mean"], 3),
           "omega": f(v["gate_probe"][3], 3), "conf": f"{v['confirmations_passed']}/{v['confirmations_total']}",
           "sel_id": s["config_id"], "git_head": R2.values()["git_head"]}
    g = R1.TA[R1.TA.run_id == FINAL7]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def shift_part(m):
    """'Part B' -> 'Part C', 'Parts B–F' -> 'Parts C–G', 'Parts E and F' -> 'Parts F and G' (one pass; Part A is kept)."""
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEF" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def main():
    R1.use_snapshot(FINAL7)
    vals = values()
    tpl = (REP / "report_template_round7.md").read_text()
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    R6.main()
    prev = (REP / "final_research_prototype_report.md").read_text()
    cut = prev.index("\n# Parts B–F: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-6 summary", "## R6-0. Round-6 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R6-0.\1 ", own, flags=re.M)
    own = re.sub(r"§0\.(\d)", r"§R6-0.\1", own)
    prev = re.sub(r"\b(Parts? )([B-F])(–| and )?([B-F])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("Part A is round 6", "Part B is round 6", 1)          # after the shift, so it is not shifted again
    full = out + "\n\n# Part B: round-6 report (2026-10-18), historical record\n\n" + prev
    # D98: an unescaped "|C − C_R2|" inside a table row splits the cell (pandoc pipe tables); rendering fix only
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round7.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
