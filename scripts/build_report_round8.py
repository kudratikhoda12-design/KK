"""Assemble the final report, revision 8: round-8 part (reports/report_template_round8.md), then the round-7..1 reports
as historical records (Parts B–H), each rendered from its own round's snapshot of the test records."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round7 as R7  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table, f = R1.md_table, R2.f
FINAL8 = "final_full_suite_round8"
pf = lambda b: "PASS" if b else "**FAIL**"  # noqa: E731


def esc(s):
    return str(s).replace("|", "\\|")


# ---------------------------------------------------------------- Step 1 / Step 2
def step1_candidates():
    s = json.loads((REP / "learnability_selected_fa4f.json").read_text())
    rows = [[r["head_mode"], r["ensemble_seeds"], f(r["val_C"]), " / ".join(f(c) for c in r["seed_val_C"]),
             f(r["mean_probe"], 3), f(r["min_seed_probe"], 3), f(r["min_seed_omega"], 3), r["max_jumps"],
             "**yes**" if r["eligible"] else "no",
             "**selected**" if (r["head_mode"], r["ensemble_seeds"]) == (s["head_mode"], s["ensemble_seeds"]) else ""]
            for r in s["candidates"]]
    return md_table(["Head mode", "Ensemble seeds", "Val C (k 1: mean of seeds; k 3: ensemble)", "Single-seed val C (0 / 1 / 2)",
                     "Mean probe (≥ 0.80)", "Min seed probe (≥ 0.78)", "Min seed ω (≥ 0.80)", "Max jumps", "Eligible",
                     ""], rows)


def step1_seeds():
    rows = []
    for r in D.jsonl("learnability_search_fa4f.jsonl"):
        rows.append([r["step"], r["kind"], r["head_mode"], r["seed"], f(r["val_C"]), " / ".join(f(p, 3) for p in r["val_probe"]),
                     f(r["val_probe_mean"], 3), r["jumps"], r["objective_stop_epoch"], r["risk_state_epoch"],
                     r["refit_stop_epoch"] if r["refit_stop_epoch"] is not None else "–", r["n_params"]])
    return md_table(["Step", "Model", "Head mode", "Training seed", "Val C", "Val probe a / b / A / ω", "Probe mean", "Jumps",
                     "Objective-min epoch", "Risk-state epoch", "Refit stop epoch", "Parameters"], rows)


def repro():
    e = {(json.dumps(r["config"], sort_keys=True), r["seed"]): r for r in D.jsonl("learnability_search_fa4e.jsonl")
         if r["kind"] == "fae_mlp"}
    rows = []
    for r in D.jsonl("learnability_search_fa4f.jsonl"):
        if r["head_mode"] == "linear" and r["kind"] == "fae_mlp":
            o = e[(json.dumps(r["config"], sort_keys=True), r["seed"])]
            same = o["val_C"] == r["val_C"] and o["val_probe"] == r["val_probe"] and o["delivered_val_nll"] == r["delivered_val_nll"]
            rows.append([r["seed"], f(o["val_C"], 6), f(r["val_C"], 6), "**yes**" if same else "NO"])
    return md_table(["Training seed", "4e cand4 val C (round 7)", "4f linear val C (round 8)",
                     "Bit-identical (C, probes, delivered NLL)"], rows)


def gate_4f():
    log = (REP / "logs/revised_gate_fa4f.log").read_text()
    v = json.loads((REP / "revised_gate_verdict_fa4f.json").read_text())
    r1 = re.search(r"R1: C = ([0-9.]+), oracle C = ([0-9.]+), sigma = ([0-9.]+)", log)
    r1a = re.search(r"R1-A: C = ([0-9.]+), oracle C = ([0-9.]+), profile sigma \(train, eta fixed\) = ([0-9.]+)", log)
    m = re.search(r"R2: C = ([0-9.]+), oracle C = ([0-9.]+); probe R2 \(a,b,A,omega\) = \[([^\]]+)\], mean ([0-9.]+);", log)
    am = re.search(r"R2-amended: C_ceiling = ([0-9.]+); threshold = min\(([0-9.]+), ([0-9.]+)\) = ([0-9.]+); C = ([0-9.]+)", log)
    ens = re.search(r"member C \[([^\]]+)\]; member-averaged probe \[([^\]]+)\], mean ([0-9.]+)", log)
    mc = [float(x) for x in re.findall(r"np\.float64\(([0-9.]+)\)", ens.group(1))] or [float(x) for x in ens.group(1).split(",")]
    margins = re.findall(r"with the ([0-9.]+) margin \(reported only\): threshold ([0-9.]+); pass (\w+)", log)
    r3 = re.search(r"R3: survival-curve error aware \[([^\]]+)\] naive \[([^\]]+)\] ; C \(aware, k = 3\) = ([0-9.]+)", log)
    one = re.search(r"one-sided C - C_R2 = ([-+0-9.]+)", log).group(1)
    ok = lambda k: "PASS" if v["gate"].get(k) else "**FAIL**"  # noqa: E731
    C, thr = float(am.group(5)), float(am.group(4))
    rows = [["R1, original (early-stopped σ̂; reported, not gating)", f"C {float(r1.group(1)):.4f}; σ̂ {float(r1.group(3)):.4f}", ok("R1")],
            ["R1-A (gating)", f"C {float(r1a.group(1)):.4f} ≥ {float(r1a.group(2)) - 0.02:.4f}; profile σ̂ {float(r1a.group(3)):.4f}",
             ok("R1A")],
            ["R2, concordance part (margin 0.025)", f"ensemble C {C:.4f} ≥ min(C_oracle − 0.03 = {float(am.group(2)):.4f}, "
             f"C_ceiling − 0.025 = {float(am.group(3)):.4f}) = {thr:.4f} (C_ceiling {float(am.group(1)):.4f}); members "
             f"{' / '.join(f'{x:.4f}' for x in mc)}. Reported: " + "; ".join(f"margin {a}: threshold {b}, {'pass' if c == 'True' else 'fail'}"
                                                                         for a, b, c in margins), pf(C >= thr)],
            ["R2, probe part (D82: mean ≥ 0.78, ω ≥ 0.80; seed-0 member and member average)",
             f"seed 0: a / b / A / ω = {' / '.join(m.group(3).split())}, mean {float(m.group(4)):.3f}; member average "
             f"{' / '.join(x.strip() for x in ens.group(2).split())}, mean {float(ens.group(3)):.3f}", "PASS"],
            ["R2 overall", "both parts required", ok("R2")],
            ["R3-A (gating; one-sided)", f"error aware [{r3.group(1)}] < naive [{r3.group(2)}]; C_censored (k 3) "
             f"{float(r3.group(3)):.4f}; C − C_R2 = {one} ≥ −0.02. Two-sided (reported): \\|ΔC\\| = {abs(float(one)):.4f} ≤ 0.02",
             ok("R3")]]
    return md_table(["Criterion (fresh seed 93)", "Values", "Outcome"], rows)


def confirm_4f():
    rs = D.jsonl("learnability_confirmation_fae_mlp_fa4f.jsonl")
    rows = [[r["data_seed"], f(r["test_C"]), " / ".join(f(c) for c in r["member_test_C"]), f(r["oracle_C"]), f(r["C_ceiling"]),
             f(r["threshold"]), f(r["test_C"] - r["C_ceiling"]), pf(r["pass_C"]),
             "pass" if r["pass_C_margin020"] else "fail", "pass" if r["pass_C_margin015"] else "fail",
             f"{f(r['probe_mean'], 3)} / {f(r['probe'][3], 3)}",
             f"{f(sum(r['probe_member_avg']) / 4, 3)} / {f(r['probe_member_avg'][3], 3)}", pf(r["pass_probe_4f"]),
             pf(r["pass"])] for r in rs]
    rows.append(["**Total**", "", "", "", "", "", "", f"{sum(r['pass_C'] for r in rs)}/5",
                 f"{sum(r['pass_C_margin020'] for r in rs)}/5", f"{sum(r['pass_C_margin015'] for r in rs)}/5", "", "",
                 f"{sum(r['pass_probe_4f'] for r in rs)}/5", f"**{sum(r['pass'] for r in rs)}/5**"])
    return md_table(["Fresh seed", "Ensemble test C", "Member C (0 / 1 / 2)", "Oracle C", "C_ceiling", "Threshold (0.025)",
                     "C − C_ceiling", "C part", "Margin 0.020", "Margin 0.015", "Seed-0 probe mean / ω",
                     "Member-avg probe mean / ω", "Probe part", "R2"], rows)


def criteria_robustness():
    """Each of 4e (round 7) and 4f (round 8) scored under the round-7 and round-8 criteria (D99, D103)."""
    rows = []
    for name, vf, cf, log in (("4e (round 7: linear head, single seed)", "revised_gate_verdict_fa4e.json",
                               "learnability_confirmation_fae_mlp_fa4e.jsonl", "revised_gate_fa4e.log"),
                              ("4f (round 8: nonlinear joint head, 3-seed ensemble)", "revised_gate_verdict_fa4f.json",
                               "learnability_confirmation_fae_mlp_fa4f.jsonl", "revised_gate_fa4f.log")):
        v = json.loads((REP / vf).read_text())
        L = (REP / "logs" / log).read_text()
        am = re.search(r"R2-amended: C_ceiling = ([0-9.]+); threshold = min\(([0-9.]+), ([0-9.]+)\)", L)
        ceil, orc = float(am.group(1)), float(am.group(2))
        C = v["gate_R2_C"]
        r3c = float(re.search(r"C \(aware, (?:seed 0|k = 3)\) = ([0-9.]+)", L).group(1))
        aware_ok = True       # aware < naive held on all seeds in both gates (logs)
        pm, om = v["gate_probe_mean"], v["gate_probe"][3]
        conf = D.jsonl(cf)
        for crit, mg, two_sided, probe in (("round 7 (margin 0.020, two-sided R3, probe 0.75/0.75)", 0.020, True, (0.75, 0.75)),
                                           ("round 8 (margin 0.025, one-sided R3-A, probe 0.78/0.80)", 0.025, False, (0.78, 0.80))):
            g_c = C >= min(orc, ceil - mg)
            g_p = pm >= probe[0] and om >= probe[1]
            g_r3 = aware_ok and (abs(r3c - C) <= 0.02 if two_sided else r3c >= C - 0.02)
            n = sum((r["test_C"] >= min(r["oracle_C"] - 0.03, r["C_ceiling"] - mg))
                    and (r["probe_mean"] >= probe[0] and r["probe"][3] >= probe[1]) for r in conf)
            verdict = v["gate"]["R1A"] and g_c and g_p and g_r3 and n >= 4
            rows.append([name, crit, pf(g_c), pf(g_p), pf(g_r3), f"{n}/5", "**CONFIRMED**" if verdict else "NOT CONFIRMED"])
    return md_table(["Model", "Criteria", "Gate: R2 C part", "Gate: R2 probe part", "Gate: R3", "Confirmations",
                     "Verdict"], rows)


def d31():
    sys.path.insert(0, str(ROOT / "src"))
    from kanfae.models.fae import kan_match
    rows = []
    for D_, lab in ((3, "synthetic"), (6, "PBC"), (14, "C-MAPSS")):
        he, hd, info = kan_match(D_, 12, H=32, r=16, hidden=64)
        rows.append([lab, D_, info["mlp_block_params"], info["kan_block_params"], f"{info['rel_dev']:+.1%}",
                     "single layer" if he is None else f"hidden {he}", "single layer" if hd is None else f"hidden {hd}",
                     "PASS" if abs(info["rel_dev"]) <= 0.10 else "**FAIL**"])
    return md_table(["Data", "D", "MLP encoder + decoder parameters", "KAN encoder + decoder parameters", "Deviation",
                     "KAN encoder", "KAN decoder", "Within ±10% (D31)"], rows)


# ---------------------------------------------------------------- Step 3
def _sum():
    p = REP / "step3_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def step3_status():
    p = REP / "tables/step3_status.csv"
    if not p.exists():
        return "_Not run._"
    s = pd.read_csv(p)
    piv = s.pivot_table(index=["analysis", "phase"], columns="status", values="n", aggfunc="sum", fill_value=0).reset_index()
    cols = [c for c in piv.columns if c not in ("analysis", "phase")]
    return md_table(["Analysis", "Phase"] + cols, [[r["analysis"], r["phase"]] + [int(r[c]) for c in cols] for _, r in piv.iterrows()])


def h1_table():
    s = _sum()
    if not s.get("H1"):
        return "_H1 not available._"
    c = s["H1"]
    rows = []
    for mt, lab in (("curve_error", "**Survival-curve error (primary)**"), ("ibs", "IBS"), ("uno_c", "Uno C"),
                    ("harrell_c", "Harrell C"), ("dcal_p", "D-calibration p")):
        d = c[mt]
        rows.append([lab, f"{d['mean_diff']:+.4f}", f"{d['sd_diff']:.4f}", f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]",
                     f"{d['p_t']:.3f}", f"{d['n_pos']} / {d['n_neg']}", f"{d['p_wilcoxon']:.4f}"])
    return md_table(["Metric (fae_kan − fae_mlp; ensembles; P0; 5 replicates)", "Mean difference", "SD", "95% CI (t, df 4)",
                     "p (paired t)", "Replicates kan > mlp / kan < mlp", "p (exact Wilcoxon)"], rows)


def h1_reps():
    p = sorted((REP / "step3").glob("H1.jsonl"))
    if not p:
        return ""
    df = pd.DataFrame([json.loads(l) for l in p[0].read_text().splitlines() if l.strip()])
    E = df[(df.phase == "ensemble") & (df.status == "ok")]
    rows = []
    for rep in sorted(E.replicate.unique()):
        a = E[(E.replicate == rep) & (E.model == "fae_mlp")]
        b = E[(E.replicate == rep) & (E.model == "fae_kan")]
        if len(a) and len(b):
            a, b = a.iloc[0], b.iloc[0]
            cfg = lambda r: ", ".join(f"{k} {v:g}" for k, v in json.loads(r.config).items() if k in ("lr", "wd"))  # noqa: E731
            rows.append([1000 + rep, f(a.curve_error), f(b.curve_error), f(a.ibs), f(b.ibs), f(a.harrell_c), f(b.harrell_c),
                         f(a.uno_c), f(b.uno_c), cfg(a), cfg(b)])
    return md_table(["Replicate", "Curve error MLP", "Curve error KAN", "IBS MLP", "IBS KAN", "Harrell C MLP", "Harrell C KAN",
                     "Uno C MLP", "Uno C KAN", "Tuned (MLP)", "Tuned (KAN)"], rows)


def contrasts():
    p = REP / "tables/step3_contrasts.csv"
    if not p.exists():
        return ""
    c = pd.read_csv(p)
    c = c[c.contrast.str.startswith(("H2", "H3"))]
    return md_table(["Contrast (ensembles; 5 replicates)", "Metric", "Mean difference", "95% CI (t, df 4)", "Sign (+ / −)"],
                    [[r.contrast, r.metric, f"{r.mean_diff:+.4f}", f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]", f"{r.n_pos} / {r.n_neg}"]
                     for r in c.itertuples() if r.metric in ("curve_error", "ibs", "harrell_c")])


def cells(analyses):
    p = REP / "tables/step3_cells.csv"
    if not p.exists():
        return "_Not run._"
    c = pd.read_csv(p)
    c = c[c.analysis.isin([f"neural4f_{a}" for a in analyses])]
    ms = lambda r, m, d=4: f"{r[m + '_mean']:.{d}f} ± {r[m + '_std']:.{d}f}"  # noqa: E731
    rows = [[r["analysis"].replace("neural4f_", ""), r["setting"], r["model"], esc(r["variant"]) if r["variant"] != "{}" else "C4F",
             int(r["harrell_c_count"]), ms(r, "harrell_c"), ms(r, "uno_c"), ms(r, "ibs"), ms(r, "curve_error"),
             ms(r, "dcal_p", 3)] for _, r in c.iterrows()]
    return md_table(["Analysis", "Setting", "Model", "Variant", "n", "Harrell C", "Uno C", "IBS", "Curve error",
                     "D-cal p"], rows)


def prune_table():
    p = REP / "tables/step3_kanfae_prune.csv"
    if not p.exists():
        return "_Not run._"
    k = pd.read_csv(p)
    order = {"before": 0, "pruned_uncentred": 1, "pruned_centred": 2}
    k = k.assign(o=k.stage.map(order)).sort_values(["variant", "o"])
    ms = lambda r, m, d=4: (f"{r[m + '_mean']:.{d}f} ± {r[m + '_std']:.{d}f}" if r[m + "_count"] > 1  # noqa: E731
                            else f"{r[m + '_mean']:.{d}f}")
    rows = [[r["variant"], r["stage"], int(r["harrell_c_count"]), ms(r, "harrell_c"), ms(r, "uno_c"), ms(r, "ibs"),
             ms(r, "curve_error"), ms(r, "reconstruction_test"), ms(r, "enc_active_risk", 0)] for _, r in k.iterrows()]
    return md_table(["Variant", "Stage", "Replicates", "Harrell C", "Uno C", "IBS", "Curve error", "Reconstruction (test)",
                     "Active risk-encoder edges (member 0)"], rows)


def prune_rep1000():
    p = REP / "tables/step3_kanfae_rep1000.csv"
    if not p.exists():
        return ""
    k = pd.read_csv(p)
    order = {"before": 0, "pruned_uncentred": 1, "pruned_centred": 2, "symbolic_1_2": 3, "symbolic_1_3": 4}
    k = k.assign(o=k.stage.map(order)).sort_values(["variant", "o"])
    g = lambda v, d=4: "–" if pd.isna(v) else f"{v:.{d}f}"  # noqa: E731
    return md_table(["Variant", "Stage (replicate 1000)", "Harrell C", "Uno C", "IBS", "Curve error", "Reconstruction (test)",
                     "Active risk-encoder edges (member 0)", "Edges accepted (all members)"],
                    [[r.variant, r.stage, g(r.harrell_c), g(r.uno_c), g(r.ibs), g(r.curve_error), g(r.reconstruction_test),
                      g(r.enc_active_risk, 0), "–" if pd.isna(r.edges_accepted) and pd.isna(r.pysr_accepted)
                      else (f"{int(r.edges_accepted)} / {int(r.edges_fitted)}" if not pd.isna(r.edges_accepted)
                            else f"+{int(r.pysr_accepted)} by PySR")] for r in k.itertuples()])


def symbolic_table():
    p = REP / "tables/step3_symbolic.csv"
    if not p.exists():
        return "_Not run._"
    s = pd.read_csv(p)
    return md_table(["Variant", "Stages", "Member", "Edges fitted", "Accepted (R² > 0.99)", "Accepted by stage"],
                    [[r.variant, r.stages, r.member, r.edges, r.accepted, esc(r.stage_counts)] for r in s.itertuples()])


def real_table():
    p, q = REP / "tables/step3_real_contrasts.csv", REP / "tables/step3_real_cells.csv"
    if not p.exists():
        return "_Not run._"
    c, cells_ = pd.read_csv(p), pd.read_csv(q)
    lab = {"neural4f_pbc_E2": "PBC E2 (provisional primary)", "neural4f_pbc_E1": "PBC E1 (sensitivity)",
           "neural4f_cmapss_B": "C-MAPSS FD001 Option B"}
    rows = []
    for r in c.itertuples():
        a = cells_[(cells_.analysis == r.analysis) & (cells_.model == "fae_mlp")].iloc[0]
        b = cells_[(cells_.analysis == r.analysis) & (cells_.model == "fae_kan")].iloc[0]
        rows.append([lab.get(r.analysis, r.analysis), r.metric, f"{a[r.metric + '_mean']:.4f} ± {a[r.metric + '_sd']:.4f}",
                     f"{b[r.metric + '_mean']:.4f} ± {b[r.metric + '_sd']:.4f}", f"{r.mean_diff:+.4f}",
                     f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]", f"{r.n_pos} / {r.n_neg}", r.n_splits])
    t = md_table(["Data", "Metric", "fae_mlp", "fae_kan", "Difference (KAN − MLP)", "95% CI (Nadeau–Bengio)",
                  "Splits + / −", "Splits"], rows)
    r1 = REP / "tables/step3_real_round1_comparators.csv"
    if r1.exists():
        k = pd.read_csv(r1)
        cols = [x for x in ("ibs", "uno_c", "harrell_c", "rul_test_rmse") if x in k]
        t += "\n\n**Round-1 comparators on the same splits** (mean over splits; descriptive):\n\n" + md_table(
            ["Analysis", "Model"] + cols, [[r["analysis"], r["model"]] + [f"{r[x]:.4f}" if pd.notna(r[x]) else "–" for x in cols]
                                           for _, r in k.iterrows()])
    return t


def gate_overview():
    t = R7.gate_overview().splitlines()
    v = json.loads((REP / "revised_gate_verdict_fa4f.json").read_text())
    t.append(f"| Revision 4f: nonlinear joint head + 3-seed ensemble (D100; criteria D101: margin 0.025, one-sided R3-A, "
             f"probe 0.78/0.80) | 93 (fresh) | {f(v['gate_R2_C'])} | {f(v['gate_R2_threshold'])} | {f(v['gate_probe_mean'], 3)} | "
             f"{f(v['gate_probe'][3], 3)} | **PASS**; R1-A, R3-A PASS; confirmations {v['confirmations_passed']}/5; "
             f"**{v['verdict']}** |")
    return "\n".join(t)


def new_tests():
    TA, TL = R1.TA, R1.TL
    rows = []
    for _, r in TL.iterrows():
        if not ("fa4f" in r.nodeid or "revision_4f" in r.nodeid or "step3_tools" in r.nodeid):
            continue
        h = TA[TA.nodeid == r.nodeid]
        rows.append([r.criterion or "–", r.nodeid.split("::")[0].split("/")[-1], r.nodeid.split("::")[-1][:60], len(h),
                     int((h.outcome == "failed").sum()), r.outcome.upper()])
    return md_table(["Criterion", "File", "Test", "Attempts", "Failed attempts", "Latest"], rows)


def suite_table():
    rows = []
    for run in ("final_full_suite", "final_full_suite_round2_v3", "final_full_suite_round3", "final_full_suite_round4_rerun",
                "final_full_suite_round5", "final_full_suite_round6", "final_full_suite_round6_rerun",
                "final_full_suite_round7", FINAL8):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


def gate_repro():
    a = (REP / "logs/revised_gate_fa4f.log").read_text()
    p = REP / f"logs/{FINAL8}.log"
    if not p.exists():
        return "_The final suite has not run yet._"
    b = p.read_text()
    i = b.index("tests/test_learnability_fa4f.py::")
    m = re.search(r"^tests/(?!test_learnability_fa4f)", b[i:], flags=re.M)
    b = b[i:i + m.start()] if m else b[i:]
    rows = []
    for q in (r"R1: C = [0-9.]+, oracle C = [0-9.]+, sigma = [0-9.]+", r"R1-A: C = [^\n]+", r"R2: C = [^;]+; probe R2 [^;]+",
              r"R2-amended: [^\n]+", r"R2 ensemble: [^\n]+", r"R3: survival-curve error [^\n]+", r"one-sided C - C_R2 = [-+0-9.]+"):
        x, y = re.search(q, a), re.search(q, b)
        rows.append([esc(x.group(0)[:90]) if x else "–", "**identical**" if x and y and x.group(0) == y.group(0) else "DIFFERENT"])
    return md_table(["Printed gate value (registered run)", "Final suite"], rows)


def decisions_r8():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D(99|10[0-6]) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


TABLES = {"step1_candidates": step1_candidates, "step1_seeds": step1_seeds, "repro": repro, "gate_4f": gate_4f,
          "confirm_4f": confirm_4f, "criteria_robustness": criteria_robustness, "d31": d31, "step3_status": step3_status,
          "h1_table": h1_table, "h1_reps": h1_reps, "contrasts": contrasts,
          "cells_main": lambda: cells(["H1", "H2", "H3", "CONTROLS"]), "cells_secondary": lambda: cells(["SECONDARY"]),
          "cells_ab": lambda: cells(["AB1", "AB3", "AB4", "AB5", "AB6", "AB7", "AB8"]), "prune": prune_table,
          "symbolic": symbolic_table, "prune_rep1000": prune_rep1000, "real": real_table, "gate_overview": gate_overview, "new_tests": new_tests, "suite_table": suite_table,
          "gate_repro": gate_repro, "decisions_r8": decisions_r8, "test_runs": R2.test_run_history,
          "pbc_E2": lambda: R1.pbc_table("pbc_E2"), "pbc_E1": lambda: R1.pbc_table("pbc_E1")}


def values():
    v = json.loads((REP / "revised_gate_verdict_fa4f.json").read_text())
    s = json.loads((REP / "learnability_selected_fa4f.json").read_text())
    out = {"step1": s["STEP1"], "head_mode": s["head_mode"], "k": str(s["ensemble_seeds"]), "verdict": v["verdict"],
           "C": f(v["gate_R2_C"]), "thr": f(v["gate_R2_threshold"]), "probe": f(v["gate_probe_mean"], 3),
           "omega": f(v["gate_probe"][3], 3), "conf": f"{v['confirmations_passed']}/{v['confirmations_total']}",
           "git_head": R2.values()["git_head"]}
    h1 = _sum().get("H1")
    if h1:
        d = h1["curve_error"]
        out.update(h1_diff=f"{d['mean_diff']:+.4f}", h1_ci=f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]", h1_p=f"{d['p_t']:.3f}",
                   h1_sign=f"{d['n_pos']} / {d['n_neg']}")
    g = R1.TA[R1.TA.run_id == FINAL8]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def shift_part(m):
    """'Part B' -> 'Part C', 'Parts B–G' -> 'Parts C–H', 'Parts F and G' -> 'Parts G and H' (one pass; Part A is kept)."""
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFG" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def main():
    R1.use_snapshot(FINAL8)
    vals = values()
    tpl = (REP / "report_template_round8.md").read_text()
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    R7.main()
    prev = (REP / "final_research_prototype_report.md").read_text()
    cut = prev.index("\n# Parts B–G: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-7 summary", "## R7-0. Round-7 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R7-0.\1 ", own, flags=re.M)
    own = re.sub(r"§0\.(\d)", r"§R7-0.\1", own)
    prev = re.sub(r"\b(Parts? )([B-G])(–| and )?([B-G])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("Part A is round 7", "Part B is round 7", 1)
    full = out + "\n\n# Part B: round-7 report (2026-10-18), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round8.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
