"""Assemble the final report, revision 17: Part A = Round 15 (Phase-2 methods; reports/report_template_round15.md);
Part B = revision 16's Part A (Round 14: the manuscript draft); Part C = Round 13; Part D = Round 12; Parts E–F = the
L39/L38 parts; Parts G–Q = rounds 11..1. Every Round-15 number is read from the saved records."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round14 as R14  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL = "final_full_suite_round15"
R15 = REP / "round15"


def S():
    return json.loads((REP / "round15_summary.json").read_text())


def jl(pattern):
    out = []
    for f in sorted(R15.glob(pattern)):
        out += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return out


def f3(v):
    return "–" if v is None else f"{v:.3f}"


def f4(v):
    return "–" if v is None else f"{v:+.4f}"


def ci(p, d=4):
    return f"[{p['ci95'][0]:+.{d}f}, {p['ci95'][1]:+.{d}f}]"


def prow(label, p, d=4):
    return [label, str(p["n"]), f"{p['mean_diff']:+.{d}f}", ci(p, d), f"{p['p_t']:.3g}", f"{p['n_neg']} / {p['n_pos']}"]


PH = ["Contrast", "Replicates", "Mean difference", "95% CI", "Paired-t p", "a < b / a > b"]


def dev_tables():
    return (R15 / "dev_tables.md").read_text()


def dev_checks():
    rows = []
    for r in jl("dev/results_check_lane*.jsonl"):
        if r.get("job") == "afcmlaml":
            for c in ("laml", "aic"):
                s = r[c]["shapes"]
                rows.append([f"C2, seed {r['seed']}", "LAML (primary)" if c == "laml" else "AIC-type (information)",
                             f3(s["phi1"]["rise_id"]), f3(s["phi2"]["rise_id"]), f"λ_x {r[c]['lam_x']:g}, λ_t {r[c]['lam_t']:g}",
                             ("meets" if r["laml_meets_ck6c_criterion"] else "does not meet") if c == "laml" else ""])
    out = md_table(["Check", "Selection", "RISE φ₁", "RISE φ₂", "Selected λ", "CK-6c criterion"], rows)
    rows = []
    for r in jl("dev/results_check_lane*.jsonl"):
        if r.get("job") in ("res", "res_dev_real"):
            for k, v in r["models"].items():
                rows.append([f"G, {r['scenario']} {r['seed']}", k, f3(v.get("curve_error")), f3(v["ibs"]),
                             str(r["res_on"]) if "cvgate" in k else ""])
    return out + "\n\n" + md_table(["Check", "Model", "Curve error", "IBS", "Channels kept by the CV gate"], rows)


def exec_table():
    s = S()
    recs = jl("results_lane*.jsonl")
    import statistics
    rows = []
    for k, lab in (("add", "S-Add"), ("sparse", "S-Sparse"), ("last", "S-Last"), ("real", "real data")):
        w = [r["wall_seconds"] for r in recs if r["job_name"].startswith(k + "_")]
        rows.append([lab, str(len(w)), f"{statistics.median(w) / 60:.1f}" if w else "–"])
    return md_table(["Jobs", "Count", "Median minutes"], rows) + f"\n\nRecords: {s['n_records']}; failed jobs: {s['n_failed']}."


def a_table():
    a = S()["family_a"]
    labels = [("after_pruning", "after pruning"), ("sub_noguard", "substituted, no refit"), ("refit_old", "old unconstrained refit"),
              ("sub_guard", "guarded substitution"), ("distill_mu0.01", "distillation, μ 0.01"),
              ("distill_mu0.1", "distillation, μ 0.1"), ("distill_mu1", "distillation, μ 1"),
              ("distill_mu10", "distillation, μ 10"), ("distill", "**distillation, μ chosen by validation (the method)**")]
    rows = [[lab, f3(a["add"]["rise_means"][k]), f3(a["sparse"]["rise_means"][k])] for k, lab in labels]
    out = md_table(["Model (B2-S Cox seed-0, pruned)", "S-Add mean RISE", "S-Sparse mean RISE"], rows)
    rows = []
    for kind, lab in (("add", "S-Add"), ("sparse", "S-Sparse")):
        x = a[kind]
        rows += [prow(f"{lab}: distilled − after pruning (RISE)", x["distill_minus_pruned_rise"]),
                 prow(f"{lab}: distilled − old refit (RISE)", x["distill_minus_oldrefit_rise"]),
                 prow(f"{lab}: distilled − old refit (curve error)", x["distill_minus_oldrefit_curve"]),
                 prow(f"{lab}: distilled − no refit (curve error)", x["distill_minus_noguard_curve_error"]),
                 prow(f"{lab}: distilled − no refit (IBS)", x["distill_minus_noguard_ibs"]),
                 prow(f"{lab}: distilled − no refit (validation NLPL)", x["distill_minus_noguard_val_loss"])]
    out += "\n\n" + md_table(PH, rows)
    rows = [[lab, f"{a[k]['n_distill_finite']} / {a[k]['n']}", f"{a[k]['n_refit_old_finite']} / {a[k]['n']}",
             ", ".join(f"μ {m}: {c}" for m, c in a[k]["mu_chosen"].items()), str(a[k]["n_guarded_total"]),
             f"{a[k]['recall_mean']:.3f}", f"{a[k]['fdr_mean']:.3f}"] for k, lab in (("add", "S-Add"), ("sparse", "S-Sparse"))]
    return out + "\n\n" + md_table(["Scenario", "Distillation refits finite", "Old refits finite", "μ chosen",
                                    "Guarded edges", "Recall", "FDR"], rows)


def b_table():
    b = S()["family_b"]
    ms = ("b2s_anchor_cox", "b2s_res_cox", "b2s_cox", "static_coxkan", "landmark_cox")
    rows = [[f"`{m}`", f3(b["last"]["curve_error_means"][m]), f3(b["sparse"]["curve_error_means"][m]),
             f3(b["last"]["ibs_means"][m]), f3(b["sparse"]["ibs_means"][m])] for m in ms]
    out = md_table(["Model", "S-Last curve error", "S-Sparse curve error", "S-Last IBS", "S-Sparse IBS"], rows)
    rows = []
    for kind, lab in (("last", "S-Last"), ("sparse", "S-Sparse")):
        rows += [prow(f"{lab}: {k} (curve error)", p) for k, p in b[kind]["contrasts"].items()]
        rows += [prow(f"{lab}: {k} (IBS)", p) for k, p in b[kind]["contrasts_ibs"].items()]
    out += "\n\n" + md_table(PH, rows)
    rows = [[lab, m, str([round(x, 2) for x in b[k]["gate_on_frequency"][m]]), f"{b[k]['gate_all_off'][m]} / {b[k]['n']}"]
            for k, lab in (("last", "S-Last"), ("sparse", "S-Sparse")) for m in ("b2s_anchor_cox", "b2s_res_cox")]
    return out + "\n\n" + md_table(["Scenario", "Model", "Share of replicates keeping channel 1 / 2 / 3 (3 = null)",
                                    "Gate all off"], rows)


def posthoc():
    p = json.loads((R15 / "posthoc_exploratory.json").read_text())
    rows = [prow(k, v) for k, v in p.items() if isinstance(v, dict) and "mean_diff" in v]
    out = md_table(PH, rows)
    rows = [[k.split(":")[0], m, f"{v:+.4f}"] for k, d in p.items() if "by chosen mu" in k for m, v in d.items()]
    out += "\n\n" + md_table(["Scenario", "μ chosen by validation", "Mean RISE change against after pruning"], rows)
    d = json.loads((R15 / "diag_3515.json").read_text())
    e = [x for x in d["edges"] if x["layer"] == "stat"][0]
    r = d["refits"]["0.01"]
    out += (f"\n\n**S-Add 3515.** The accepted formula on the baseline-covariate edge was `{e['f']}` with affine "
            f"parameters a = {e['p'][0]:.1f}, b = {e['p'][1]:.1f}, c = {e['p'][2]:.4f}, d = {e['p'][3]:.3f}. The first "
            f"non-finite evaluation was L-BFGS evaluation {r['first_nonfinite_eval']} (training loss before it "
            f"{r['last_finite']['loss']:.3f}); afterwards the substituted parameters were of order 1e+180.")
    return out


def real_table():
    s = S()
    rows = []
    for ds, v in s["real"].items():
        for m in ("b2s_anchor_cox", "b2s_res_cox", "b2s_cox", "landmark_cox", "static_coxkan"):
            mm = v["means"].get(m)
            if mm:
                rows.append([ds, f"`{m}`", f"{mm['ibs']:.3f}", f"{mm['uno_c']:.3f}"])
    out = md_table(["Data set", "Model (Round-15 new: anchor, res; others: Round-13 records)", "IBS", "Uno C"], rows)
    rows = []
    for ds, v in s["real"].items():
        for k in ("ibs b2s_anchor_cox - b2s_cox", "ibs b2s_anchor_cox - landmark_cox", "uno_c b2s_anchor_cox - b2s_cox",
                  "uno_c b2s_anchor_cox - landmark_cox"):
            p = v["contrasts"][k]
            rows.append([ds, k, f"{p['mean_diff']:+.3f}", ci(p, 3)])
    out += "\n\n" + md_table(["Data set", "Contrast", "Mean difference", "Nadeau–Bengio 95% interval"], rows)
    rows = [[ds, c, a["direction"], f"{a['share_monotone']:.1f}", str(a["n_splits"])]
            for ds, v in s["real"].items() for c, a in v["last_value_alignment"].items()]
    out += "\n\n" + md_table(["Data set", "Last-value edge", "Stated direction of risk", "Share of splits monotone",
                              "Splits"], rows)
    rep = s["real_repro_landmark_cox_split0"]
    return out + "\n\nSplit-0 reproduction of the Round-13 landmark Cox records (maximum absolute metric difference): " + \
        "; ".join(f"{k} {v['max_abs_diff']:.1g}" for k, v in rep.items()) + "."


def decisions():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D13[3-9] \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    rows = []
    for run in ("final_full_suite_round14_rerun2", FINAL):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


TABLES = {"dev_tables": dev_tables, "dev_checks": dev_checks, "exec_table": exec_table, "a_table": a_table,
          "b_table": b_table, "posthoc": posthoc, "real_table": real_table, "decisions": decisions,
          "suite_table": suite_table, "test_runs": R2.test_run_history}


def values():
    s = S()
    a, b = s["family_a"], s["family_b"]
    out = {"git_head": R2.values()["git_head"], "n_records": str(s["n_records"]), "n_failed": str(s["n_failed"]),
           "a_finite": f"{a['H_R15a']['n_finite']} of {a['H_R15a']['n_cells']}",
           "a_diff": f4(a["add"]["distill_minus_pruned_rise"]["mean_diff"]), "a_ci": ci(a["add"]["distill_minus_pruned_rise"], 3),
           "a_old": f4(a["add"]["distill_minus_oldrefit_rise"]["mean_diff"]),
           "b1_diff": f4(b["last"]["contrasts"]["b2s_anchor_cox - b2s_cox"]["mean_diff"]),
           "b1_ci": ci(b["last"]["contrasts"]["b2s_anchor_cox - b2s_cox"]),
           "b2_diff": f4(b["sparse"]["contrasts"]["b2s_anchor_cox - b2s_cox"]["mean_diff"]),
           "b2_ci": ci(b["sparse"]["contrasts"]["b2s_anchor_cox - b2s_cox"]),
           "res_sparse": f4(b["sparse"]["contrasts"]["b2s_res_cox - b2s_cox"]["mean_diff"])}
    g = R1.TA[R1.TA.run_id == FINAL]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def shift_part(m):
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJKLMNOP" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def main():
    R14.main()                                   # revision 16 (itself rebuilding revisions 15 .. 1 from their records)
    prev = (REP / "final_research_prototype_report.md").read_text()
    R1.use_snapshot(FINAL)
    vals = values()
    tpl = (REP / "report_template_round15.md").read_text()
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    cut = prev.index("\n# Parts B–P: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-14 summary", "## R14-0. Round-14 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R14-0.\1 ", own, flags=re.M)
    own = own.replace("\n# Part A: Manuscript draft (Round 14)\n", "\n")
    prev = re.sub(r"\b(Parts? )([B-P])(–| and )?([B-P])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("Part A is Round 14", "Part B is Round 14", 1)
    full = out + "\n\n# Part B: revision 16 (Round 14 manuscript draft, 2026-10-23), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round15.json").write_text(json.dumps(vals, indent=1, ensure_ascii=False))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
