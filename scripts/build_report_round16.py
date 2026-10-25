"""Assemble the final report, revision 18: Part A = Round 16 (fixed-mu NaN-safe distillation and the non-cliff gate,
D138/D139) and the updated standalone manuscript (docs/manuscript_round16.md) with its verification
(reports/report_template_round16.md); Part B = revision 17's Part A (Round 15); Part C = Round 14 (the first manuscript
draft); Part D = Round 13; Part E = Round 12; Parts F–G = the L39/L38 parts; Parts H–R = rounds 11..1. Every Round-16
number is read from the saved records."""
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round15 as R15  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import manuscript_round16 as M  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL = "final_full_suite_round16"
R16 = REP / "round16"
VER = REP / "manuscript_verification_round16.json"
f3, f4, ci, prow, PH = R15.f3, R15.f4, R15.ci, R15.prow, R15.PH


def S():
    return json.loads((REP / "round16_summary.json").read_text())


def recs():
    out = []
    for f in sorted(R16.glob("results_lane*.jsonl")):
        out += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return out


def dev_checks():
    s = S()
    d = s["dev_3515"]
    rows = [[v, f3(d[k]["shape"]["mean_id"]) if d[k].get("shape") else "–", f"{d[k]['curve_error']:.4f}", f"{d[k]['ibs']:.4f}"]
            for k, v in (("after_pruning", "after pruning"), ("distill_fixed", "distillation, μ = 10, NaN-safe"))]
    out = md_table(["S-Add 3515 (Round-15 failure), Round-16 pipeline", "Identifiable RISE", "Curve error", "IBS"], rows)
    out += (f"\n\nFallback used: {'yes' if d['refit_fallback'] else 'no'}. Library formulas rejected by the |a x + b| ≤ 50 "
            "rule: " + "; ".join(f"`{r['formula']}` ({r['layer']} edge {r['i']}→{r['o']})" for r in d["rejected"]) + ".")
    g = s["dev_gate_rule_on_round15"]
    rows = [[lab, str(g[k]["n"]), str(g[k]["old_all_off"]), str(g[k]["new_all_off"]),
             "; ".join(f"seed {c[0]}: {c[1]} → {c[2]}" for c in g[k]["changed"]) or "–"]
            for k, lab in (("sparse", "S-Sparse (Round 15, 3600–3619)"), ("last", "S-Last (Round 15, 3700–3719)"))]
    return out + "\n\n" + md_table(["Saved Round-15 gate grids", "Replicates", "All off (D133 rule)",
                                    "All off (non-cliff rule)", "Changed decisions (channels kept)"], rows)


def exec_table():
    s = S()
    rr = recs()
    rows = []
    for k, lab in (("add", "S-Add"), ("sparse", "S-Sparse"), ("last", "S-Last")):
        w = [r["wall_seconds"] for r in rr if r["job_name"].startswith(k + "_")]
        rows.append([lab, str(len(w)), f"{statistics.median(w) / 60:.1f}" if w else "–"])
    t = (R16 / "started_utc.txt").read_text().strip()
    return md_table(["Jobs", "Count", "Median minutes"], rows) + \
        f"\n\nRecords: {s['n_records']}; failed jobs: {s['n_failed']}. Launched {t}."


def a_table():
    a = S()["family_a"]
    labels = [("after_pruning", "after pruning"), ("sub_noguard", "substituted, no refit (unrestricted formulas)"),
              ("refit_old", "old unconstrained refit (unrestricted formulas)"),
              ("sub_guard_r16", "guarded substitution (formulas with |a x + b| ≤ 50)"),
              ("distill_fixed", "**distillation, μ = 10, NaN-safe (the method)**")]
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
                 prow(f"{lab}: distilled − no refit (Uno C)", x["distill_minus_noguard_uno_c"]),
                 prow(f"{lab}: distilled − no refit (validation NLPL)", x["distill_minus_noguard_val_loss"])]
    out += "\n\n" + md_table(PH, rows)
    rows = [[lab, f"{a[k]['n_distill_finite']} / {a[k]['n']}", f"{a[k]['n_refit_old_finite']} / {a[k]['n']}",
             str(a[k]["n_fallback"]), str(a[k]["n_rejected_total"]), str(a[k]["n_guarded_total"]),
             f"{a[k]['n_substituted_mean_r15']:.2f} / {a[k]['n_substituted_mean_r16']:.2f}",
             f"{a[k]['recall_mean']:.3f}", f"{a[k]['fdr_mean']:.3f}"] for k, lab in (("add", "S-Add"), ("sparse", "S-Sparse"))]
    out += "\n\n" + md_table(["Scenario", "Distillation refits finite", "Old refits finite", "Fallbacks",
                              "Formulas rejected (|a x + b| > 50)", "Guarded edges", "Edges substituted (unrestricted / R16)",
                              "Recall", "FDR"], rows)
    rej = [[r["job_name"], f"`{f['formula']}`", f"{f['layer']} {f['i']}→{f['o']}"] for r in recs() if r["status"] == "ok"
           for f in r.get("symbolic", {}).get("formulas_r15_rejected", [])]
    if rej:
        out += "\n\n**Formulas accepted without the bound but rejected with it**\n\n" + md_table(["Job", "Formula", "Edge"], rej)
    return out


def b_table():
    b = S()["family_b"]
    ms = ("b2s_anchor_cox_noncliff", "b2s_anchor_cox_d133", "b2s_cox", "static_coxkan", "landmark_cox")
    rows = [[f"`{m}`", f"{b['last']['curve_error_means'][m]:.4f}", f"{b['sparse']['curve_error_means'][m]:.4f}",
             f"{b['last']['ibs_means'][m]:.4f}", f"{b['sparse']['ibs_means'][m]:.4f}"] for m in ms]
    out = md_table(["Model", "S-Last curve error", "S-Sparse curve error", "S-Last IBS", "S-Sparse IBS"], rows)
    rows = []
    for kind, lab in (("last", "S-Last"), ("sparse", "S-Sparse")):
        rows += [prow(f"{lab}: {k} (curve error)", p) for k, p in b[kind]["contrasts"].items()]
        rows += [prow(f"{lab}: {k} (IBS)", p) for k, p in b[kind]["contrasts_ibs"].items()]
    out += "\n\n" + md_table(PH, rows)
    rows = [[lab, m, str([round(x, 2) for x in b[k]["gate_on_frequency"][m]]), f"{b[k]['gate_all_off'][m]} / {b[k]['n']}"]
            for k, lab in (("last", "S-Last"), ("sparse", "S-Sparse")) for m in ("b2s_anchor_cox_noncliff", "b2s_anchor_cox_d133")]
    out += "\n\n" + md_table(["Scenario", "Gate rule", "Share of replicates keeping channel 1 / 2 / 3 (3 = null)",
                              "Gate all off"], rows)
    return out + "\n\nReplicates in which the two rules chose different gates: " + \
        "; ".join(f"{lab} {b[k]['gate_decisions_differ']} of {b[k]['n']}" for k, lab in (("last", "S-Last"), ("sparse", "S-Sparse"))) + "."


def ms_verification():
    v = json.loads(VER.read_text())
    rows = [["Ledger: every logged value re-read and re-formatted independently", f"{v['ledger']['n_values']} values",
             "PASS" if v["ledger"]["passed"] else "FAIL"],
            ["Re-render from the records reproduces the manuscript and the ledger", "byte-identical" if
             v["rerender"]["manuscript_identical"] else "differs", "PASS" if v["rerender"]["passed"] else "FAIL"],
            ["Number audit: every other number is a registered constant in a registered context",
             f"{v['numbers']['n_tokens']} numeric tokens; {v['numbers']['n_template_literals']} typed literals",
             "PASS" if v["numbers"]["passed"] else "FAIL"],
            ["Citations: every reference cited, every key listed", "–", "PASS" if v["citations"]["passed"] else "FAIL"],
            ["Required sections and figure files", f"{v['sections']['n_figures']} figure panels",
             "PASS" if v["sections"]["passed"] else "FAIL"]]
    if "pdf" in v:
        p = v["pdf"]
        rows.append(["Manuscript PDF: no blank page, all sections, all figures embedded",
                     f"{p['pages']} pages; {p['images']} images; blank pages: {len(p['blank_pages'])}",
                     "PASS" if p["passed"] else "FAIL"])
    rows.append(["**Overall**", "", "**ALL_PASS**" if v["ALL_PASS"] else "**FAIL**"])
    return md_table(["Check (`scripts/verify_manuscript.py --pdf`)", "Scope", "Result"], rows)


def decisions():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D1(3[8-9]|4[0-9]) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    rows = []
    for run in ("final_full_suite_round15", FINAL):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


TABLES = {"dev_checks": dev_checks, "exec_table": exec_table, "a_table": a_table, "b_table": b_table,
          "ms_verification": ms_verification, "decisions": decisions, "suite_table": suite_table,
          "test_runs": R2.test_run_history}


def verdict(x):
    return "Supported" if x else "Not supported"


def values():
    s = S()
    a, b = s["family_a"], s["family_b"]
    led = json.loads(M.B.LEDGER_PATH.read_text())
    ver = json.loads(VER.read_text())
    pages = re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(REP / "manuscript_round16.pdf")],
                                                         capture_output=True, text=True).stdout).group(1)
    pa, pb1, pb2 = (a["add"]["distill_minus_pruned_rise"], b["last"]["contrasts"]["b2s_anchor_cox_noncliff - b2s_cox"],
                    b["sparse"]["contrasts"]["b2s_anchor_cox_noncliff - b2s_cox"])
    out = {"git_head": R2.values()["git_head"], "n_records": str(s["n_records"]), "n_failed": str(s["n_failed"]),
           "a_verdict": verdict(a["H_R16a"]["supported"]), "b_verdict": verdict(b["H_R16b"]["supported"]),
           "a_finite": f"{a['H_R16a']['n_finite']} of {a['H_R16a']['n_cells']}",
           "a_fallback": str(a["add"]["n_fallback"] + a["sparse"]["n_fallback"]),
           "a_diff": f4(pa["mean_diff"]), "a_ci": ci(pa), "a_old": f4(a["add"]["distill_minus_oldrefit_rise"]["mean_diff"]),
           "b1_diff": f4(pb1["mean_diff"]), "b1_ci": ci(pb1), "b2_diff": f4(pb2["mean_diff"]), "b2_ci": ci(pb2),
           "b2_alloff": f"{b['sparse']['gate_all_off']['b2s_anchor_cox_noncliff']} of {b['sparse']['n']}",
           "b2_alloff_d133": f"{b['sparse']['gate_all_off']['b2s_anchor_cox_d133']} of {b['sparse']['n']}",
           "dev_old": str(s["dev_gate_rule_on_round15"]["sparse"]["old_all_off"]),
           "dev_rescued": str(s["dev_gate_rule_on_round15"]["sparse"]["old_all_off"] - s["dev_gate_rule_on_round15"]["sparse"]["new_all_off"]),
           "ms_pages": pages, "ms_values": str(led["n_values"]), "ms_refs": str(len(led["cited"])), "ms_title": M.B.TITLE,
           "ms_check": "verification ALL_PASS" if ver["ALL_PASS"] else "verification FAILED"}
    g = R1.TA[R1.TA.run_id == FINAL]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def shift_part(m):
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJKLMNOPQ" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def manuscript_for_report():
    txt = (ROOT / "docs/manuscript_round16.md").read_text()
    return txt.replace("](../reports/figures/", "](figures/")


def main():
    R15.main()                                   # revision 17 (itself rebuilding revisions 16 .. 1 from their records)
    prev = (REP / "final_research_prototype_report.md").read_text()
    R1.use_snapshot(FINAL)
    vals = values()
    tpl = (REP / "report_template_round16.md").read_text()
    out = tpl.replace("{{MANUSCRIPT}}", manuscript_for_report())
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), out)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    cut = prev.index("\n# Parts B–Q: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-15 summary", "## R15-0. Round-15 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R15-0.\1 ", own, flags=re.M)
    prev = re.sub(r"\b(Parts? )([B-Q])(–| and )?([B-Q])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("Part A is Round 15", "Part B is Round 15", 1)
    full = out + "\n\n# Part B: revision 17 (Round 15, 2026-10-25), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round16.json").write_text(json.dumps(vals, indent=1, ensure_ascii=False))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
