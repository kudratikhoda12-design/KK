"""Assemble the final report, revision 19: Part A = Round 17 (two-way gated landmark anchoring for Cox and log-normal
KAN-AFT; fixed-mu symbolic distillation for KAN-AFT; D141-D143) and the updated standalone manuscript
(docs/manuscript_round17.md) with its verification (reports/report_template_round17.md); Part B = revision 18's Part A
(Round 16 and the Round-16 manuscript); Part C = Round 15; Part D = Round 14; Part E = Round 13; Part F = Round 12;
Parts G–H = the L39/L38 parts; Parts I–S = rounds 11..1. Every Round-17 number is read from the saved records."""
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round15 as R15  # noqa: E402
import build_report_round16 as R16B  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import manuscript_round17 as M  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL = "final_full_suite_round17"
R17 = REP / "round17"
VER = REP / "manuscript_verification_round17.json"
f3, f4, ci, prow, PH = R15.f3, R15.f4, R15.ci, R15.prow, R15.PH
LAB = {"last": "S-Last", "sparse": "S-Sparse", "misspec": "S-Misspec"}


def S():
    return json.loads((REP / "round17_summary.json").read_text())


def DEVS():
    return json.loads((R17 / "dev_summary.json").read_text())


def recs(path=R17):
    out = []
    for f in sorted(path.glob("results_lane*.jsonl")):
        out += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return out


def dev_checks():
    d = DEVS()
    rows = []
    for loss in ("cox", "aft"):
        for sc in (("sparse", "last") if loss == "cox" else ("misspec", "sparse", "last")):
            for l, p in d["sweep"][loss][sc].items():
                rows.append([f"{loss.upper()} {LAB[sc]}", l, f"{p['n_anchored']} / {p['n']}", f"{p['mean_diff']:+.4f}",
                             ci(p) + (" ←" if float(l) == d["last_tol"][loss] else "")])
    out = md_table(["Development replicates", "last_tol", "Anchored", "Two-way − comparator (curve error)",
                    "95% CI (← chosen)"], rows)
    a = d["aft_symbolic"]
    out += (f"\n\nChosen: Cox last_tol {d['last_tol']['cox']} ({d['last_tol']['cox_reason']}), AFT last_tol "
            f"{d['last_tol']['aft']} ({d['last_tol']['aft_reason']}). Round-16 b2s_cox reproduction: "
            f"{d['r16_reproduction_b2s_cox']['n']} replicates, maximum curve-error difference "
            f"{d['r16_reproduction_b2s_cox']['max_abs_diff']:.1g}. AFT distillation on S-Misspec 961–969: "
            f"{a['n_distill_finite']} / {a['n']} finite, {a['n_fallback']} fallbacks; mean RISE after pruning "
            f"{a['rise_means']['after_pruning']:.3f}, distilled {a['rise_means']['distill_fixed']:.3f}, unanchored refit "
            f"{a['rise_means']['refit_plain']:.3f}. Go/no-go: {'GO' if d['go']['go'] else 'NO-GO'}.")
    return out


def exec_table():
    s = S()
    rr = recs()
    rows = []
    for k in ("last", "sparse", "misspec"):
        w = [r["wall_seconds"] for r in rr if r["job_name"].startswith(k + "_")]
        rows.append([LAB[k], str(len(w)), f"{statistics.median(w) / 60:.1f}" if w else "–"])
    t = (R17 / "started_utc.txt").read_text().strip()
    return md_table(["Jobs", "Count", "Median minutes"], rows) + \
        f"\n\nRecords: {s['n_records']}; failed jobs: {s['n_failed']}. Launched {t}."


def a_table():
    s = S()
    rows = []
    for loss, kinds in (("cox", ("last", "sparse")), ("aft", ("last", "sparse", "misspec"))):
        for k in kinds:
            x = s[loss][k]
            cm, im = x["curve_error_means"], x["ibs_means"]
            rows.append([f"{LAB[k]}, {loss.upper()}"] + [f"{cm.get(m, float('nan')):.4f}" for m in
                                                         (f"b2s_{loss}", f"b2s_anchor_{loss}_twoway", "landmark_cox")] +
                        [f"{im.get(m, float('nan')):.4f}" for m in (f"b2s_{loss}", f"b2s_anchor_{loss}_twoway")])
    out = md_table(["Scenario, family", "Curve error: comparator", "two-way", "landmark Cox", "IBS: comparator", "two-way"],
                   rows).replace("nan", "–")
    rows = []
    for loss, kinds in (("cox", ("last", "sparse")), ("aft", ("last", "sparse", "misspec"))):
        for k in kinds:
            x = s[loss][k]
            rows += [prow(f"{LAB[k]} {loss.upper()}: two-way − comparator (curve error)", x["contrast"]),
                     prow(f"{LAB[k]} {loss.upper()}: two-way − comparator (IBS)", x["contrast_ibs"]),
                     prow(f"{LAB[k]} {loss.upper()}: anchored alternative − comparator (curve error)",
                          x["alternative_minus_comparator"])]
            if "twoway_minus_landmark" in x:
                rows.append(prow(f"{LAB[k]} {loss.upper()}: two-way − landmark Cox (curve error)", x["twoway_minus_landmark"]))
    out += "\n\n" + md_table(PH, rows)
    rows = []
    for loss, kinds in (("cox", ("last", "sparse")), ("aft", ("last", "sparse", "misspec"))):
        for k in kinds:
            g = s[loss][k]["gates"]
            rows.append([f"{LAB[k]}, {loss.upper()}", f"{g['n_reduced']} / {g['n']}",
                         str([round(v, 2) for v in g["last_on_frequency"]]), str([round(v, 2) for v in g["traj_on_frequency"]]),
                         f"{g['anchored_kind_counts']['joint']} / {g['anchored_kind_counts']['static']}",
                         f"{g['val_gap_mean']:+.4f}"])
    return out + "\n\n" + md_table(["Scenario, family", "Reduced to the comparator", "Last-value edge kept (ch. 1, 2, 3)",
                                    "Trajectory channel kept (ch. 1, 2, 3)", "Anchored state joint / static",
                                    "Mean (comparator − anchored) validation loss"], rows)


def b_table():
    a = S()["aft_symbolic"]
    rows = [[lab, f3(a["rise_means"][k])] for k, lab in (("after_pruning", "after pruning"),
                                                       ("sub_guard", "guarded substitution (|a x + b| ≤ 50), no refit"),
                                                       ("refit_plain", "unanchored refit"),
                                                       ("distill_fixed", "**distillation, μ = 10, NaN-safe**"))]
    out = md_table(["Log-normal KAN-AFT, S-Misspec 4300–4319 (seed-0, pruned)", "Mean identifiable RISE"], rows)
    out += "\n\n" + md_table(PH, [prow("distilled − after pruning (RISE)", a["distill_minus_pruned_rise"]),
                                  prow("distilled − unanchored refit (RISE)", a["distill_minus_plain_rise"])] +
                             [prow(f"distilled − no refit ({m})", a[f"distill_minus_sub_{m}"])
                              for m in ("curve_error", "ibs", "uno_c", "val_loss")])
    return out + "\n\n" + md_table(["Refits finite", "Fallbacks", "Formulas rejected", "Guarded edges",
                                    "Edges substituted (mean)", "Recall", "FDR"],
                                   [[f"{a['n_distill_finite']} / {a['n']}", str(a["n_fallback"]), str(a["n_rejected"]),
                                     str(a["n_guarded"]), f"{a['n_substituted_mean']:.2f}", f"{a['recall_mean']:.3f}",
                                     f"{a['fdr_mean']:.3f}"]])


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
    return md_table(["Check (`scripts/verify_manuscript_round17.py --pdf`)", "Scope", "Result"], rows)


def decisions():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D14[1-9] \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    rows = []
    for run in ("final_full_suite_round16", FINAL):
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
    s, d = S(), DEVS()
    ha, hb = s["H_R17a"], s["H_R17b"]
    led = json.loads(M.B.LEDGER_PATH.read_text())
    ver = json.loads(VER.read_text())
    pages = re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(REP / "manuscript_round17.pdf")],
                                                         capture_output=True, text=True).stdout).group(1)
    c = lambda p: f"[{p[0]:+.4f}, {p[1]:+.4f}]"  # noqa: E731
    out = {"git_head": R2.values()["git_head"], "n_records": str(s["n_records"]), "n_failed": str(s["n_failed"]),
           "lt_cox": str(d["last_tol"]["cox"]), "lt_aft": str(d["last_tol"]["aft"]),
           "a_verdict": verdict(ha["supported"]), "b_verdict": verdict(hb["supported"]),
           "a1_ci": c(ha["part1_ci95"]), "a2_ci": c(ha["part2_ci95"]),
           "b1_finite": f"{hb['part1_n_finite']} of {hb['part1_n_cells']}", "b1_up": f"{hb['part1_upper95']:+.4f}",
           "b2m_ci": c(hb["part2_misspec_ci95"]), "b2l_ci": c(hb["part2_last_ci95"]),
           "ms_pages": pages, "ms_values": str(led["n_values"]), "ms_refs": str(len(led["cited"])), "ms_title": M.B.TITLE,
           "ms_check": "verification ALL_PASS" if ver["ALL_PASS"] else "verification FAILED"}
    g = R1.TA[R1.TA.run_id == FINAL]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def shift_part(m):
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJKLMNOPQR" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def manuscript_for_report():
    txt = (ROOT / "docs/manuscript_round17.md").read_text()
    return txt.replace("](../reports/figures/", "](figures/")


def main():
    R16B.main()                                  # revision 18 (itself rebuilding revisions 17 .. 1 from their records)
    prev = (REP / "final_research_prototype_report.md").read_text()
    R1.use_snapshot(FINAL)
    vals = values()
    tpl = (REP / "report_template_round17.md").read_text()
    out = tpl.replace("{{MANUSCRIPT}}", manuscript_for_report())
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), out)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    cut = prev.index("\n# Parts B–R: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-16 summary", "## R16-0. Round-16 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R16-0.\1 ", own, flags=re.M)
    own = own.replace("\n# Part A: Round 16\n", "\n")
    own = own.replace("\n# Part A (continued): the standalone manuscript, Rounds 1–16\n",
                      "\n# Part B (continued): the Round-16 standalone manuscript, Rounds 1–16 (revision 18)\n")
    own = own.replace("{#manuscript-title-r16}", "{#manuscript-title-r16-hist}")
    prev = re.sub(r"\b(Parts? )([B-R])(–| and )?([B-R])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("# Part C (continued): the Round-16", "# Part B (continued): the Round-16", 1)
    prev = prev.replace("Part A is Round 16", "Part B is Round 16", 1)
    full = out + "\n\n# Part B: revision 18 (Round 16, 2026-10-26), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round17.json").write_text(json.dumps(vals, indent=1, ensure_ascii=False))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
