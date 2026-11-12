"""Assemble the final report, revision 26: Part A = Round 24 (a Bayesian joint model, JMbayes2, as the standard
dynamic-prediction comparator on the six clinical cohorts and the Round-23 partitions; D163-D165), the Round-24 journal
main manuscript (docs/manuscript_journal_main_round24.md, embedded) and the verification of both Round-24 manuscripts (the
cumulative Round-24 manuscript, docs/manuscript_round24.md, is its Supplementary Material and a separate deliverable,
reports/manuscript_round24.pdf) (reports/report_template_round24.md); Part B = revision 25's Part A (Round 23 and the
Round-23 journal manuscript); Parts C-Z = revision 25's Parts B-Y. Every Round-24 number is read from the saved records."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round15 as R15  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round22 as R22B  # noqa: E402
import build_report_round23 as R23B  # noqa: E402
import manuscript_journal_main_round24 as J  # noqa: E402
import manuscript_round24 as M  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL = "final_full_suite_round24"
VER = REP / "manuscript_verification_round24.json"
VERJ = REP / "manuscript_verification_journal_main_round24.json"
f4, ci = R15.f4, R15.ci
CLIN = ("pbc_E2", "pbc_E1", "prothro", "aids", "renal", "heart_valve")
MODELS = ("b2s_npaft", "landmark_cox", "stack", "jm", "jm_lin", "jm_ns2")
con_rows, CON_H = R23B.con_rows, R23B.CON_H


def S():
    return json.loads((REP / "round24/analysis_summary.json").read_text())


def ibs_table():
    s = S()
    rows = []
    for ds in CLIN:
        X = s["cohorts"][ds]
        m = X["means"]
        rows.append([ds, str(X["n_splits"]), f"{X['visits_median']:.0f}"] + [f"{m['ibs_cond'][k]:.4f}" for k in MODELS] +
                    [f"{X['chosen_form']['lin']} / {X['chosen_form']['ns2']}"])
    out = md_table(["Cohort", "Partitions", "Median visits", "Cond. IBS NP-KAN-AFT", "Landmark Cox", "Blend", "JM (chosen)",
                    "JM lin", "JM ns2", "lin / ns2 chosen"], rows)
    rows = []
    for ds in CLIN:
        X = s["cohorts"][ds]
        rows += con_rows(f"{ds}: cond. IBS ", X["contrasts"])
        rows += con_rows(f"{ds}: Harrell C ", {"b2s_npaft - jm": X["c_contrasts"]["b2s_npaft - jm"]})
    return out + "\n\n" + md_table(CON_H, rows)


def conv_table():
    s = S()
    rows = []
    for ds in CLIN:
        X = s["cohorts"][ds]
        m = X["means"]
        rows.append([ds, f"{m['harrell_c']['b2s_npaft']:.3f}", f"{m['harrell_c']['landmark_cox']:.3f}", f"{m['harrell_c']['jm']:.3f}",
                     " / ".join(str(X["dcal_pass"][k]) for k in ("b2s_npaft", "landmark_cox", "jm")),
                     f"{X['rhat_max_median']:.3f}", str(X["n_rhat_above_1_1"]), str(X["jm_failed_partitions"]),
                     f"{X['jm_seconds_median']:.0f}"])
    out = md_table(["Cohort", "Harrell C NP", "Landmark", "JM", "D-cal p ≥ 0.05: NP / landmark / JM", "JM median max Rhat",
                    "Partitions with Rhat > 1.1", "JM failed partitions", "JM median seconds (both forms)"], rows)
    fails = s["failures"]
    rows = [[f["job"], f.get("model", "–"), str(f.get("error", "")).replace("\n", " ").replace("|", "/")] for f in fails]
    return out + "\n\n**Failed fits (recorded, not refitted).**\n\n" + md_table(["Job", "Model", "Error"], rows)


def hyp_table():
    s = S()
    v = lambda x: "Supported" if x else "Not supported"  # noqa: E731
    a, b = s["H_R24a"], s["H_R24b"]
    rows = [["H-R24a", "renal NP − JM cond. IBS: mean < 0 and NB upper < 0; evaluable if JM fails on ≤ 2 of 20 partitions",
             f"{a['mean_diff']:+.4f} [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}], evaluable {a['evaluable']} → **{v(a['supported'])}**"]]
    rows += [[f"H-R24b ({ds})", "NP − JM cond. IBS: NB upper ≤ +0.005",
              f"{x['mean_diff']:+.4f}, upper {x['upper']:+.4f}, evaluable {x['evaluable']} → {x['met']}"]
             for ds, x in b["per_cohort"].items()]
    rows.append(["H-R24b", "all six cohorts", f"**{v(b['supported'])}**"])
    rows.append(["Records", "jobs / ok / recorded fit failures / partitions paired / not paired",
                 f"{s['n_records']} / {s['n_ok']} / {s['n_fit_failed']} / {s['n_partitions_paired']} / "
                 f"{len(s['mismatched_partitions'])}"])
    return md_table(["Hypothesis", "Criterion (D163)", "Result"], rows)


def ms_verification():
    out = "**Supplementary Material (`docs/manuscript_round24.md`)**\n\n" + R22B._ver(VER, "verify_manuscript_round24.py")
    v = json.loads(VERJ.read_text())
    out += ("\n\n**Journal main manuscript (`docs/manuscript_journal_main_round24.md`)**\n\n" +
            R22B._ver(VERJ, "verify_manuscript_journal_main_round24.py"))
    rows = [["Every logged value of the main manuscript is a verified value of the Supplementary Material",
             f"{v['supplement']['n_values']} values", "PASS" if v["supplement"]["passed"] else "FAIL"],
            ["Every reference row identical to its row in `docs/references_round22.md`", f"{v['references']['n_rows']} rows",
             "PASS" if v["references"]["passed"] else "FAIL"]]
    if "pdf" in v:
        rows.append([f"PDF length within {v['pdf']['page_range'][0]}–{v['pdf']['page_range'][1]} pages", f"{v['pdf']['pages']} pages",
                     "PASS" if v["pdf"]["page_range"][0] <= v["pdf"]["pages"] <= v["pdf"]["page_range"][1] else "FAIL"])
    return out + "\n\n" + md_table(["Journal-specific check", "Scope", "Result"], rows)


def decisions():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D16[3-9] \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    rows = []
    for run in ("final_full_suite_round23", FINAL):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


TABLES = {"ibs": ibs_table, "conv": conv_table, "hyp": hyp_table, "ms_verification": ms_verification,
          "decisions": decisions, "suite_table": suite_table, "test_runs": R2.test_run_history}


def values():
    s = S()
    led, ledj = json.loads(M.B.LEDGER_PATH.read_text()), json.loads(J.B.LEDGER_PATH.read_text())
    ver, verj = json.loads(VER.read_text()), json.loads(VERJ.read_text())
    v = lambda b: "Supported" if b else "Not supported"  # noqa: E731
    a, b = s["H_R24a"], s["H_R24b"]
    rn = s["cohorts"]["renal"]
    out = {"git_head": R2.values()["git_head"], "n_records": str(s["n_records"]), "n_ok": str(s["n_ok"]),
           "n_failed": str(s["n_fit_failed"]), "hA": v(a["supported"]), "hB": v(b["supported"]),
           "a_diff": f"NP-KAN-AFT − JM {a['mean_diff']:+.4f} [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}]",
           "a_lc": f"landmark Cox − JM {rn['contrasts']['landmark_cox - jm']['mean_diff']:+.4f}",
           "a_dcal": f"{rn['dcal_pass']['jm']} of 20",
           "b_uppers": ", ".join(f"{ds} {x['upper']:+.4f}" for ds, x in b["per_cohort"].items()),
           "ms_pages": R22B._pages(REP / "manuscript_round24.pdf"), "ms_values": str(led["n_values"]),
           "ms_refs": str(len(led["cited"])), "ms_check": "verification ALL_PASS" if ver["ALL_PASS"] else "verification FAILED",
           "j_pages": R22B._pages(REP / "manuscript_journal_main_round24.pdf"), "j_values": str(ledj["n_values"]),
           "j_refs": str(len(ledj["cited"])), "j_title": J.B.TITLE,
           "j_check": "verification ALL_PASS" if verj["ALL_PASS"] else "verification FAILED"}
    for k in ("a_diff", "a_lc", "b_uppers"):
        out[k] = out[k].replace("-", "−").replace("NP−KAN−AFT", "NP-KAN-AFT")
    gg = R1.TA[R1.TA.run_id == FINAL]
    gg = gg if len(gg) else R1.TL
    out["n_tests"] = str(len(gg))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((gg.outcome == o).sum()))
    return out


def shift_part(m):
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJKLMNOPQRSTUVWXY" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def manuscript_for_report():
    return (ROOT / "docs/manuscript_journal_main_round24.md").read_text().replace("](../reports/figures/", "](figures/")


def main():
    R23B.main()                                  # revision 25 (itself rebuilding revisions 24 .. 1 from their records)
    prev = (REP / "final_research_prototype_report.md").read_text()
    R1.use_snapshot(FINAL)
    vals = values()
    tpl = (REP / "report_template_round24.md").read_text()
    out = tpl.replace("{{MANUSCRIPT}}", manuscript_for_report())
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), out)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    cut = prev.index("\n# Parts B–Y: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-23 summary", "## R23-0. Round-23 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R23-0.\1 ", own, flags=re.M)
    own = own.replace("\n# Part A: Round 23\n", "\n")
    own = own.replace("\n# Part A (continued): the journal main manuscript (Round 23)\n",
                      "\n# Part B (continued): the Round-23 journal main manuscript (revision 25)\n")
    prev = re.sub(r"\b(Parts? )([B-Y])(–| and )?([B-Y])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("# Part C (continued): the Round-23 journal", "# Part B (continued): the Round-23 journal", 1)
    prev = prev.replace("Part A is Round 23", "Part B is Round 23", 1)
    prev = prev.replace("Part A continues with the Round-23", "Part B continues with the Round-23", 1)
    full = out + "\n\n# Part B: revision 25 (Round 23, 2026-11-10), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round24.json").write_text(json.dumps(vals, indent=1, ensure_ascii=False))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
