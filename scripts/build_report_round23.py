"""Assemble the final report, revision 25: Part A = Round 23 (untouched C-MAPSS FD002 / FD004, renal history thinning and
the landmark / NP-KAN-AFT blend on six clinical cohorts; D160-D162), the Round-23 journal main manuscript
(docs/manuscript_journal_main_round23.md, embedded) and the verification of both Round-23 manuscripts (the cumulative
Round-23 manuscript, docs/manuscript_round23.md, is its Supplementary Material and a separate deliverable,
reports/manuscript_round23.pdf) (reports/report_template_round23.md); Part B = revision 24's Part A (Round 22 and the
Round-22 journal manuscript); Parts C-Y = revision 24's Parts B-X. Every Round-23 number is read from the saved records."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round15 as R15  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round22 as R22B  # noqa: E402
import manuscript_journal_main_round23 as J  # noqa: E402
import manuscript_round23 as M  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL = "final_full_suite_round23"
VER = REP / "manuscript_verification_round23.json"
VERJ = REP / "manuscript_verification_journal_main_round23.json"
f4, ci = R15.f4, R15.ci
CM = ("FD002", "FD004")
CLIN = ("pbc_E2", "pbc_E1", "prothro", "aids", "renal", "heart_valve")
CM_MODELS = ("b2s_npaft", "landmark_cox", "landmark_cox_raw", "landmark_cox_smooth", "stack", "sum_aft", "fpca_aft")


def S():
    return json.loads((REP / "round23/analysis_summary.json").read_text())


def con_rows(prefix, contrasts):
    return [[f"{prefix}{k}", str(c.get("n", "–")), f"{c['mean_diff']:+.4f}" if "mean_diff" in c else "–", ci(c),
             f"{c.get('p', float('nan')):.3g}", f"{c.get('n_neg', '–')} / {c.get('n_pos', '–')}",
             f"{c['n_missing_a']} / {c['n_missing_b']}"] for k, c in contrasts.items()]


CON_H = ["Contrast (a − b)", "Splits", "Mean", "Nadeau–Bengio 95% CI", "p", "a < b / a > b", "Missing a / b"]


def cm_table():
    s, out = S(), []
    for fd in CM:
        X = s["cm"].get(fd)
        if not X:
            out.append(f"**{fd}**: no records.")
            continue
        rows = [[m, f4(X["means"]["ibs"][m]), f"{X['means']['harrell_c'][m]:.3f}", f"{X['means']['uno_c'][m]:.3f}",
                 str(X["dcal_pass"][m]), ", ".join(map(str, X["model_failures"][m])) or "–"] for m in CM_MODELS]
        out.append(f"**{fd}** ({X['n_splits']} engine splits)\n\n" +
                   md_table(["Model", "IBS", "Harrell C", "Uno C", "D-cal p ≥ 0.05", "Failed splits"], rows))
        out.append(md_table(CON_H, con_rows("IBS ", X["contrasts"]) + con_rows("Harrell C ", X["c_contrasts"])))
        out.append(f"Landmark version chosen per split: {X['landmark_chosen']}. Blend weight per split: {X['w']}.")
    return "\n\n".join(out)


def renal_table():
    R = S()["clinical"]["renal"]
    rows = [[m, f4(R["means"]["ibs_cond"][m]), f4(R["means"]["ibs"][m]), f"{R['means']['harrell_c'][m]:.3f}"]
            for m in ("b2s_npaft", "b2s_npaft_sparse3", "landmark_cox", "stack")]
    out = md_table(["Renal model", "Cond. IPCW IBS", "Marg. IPCW IBS", "Harrell C"], rows)
    keys = ("b2s_npaft - b2s_npaft_sparse3", "b2s_npaft_sparse3 - landmark_cox", "b2s_npaft - landmark_cox")
    return out + "\n\n" + md_table(CON_H, con_rows("cond. IBS ", {k: R["contrasts"][k] for k in keys}))


def clin_table():
    s, out = S(), []
    rows = []
    for ds in CLIN:
        X = s["clinical"].get(ds)
        if not X:
            continue
        m = X["means"]
        rows.append([ds, str(X["n_splits"]), f"{X['visits_median']:.0f}", f4(m["ibs_cond"]["b2s_npaft"]),
                     f4(m["ibs_cond"]["landmark_cox"]), f4(m["ibs_cond"]["stack"]), f"{m['harrell_c']['b2s_npaft']:.3f}",
                     f"{m['harrell_c']['landmark_cox']:.3f}", f"{m['harrell_c']['stack']:.3f}",
                     " / ".join(str(X["w_counts"][w]) for w in ("0.0", "0.25", "0.5", "0.75", "1.0"))])
    out.append(md_table(["Cohort", "Partitions", "Median visits", "Cond. IBS NP-KAN-AFT", "Landmark Cox", "Blend",
                         "Harrell C NP", "Landmark", "Blend", "w = 0 / .25 / .5 / .75 / 1"], rows))
    rows = []
    for ds in CLIN:
        X = s["clinical"].get(ds)
        if X:
            rows += con_rows(f"{ds}: cond. IBS ", {k: X["contrasts"][k] for k in ("stack - landmark_cox", "b2s_npaft - landmark_cox")})
            rows += con_rows(f"{ds}: Harrell C ", X["c_contrasts"])
    out.append(md_table(CON_H, rows))
    return "\n\n".join(out)


def hyp_table():
    s = S()
    v = lambda x: "Supported" if x else "Not supported"  # noqa: E731
    a, b, c = s["H_R23a"], s["H_R23b"], s["H_R23c"]
    rows = [[f"H-R23a ({fd})", "NP − landmark Cox test IBS: mean < 0 and NB upper < 0; evaluable if ≤ 2 of 10 fits missing",
             f"{x['mean_diff']:+.4f}, upper {x['upper']:+.4f}, evaluable {x['evaluable']} → {x['met']}"] for fd, x in a["per_dataset"].items()]
    rows.append(["H-R23a", "both data sets", f"**{v(a['supported'])}**"])
    rows.append(["H-R23b", "renal NP full − NP three visits, cond. IBS: mean < 0 and NB upper < 0",
                 f"{b['mean_diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] → **{v(b['supported'])}**"])
    rows.append(["H-R23c (i)", "blend − landmark Cox cond. IBS: NB upper ≤ +0.005 on every clinical cohort",
                 "; ".join(f"{ds} {x['upper']:+.4f}" for ds, x in c["part_i"]["per_cohort"].items()) +
                 f" → **{v(c['part_i']['supported'])}**"])
    rows.append(["H-R23c (ii)", "renal blend − landmark Cox: NB upper < 0",
                 f"{c['part_ii']['mean_diff']:+.4f} [{c['part_ii']['ci95'][0]:+.4f}, {c['part_ii']['ci95'][1]:+.4f}] → "
                 f"**{v(c['part_ii']['supported'])}**"])
    fails = s["failed_records"]
    rows.append(["Failed jobs", "reported with their errors", "; ".join(f"{f['job']}: {f['error']}" for f in fails) or "none"])
    return md_table(["Hypothesis", "Criterion (D160)", "Result"], rows)


def ms_verification():
    out = "**Supplementary Material (`docs/manuscript_round23.md`)**\n\n" + R22B._ver(VER, "verify_manuscript_round23.py")
    v = json.loads(VERJ.read_text())
    out += ("\n\n**Journal main manuscript (`docs/manuscript_journal_main_round23.md`)**\n\n" +
            R22B._ver(VERJ, "verify_manuscript_journal_main_round23.py"))
    rows = [["Every logged value of the main manuscript is a verified value of the Supplementary Material",
             f"{v['supplement']['n_values']} values", "PASS" if v["supplement"]["passed"] else "FAIL"],
            ["Every reference row identical to its row in `docs/references_round22.md`", f"{v['references']['n_rows']} rows",
             "PASS" if v["references"]["passed"] else "FAIL"]]
    if "pdf" in v:
        rows.append([f"PDF length within {v['pdf']['page_range'][0]}–{v['pdf']['page_range'][1]} pages", f"{v['pdf']['pages']} pages",
                     "PASS" if v["pdf"]["page_range"][0] <= v["pdf"]["pages"] <= v["pdf"]["page_range"][1] else "FAIL"])
    return out + "\n\n" + md_table(["Journal-specific check", "Scope", "Result"], rows)


def decisions():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D16[0-9] \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    rows = []
    for run in ("final_full_suite_round22", FINAL):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


TABLES = {"cm": cm_table, "renal": renal_table, "clin": clin_table, "hyp": hyp_table, "ms_verification": ms_verification,
          "decisions": decisions, "suite_table": suite_table, "test_runs": R2.test_run_history}


def values():
    s = S()
    led, ledj = json.loads(M.B.LEDGER_PATH.read_text()), json.loads(J.B.LEDGER_PATH.read_text())
    ver, verj = json.loads(VER.read_text()), json.loads(VERJ.read_text())
    v = lambda b: "Supported" if b else "Not supported"  # noqa: E731
    a = s["H_R23a"]["per_dataset"]
    out = {"git_head": R2.values()["git_head"], "n_records": str(s["n_records"]), "n_ok": str(s["n_ok"]),
           "n_failed": str(len(s["failed_records"])), "hA": v(s["H_R23a"]["supported"]), "hB": v(s["H_R23b"]["supported"]),
           "hC1": v(s["H_R23c"]["part_i"]["supported"]), "hC2": v(s["H_R23c"]["part_ii"]["supported"]),
           "a_fd002": f"{a['FD002']['mean_diff']:+.4f}, upper {a['FD002']['upper']:+.4f}",
           "a_fd004": f"{a['FD004']['mean_diff']:+.4f}, upper {a['FD004']['upper']:+.4f}",
           "b_diff": f"full − three visits {s['H_R23b']['mean_diff']:+.4f} [{s['H_R23b']['ci95'][0]:+.4f}, {s['H_R23b']['ci95'][1]:+.4f}]",
           "ms_pages": R22B._pages(REP / "manuscript_round23.pdf"), "ms_values": str(led["n_values"]),
           "ms_refs": str(len(led["cited"])), "ms_check": "verification ALL_PASS" if ver["ALL_PASS"] else "verification FAILED",
           "j_pages": R22B._pages(REP / "manuscript_journal_main_round23.pdf"), "j_values": str(ledj["n_values"]),
           "j_refs": str(len(ledj["cited"])), "j_title": J.B.TITLE,
           "j_check": "verification ALL_PASS" if verj["ALL_PASS"] else "verification FAILED"}
    gg = R1.TA[R1.TA.run_id == FINAL]
    gg = gg if len(gg) else R1.TL
    for k in ("a_fd002", "a_fd004", "b_diff"):
        out[k] = out[k].replace("-", "−")
    out["n_tests"] = str(len(gg))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((gg.outcome == o).sum()))
    return out


def shift_part(m):
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJKLMNOPQRSTUVWX" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def manuscript_for_report():
    return (ROOT / "docs/manuscript_journal_main_round23.md").read_text().replace("](../reports/figures/", "](figures/")


def main():
    R22B.main()                                  # revision 24 (itself rebuilding revisions 23 .. 1 from their records)
    prev = (REP / "final_research_prototype_report.md").read_text()
    R1.use_snapshot(FINAL)
    vals = values()
    tpl = (REP / "report_template_round23.md").read_text()
    out = tpl.replace("{{MANUSCRIPT}}", manuscript_for_report())
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), out)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    cut = prev.index("\n# Parts B–X: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-22 summary", "## R22-0. Round-22 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R22-0.\1 ", own, flags=re.M)
    own = own.replace("\n# Part A: Round 22\n", "\n")
    own = own.replace("\n# Part A (continued): the journal main manuscript (Round 22)\n",
                      "\n# Part B (continued): the Round-22 journal main manuscript (revision 24)\n")
    own = own.replace("{#manuscript-title-journal}", "{#manuscript-title-journal-r22}")
    prev = re.sub(r"\b(Parts? )([B-X])(–| and )?([B-X])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("# Part C (continued): the Round-22 journal", "# Part B (continued): the Round-22 journal", 1)
    prev = prev.replace("Part A is Round 22", "Part B is Round 22", 1)
    prev = prev.replace("Part A continues with the journal main manuscript", "Part B continues with the journal main manuscript", 1)
    full = out + "\n\n# Part B: revision 24 (Round 22, 2026-11-07), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round23.json").write_text(json.dumps(vals, indent=1, ensure_ascii=False))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
