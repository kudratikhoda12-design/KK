"""Assemble the final report, revision 16: Part A = Round 14, the manuscript draft (docs/manuscript_round14.md) with its
verification, decisions and tests (reports/report_template_round14.md); Part B = revision 15's Part A (Round 13);
Part C = revision 14's Part A (Round 12); Parts D–E = the L39/L38 parts; Parts F–P = rounds 11..1."""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round13 as R13  # noqa: E402
import manuscript_round14 as M  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL = "final_full_suite_round14_rerun2"
VER = REP / "manuscript_verification_round14.json"


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
    return md_table(["Check (`scripts/verify_manuscript_round14.py --pdf`)", "Scope", "Result"], rows)


def decisions():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D13[1-9] \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    rows = []
    for run in ("final_full_suite_round13", "final_full_suite_round14", "final_full_suite_round14_rerun", FINAL):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


TABLES = {"ms_verification": ms_verification, "decisions": decisions, "suite_table": suite_table,
          "test_runs": R2.test_run_history}


def values():
    led = json.loads(M.LEDGER_PATH.read_text())
    ver = json.loads(VER.read_text())
    pages = re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(REP / "manuscript_round14.pdf")],
                                                         capture_output=True, text=True).stdout).group(1)
    out = {"git_head": R2.values()["git_head"], "ms_pages": pages, "ms_values": str(led["n_values"]),
           "ms_refs": str(len(led["cited"])), "ms_title": M.TITLE,
           "ms_check": "verification ALL_PASS" if ver["ALL_PASS"] else "verification FAILED"}
    g = R1.TA[R1.TA.run_id == FINAL]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def shift_part(m):
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJKLMNO" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def manuscript_for_report():
    txt = (ROOT / "docs/manuscript_round14.md").read_text()
    return txt.replace("](../reports/figures/", "](figures/")


def main():
    R13.main()                                   # revision 15 (itself rebuilding revisions 14 .. 1 from their records)
    prev = (REP / "final_research_prototype_report.md").read_text()
    R1.use_snapshot(FINAL)
    vals = values()
    tpl = (REP / "report_template_round14.md").read_text()
    out = tpl.replace("{{MANUSCRIPT}}", manuscript_for_report())
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), out)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    cut = prev.index("\n# Parts B–O: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-13 summary", "## R13-0. Round-13 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R13-0.\1 ", own, flags=re.M)
    own = re.sub(r"§0\.(\d)", r"§R13-0.\1", own)
    prev = re.sub(r"\b(Parts? )([B-O])(–| and )?([B-O])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("Part A is Round 13", "Part B is Round 13", 1)
    full = out + "\n\n# Part B: revision 15 (Round 13, 2026-10-22), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round14.json").write_text(json.dumps(vals, indent=1, ensure_ascii=False))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
