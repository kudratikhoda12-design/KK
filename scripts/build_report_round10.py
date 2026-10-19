"""Assemble the final report, revision 10: round-10 part (reports/report_template_round10.md: literature review, design
note and pre-registration of Rounds 11-13), then the round-9..1 reports as historical records (Parts B–J)."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round9 as R9  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL10 = "final_full_suite_round10"


def suite_table():
    rows = []
    for run in ("final_full_suite", "final_full_suite_round2_v3", "final_full_suite_round3", "final_full_suite_round4_rerun",
                "final_full_suite_round5", "final_full_suite_round6", "final_full_suite_round6_rerun",
                "final_full_suite_round7", "final_full_suite_round8", "final_full_suite_round9", FINAL10):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


def decisions_r10():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D(11[0-5]) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


TABLES = {"suite_table": suite_table, "decisions_r10": decisions_r10, "test_runs": R2.test_run_history}


def values():
    out = {"git_head": R2.values()["git_head"]}
    g = R1.TA[R1.TA.run_id == FINAL10]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def shift_part(m):
    """'Part B' -> 'Part C', 'Parts B–I' -> 'Parts C–J', 'Parts H and I' -> 'Parts I and J' (one pass; Part A is kept)."""
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHI" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def main():
    R1.use_snapshot(FINAL10)
    vals = values()
    tpl = (REP / "report_template_round10.md").read_text()
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    R9.main()
    prev = (REP / "final_research_prototype_report.md").read_text()
    cut = prev.index("\n# Parts B–I: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-9 summary", "## R9-0. Round-9 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R9-0.\1 ", own, flags=re.M)
    own = re.sub(r"§0\.(\d)", r"§R9-0.\1", own)
    prev = re.sub(r"\b(Parts? )([B-I])(–| and )?([B-I])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("Part A is round 9", "Part B is round 9", 1)
    full = out + "\n\n# Part B: round-9 report (2026-10-19), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round10.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
