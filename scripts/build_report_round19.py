"""Assemble the final report, revision 21: Part A = Round 19 (the confirmation-split decision with a full-validation
estimator; development check only -- NO-GO, closed by the owner as a development result; D147-D149) and the updated standalone manuscript (docs/manuscript_round19.md) with its verification
(reports/report_template_round19.md); Part B = revision 20's Part A (Round 18 and the Round-18 manuscript); Parts C–U =
revision 20's Parts B–T. Every Round-19 number is read from the saved records."""
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round15 as R15  # noqa: E402
import build_report_round17 as R17B  # noqa: E402
import build_report_round18 as R18B  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import manuscript_round19 as M  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL = "final_full_suite_round19"
R19 = REP / "round19"
VER = REP / "manuscript_verification_round19.json"
f3, f4, ci = R15.f3, R15.f4, R15.ci
LAB = {"last": "S-Last", "sparse": "S-Sparse", "misspec": "S-Misspec"}
FAMS = (("cox", ("last", "sparse")), ("aft", ("last", "misspec")))
PH = ["Contrast", "Replicates", "Mean difference", "95% CI", "Paired-t p (two-sided)", "a < b / a > b / ties"]


def prow(label, p, d=4):
    if p.get("n", 0) < 2:
        return [label, str(p.get("n", 0)), "–", "–", "–", "–"]
    pt = "–" if p["p_t"] != p["p_t"] else f"{p['p_t']:.3g}"
    return [label, str(p["n"]), f"{p['mean_diff']:+.{d}f}", ci(p, d), pt, f"{p['n_neg']} / {p['n_pos']} / {p['n_zero']}"]


def DEVS():
    return json.loads((R19 / "dev_summary.json").read_text())


def dev_checks():
    d = DEVS()
    rows = []
    for how, sw in d["sweep"].items():
        for loss in ("cox", "aft"):
            for sc in (("sparse", "last") if loss == "cox" else ("misspec", "last")):
                for l, p in sw[loss][sc].items():
                    chosen = how == d["variant"] and float(l) == d["margin"][loss]
                    rows.append([how, f"{loss.upper()} {LAB[sc]}", l, f"{p['n_anchored']} / {p['n']}", f"{p['mean_diff']:+.4f}",
                                 ci(p) + (" ←" if chosen else "")])
    out = md_table(["Refit variant", "Development replicates (Round-18 seeds)", "Margin", "Anchored",
                    "Estimator − full-validation B2-S (curve error)", "95% CI (← chosen)"], rows)
    rows = [[k, f"{v['select_gap_mean']:+.4f}", f"{v['confirm_gap_mean']:+.4f}", f"{v['alternative_test_diff_mean']:+.4f}"]
            for k, v in d["diagnostics"].items()]
    out += "\n\n" + md_table(["Family / scenario", "Comparator − anchored loss: val_select", "val_confirm",
                              "Refitted anchored state − full-validation B2-S (test)"], rows)
    rows = [[k, f"{p['mean_diff']:+.4f}", ci(p)] for k, p in d["select_minus_fullval_comparator"].items()]
    out += "\n\n" + md_table(["Family / scenario", "val_select B2-S − full-validation B2-S", "95% CI"], rows)
    rep = json.loads((R19 / "dev/repro_round18_check.json").read_text())
    out += (f"\n\nChosen: refit variant **{d['variant']}**, margins Cox {d['margin']['cox']}, AFT {d['margin']['aft']} "
            f"(by variant: {json.dumps(d['choice_by_variant'])}). Go/no-go: {'GO' if d['go']['go'] else 'NO-GO'}. "
            f"Round-18 reproduction (job sparse_4500 with the Round-19 code): {rep['values_compared']} values, "
            f"{rep['differences']} differences.")
    return out


def ms_verification():
    t = R17B.ms_verification.__globals__["VER"]
    R17B.ms_verification.__globals__["VER"] = VER
    try:
        return R17B.ms_verification().replace("verify_manuscript_round17.py", "verify_manuscript_round19.py")
    finally:
        R17B.ms_verification.__globals__["VER"] = t


def decisions():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D1(4[7-9]|5[0-9]) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    rows = []
    for run in ("final_full_suite_round18", FINAL):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


TABLES = {"dev_checks": dev_checks, "ms_verification": ms_verification,
          "decisions": decisions, "suite_table": suite_table, "test_runs": R2.test_run_history}


def verdict(x):
    return "Supported" if x else "Not supported"


def values():
    d = DEVS()
    led = json.loads(M.B.LEDGER_PATH.read_text())
    ver = json.loads(VER.read_text())
    pages = re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(REP / "manuscript_round19.pdf")],
                                                         capture_output=True, text=True).stdout).group(1)
    g = d["go"]
    out = {"git_head": R2.values()["git_head"], "n_records": str(d["n_records"]), "n_failed": str(d["n_failed"]),
           "variant": d["variant"], "m_cox": str(d["margin"]["cox"]), "m_aft": str(d["margin"]["aft"]),
           "go": "GO" if g["go"] else "NO-GO", "u_cox_sparse": f"{g['cox_sparse_upper']:+.4f}",
           "u_cox_last": f"{g['cox_last_upper']:+.4f}", "u_aft_misspec": f"{g['aft_misspec_upper']:+.4f}",
           "u_aft_last": f"{g['aft_last_upper']:+.4f}",
           "ms_pages": pages, "ms_values": str(led["n_values"]), "ms_refs": str(len(led["cited"])), "ms_title": M.B.TITLE,
           "ms_check": "verification ALL_PASS" if ver["ALL_PASS"] else "verification FAILED"}
    gg = R1.TA[R1.TA.run_id == FINAL]
    gg = gg if len(gg) else R1.TL
    out["n_tests"] = str(len(gg))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((gg.outcome == o).sum()))
    return out


def shift_part(m):
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJKLMNOPQRST" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def manuscript_for_report():
    return (ROOT / "docs/manuscript_round19.md").read_text().replace("](../reports/figures/", "](figures/")


def main():
    R18B.main()                                  # revision 20 (itself rebuilding revisions 19 .. 1 from their records)
    prev = (REP / "final_research_prototype_report.md").read_text()
    R1.use_snapshot(FINAL)
    vals = values()
    tpl = (REP / "report_template_round19.md").read_text()
    out = tpl.replace("{{MANUSCRIPT}}", manuscript_for_report())
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), out)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    cut = prev.index("\n# Parts B–T: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-18 summary", "## R18-0. Round-18 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R18-0.\1 ", own, flags=re.M)
    own = own.replace("\n# Part A: Round 18\n", "\n")
    own = own.replace("\n# Part A (continued): the standalone manuscript, Rounds 1–18\n",
                      "\n# Part B (continued): the Round-18 standalone manuscript, Rounds 1–18 (revision 20)\n")
    own = own.replace("{#manuscript-title-r18}", "{#manuscript-title-r18-hist}")
    prev = re.sub(r"\b(Parts? )([B-T])(–| and )?([B-T])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("# Part C (continued): the Round-18", "# Part B (continued): the Round-18", 1)
    prev = prev.replace("Part A is Round 18", "Part B is Round 18", 1)
    full = out + "\n\n# Part B: revision 20 (Round 18, 2026-10-28), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round19.json").write_text(json.dumps(vals, indent=1, ensure_ascii=False))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
