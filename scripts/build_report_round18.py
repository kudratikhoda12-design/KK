"""Assemble the final report, revision 20: Part A = Round 18 (landmark anchoring decided on a confirmation split;
D144-D146) and the updated standalone manuscript (docs/manuscript_round18.md) with its verification
(reports/report_template_round18.md); Part B = revision 19's Part A (Round 17 and the Round-17 manuscript); Part C =
Round 16 and its manuscript; Part D = Round 15; Part E = Round 14; Part F = Round 13; Part G = Round 12; Parts H–I = the
L39/L38 parts; Parts J–T = rounds 11..1. Every Round-18 number is read from the saved records."""
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
import build_report_round2 as R2  # noqa: E402
import manuscript_round18 as M  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL = "final_full_suite_round18"
R18 = REP / "round18"
VER = REP / "manuscript_verification_round18.json"
f3, f4, ci = R15.f3, R15.f4, R15.ci
LAB = {"last": "S-Last", "sparse": "S-Sparse", "misspec": "S-Misspec"}
FAMS = (("cox", ("last", "sparse")), ("aft", ("last", "sparse", "misspec")))
PH = ["Contrast", "Replicates", "Mean difference", "95% CI", "Paired-t p (two-sided)", "a < b / a > b / ties"]


def prow(label, p, d=4):
    if p.get("n", 0) < 2:
        return [label, str(p.get("n", 0)), "–", "–", "–", "–"]
    pt = "–" if p["p_t"] != p["p_t"] else f"{p['p_t']:.3g}"
    return [label, str(p["n"]), f"{p['mean_diff']:+.{d}f}", ci(p, d), pt, f"{p['n_neg']} / {p['n_pos']} / {p['n_zero']}"]


def S():
    return json.loads((REP / "round18_summary.json").read_text())


def DEVS():
    return json.loads((R18 / "dev_summary.json").read_text())


def dev_checks():
    d = DEVS()
    rows = []
    for loss in ("cox", "aft"):
        for sc in (("sparse", "last") if loss == "cox" else ("misspec", "sparse", "last")):
            for l, p in d["sweep"][loss][sc].items():
                rows.append([f"{loss.upper()} {LAB[sc]}", l, f"{p['n_anchored']} / {p['n']}", f"{p['mean_diff']:+.4f}",
                             ci(p) + (" ←" if float(l) == d["margin"][loss] else "")])
    out = md_table(["Development replicates (Round-17 seeds)", "Margin", "Anchored", "Confirm − comparator (curve error)",
                    "95% CI (← chosen)"], rows)
    rows = [[k, f"{v['select_gap_mean']:+.4f}", f"{v['confirm_gap_mean']:+.4f}", f"{v['alternative_test_diff_mean']:+.4f}",
             "–" if v["corr_select_gap_test"] is None else f"{v['corr_select_gap_test']:+.2f}",
             "–" if v["corr_confirm_gap_test"] is None else f"{v['corr_confirm_gap_test']:+.2f}"]
            for k, v in d["diagnostics"].items()]
    out += "\n\n" + md_table(["Family / scenario", "Comparator − anchored loss: val_select", "val_confirm",
                              "Anchored alternative − comparator (test)", "corr(val_select gap, test)", "corr(val_confirm gap, test)"], rows)
    rep = json.loads((R18 / "dev/repro_round17_check.json").read_text())
    out += (f"\n\nChosen margins: Cox {d['margin']['cox']} ({d['margin']['cox_reason']}), AFT {d['margin']['aft']} "
            f"({d['margin']['aft_reason']}). Go/no-go: {'GO' if d['go']['go'] else 'NO-GO'}. Round-17 reproduction "
            f"(job last_4100 with the Round-18 code): {rep['values_compared']} values, {rep['differences']} differences.")
    return out


def exec_table():
    s = S()
    rr = R17B.recs(R18)
    rows = []
    for k in ("last", "sparse", "misspec"):
        w = [r["wall_seconds"] for r in rr if r["job_name"].startswith(k + "_")]
        rows.append([LAB[k], str(len(w)), f"{statistics.median(w) / 60:.1f}" if w else "–"])
    t = (R18 / "started_utc.txt").read_text().strip()
    return md_table(["Jobs", "Count", "Median minutes"], rows) + \
        f"\n\nRecords: {s['n_records']}; failed jobs: {s['n_failed']}. Launched {t}."


def a_table():
    s = S()
    rows = []
    for loss, kinds in FAMS:
        for k in kinds:
            x = s[loss][k]
            cm = x["curve_error_means"]
            rows.append([f"{LAB[k]}, {loss.upper()}"] + [f"{cm[m]:.4f}" if m in cm else "–" for m in
                                                         (f"b2s_{loss}", f"b2s_{loss}_fullval", f"b2s_anchor_{loss}_confirm",
                                                          "landmark_cox")] +
                        [f"{x['anchor_rate']['n_anchored']} / {x['anchor_rate']['n']}",
                         str([round(v, 2) for v in x["last_on_frequency"]]), str([round(v, 2) for v in x["traj_on_frequency"]])])
    out = md_table(["Scenario, family", "Comparator (val_select)", "Comparator (whole validation)", "Confirm-gated",
                    "Landmark Cox", "Anchored", "Last-value edge kept (ch. 1, 2, 3)", "Trajectory kept (ch. 1, 2, 3)"], rows)
    rows = []
    for loss, kinds in FAMS:
        for k in kinds:
            x = s[loss][k]
            L = f"{LAB[k]} {loss.upper()}"
            rows += [prow(f"{L}: confirm − comparator (curve error)", x["contrast"]),
                     prow(f"{L}: confirm − comparator (IBS)", x["contrast_ibs"]),
                     prow(f"{L}: anchored alternative − comparator", x["alternative_minus_comparator"]),
                     prow(f"{L}: confirm − comparator on the whole validation set", x["confirm_minus_fullval"]),
                     prow(f"{L}: comparator on the whole validation set − comparator", x["fullval_minus_comparator"])]
            if "confirm_minus_landmark" in x:
                rows.append(prow(f"{L}: confirm − landmark Cox", x["confirm_minus_landmark"]))
    out += "\n\n" + md_table(PH, rows)
    rows = [[f"{LAB[k]}, {loss.upper()}", f"{v['select_gap_mean']:+.4f}", f"{v['confirm_gap_mean']:+.4f}",
             f"{v['alternative_test_diff_mean']:+.4f}", f"{v['n_select_gap_positive']} / {v['n_confirm_gap_positive']}",
             "–" if v["corr_confirm_gap_test"] is None else f"{v['corr_confirm_gap_test']:+.2f}"]
            for loss, kinds in FAMS for k in kinds for v in [s[loss][k]["diagnostics"]]]
    return out + "\n\n" + md_table(["Scenario, family", "Comparator − anchored loss: val_select", "val_confirm",
                                    "Anchored alternative − comparator (test)",
                                    "Replicates favouring anchoring: val_select / val_confirm", "corr(val_confirm gap, test)"], rows)


def ms_verification():
    t = R17B.ms_verification.__globals__["VER"]
    R17B.ms_verification.__globals__["VER"] = VER
    try:
        return R17B.ms_verification().replace("verify_manuscript_round17.py", "verify_manuscript_round18.py")
    finally:
        R17B.ms_verification.__globals__["VER"] = t


def decisions():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D14[4-9] \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    rows = []
    for run in ("final_full_suite_round17", FINAL):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


TABLES = {"dev_checks": dev_checks, "exec_table": exec_table, "a_table": a_table, "ms_verification": ms_verification,
          "decisions": decisions, "suite_table": suite_table, "test_runs": R2.test_run_history}


def verdict(x):
    return "Supported" if x else "Not supported"


def values():
    s, d = S(), DEVS()
    ha, hb = s["H_R18a"], s["H_R18b"]
    led = json.loads(M.B.LEDGER_PATH.read_text())
    ver = json.loads(VER.read_text())
    pages = re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(REP / "manuscript_round18.pdf")],
                                                         capture_output=True, text=True).stdout).group(1)
    c = lambda p: f"[{p[0]:+.4f}, {p[1]:+.4f}]"  # noqa: E731
    ar = lambda loss, k: f"{s[loss][k]['anchor_rate']['n_anchored']} of {s[loss][k]['anchor_rate']['n']}"  # noqa: E731
    out = {"git_head": R2.values()["git_head"], "n_records": str(s["n_records"]), "n_failed": str(s["n_failed"]),
           "m_cox": str(d["margin"]["cox"]), "m_aft": str(d["margin"]["aft"]),
           "go": "GO" if d["go"]["go"] else "NO-GO",
           "a_verdict": verdict(ha["supported"]), "b_verdict": verdict(hb["supported"]),
           "a1_ci": c(ha["part1_ci95"]), "a2_ci": c(ha["part2_ci95"]), "b1_ci": c(hb["part1_misspec_ci95"]),
           "b2_ci": c(hb["part2_last_ci95"]), "ar_cox_last": ar("cox", "last"), "ar_cox_sparse": ar("cox", "sparse"),
           "ar_aft_last": ar("aft", "last"), "ar_aft_misspec": ar("aft", "misspec"),
           "ms_pages": pages, "ms_values": str(led["n_values"]), "ms_refs": str(len(led["cited"])), "ms_title": M.B.TITLE,
           "ms_check": "verification ALL_PASS" if ver["ALL_PASS"] else "verification FAILED"}
    g = R1.TA[R1.TA.run_id == FINAL]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def shift_part(m):
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJKLMNOPQRS" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def manuscript_for_report():
    return (ROOT / "docs/manuscript_round18.md").read_text().replace("](../reports/figures/", "](figures/")


def main():
    R17B.main()                                  # revision 19 (itself rebuilding revisions 18 .. 1 from their records)
    prev = (REP / "final_research_prototype_report.md").read_text()
    R1.use_snapshot(FINAL)
    vals = values()
    tpl = (REP / "report_template_round18.md").read_text()
    out = tpl.replace("{{MANUSCRIPT}}", manuscript_for_report())
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), out)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    cut = prev.index("\n# Parts B–S: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-17 summary", "## R17-0. Round-17 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R17-0.\1 ", own, flags=re.M)
    own = own.replace("\n# Part A: Round 17\n", "\n")
    own = own.replace("\n# Part A (continued): the standalone manuscript, Rounds 1–17\n",
                      "\n# Part B (continued): the Round-17 standalone manuscript, Rounds 1–17 (revision 19)\n")
    own = own.replace("{#manuscript-title-r17}", "{#manuscript-title-r17-hist}")
    prev = re.sub(r"\b(Parts? )([B-S])(–| and )?([B-S])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("# Part C (continued): the Round-17", "# Part B (continued): the Round-17", 1)
    prev = prev.replace("Part A is Round 17", "Part B is Round 17", 1)
    full = out + "\n\n# Part B: revision 19 (Round 17, 2026-10-27), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round18.json").write_text(json.dumps(vals, indent=1, ensure_ascii=False))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
