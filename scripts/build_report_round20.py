"""Assemble the final report, revision 22: Part A = Round 20 (a semiparametric functional KAN AFT with a Kaplan-Meier
error law, H-R20a, and the two-half cross-confirmed landmark anchor against the 3-seed B2-S ensembles, H-R20b;
development stages A and B and the formal run; D150-D152) and the updated standalone manuscript
(docs/manuscript_round20.md) with its verification (reports/report_template_round20.md); Part B = revision 21's Part A
(Round 19 and the Round-19 manuscript); Parts C–V = revision 21's Parts B–U. Every Round-20 number is read from the
saved records."""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round15 as R15  # noqa: E402
import build_report_round17 as R17B  # noqa: E402
import build_report_round19 as R19B  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import manuscript_round20 as M  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL = "final_full_suite_round20"
R20 = REP / "round20"
VER = REP / "manuscript_verification_round20.json"
f3, f4, ci = R15.f3, R15.f4, R15.ci
LAB = {"last": "S-Last", "sparse": "S-Sparse", "misspec": "S-Misspec"}
FL = {"cox": "Cox", "lnaft": "log-normal AFT", "npaft": "NP-AFT"}
TRAJ = {"cox": "sparse", "lnaft": "misspec", "npaft": "misspec"}
PH = R19B.PH
prow = R19B.prow


def J(p):
    return json.loads(p.read_text())


def DA():
    return J(R20 / "devA_summary.json")


def DB():
    return J(R20 / "devB_summary.json")


def S():
    return J(REP / "round20_summary.json")


def dev_a():
    d = DA()
    rows = [[LAB[sc]] + [f4(d["means"][sc][m]) if m in d["means"][sc] else "–"
                         for m in ("b2s_cox_fullval", "b2s_weibull_fullval", "b2s_npaft_bj_fullval", "b2s_npaft_rank_fullval")]
            for sc in ("sparse", "misspec")]
    out = md_table(["Development replicates", "B2-S Cox", "B2-S Weibull AFT", "NP-AFT bj_km", "NP-AFT rank_km"], rows)
    rows = []
    for o in ("bj_km", "rank_km"):
        c = d["contrasts"][o]
        rows += [prow(f"S-Misspec: NP ({o}) − Cox", c["misspec_vs_cox"]), prow(f"S-Misspec: NP ({o}) − Weibull", c["misspec_vs_weibull"]),
                 prow(f"S-Sparse: NP ({o}) − Cox", c["sparse_vs_cox"])]
    out += "\n\n" + md_table(PH, rows)
    g = d["go_a"]
    return out + (f"\n\nObjective rule: scores {json.dumps({k: round(v, 5) for k, v in d['objective_score'].items()})} -> "
                  f"**{d['objective']}**. GO-a: S-Misspec upper bounds {g['misspec_vs_cox_upper']:+.4f} (vs Cox), "
                  f"{g['misspec_vs_weibull_upper']:+.4f} (vs Weibull), S-Sparse {g['sparse_vs_cox_upper']:+.4f} (vs Cox, "
                  f"rule < +0.005) -> **{'GO' if g['go_a'] else 'NO-GO'}**.")


def dev_b():
    d = DB()
    rows = []
    for f in ("cox", "lnaft", "npaft"):
        lm = d["margin"][f]["margin"]
        for sc, sw in d["sweep"][f].items():
            for l, p in sw.items():
                rows.append([FL[f], LAB[sc], l + (" ←" if float(l) == lm else ""), f"{p['n_confirmed']} / {p['n']}",
                             f"{p['mean_diff']:+.4f}", ci(p)])
    out = md_table(["Family", "Development replicates", "Margin (← chosen)", "Confirmed",
                    "Cross-confirmed − full-validation B2-S (curve error)", "95% CI"], rows)
    rows = [[k, f"{v['refit_always']['mean_diff']:+.4f}", ci(v["refit_always"]), f"{v['h1_b2s_vs_fullval']['mean_diff']:+.4f}",
             str(v["n_h1_favours"]), str(v["n_h2_favours"]), str(v["n_both_favour"]), str(v["n_r18_rule"])]
            for k, v in d["diag"].items()]
    out += "\n\n" + md_table(["Family / scenario", "Always refit − full-validation B2-S", "95% CI", "h1 B2-S − full-validation B2-S",
                              "h1 favours", "h2 favours", "Both (margin 0)", "Round-18 rule"], rows)
    gb = d["go_b"]
    return out + "\n\nGO-b: " + "; ".join(
        f"{FL[f]} margin {d['margin'][f]['margin']} ({d['margin'][f]['reason']}): S-Last mean {gb[f]['last_mean']:+.4f}, "
        f"{LAB[gb[f]['traj']]} upper {gb[f]['traj_upper']:+.4f} -> {'pass' if gb[f]['pass'] else 'fail'}"
        for f in ("cox", "lnaft", "npaft")) + f" -> **{'GO' if gb['go'] else 'NO-GO'}**."


def formal_a():
    s = S()
    a = s["H_R20a"]
    if not isinstance(a, dict):
        return a
    rows = [[LAB[sc]] + [f4(s["means"][sc][m]) if m in s["means"][sc] else "–"
                         for m in ("b2s_npaft_fullval", "b2s_cox_fullval", "b2s_weibull_fullval", "b2s_lnaft_fullval")]
            for sc in ("misspec", "sparse")]
    out = md_table(["Fresh replicates", "NP-AFT (rank_km)", "B2-S Cox", "B2-S Weibull AFT", "B2-S log-normal AFT"], rows)
    rows = [prow(lab, a[k]) for k, lab in (("misspec_vs_cox", "S-Misspec: NP-AFT − Cox (H-R20a)"),
                                            ("misspec_vs_weibull", "S-Misspec: NP-AFT − Weibull AFT (H-R20a)"),
                                            ("sparse_vs_cox", "S-Sparse: NP-AFT − Cox (secondary)"),
                                            ("misspec_vs_lognormal", "S-Misspec: NP-AFT − log-normal AFT (exploratory)"),
                                            ("misspec_vs_cox_ibs", "S-Misspec IBS: NP-AFT − Cox (exploratory)"),
                                            ("misspec_vs_weibull_ibs", "S-Misspec IBS: NP-AFT − Weibull (exploratory)"))]
    return out + "\n\n" + md_table(PH, rows)


def formal_b():
    s = S()
    b = s["H_R20b"]
    if not isinstance(b, dict):
        return b
    rows, rows2 = [], []
    for f in ("cox", "lnaft", "npaft"):
        for k, sc in (("last", "last"), ("traj", TRAJ[f])):
            x = b[f][k]
            m = s["means"][sc]
            rows.append([f"{LAB[sc]}, {FL[f]}", f4(m[f"b2s_{f}_fullval"]), f4(m[f"b2s_{f}_h1"]), f4(m[f"b2s_anchor_{f}_cross"]),
                         f"{x['n_confirmed']} / {x['n']}", f"{x['n_h1_favours']} / {x['n_h2_favours']} / {x['n_r18_rule']}"])
            rows2 += [prow(f"{LAB[sc]}, {FL[f]}: cross-confirmed − full-validation B2-S", x["contrast"]),
                      prow(f"{LAB[sc]}, {FL[f]}: always refit − full-validation B2-S (exploratory)", x["refit_always"]),
                      prow(f"{LAB[sc]}, {FL[f]}: cross-confirmed − h1 B2-S (secondary)", x["vs_h1"])]
            if "vs_landmark" in x:
                rows2.append(prow(f"{LAB[sc]}, {FL[f]}: cross-confirmed − landmark Cox (secondary)", x["vs_landmark"]))
    return (md_table(["Scenario, family", "B2-S full validation", "B2-S h1", "Cross-confirmed", "Confirmed",
                      "h1 / h2 favour; Round-18 rule"], rows) + "\n\n" + md_table(PH, rows2))


def ms_verification():
    t = R17B.ms_verification.__globals__["VER"]
    R17B.ms_verification.__globals__["VER"] = VER
    try:
        return R17B.ms_verification().replace("verify_manuscript_round17.py", "verify_manuscript_round20.py")
    finally:
        R17B.ms_verification.__globals__["VER"] = t


def decisions():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D15[0-9] \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    rows = []
    for run in ("final_full_suite_round19", FINAL):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


TABLES = {"dev_a": dev_a, "dev_b": dev_b, "formal_a": formal_a, "formal_b": formal_b, "ms_verification": ms_verification,
          "decisions": decisions, "suite_table": suite_table, "test_runs": R2.test_run_history}


def ci_s(p):
    return f"{p['mean_diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"


def values():
    da, db, s = DA(), DB(), S()
    led = json.loads(M.B.LEDGER_PATH.read_text())
    ver = json.loads(VER.read_text())
    pages = re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(REP / "manuscript_round20.pdf")],
                                                         capture_output=True, text=True).stdout).group(1)
    out = {"git_head": R2.values()["git_head"], "nA": str(da["n_records"]), "nA_failed": str(da["n_failed"]),
           "nB": str(db["n_records"]), "nB_failed": str(db["n_failed"]), "objective": da["objective"],
           "go_a": "GO" if da["go_a"]["go_a"] else "NO-GO", "go_b": "GO" if db["go_b"]["go"] else "NO-GO",
           "m_cox": str(db["margin"]["cox"]["margin"]), "m_lnaft": str(db["margin"]["lnaft"]["margin"]),
           "m_npaft": str(db["margin"]["npaft"]["margin"]),
           "nF": str(s["n_records"]), "nF_failed": str(s["n_failed"]),
           "ms_pages": pages, "ms_values": str(led["n_values"]), "ms_refs": str(len(led["cited"])), "ms_title": M.B.TITLE,
           "ms_check": "verification ALL_PASS" if ver["ALL_PASS"] else "verification FAILED"}
    a, b = s["H_R20a"], s["H_R20b"]
    if isinstance(a, dict):
        out.update(hA="Supported" if a["supported"] else "Not supported", a_cox=ci_s(a["misspec_vs_cox"]),
                   a_wei=ci_s(a["misspec_vs_weibull"]), a_sparse=ci_s(a["sparse_vs_cox"]), a_ln=ci_s(a["misspec_vs_lognormal"]))
    if isinstance(b, dict):
        out["hB"] = "Supported" if b["supported"] else "Not supported"
        for f in ("cox", "lnaft", "npaft"):
            out[f"b_{f}_last"] = ci_s(b[f]["last"]["contrast"])
            out[f"b_{f}_traj"] = ci_s(b[f]["traj"]["contrast"])
            out[f"b_{f}_ok"] = "met" if b[f]["supported"] else "not met"
            out[f"b_{f}_conf"] = f"{b[f]['last']['n_confirmed']} and {b[f]['traj']['n_confirmed']}"
    gg = R1.TA[R1.TA.run_id == FINAL]
    gg = gg if len(gg) else R1.TL
    out["n_tests"] = str(len(gg))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((gg.outcome == o).sum()))
    return out


def shift_part(m):
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJKLMNOPQRSTU" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def manuscript_for_report():
    return (ROOT / "docs/manuscript_round20.md").read_text().replace("](../reports/figures/", "](figures/")


def main():
    R19B.main()                                  # revision 21 (itself rebuilding revisions 20 .. 1 from their records)
    prev = (REP / "final_research_prototype_report.md").read_text()
    R1.use_snapshot(FINAL)
    vals = values()
    tpl = (REP / "report_template_round20.md").read_text()
    out = tpl.replace("{{MANUSCRIPT}}", manuscript_for_report())
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), out)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    cut = prev.index("\n# Parts B–U: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-19 summary", "## R19-0. Round-19 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R19-0.\1 ", own, flags=re.M)
    own = own.replace("\n# Part A: Round 19\n", "\n")
    own = own.replace("\n# Part A (continued): the standalone manuscript, Rounds 1–19\n",
                      "\n# Part B (continued): the Round-19 standalone manuscript, Rounds 1–19 (revision 21)\n")
    own = own.replace("{#manuscript-title-r19}", "{#manuscript-title-r19-hist}")
    prev = re.sub(r"\b(Parts? )([B-U])(–| and )?([B-U])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("# Part C (continued): the Round-19", "# Part B (continued): the Round-19", 1)
    prev = prev.replace("Part A is Round 19", "Part B is Round 19", 1)
    full = out + "\n\n# Part B: revision 21 (Round 19, 2026-10-30), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round20.json").write_text(json.dumps(vals, indent=1, ensure_ascii=False))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
