"""Assemble the final report, revision 23: Part A = Round 21 (the Round-20 NP-KAN-AFT, log-normal KAN-AFT, two-half
cross-confirmed anchor and fixed-mu distillation on the real benchmarks and C-MAPSS FD001 Option B; D153-D155) and the
updated standalone manuscript (docs/manuscript_round21.md) with its verification (reports/report_template_round21.md);
Part B = revision 22's Part A (Round 20 and the Round-20 manuscript); Parts C–W = revision 22's Parts B–V. Every
Round-21 number is read from the saved records."""
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
import build_report_round20 as R20B  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import manuscript_round21 as M  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL = "final_full_suite_round21"
VER = REP / "manuscript_verification_round21.json"
f4, ci = R15.f4, R15.ci
PH, prow = R19B.PH, R19B.prow
DS = ("pbc_E2", "pbc_E1", "prothro", "aids")
MODELS = ("b2s_cox", "b2s_waft", "b2s_lnaft", "b2s_npaft", "b2s_anchor_cox_cross", "b2s_anchor_lnaft_cross",
          "b2s_anchor_npaft_cross", "landmark_cox", "r15_b2s_anchor_cox", "afcm_v3", "fpca_cox", "lme2_cox", "sum_cox",
          "sum_rsf")


def S():
    return json.loads((REP / "round21/analysis_summary.json").read_text())


def clinical_tables():
    s, out = S(), []
    for ds in DS:
        X = s["clinical"][ds]
        rows = [[m, f4(X["means"][m]["ibs_cond"]), f4(X["means"][m]["ibs"]), f"{X['means'][m]['harrell_c']:.3f}",
                 f"{X['means'][m]['uno_c']:.3f}", str(X["dcal_pass"].get(m, "–"))] for m in MODELS if m in X["means"]]
        out.append(f"**{ds}** (n_val per split {X['n_val'][0]}, halves {X['n_halves'][0][0]} / {X['n_halves'][0][1]})\n\n" +
                   md_table(["Model", "Cond. IPCW IBS", "Marg. IPCW IBS", "Harrell C", "Uno C", "D-cal p ≥ 0.05 (of 5)"], rows))
        rows = []
        for k, v in X["contrasts"].items():
            if k.startswith("ibs_cond "):
                p = v["nb"]
                rows.append([k[9:], str(p.get("n", "–")), f"{p['mean_diff']:+.4f}" if "mean_diff" in p else "–", ci(p),
                             ci(v["paired"]), f"{p.get('n_neg', '–')} / {p.get('n_pos', '–')}"])
        out.append(md_table(["Contrast (conditional IBS)", "Splits", "Mean", "Nadeau–Bengio 95% CI", "Plain 95% CI",
                             "a < b / a > b"], rows))
        rows = [[f, f"{g['n_confirmed']} / {g['n']}", str(g["confirmed_splits"]), str(g["fallback_exact"]),
                 f"{g['n_h1_favours']} / {g['n_h2_favours']}"] for f, g in X["gates"].items()]
        out.append(md_table(["Cross-confirmed family", "Confirmed", "Confirmed splits", "Exact fallback elsewhere",
                             "h1 / h2 favour anchoring"], rows))
        rows = [[f, f"{v['replacement_rate_mean']:.2f}", str(v["n_rollback"]), str(v["n_refit_fallback"]),
                 f4(v["ibs_cond_before_mean"]), f4(v["ibs_cond_after_mean"]), ci(v["after_minus_before"]["nb"])]
                for f, v in X["symbolic"].items()]
        out.append(md_table(["Distilled model (seed 0)", "Substituted share", "Rollbacks", "Refit fallbacks",
                             "Cond. IBS before", "after", "after − before, NB 95% CI"], rows))
        out.append("Reproduction against Round 13: " + "; ".join(f"{k}: {v['n_values']} values, max |diff| {v['max_abs_diff']}"
                                                                for k, v in X["reproduction"].items()) +
                   ". Mean seconds per model: " + ", ".join(f"{m} {round(v)}" for m, v in X["seconds_mean"].items()) + ".")
    return "\n\n".join(out)


def formulas():
    s, rows = S(), []
    for ds in DS:
        for f, v in s["clinical"][ds]["symbolic"].items():
            fs = v["formulas"].get("0") or v["formulas"].get(0) or []
            for e in fs[:6]:
                rows.append([ds, f, e["layer"], str(e["channel_index"]), e["stage"], f"`{e['formula']}`",
                             f"{e['r2_val']:.3f}" if e["r2_val"] is not None else "–"])
    return md_table(["Data set", "Model", "Layer", "Input index", "Stage", "Accepted formula (split 0)", "Validation R²"], rows)


def cmapss_table():
    c = S()["cmapss"]
    rows = [[m, f4(v["ibs"]), f"{v['harrell_c']:.3f}", f"{v['uno_c']:.3f}", f"{v['dcal_chi2']:.1f}", f"{v['dcal_p']:.4f}",
             str(c["dcal_pass"][m]), str(round(c["seconds_mean"][m]))] for m, v in c["means"].items()]
    out = md_table(["Model", "IBS", "Harrell C", "Uno C", "D-cal χ²", "D-cal p (mean)", "p ≥ 0.05 (of 5)", "Mean seconds"], rows)
    rows = [prow(k, v["nb"]) for k, v in c["contrasts"].items()]
    out += "\n\n" + md_table(PH, rows)
    crit = c["criterion"]
    out += "\n\n" + md_table(["Model", "D-cal improved", "IBS ≤ 0.055", "Harrell C ≥ 0.88", "Owner criterion", "Relative criterion"],
                             [[m, str(v["dcal_improved"]), str(v["ibs_le_0055"]), str(v["harrell_ge_088"]),
                               str(v["owner_criterion"]), str(v["relative_criterion"])] for m, v in crit.items()])
    return out + (f"\n\nCross-confirmed NP anchor: {c['gate']['n_confirmed']} / {c['gate']['n']} confirmed; exact fallback "
                  f"{c['gate']['fallback_exact']}. Training windows per split {c['n_train']}, test windows {c['n_test']}. "
                  "Seconds for cmapss_1 and cmapss_2 include parts reloaded after container restarts (flagged in the records).")


def ms_verification():
    t = R17B.ms_verification.__globals__["VER"]
    R17B.ms_verification.__globals__["VER"] = VER
    try:
        return R17B.ms_verification().replace("verify_manuscript_round17.py", "verify_manuscript_round21.py")
    finally:
        R17B.ms_verification.__globals__["VER"] = t


def decisions():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D15[3-9] \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    rows = []
    for run in ("final_full_suite_round20", FINAL):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


TABLES = {"clinical": clinical_tables, "formulas": formulas, "cmapss": cmapss_table, "ms_verification": ms_verification,
          "decisions": decisions, "suite_table": suite_table, "test_runs": R2.test_run_history}


def values():
    s = S()
    led = json.loads(M.B.LEDGER_PATH.read_text())
    ver = json.loads(VER.read_text())
    pages = re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(REP / "manuscript_round21.pdf")],
                                                         capture_output=True, text=True).stdout).group(1)
    h = s["hypotheses"]
    v = lambda b: "Supported" if b else "Not supported"  # noqa: E731
    out = {"git_head": R2.values()["git_head"], "n_records": str(s["n_records"]), "n_failed": str(s["n_failed"]),
           "hA": v(h["H_R21a"]["supported"]), "hB": v(h["H_R21b"]["supported"]), "hC": v(h["H_R21c"]["supported"]),
           "hC_rel": v(h["H_R21c"]["secondary_relative_supported"]),
           "a_cox": f"{h['H_R21a']['part2_mean_vs_cox']:+.4f}", "a_waft": f"{h['H_R21a']['part2_mean_vs_waft']:+.4f}",
           "a_dcal": str(h["H_R21a"]["part3_dcal_pass"]), "a_n": str(h["H_R21a"]["n_npaft_fits"]),
           "ms_pages": pages, "ms_values": str(led["n_values"]), "ms_refs": str(len(led["cited"])), "ms_title": M.B.TITLE,
           "ms_check": "verification ALL_PASS" if ver["ALL_PASS"] else "verification FAILED"}
    for ds in DS:
        b = h["H_R21b"][ds]
        out[f"b_{ds}"] = (f"NP {b['npaft_mean_shift']:+.4f}, Cox {b['cox_mean_shift']:+.4f}; confirmed Cox {b['confirmed']['cox']}, "
                          f"NP {b['confirmed']['npaft']}")
    c = s["cmapss"]["means"]
    for m in ("b2s_cox", "b2s_npaft", "b2s_lnaft"):
        out[f"c_{m}"] = f"IBS {c[m]['ibs']:.4f}, Harrell C {c[m]['harrell_c']:.3f}, D-cal χ² {c[m]['dcal_chi2']:.1f}"
    gg = R1.TA[R1.TA.run_id == FINAL]
    gg = gg if len(gg) else R1.TL
    out["n_tests"] = str(len(gg))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((gg.outcome == o).sum()))
    return out


def shift_part(m):
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJKLMNOPQRSTUV" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def manuscript_for_report():
    return (ROOT / "docs/manuscript_round21.md").read_text().replace("](../reports/figures/", "](figures/")


def main():
    R20B.main()                                  # revision 22 (itself rebuilding revisions 21 .. 1 from their records)
    prev = (REP / "final_research_prototype_report.md").read_text()
    R1.use_snapshot(FINAL)
    vals = values()
    tpl = (REP / "report_template_round21.md").read_text()
    out = tpl.replace("{{MANUSCRIPT}}", manuscript_for_report())
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), out)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    cut = prev.index("\n# Parts B–V: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-20 summary", "## R20-0. Round-20 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R20-0.\1 ", own, flags=re.M)
    own = own.replace("\n# Part A: Round 20\n", "\n")
    own = own.replace("\n# Part A (continued): the standalone manuscript, Rounds 1–20\n",
                      "\n# Part B (continued): the Round-20 standalone manuscript, Rounds 1–20 (revision 22)\n")
    own = own.replace("{#manuscript-title-r20}", "{#manuscript-title-r20-hist}")
    prev = re.sub(r"\b(Parts? )([B-V])(–| and )?([B-V])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("# Part C (continued): the Round-20", "# Part B (continued): the Round-20", 1)
    prev = prev.replace("Part A is Round 20", "Part B is Round 20", 1)
    full = out + "\n\n# Part B: revision 22 (Round 20, 2026-11-02), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round21.json").write_text(json.dumps(vals, indent=1, ensure_ascii=False))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
