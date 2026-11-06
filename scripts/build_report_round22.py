"""Assemble the final report, revision 24: Part A = Round 22 (per-channel REML AFCM and CK-6d; the untouched joineRML
cohorts renal and heart.valve with C-MAPSS FD003 Option B; the three-law NP-KAN-AFT simulation; D156-D159), the journal
main manuscript (docs/manuscript_journal_main.md, embedded) and the verification of both manuscripts (the cumulative
Round-22 manuscript, docs/manuscript_round22.md, is its Supplementary Material and a separate deliverable,
reports/manuscript_round22.pdf) (reports/report_template_round22.md); Part B = revision 23's Part A (Round 21 and the
Round-21 manuscript); Parts C–X = revision 23's Parts B–W. Every Round-22 number is read from the saved records."""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round15 as R15  # noqa: E402
import build_report_round17 as R17B  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round21 as R21B  # noqa: E402
import manuscript_journal_main as J  # noqa: E402
import manuscript_round22 as M  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL = "final_full_suite_round22"
VER = REP / "manuscript_verification_round22.json"
VERJ = REP / "manuscript_verification_journal_main.json"
f4, ci = R15.f4, R15.ci
CO = ("renal", "heart_valve")
LAWS = ("bimodal", "studentt3", "logistic05")
MODELS = ("b2s_cox", "b2s_waft", "b2s_lnaft", "b2s_npaft", "b2s_anchor_cox_cross", "b2s_anchor_lnaft_cross",
          "b2s_anchor_npaft_cross", "landmark_cox", "fpca_cox", "lme2_cox", "sum_cox", "sum_rsf")


def S():
    return json.loads((REP / "round22/analysis_summary.json").read_text())


def afcm_table():
    g = json.loads((REP / "round22/afcm_gate.json").read_text())
    ck = json.loads((REP / "round22/ck6d.json").read_text())
    rows = []
    for s, r in g["seeds"].items():
        f = json.loads((REP / f"round22/dev/results_afcmdev_{s}.jsonl").read_text().splitlines()[0])["afcm_reml"]["fit"]
        rows.append([s, f"{r['afcm_reml']['phi1']:.3f} / {r['afcm_reml']['phi2']:.3f}", str(r["afcm_reml"]["meets"]),
                     f"{f['chosen_iter']} ({'converged' if f['converged'] else 'not converged'})",
                     " / ".join(f"{e:.2f}" for e in f["edf_channel"]),
                     " / ".join(f"{x:.2e}" for x in f["lam_x"]), " / ".join(f"{x:.2e}" for x in f["lam_t"]),
                     f"{r['afcm_reml_ens']['phi1']:.3f} / {r['afcm_reml_ens']['phi2']:.3f}", str(r["afcm_reml_ens"]["meets"]),
                     str(r["pilot_equals_round15_laml"])])
    rows.append(["24 (CK-6d)", f"{ck['shapes']['phi1']['rise_id']:.3f} / {ck['shapes']['phi2']['rise_id']:.3f}", str(ck["passed"]),
                 "–", "–", " / ".join(f"{x:.2e}" for x in ck["lam_x"]), " / ".join(f"{x:.2e}" for x in ck["lam_t"]), "–", "–", "–"])
    out = md_table(["Seed", "afcm_reml φ₁ / φ₂", "meets", "FS iteration chosen", "EDF by channel", "λ_x", "λ_t",
                    "afcm_reml_ens φ₁ / φ₂", "meets", "pilot = Round-15 LAML"], rows)
    return out + (f"\n\nGate (rule fixed in D156): **{'GO, ' + g['go_variant'] if g['go'] else 'NO-GO'}**. CK-6d on seed 24: "
                  f"**{'PASS' if ck['passed'] else 'FAIL'}** (φ₁ < 0.15 and φ₂ < 0.10 required). H-R22a: "
                  f"{S()['H_R22a']['status']}.")


def cohort_tables():
    s, out = S(), []
    for ds in CO:
        X = s["cohorts"].get(ds)
        if not X:
            out.append(f"**{ds}**: no records.")
            continue
        c = X["cohort"]
        out.append(f"**{ds}** (patients {c['n_all']}; landmark {c['t_L']} years; at risk {c['n']}, events {c['events']}; median "
                   f"visits before the landmark {c['obs_per_subject_median']}; patients with no visit per channel "
                   f"{c['unseen_per_channel']}; n_val per split {X['n_val']}, halves {X['n_halves']})")
        rows = [[m, f4(X["means"][m]["ibs_cond"]), f4(X["means"][m]["ibs"]), f"{X['means'][m]['harrell_c']:.3f}",
                 f"{X['means'][m]['uno_c']:.3f}", str(X["dcal_pass"].get(m, "–"))] for m in MODELS if m in X["means"]]
        out.append(md_table(["Model", "Cond. IPCW IBS", "Marg. IPCW IBS", "Harrell C", "Uno C", "D-cal p ≥ 0.05 (of 5)"], rows))
        rows = []
        for k, v in X["contrasts"].items():
            if k.startswith("ibs_cond "):
                p = v["nb"]
                rows.append([k[9:], str(p.get("n", "–")), f"{p['mean_diff']:+.4f}" if "mean_diff" in p else "–", ci(p),
                             ci(v["paired"]), f"{p.get('n_neg', '–')} / {p.get('n_pos', '–')}"])
        out.append(md_table(["Contrast (conditional IBS)", "Splits", "Mean", "Nadeau–Bengio 95% CI", "Plain 95% CI",
                             "a < b / a > b"], rows))
        rows = [[f, f"{g['n_confirmed']} / {g['n']}", str(g["confirmed_splits"]), str(g["fallback_exact"]),
                 f"{g['fallback_max_abs_diff']:.1e}"] for f, g in X["gates"].items()]
        out.append(md_table(["Cross-confirmed family", "Confirmed", "Confirmed splits", "Exact fallback elsewhere",
                             "Max |diff| on fallback splits"], rows))
        rows = [[f, str(v["n"]), f"{v['replacement_rate_mean']:.2f}", str(v["n_rollback"]), str(v["all_finite"]),
                 f4(v["ibs_cond_before_mean"]), f4(v["ibs_cond_after_mean"]), ci(v["after_minus_before"]["nb"])]
                for f, v in X["symbolic"].items()]
        out.append(md_table(["Distilled model (seed 0)", "Splits", "Substituted share", "Rollbacks", "All finite",
                             "Cond. IBS before", "after", "after − before, NB 95% CI"], rows))
        out.append("Mean seconds per model: " + ", ".join(f"{m} {round(v)}" for m, v in X["seconds_mean"].items()) + ".")
    return "\n\n".join(out)


def fd003_table():
    F = S().get("fd003")
    if not F:
        return "No FD003 records."
    rows = [[m, f4(v["ibs"]), f"{v['harrell_c']:.3f}", f"{v['uno_c']:.3f}", f"{v['dcal_p']:.4f}", str(F["dcal_pass"][m])]
            for m, v in F["means"].items()]
    out = md_table(["Model", "IBS", "Harrell C", "Uno C", "D-cal p (mean)", "p ≥ 0.05 (of 5)"], rows)
    rows = [[k, str(v["nb"].get("n", "–")), f"{v['nb']['mean_diff']:+.4f}" if "mean_diff" in v["nb"] else "–", ci(v["nb"]),
             ci(v["paired"]), f"{v['nb'].get('n_neg', '–')} / {v['nb'].get('n_pos', '–')}"] for k, v in F["contrasts"].items()]
    out += "\n\n" + md_table(["Contrast", "Splits", "Mean", "Nadeau–Bengio 95% CI", "Plain 95% CI", "a < b / a > b"], rows)
    return out + (f"\n\nCross-confirmed NP anchor: {F['gate']['n_confirmed']} / {F['gate']['n']} confirmed; exact fallback "
                  f"{F['gate']['fallback_exact']}. Training windows per split {F['n_train']}, test windows {F['n_test']}; "
                  f"{F['n_part_records']} part records (one per split and family).")


def sims_table():
    s, out = S(), []
    rows = []
    for law in LAWS:
        X = s["sims"].get(law)
        if not X:
            continue
        for m, v in X["means"].items():
            rows.append([law, m, f4(v["curve_error"]), f4(v["ibs"]), f"{v['harrell_c']:.3f}", f"{v['uno_c']:.3f}",
                         str(X["dcal_pass"][m])])
    out.append(md_table(["Law", "Model", "Curve error", "IBS", "Harrell C", "Uno C", "D-cal p ≥ 0.05 (of 20)"], rows))
    rows = []
    for law in LAWS:
        X = s["sims"].get(law)
        if not X:
            continue
        for k, p in X["contrasts"].items():
            rows.append([law, k, str(p.get("n", "–")), f"{p['mean_diff']:+.4f}" if "mean_diff" in p else "–", ci(p),
                         f"{p.get('p_t', float('nan')):.3g}", f"{p.get('n_neg', '–')} / {p.get('n_pos', '–')}"])
    out.append(md_table(["Law", "Contrast (curve error)", "Replicates", "Mean", "95% CI", "Paired-t p", "a < b / a > b"], rows))
    cens = "; ".join(f"{law} {s['sims'][law]['censored_frac_mean']:.3f}" for law in LAWS if law in s["sims"])
    return "\n\n".join(out) + f"\n\nMean censored fraction: {cens}."


def hyp_table():
    s = S()
    b, c = s["H_R22b"], s["H_R22c"]
    v = lambda x: "Supported" if x else "Not supported"  # noqa: E731
    rows = [["H-R22a", "B2-S Cox − per-channel REML AFCM, RISE, S-Add 3000–3019 (only if CK-6d passes)", s["H_R22a"]["status"]],
            ["H-R22b (1)", "every external job complete, every metric and prediction finite", str(b["part1_all_finite"]) +
             (f"; failed jobs {b['failed_jobs']}" if b["failed_jobs"] else "") + (f"; missing {b['missing_jobs']}" if b["missing_jobs"] else "")],
            ["H-R22b (2)", "anchored − unanchored mean cond. IBS ≤ +0.005 and exact fallback, Cox / log-normal / NP",
             "; ".join(f"{ds} {f} {x['mean_shift']:+.4f} ({x['confirmed']}, exact {x['fallback_exact']})"
                       for ds, d in b["part2"].items() for f, x in d.items()) + f" → {b['part2_ok']}"],
            ["H-R22b (3)", "distillation finite on 5 splits, mean change ≤ +0.002 (NP and Cox)",
             "; ".join(f"{ds} {f} {x['mean_change']:+.4f}" for ds, d in b["part3"].items() for f, x in d.items()) + f" → {b['part3_ok']}"],
            ["H-R22b", "all three parts", f"**{v(b['supported'])}**"]]
    if c.get("complete"):
        rows += [["H-R22c (1)", "bimodal: upper 95% bound < 0 vs Cox, Weibull, log-normal",
                  ", ".join(f"{k} {x:+.4f}" for k, x in c["upper_bounds_bimodal"].items()) + f" → {c['part1']}"],
                 ["H-R22c (2)", "t₃ and logistic: < 0 vs Cox and Weibull, < +0.005 vs log-normal",
                  "; ".join(f"{law}: " + ", ".join(f"{k} {x:+.4f}" for k, x in d.items()) for law, d in c["upper_bounds_other"].items())
                  + f" → {c['part2']}"]]
    rows.append(["H-R22c", "both parts", f"**{v(c.get('supported'))}**" + ("" if c.get("complete") else " (incomplete)")])
    return md_table(["Hypothesis", "Criterion (D156)", "Result"], rows)


def _ver(path, script):
    t = R17B.ms_verification.__globals__["VER"]
    R17B.ms_verification.__globals__["VER"] = path
    try:
        return R17B.ms_verification().replace("verify_manuscript_round17.py", script)
    finally:
        R17B.ms_verification.__globals__["VER"] = t


def ms_verification():
    out = "**Supplementary Material (`docs/manuscript_round22.md`)**\n\n" + _ver(VER, "verify_manuscript_round22.py")
    v = json.loads(VERJ.read_text())
    out += "\n\n**Journal main manuscript (`docs/manuscript_journal_main.md`)**\n\n" + _ver(VERJ, "verify_manuscript_journal_main.py")
    rows = [["Every logged value of the main manuscript is a verified value of the Supplementary Material",
             f"{v['supplement']['n_values']} values", "PASS" if v["supplement"]["passed"] else "FAIL"],
            ["Every reference row identical to its row in `docs/references_round22.md`", f"{v['references']['n_rows']} rows",
             "PASS" if v["references"]["passed"] else "FAIL"]]
    if "pdf" in v:
        rows.append([f"PDF length within {v['pdf']['page_range'][0]}–{v['pdf']['page_range'][1]} pages", f"{v['pdf']['pages']} pages",
                     "PASS" if v["pdf"]["page_range"][0] <= v["pdf"]["pages"] <= v["pdf"]["page_range"][1] else "FAIL"])
    return out + "\n\n" + md_table(["Journal-specific check", "Scope", "Result"], rows)


def decisions():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D(15[6-9]|16\d) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    rows = []
    for run in ("final_full_suite_round21", FINAL):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


TABLES = {"afcm": afcm_table, "cohorts": cohort_tables, "fd003": fd003_table, "sims": sims_table, "hyp": hyp_table,
          "ms_verification": ms_verification, "decisions": decisions, "suite_table": suite_table,
          "test_runs": R2.test_run_history}


def _pages(pdf):
    return re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1)


def values():
    s = S()
    led, ledj = json.loads(M.B.LEDGER_PATH.read_text()), json.loads(J.B.LEDGER_PATH.read_text())
    ver, verj = json.loads(VER.read_text()), json.loads(VERJ.read_text())
    g = json.loads((REP / "round22/afcm_gate.json").read_text())
    ck = json.loads((REP / "round22/ck6d.json").read_text())
    v = lambda b: "Supported" if b else "Not supported"  # noqa: E731
    out = {"git_head": R2.values()["git_head"], "n_ok": str(s["jobs"]["ok"]), "n_failed": str(s["jobs"]["failed"]),
           "gate": f"GO ({g['go_variant']})" if g["go"] else "NO-GO", "ck6d": "PASS" if ck["passed"] else "FAIL",
           "ck_phi1": f"{ck['shapes']['phi1']['rise_id']:.3f}", "ck_phi2": f"{ck['shapes']['phi2']['rise_id']:.3f}",
           "hA": s["H_R22a"]["status"], "hB": v(s["H_R22b"]["supported"]), "hC": v(s["H_R22c"].get("supported")),
           "ms_pages": _pages(REP / "manuscript_round22.pdf"), "ms_values": str(led["n_values"]), "ms_refs": str(len(led["cited"])),
           "ms_check": "verification ALL_PASS" if ver["ALL_PASS"] else "verification FAILED",
           "j_pages": _pages(REP / "manuscript_journal_main.pdf"), "j_values": str(ledj["n_values"]),
           "j_refs": str(len(ledj["cited"])), "j_title": J.B.TITLE,
           "j_check": "verification ALL_PASS" if verj["ALL_PASS"] else "verification FAILED"}
    gg = R1.TA[R1.TA.run_id == FINAL]
    gg = gg if len(gg) else R1.TL
    out["n_tests"] = str(len(gg))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((gg.outcome == o).sum()))
    return out


def shift_part(m):
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJKLMNOPQRSTUVW" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def manuscript_for_report():
    return (ROOT / "docs/manuscript_journal_main.md").read_text().replace("](../reports/figures/", "](figures/")


def main():
    R21B.main()                                  # revision 23 (itself rebuilding revisions 22 .. 1 from their records)
    prev = (REP / "final_research_prototype_report.md").read_text()
    R1.use_snapshot(FINAL)
    vals = values()
    tpl = (REP / "report_template_round22.md").read_text()
    out = tpl.replace("{{MANUSCRIPT}}", manuscript_for_report())
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), out)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    cut = prev.index("\n# Parts B–W: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-21 summary", "## R21-0. Round-21 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R21-0.\1 ", own, flags=re.M)
    own = own.replace("\n# Part A: Round 21\n", "\n")
    own = own.replace("\n# Part A (continued): the standalone manuscript, Rounds 1–21\n",
                      "\n# Part B (continued): the Round-21 standalone manuscript, Rounds 1–21 (revision 23)\n")
    own = own.replace("{#manuscript-title-r21}", "{#manuscript-title-r21-hist}")
    prev = re.sub(r"\b(Parts? )([B-W])(–| and )?([B-W])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("# Part C (continued): the Round-21", "# Part B (continued): the Round-21", 1)
    prev = prev.replace("Part A is Round 21", "Part B is Round 21", 1)
    full = out + "\n\n# Part B: revision 23 (Round 21, 2026-11-05), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round22.json").write_text(json.dumps(vals, indent=1, ensure_ascii=False))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
