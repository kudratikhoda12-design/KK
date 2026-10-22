"""Assemble the final report, revision 14: Part A = Round 12 (the simulation study, D126/D127;
reports/report_template_round12c.md); Part B = revision 13's Part A (L39); Part C = revision 12's Part A (L38);
Parts D–N = rounds 11..1 as historical records."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round12b as R12B  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL = "final_full_suite_round12c_rerun"
SUM = REP / "round12_summary.json"
CSV = REP / "tables/r12_summary_by_model.csv"
LABEL = {"b2s_cox": "B2-S Cox (functional CoxKAN)", "b2s_aft": "B2-S KAN-AFT (Weibull)", "b2s_aft_rc": "B2-S KAN-AFT, residual-calibrated",
         "b2h_cox": "B2-H Cox", "b2h_aft": "B2-H KAN-AFT", "b1_fpca1_cox": "B1-FPCA-1 Cox", "b1_fpca1_aft": "B1-FPCA-1 AFT",
         "b1_fpca2_cox": "B1-FPCA-2 Cox", "b2deep_cox": "B2-deep Cox", "b1_sum_cox": "B1-SUM Cox", "b1_sum_aft": "B1-SUM AFT",
         "static_coxkan": "static CoxKAN", "afcm_v3": "AFCM (repaired, V3) — CK-6c FAIL", "afcm_v0": "AFCM (Round-12a, V0)",
         "fpca_cox": "fpca_cox", "sum_cox": "sum_cox", "sum_rsf": "sum_rsf", "flat_ae": "flat_ae",
         "b2s_cox_quantile": "B2-S Cox, quantile input (sensitivity)"}


def S():
    return json.loads(SUM.read_text())


def f3(v):
    return "–" if v is None or (isinstance(v, float) and v != v) else f"{v:.3f}"


def f4(v):
    return "–" if v is None or (isinstance(v, float) and v != v) else f"{v:.4f}"


def ptab(rows):
    out = []
    for lab, p in rows:
        if not p or p.get("n", 0) < 2:
            out.append([lab, str(p.get("n", 0) if p else 0)] + ["–"] * 8)
            continue
        out.append([lab, str(p["n"]), f4(p["mean_a"]), f4(p["mean_b"]), f4(p["mean_diff"]),
                    f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]", f"{p['p_t']:.3g}", f"{p['p_wilcoxon']:.3g}",
                    f"{p['n_neg']} / {p['n_pos']}", (f"{p['p_tost']:.3g}" if "p_tost" in p else "–")
                    + (f" ({'equivalent' if p.get('equivalent') else 'not equivalent'})" if "p_tost" in p else "")])
    return md_table(["Contrast (a − b)", "n", "mean a", "mean b", "mean difference", "95% CI", "paired-t p",
                     "exact Wilcoxon p", "a < b / a > b", "TOST p (margin)"], out)


def hr12_table():
    s = S()
    rows = [("**H-R12: B2-S Cox ensemble − AFCM V3, mean identifiable RISE (confirmatory)**", s["H_R12"])]
    lab = {"single_seed_b2s_vs_afcm_v3": "single-seed (seed 0) B2-S Cox − AFCM V3", "b2s_vs_afcm_v0": "B2-S Cox ensemble − AFCM V0",
           "raw_rise_b2s_vs_afcm_v3": "raw RISE: B2-S Cox ensemble − AFCM V3",
           "phi1_b2s_vs_afcm_v3": "φ₁ only (identifiable): B2-S − AFCM V3", "phi2_b2s_vs_afcm_v3": "φ₂ only (identifiable): B2-S − AFCM V3"}
    rows += [(lab[k], v) for k, v in s["H_R12_sensitivity"].items()]
    return ptab(rows) + "\n\nInterpretation (D113 rule): **" + s["H_R12"].get("interpretation", "–") + "** (confirmatory row only)."


def secondary_table():
    s = S()
    return ptab(list(s["secondary"].items()))


def p0_table():
    p = S().get("P0")
    if not p or "curve_error" not in p:
        return "No P0 records."
    return ptab([("S-Int/P0 survival-curve error: fae_kan_s3 − fae_mlp_s3 (TOST ±0.005)", p["curve_error"]),
                 ("S-Int/P0 IBS: fae_kan_s3 − fae_mlp_s3", p["ibs"]),
                 ("S-Int/P0 Harrell C: fae_kan_s3 − fae_mlp_s3", p["harrell_c"])])


def model_table(scen, sub=None, shape=True):
    T = pd.read_csv(CSV).fillna({"sub": ""})
    g = T[(T.scenario == scen) & (T["sub"] == (sub or ""))]
    rows = []
    for _, r in g.iterrows():
        if r.model == "prune_symbolic":
            continue
        row = [LABEL.get(r.model, r.model), r.front, f"{int(r.n_ok)}" + (f" ({int(r.n_failed)} failed)" if r.n_failed else "")
               + (f" ({int(r.n_dropped)} dropped)" if r.get("n_dropped", 0) else ""),
               f3(r.get("curve_error_mean")), f3(r.get("ibs_mean")), f3(r.get("harrell_c_mean")), f3(r.get("uno_c_mean")),
               f3(r.get("dcal_frac_p_lt_005"))]
        if shape:
            row += [f3(r.get("rise_mean_id_mean")), f3(r.get("rise_phi1.rise_primary_mean")),
                    f3(r.get("rise_phi2.rise_primary_mean")), f3(r.get("rise_mean_raw_mean"))]
        rows.append(row)
    hdr = ["Model", "Front end", "Replicates ok", "Curve error", "IBS", "Harrell C", "Uno C", "Share D-cal p < 0.05"]
    if shape:
        hdr += ["Identifiable RISE (mean φ₁, φ₂)", "φ₁ (primary)", "φ₂ (primary)", "Raw RISE (mean)"]
    return md_table(hdr, rows)


def sel_table():
    s = S()["selection_symbolic"]
    rows = []
    for sc, v in s.items():
        if not v:
            continue
        rows.append([sc, str(v["n"]), f3(v["recall_mean"]), f3(v["fdr_mean"]),
                     "; ".join(f"{k}: {n}" for k, n in v["selected_patterns"].items()),
                     f3(v["acceptance_rate_mean"]), f4(v["r2_val_accepted_mean"]),
                     ", ".join(f"{k} {n}" for k, n in v["stages_total"].items()),
                     f3(v["rise_after_pruning"]), f3(v["rise_after_substitution_before_refit"]),
                     f3(v["rise_after_substitution"]) + f" ({v['n_refit_nonfinite']} non-finite refits)",
                     f3(v["curve_error_after_pruning"]), f3(v["curve_error_after_substitution"])])
    return md_table(["Scenario", "n", "Recall {1, 2}", "FDR (channel 3)", "Selected patterns", "Symbolic acceptance rate",
                     "Mean validation R² (accepted)", "Stages (all edges)", "RISE after pruning",
                     "RISE after substitution, before refit", "RISE after substitution + refit",
                     "Curve error after pruning", "Curve error after substitution + refit"], rows)


def exec_table():
    s = S()
    return md_table(["Records", "Failed cells", "Dropped by the budget rule"],
                    [[str(s["n_records"]), str(s["n_failed"]), str(s["n_dropped"])]]) + (
        "\n\n" + md_table(["Scenario", "Sub", "Replicate", "Model", "Front", "Error"],
                          [[f["scenario"], str(f["sub"] or ""), str(f["replicate"]), f["model"], f["front"], (f["error"] or "")[:120]]
                           for f in s["failed"]]) if s["failed"] else "")


def decisions():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D(12[6-9]|13[0-9]) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    rows = []
    for run in ("final_full_suite_round11", "final_full_suite_round12", "final_full_suite_round12b", "final_full_suite_round12c", FINAL):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


TABLES = {"hr12_table": hr12_table, "secondary_table": secondary_table, "p0_table": p0_table, "sel_table": sel_table,
          "exec_table": exec_table, "decisions": decisions, "suite_table": suite_table, "test_runs": R2.test_run_history,
          "sadd_table": lambda: model_table("S-Add"), "sint_table": lambda: model_table("S-Int", shape=False),
          "ssparse_table": lambda: model_table("S-Sparse"),
          "smis_ln_table": lambda: model_table("S-Misspec", "lognormal"),
          "smis_ll_table": lambda: model_table("S-Misspec", "loglogistic")}


def values():
    out = {"git_head": R2.values()["git_head"]}
    g = R1.TA[R1.TA.run_id == FINAL]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def shift_part(m):
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJKLM" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def main():
    R1.use_snapshot(FINAL)
    vals = values()
    tpl = (REP / "report_template_round12c.md").read_text()
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    R12B.main()
    prev = (REP / "final_research_prototype_report.md").read_text()
    cut = prev.index("\n# Parts B–M: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Summary", "## R12b-0. Summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R12b-0.\1 ", own, flags=re.M)
    own = re.sub(r"§0\.(\d)", r"§R12b-0.\1", own)
    prev = re.sub(r"\b(Parts? )([B-M])(–| and )?([B-M])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("Part A covers the owner decisions closing L39", "Part B covers the owner decisions closing L39", 1)
    full = out + "\n\n# Part B: revision 13 (L39 decisions, 2026-10-21), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round12c.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
