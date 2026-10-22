"""Assemble the final report, revision 15: Part A = Round 13 (real data; reports/report_template_round13.md); Part B =
revision 14's Part A (Round 12); Parts C–D = revisions 13/12 Part A (L39/L38); Parts E–O = rounds 11..1."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round12c as R12C  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL = "final_full_suite_round13"
SUM = REP / "round13_summary.json"
CSV = REP / "tables/r13_single_by_model.csv"
NAMES = {"pbc_E2": "PBC (E2, primary)", "pbc_E1": "PBC (E1, sensitivity)", "prothro": "prothro", "aids": "aids"}


def S():
    return json.loads(SUM.read_text())


def f3(v):
    return "–" if v is None or (isinstance(v, float) and v != v) else f"{v:.3f}"


def single_table(ds):
    T = pd.read_csv(CSV)
    g = T[T.dataset == ds]
    rows = [[r.model, str(int(r.n_splits)), f"{r.ibs_mean:.3f} ({r.ibs_sd:.3f})", f3(r.get("ibs_cond_mean")),
             f"{r.harrell_c_mean:.3f}", f"{r.uno_c_mean:.3f} ({r.uno_c_sd:.3f})", f"{int(r.dcal_n_p_lt_005)}/{int(r.n_splits)}"]
            for _, r in g.sort_values("ibs_mean").iterrows()]
    return md_table(["Model", "Splits", "IBS mean (SD)", "Conditional IBS", "Harrell C", "Uno C mean (SD)",
                     "D-cal p < 0.05"], rows)


def contrast_table():
    rows = []
    for k, p in S()["contrasts"].items():
        if p.get("n", 0) < 2:
            continue
        rows.append([k, str(p["n"]), f"{p['mean_diff']:+.4f}", f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]", f"{p['p']:.3g}",
                     f"{p['n_neg']} / {p['n_pos']}"])
    return md_table(["Contrast (a − b)", "Splits", "Mean difference", "NB-corrected 95% CI", "p", "a < b / a > b"], rows)


def prune_table():
    rows = []
    for ds, v in S()["prune_symbolic"].items():
        rows.append([NAMES.get(ds, ds), str(v["n_splits"]),
                     "; ".join(f"{c}: {f:.1f}" for c, f in zip(v["channels"], v["selection_frequency"])),
                     f3(v["acceptance_rate"]), f3(v["mean_r2_val"]), f3(v["fidelity_corr_no_refit"]), f3(v["fidelity_corr_refit"]),
                     f"{f3(v['uno_c_pruned'])} / {f3(v['uno_c_no_refit'])} / {f3(v['uno_c_refit'])}",
                     f"{f3(v['ibs_pruned'])} / {f3(v['ibs_no_refit'])} / {f3(v['ibs_refit'])}", str(v["n_refit_nonfinite"])])
    return md_table(["Data set", "Splits", "Selection frequency per channel", "Symbolic acceptance", "Mean validation R²",
                     "Risk correlation, substituted (no refit) vs pruned", "Same, refitted", "Uno C: pruned / substituted / refitted",
                     "IBS: pruned / substituted / refitted", "Non-finite refits"], rows)


def formula_table():
    rows = []
    for ds, v in S()["prune_symbolic"].items():
        for e in v["formulas_split0"]:
            rows.append([NAMES.get(ds, ds), e.get("channel") or f"{e['layer']} {e['i']}", e["layer"], e["stage"],
                         "`" + e["formula"][:90] + "`", f"{e['r2_val']:.4f}", f"{e['share']:.2f}"])
    return md_table(["Data set", "Input", "Layer", "Stage", "Formula (original units of the input)", "Validation R²",
                     "Share of the layer's score"], rows) if rows else "No accepted formula carried ≥ 5% of its layer's score."


def b3_table():
    rows = []
    for ds, res in S()["b3"].items():
        for lm in res.get("b3_coxkan", {}):
            a, b = res["b3_coxkan"][lm], res.get("b3_cox", {}).get(lm, {})
            rows.append([NAMES.get(ds, ds), f"{float(lm):g}", f"{a['n_test']:.0f}", f3(a["brier"]), f3(b.get("brier")),
                         f3(a["harrell_c"]), f3(b.get("harrell_c")),
                         f"{f3(a['td_auc'])} ({a['n_auc']})", f"{f3(b.get('td_auc'))} ({b.get('n_auc', 0)})"])
    return md_table(["Data set", "Landmark", "Test rows (mean)", "Brier KAN", "Brier Cox", "C KAN", "C Cox",
                     "tdAUC KAN (splits)", "tdAUC Cox (splits)"], rows)


def cmapss_table():
    s = S()
    rows = [[r["model"], str(r["n_splits"]), f3(r["ibs_mean"]), f3(r["harrell_c_mean"]), f3(r["uno_c_mean"]),
             f3(r["dcal_p_mean"]), f"{r['dcal_n_p_lt_005']}/{r['n_splits']}"] for r in s["cmapss"]]
    out = md_table(["Model", "Splits", "IBS", "Harrell C", "Uno C", "Mean D-cal p", "D-cal p < 0.05"], rows)
    crow = [[k, str(p["n"]), f"{p['mean_diff']:+.4f}", f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]", f"{p['p']:.3g}"]
            for k, p in s["cmapss_contrasts"].items() if p.get("n", 0) >= 2]
    return out + "\n\n" + md_table(["Contrast", "Splits", "Mean difference", "NB-corrected 95% CI", "p"], crow)


def stab_table():
    rows = []
    for key, res in S()["stability"].items():
        ds, model = key.split("/")
        for c, v in res.items():
            rows.append([NAMES.get(ds, ds), model, c, str(v["n_fits"]), f3(v["r2_mean"]), f3(v["cos_mean"]),
                         f3(v["r2_mean_surface"]), v.get("direction", "–"),
                         (f"{v['frac_aligned']:.2f} ({'aligned' if v['aligned'] else 'not aligned'})" if "frac_aligned" in v else "–")])
    return md_table(["Data set", "Model", "Channel", "Fits", "Stability R² (time-averaged)", "Cosine",
                     "R² (full surface)", "Stated direction", "Share of fits monotone (≥ 0.80 = aligned)"], rows)


def figures():
    out = []
    for f in sorted((REP / "figures").glob("r13_effects_*.png")):
        out.append(f"![{f.stem}](figures/{f.name})")
    return "\n\n".join(out) if out else "No figure."


def exec_table():
    s = S()
    out = md_table(["Records", "Failed cells"], [[str(s["n_records"]), str(s["n_failed"])]])
    if s["failed"]:
        out += "\n\n" + md_table(["Job", "Data set", "Split", "Model", "Error"],
                                 [[f["job"], str(f["dataset"]), str(f["split"]), f["model"], (f["error"] or "")[:120]]
                                  for f in s["failed"]])
    return out


def decisions():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D(12[8-9]|13[0-9]) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    rows = []
    for run in ("final_full_suite_round12b", "final_full_suite_round12c", "final_full_suite_round12c_rerun", FINAL):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


TABLES = {"single_pbc_E2": lambda: single_table("pbc_E2"), "single_pbc_E1": lambda: single_table("pbc_E1"),
          "single_prothro": lambda: single_table("prothro"), "single_aids": lambda: single_table("aids"),
          "contrast_table": contrast_table, "prune_table": prune_table, "formula_table": formula_table,
          "b3_table": b3_table, "cmapss_table": cmapss_table, "stab_table": stab_table, "figures": figures,
          "exec_table": exec_table, "decisions": decisions, "suite_table": suite_table, "test_runs": R2.test_run_history}


def values():
    out = {"git_head": R2.values()["git_head"]}
    g = R1.TA[R1.TA.run_id == FINAL]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def shift_part(m):
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJKLMN" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def main():
    R1.use_snapshot(FINAL)
    vals = values()
    tpl = (REP / "report_template_round13.md").read_text()
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    R12C.main()
    prev = (REP / "final_research_prototype_report.md").read_text()
    cut = prev.index("\n# Parts B–N: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-12 summary", "## R12c-0. Round-12 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R12c-0.\1 ", own, flags=re.M)
    own = re.sub(r"§0\.(\d)", r"§R12c-0.\1", own)
    prev = re.sub(r"\b(Parts? )([B-N])(–| and )?([B-N])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("Part A is Round 12", "Part B is Round 12", 1)
    full = out + "\n\n# Part B: revision 14 (Round 12, 2026-10-22), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round13.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
