"""Assemble the final report: round-2 part (reports/report_template_round2.md) + round-1 report (historical).

Placeholders in the round-2 template: {{T:name}} table from saved records, {{V:key}} single value,
{{F:path}} embed a Markdown document (its headings demoted by two levels). Unknown placeholders raise an error.
The round-1 part is rendered by scripts/build_report.py and appended with its headings renumbered "II.x".
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table


def f(x, d=4):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


# ---------------------------------------------------------------- learnability
def gate_history():
    rows = []
    for r in D.gate_history():
        rows.append([r["label"], r["revision"], r["decision"], f(r["C"]), f(r["threshold"]),
                     "PASS" if r["C"] >= r["threshold"] else "FAIL", " / ".join(f(p, 3) for p in r["probe"]),
                     f(r["probe_mean"], 3), r["epochs"], f(r["R1C"]), f(r["R3dC"]), f"`{r['log']}`"])
    return md_table(["Gate run", "Design", "Decision", "R2 test C", "Threshold", "R2 C", "Probe R² a / b / A / ω",
                     "Probe mean (≥ 0.8)", "Epochs", "R1 C", "R3 \\|ΔC\\|", "Log"], rows)


def _search_rows(recs, base_keys=("r",)):
    rows = []
    for r in recs:
        if r["status"] != "ok":
            rows.append([r["step"], r["config_id"], json.dumps(r["config"]), r["seed"], "FAILED", "", "", "", "", "",
                         r.get("error", "")])
            continue
        cfg = ", ".join(f"{k}={v}" for k, v in r["config"].items())
        rows.append([r["step"], f"`{r['config_id']}`", cfg, r["seed"], f(r["val_C"]),
                     f(r["val_C"] - r["val_oracle_C"]), " / ".join(f(p, 2) for p in r["val_probe"]),
                     f(r["val_probe_mean"], 3), r["epochs"][0], r["runtime_s"], ""])
    return rows


def search_table():
    return md_table(["Step", "Config ID", "Configuration", "Seed", "Val C", "Val C − oracle", "Val probe a / b / A / ω",
                     "Probe mean", "Epochs", "Runtime (s)", "Error"], _search_rows(D.search_records()))


def search_superseded_compare():
    old = D.jsonl("superseded_D53/learnability_search.jsonl")
    new = D.search_records()

    def norm(r):                       # the recomputation pins r = 4 explicitly where the original inherited it
        c = dict(r["config"])
        c.setdefault("r", 4) if r["step"] in ("S1-front", "S2-latent") else None
        return r["step"].split("-")[0], r["kind"], r["seed"], json.dumps(c, sort_keys=True)

    by_key = {norm(n): n for n in new}
    rows = []
    for o in old:
        n = by_key.get(norm(o))
        if n is None:
            rows.append([o["step"], o.get("config_id"), f(o["val_C"]), "not refitted (identical to the S1 A3 fit; "
                         "deduplicated by effective-config ID)", "", f(o["val_probe_mean"], 3), ""])
            continue
        rows.append([o["step"], o.get("config_id"), f(o["val_C"]), f(n["val_C"]), f(n["val_C"] - o["val_C"]),
                     f(o["val_probe_mean"], 3), f(n["val_probe_mean"], 3)])
    return md_table(["Step", "Original config ID", "Val C (original)", "Val C (recomputed)", "Δ", "Probe mean (original)",
                     "Probe mean (recomputed)"], rows)


def confirm_table(which):
    sets = D.confirmations()
    old = {"D41 (seeds 101–105)": D.jsonl("superseded_D53/learnability_confirmation_fae_mlp.jsonl"),
           "D46 (seeds 106–110)": D.jsonl("superseded_D53/learnability_confirmation_fae_mlp_ext.jsonl")}[which]
    oldC = {r["data_seed"]: r["test_C"] for r in old}
    rows = []
    for r in sets[which]:
        rows.append([r["data_seed"], f(r["test_C"]), f(r["oracle_C"]), f(r["threshold"]), f(r["test_C"] - r["threshold"]),
                     " / ".join(f(p, 3) for p in r["probe"]), f(r["probe_mean"], 3), "PASS" if r["pass"] else "FAIL",
                     f(oldC.get(r["data_seed"]))])
    n = sum(r["pass"] for r in sets[which])
    rows.append(["**Total**", "", "", "", "", "", "", f"**{n}/{len(sets[which])}**", ""])
    return md_table(["Data seed", "Test C", "Oracle C", "Threshold", "C − threshold", "Probe a / b / A / ω",
                     "Probe mean", "Pass", "Superseded test C (D53)"], rows)


# ---------------------------------------------------------------- symbolic / KAN-AFT
def kanaft_summary():
    rs = D.symbolic_synth()
    rows = []
    for loss in ("parametric", "bj", "ipcw", "transform"):
        sub = [r for r in rs if r["loss"] == loss]
        if not sub:
            continue
        g = lambda k, m: np.array([r[k][m] for r in sub])  # noqa: E731
        acc = sum(s["accepted"] for r in sub for s in r["symbolic"])
        tot = sum(len(r["symbolic"]) for r in sub)
        tr = [s["truth_recovery_r2"] for r in sub for s in r["symbolic"] if s["truth_recovery_r2"] is not None]
        rows.append([loss, len(sub), ", ".join(str(r["lam_selected"]) for r in sub),
                     f"{g('unpruned_test', 'harrell_c').mean():.4f}", f"{g('pruned_test', 'harrell_c').mean():.4f}",
                     f"{g('symbolic_test', 'harrell_c').mean():.4f}", f"{g('unpruned_test', 'ibs').mean():.4f}",
                     f"{g('symbolic_test', 'ibs').mean():.4f}", f"{g('unpruned_test', 'curve_error').mean():.4f}",
                     f"{g('symbolic_test', 'curve_error').mean():.4f}", sum(r["prune"]["edges_pruned"] for r in sub),
                     f"{acc}/{tot}", f"{np.mean(tr):.4f} (min {np.min(tr):.4f})"])
    return md_table(["Loss", "Seeds", "λ selected (val C)", "Test C spline", "Test C pruned", "Test C formulas",
                     "Test IBS spline", "Test IBS formulas", "Curve error spline", "Curve error formulas",
                     "Edges pruned (θ=1e-2)", "Formulas accepted", "Truth-recovery R² (true edges)"], rows)


def _sym_row(prefix, s):
    best = ""
    if not s["accepted"]:
        ex = s.get("extra") or {}
        bl, bp = ex.get("best_library"), ex.get("best_pysr")
        parts = []
        if bl:
            parts.append(f"best library {bl.get('f')} val R² {bl.get('r2_val', float('nan')):.4f}")
        if bp and bp.get("formula"):
            parts.append(f"best PySR val R² {bp.get('r2_val', float('nan')):.4f}")
        best = "; ".join(parts) or "no candidate"
    xr = s.get("x_range") or [np.nan, np.nan]
    return prefix + [s["stage"] if s["accepted"] else "rejected → spline kept",
                     f"`{s['formula']}`" if s["accepted"] else best, s["complexity"] if s["accepted"] else "–",
                     f(s.get("r2_fit")), f(s.get("r2_val")), f(s.get("r2_grid")), f(s.get("rmse_val")),
                     f(s.get("maxabs_grid")), f"[{xr[0]:.2f}, {xr[1]:.2f}]" if xr[0] == xr[0] else "–"]


SYM_HDR = ["Stage / outcome", "Formula (standardised input x)", "Complexity", "R² train", "R² val", "R² grid",
           "RMSE val", "Max \\|err\\| grid", "x range"]


def kanaft_formulas():
    rows = []
    for r in D.symbolic_synth():
        for s in r["symbolic"]:
            tr = s["truth_recovery_r2"]
            rows.append(_sym_row([r["seed"], r["loss"], s["true_component"], f(tr) if tr is not None else "n/a"], s))
    return md_table(["Seed", "Loss", "True component", "Truth R²"] + SYM_HDR, rows)


def censoring_table():
    cv = D.censoring_variants()
    rows = []
    for sc, v in cv.items():
        for k, e in v["est"].items():
            rows.append([sc, f"{v['censored']:.1%}", k + (" (oracle reference)" if "oracle" in k else ""),
                         f(e["bias"][0]), f(e["rmse"][0]), f(e["bias"][1]), f(e["rmse"][1])])
    return md_table(["Scenario", "Censored", "Estimator", "Bias β₁ (0.8)", "RMSE β₁", "Bias β₂ (−0.5)", "RMSE β₂"], rows)


def pruning_diag():
    rs = D.jsonl("pruning_threshold_diag.jsonl")
    rows = []
    for r in rs:
        u, c = list(r["edge_l1"].values()), list(r.get("edge_l1_centred", {}).values())
        rows.append([r["seed"], r["lam"], " / ".join(f(x, 3) for x in u), " / ".join(f(x, 3) for x in c),
                     "yes" if u[3] < 1e-2 else "no", "yes" if c and c[3] < min(c[:3]) else "no"])
    return md_table(["Seed", "λ", "Edge L1 (z1 / z2 / z3 / null z4)", "Centred L1 (z1 / z2 / z3 / null z4)",
                     "Null pruned at θ=1e-2", "Centred: null < every true edge"], rows)


def pbc_kan_table():
    rows = []
    for r in D.kanaft_pbc():
        li, k = r["linear"]["test"], r["kan"]["test"]
        rows.append([r["endpoint"], r["split"], r["events_test"], r["linear"]["l2"], r["kan"]["lam"], f(li["ibs"]), f(k["ibs"]),
                     f(r["kan_symbolic_test"]["ibs"]), f(li["ibs_conditional"]), f(k["ibs_conditional"]), f(li["uno_c"], 3),
                     f(k["uno_c"], 3), f(r["kan_symbolic_test"]["uno_c"], 3), sum(s["accepted"] for s in r["symbolic"])])
    return md_table(["Endpoint", "Split", "Test events", "Linear l2", "KAN λ", "IBS linear", "IBS KAN", "IBS KAN formulas",
                     "Cond. IBS linear", "Cond. IBS KAN", "Uno C linear", "Uno C KAN", "Uno C KAN formulas",
                     "Formulas accepted /8"], rows)


def pbc_kan_paired():
    rows = []
    rs = D.kanaft_pbc()
    for ep in ("E1", "E2"):
        sub = [r for r in rs if r["endpoint"] == ep]
        for m, lab in (("ibs", "IBS"), ("ibs_conditional", "Conditional IBS"), ("uno_c", "Uno C")):
            dlt = np.array([r["kan"]["test"][m] - r["linear"]["test"][m] for r in sub])
            better = (dlt < 0) if m != "uno_c" else (dlt > 0)
            rows.append([ep, lab, f"{np.mean([r['linear']['test'][m] for r in sub]):.4f}",
                         f"{np.mean([r['kan']['test'][m] for r in sub]):.4f}", f"{dlt.mean():+.4f} ± {dlt.std(ddof=1):.4f}",
                         f"{int(better.sum())}/{len(sub)}"])
    return md_table(["Endpoint", "Metric", "Linear AFT (mean)", "KAN-AFT (mean)", "KAN − linear (mean ± sd over splits)",
                     "Splits where KAN better"], rows)


def pbc_formula_stability():
    rs = D.kanaft_pbc()
    rows = []
    for ep in ("E1", "E2"):
        sub = [r for r in rs if r["endpoint"] == ep]
        for cov in sub[0]["covariates"]:
            ss = [s for r in sub for s in r["symbolic"] if s["covariate"] == cov]
            acc = [s for s in ss if s["accepted"]]
            fams = {}
            for s in acc:
                key = s["stage"] if s["stage"] != "library" else s["params"].get("f", "library")
                fams[key] = fams.get(key, 0) + 1
            rows.append([ep, cov, f"{len(acc)}/{len(ss)}", ", ".join(f"{k} ×{v}" for k, v in sorted(fams.items())) or "–"])
    return md_table(["Endpoint", "Covariate", "Accepted (splits)", "Accepted function families"], rows)


def pbc_formulas():
    rows = []
    for r in D.kanaft_pbc():
        for s in r["symbolic"]:
            rows.append(_sym_row([r["endpoint"], r["split"], s["covariate"]], s))
    return md_table(["Endpoint", "Split", "Covariate"] + SYM_HDR, rows)


# ---------------------------------------------------------------- tests, decisions, misc
ROUND2_CRITERIA = ["AUD", "LS-", "KREG", "SYM", "F6", "F7", "F8", "F9", "INT", "R1", "R2", "R3"]


def round2_tests():
    TA, TL = R1.TA, R1.TL
    rows = []
    for _, r in TL.iterrows():
        c = str(r.criterion)
        if not any(c.startswith(p) for p in ROUND2_CRITERIA):
            continue
        hist = TA[TA.nodeid == r.nodeid]
        n_fail = int((hist.outcome == "failed").sum())
        rows.append([c, r.nodeid.split("::")[-1][:60], r.kind, len(hist), n_fail, r.outcome.upper()])
    return md_table(["Criterion", "Test", "Kind", "Attempts recorded", "Failed attempts", "Latest outcome"], rows)


def test_run_history():
    TA = R1.TA
    rows = []
    for rid, g in TA.groupby("run_id", sort=False):
        rows.append([rid, int((g.outcome == "passed").sum()), int((g.outcome == "failed").sum()),
                     int((g.outcome == "skipped").sum())])
    return md_table(["Test run (chronological)", "Passed", "Failed", "Skipped"], rows)


def decisions_round2():
    lines = (ROOT / "docs/decisions.md").read_text().splitlines()
    rows = [l for l in lines if re.match(r"\| D(3[8-9]|[4-9]\d) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def fd003_table():
    cols = [("IBS ↓", "ibs", 3), ("Harrell C ↑", "harrell_c", 3), ("RUL RMSE internal ↓", "rul_test_rmse", 1),
            ("RUL RMSE official ↓", "rul_official_rmse", 1), ("n official", "rul_official_n", 0)]
    return {a: R1.model_table("cmapss_FD003", a, cols) for a in ("cmapss_A", "cmapss_B", "cmapss_A_artificial")}


FD3 = None


def fd3(a):
    global FD3
    FD3 = FD3 or fd003_table()
    return FD3[a]


TABLES = {"gate_history": gate_history, "search": search_table, "search_compare": search_superseded_compare,
          "confirm_d41": lambda: confirm_table("D41 (seeds 101–105)"),
          "confirm_d46": lambda: confirm_table("D46 (seeds 106–110)"),
          "kanaft_summary": kanaft_summary, "kanaft_formulas": kanaft_formulas, "censoring": censoring_table,
          "pruning_diag": pruning_diag, "pbc_kan": pbc_kan_table, "pbc_kan_paired": pbc_kan_paired,
          "pbc_formula_stability": pbc_formula_stability, "pbc_formulas": pbc_formulas, "round2_tests": round2_tests,
          "test_runs": test_run_history, "decisions_r2": decisions_round2,
          "fd003_A": lambda: fd3("cmapss_A"), "fd003_B": lambda: fd3("cmapss_B"),
          "fd003_A_art": lambda: fd3("cmapss_A_artificial")}


def values():
    v = {}
    g = D.gate_history()[-1]
    v.update(gate_C=f(g["C"]), gate_thr=f(g["threshold"]), gate_oracle=f(g["oracle"]), gate_probe_mean=f(g["probe_mean"], 3),
             gate_probe=" / ".join(f(p, 3) for p in g["probe"]), gate_R1C=f(g["R1C"]), gate_R3dC=f(g["R3dC"]),
             gate_label=g["label"])
    for k, rs in D.confirmations().items():
        tag = "d41" if k.startswith("D41") else "d46"
        v[f"{tag}_pass"] = f"{sum(r['pass'] for r in rs)}/{len(rs)}"
        v[f"{tag}_gap_max"] = f(max(r["test_C"] - r["threshold"] for r in rs))
        v[f"{tag}_gap_min"] = f(min(r["test_C"] - r["threshold"] for r in rs))
    gaps = [r["test_C"] - r["threshold"] for rs in D.confirmations().values() for r in rs] + [g["C"] - g["threshold"]]
    v["gap_range"] = f"{-max(gaps):.3f}–{-min(gaps):.3f}"
    v["n_search_fits"] = str(len(D.search_records()))
    sel = json.loads((REP / "learnability_selected_ext.json").read_text())
    v["selected_id"], v["selected_cfg"] = sel["config_id"], json.dumps(sel["config"])
    # historical accuracy (round 3): the round-2 text quotes the counts of ITS final full-suite run
    TL = R1.TA[R1.TA.run_id == "final_full_suite_round2_v3"]
    TL = TL if len(TL) else R1.TL
    v["n_tests_latest"] = str(len(TL))
    for o in ("passed", "failed", "skipped"):
        v[f"latest_{o}"] = str(int((TL.outcome == o).sum()))
    v["n_test_records"], v["n_test_runs"] = str(len(R1.TA)), str(R1.TA.run_id.nunique())
    v["git_head"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                   text=True).stdout.strip()
    sy = D.symbolic_synth()
    v["sym_acc"] = f"{sum(s['accepted'] for r in sy for s in r['symbolic'])}/{sum(len(r['symbolic']) for r in sy)}"
    pb = D.kanaft_pbc()
    v["pbc_acc"] = f"{sum(s['accepted'] for r in pb for s in r['symbolic'])}/{sum(len(r['symbolic']) for r in pb)}"
    return v


def embed(path):
    txt = (ROOT / path).read_text()
    txt = re.sub(r"^(#{1,4}) ", lambda m: "#" * min(len(m.group(1)) + 2, 6) + " ", txt, flags=re.M)
    return txt


def render_round1():
    R1.main()
    txt = (REP / "final_research_prototype_report.md").read_text()
    txt = re.sub(r"^## (\d+)\. ", r"## II.\1 ", txt, flags=re.M)
    txt = re.sub(r"^### (\d+)\.(\d+) ", r"### II.\1.\2 ", txt, flags=re.M)
    txt = re.sub(r"^#### (\d+)\.(\d+)\.(\d+) ", r"#### II.\1.\2.\3 ", txt, flags=re.M)
    return txt


def main():
    vals = values()
    tpl = (REP / "report_template_round2.md").read_text()

    def rep_t(m):
        if m.group(1) not in TABLES:
            raise KeyError(f"unknown table {m.group(1)}")
        return TABLES[m.group(1)]()

    def rep_v(m):
        if m.group(1) not in vals:
            raise KeyError(f"unknown value {m.group(1)}")
        return vals[m.group(1)]

    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", rep_t, tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", rep_v, out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: embed(m.group(1)), out)
    r1 = render_round1()
    full = out + "\n\n" + r1
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round2.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
