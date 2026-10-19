"""Assemble the final report, revision 9: round-9 part (reports/report_template_round9.md), then the round-8..1 reports
as historical records (Parts B–I), each rendered from its own round's snapshot of the test records."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round8 as R8  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table, f = R1.md_table, R2.f
FINAL9 = "final_full_suite_round9"
LAB = {"mlp": "fae_mlp (C4F)", "kan": "fae_kan (C4F; D31 decoder 16→3→36)", "kan_shared": "fae_kan, Mode B (MLP decoder)",
       "kan_unb": "fae_kan, Mode A (KAN decoder 16→36)", "mlp_wide": "fae_mlp_wide (hidden 107)"}
MLAB = {"curve_error": "Curve error", "ibs": "IBS", "harrell_c": "Harrell C", "uno_c": "Uno C", "dcal_p": "D-cal p",
        "reconstruction_test": "Reconstruction (test)"}


def arms_table():
    rows = []
    for a, (blk_e, blk_d, bp) in {"mlp": ("MLP 32→64→16", "MLP 16→64→36", 6580), "kan": ("KAN 32→16", "KAN 16→3→36", 6735),
                                  "kan_shared": ("KAN 32→16", "MLP 16→64→36", 8564),
                                  "kan_unb": ("KAN 32→16", "KAN 16→36", 10932),
                                  "mlp_wide": ("MLP 32→107→16", "MLP 16→107→36", 10966)}.items():
        rows.append([LAB[a], blk_e, blk_d, bp])
    return md_table(["Arm", "Encoder block", "Decoder block", "Encoder + decoder parameters"], rows)


def cells():
    p = REP / "tables/r9_cells.csv"
    if not p.exists():
        return "_Not run._"
    c = pd.read_csv(p)
    order = {a: i for i, a in enumerate(LAB)}
    c = c.assign(o=c.arm.map(order)).sort_values(["o", "strategy"])
    ms = lambda r, m, d=4: f"{r[m + '_mean']:.{d}f} ± {r[m + '_std']:.{d}f}"  # noqa: E731
    return md_table(["Arm", "Strategy", "n", "Parameters (whole model)", "Harrell C", "Uno C", "IBS", "Curve error",
                     "D-cal p", "Reconstruction (test)"],
                    [[LAB[r["arm"]], r["strategy"], int(r["harrell_c_count"]), int(r["n_params"]), ms(r, "harrell_c"),
                      ms(r, "uno_c"), ms(r, "ibs"), ms(r, "curve_error"), ms(r, "dcal_p", 3), ms(r, "reconstruction_test")]
                     for _, r in c.iterrows()])


def contrasts(ids):
    p = REP / "tables/r9_contrasts.csv"
    if not p.exists():
        return "_Not run._"
    c = pd.read_csv(p)
    c = c[c.id.isin(ids)]
    return md_table(["ID", "Contrast (paired, 5 replicates, ensembles)", "Metric", "Mean diff", "SD", "95% CI (t, df 4)",
                     "p (t)", "+ / −", "p (exact Wilcoxon)"],
                    [[r.id, r.contrast, MLAB[r.metric], f"{r.mean_diff:+.4f}", f"{r.sd_diff:.4f}",
                      f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]", f"{r.p_t:.3f}", f"{r.n_pos} / {r.n_neg}", f"{r.p_wilcoxon:.4f}"]
                     for r in c.itertuples()])


def determinism():
    p = REP / "tables/r9_determinism.csv"
    if not p.exists():
        return ""
    d = pd.read_csv(p)
    return md_table(["Arm (S2, repeated)", "Records", "H1 records", "Round-9 records", "Survival metrics bit-identical"],
                    [[LAB[r.arm], r.phase, r.n_h1, r.n_r9, "**yes**" if r.bit_identical else "NO"] for r in d.itertuples()])


def status():
    p = REP / "tables/r9_status.csv"
    if not p.exists():
        return ""
    s = pd.read_csv(p)
    piv = s.pivot_table(index=["arm", "strategy"], columns=["phase", "status"], values="n", aggfunc="sum", fill_value=0)
    piv.columns = [f"{a} {b}" for a, b in piv.columns]
    piv = piv.reset_index()
    return md_table(["Arm", "Strategy"] + list(piv.columns[2:]),
                    [[LAB[r["arm"]], r["strategy"]] + [int(r[c]) for c in piv.columns[2:]] for _, r in piv.iterrows()])


def tuning():
    p = REP / "tables/r9_tuning.csv"
    if not p.exists():
        return ""
    t = pd.read_csv(p)
    return md_table(["Arm", "Strategy", "Selected lr, wd per replicate (1000–1004)"],
                    [[LAB[r.arm], r.strategy, "; ".join(eval(r.cfg))] for r in t.itertuples()])


def pbc_invariance():
    import numpy as np
    R = [json.loads(l) for l in (REP / "kanaft_pbc.jsonl").read_text().splitlines() if l.strip()]
    rows = []
    for ep in ("E2", "E1"):
        rs = [r for r in R if r["endpoint"] == ep]
        for m in ("harrell_c", "uno_c", "ibs"):
            L = np.array([r["linear"]["test"][m] for r in rs]); K = np.array([r["kan"]["test"][m] for r in rs])
            d = K - L
            rows.append([f"KAN-AFT − linear AFT ({ep})", MLAB[m], f"{L.mean():.4f}", f"{K.mean():.4f}", f"{d.mean():+.4f}",
                         f"{(d > 0).sum()} / {(d < 0).sum()}"])
    c = pd.read_csv(REP / "tables/step3_real_contrasts.csv")
    for r in c[c.analysis.str.startswith("neural4f_pbc")].itertuples():
        if r.metric in ("harrell_c", "uno_c", "ibs"):
            rows.append([f"fae_kan − fae_mlp ({r.analysis[-2:]}; NB 95% CI [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}])", MLAB[r.metric],
                         "", "", f"{r.mean_diff:+.4f}", f"{r.n_pos} / {r.n_neg}"])
    return md_table(["Comparison (5 outer splits)", "Metric", "Reference", "KAN", "Mean difference", "Splits + / −"], rows)


def new_tests():
    TA, TL = R1.TA, R1.TL
    rows = []
    for _, r in TL.iterrows():
        if "kan_fairness" not in r.nodeid:
            continue
        h = TA[TA.nodeid == r.nodeid]
        rows.append([r.criterion or "–", r.nodeid.split("::")[-1][:70], len(h), int((h.outcome == "failed").sum()),
                     r.outcome.upper()])
    return md_table(["Criterion", "Test (tests/test_kan_fairness.py)", "Attempts", "Failed attempts", "Latest"], rows)


def suite_table():
    rows = []
    for run in ("final_full_suite", "final_full_suite_round2_v3", "final_full_suite_round3", "final_full_suite_round4_rerun",
                "final_full_suite_round5", "final_full_suite_round6", "final_full_suite_round6_rerun",
                "final_full_suite_round7", "final_full_suite_round8", FINAL9):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


def decisions_r9():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D(10[7-9]|1[1-9]\d) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


TABLES = {"arms": arms_table, "cells": cells, "F1": lambda: contrasts(["F1"]), "F2": lambda: contrasts(["F2"]),
          "F3": lambda: contrasts(["F3"]), "S": lambda: contrasts(["S"]), "determinism": determinism, "status": status,
          "tuning": tuning, "pbc_invariance": pbc_invariance, "new_tests": new_tests, "suite_table": suite_table,
          "decisions_r9": decisions_r9, "test_runs": R2.test_run_history, "h1_table": R8.h1_table}


def values():
    out = {"git_head": R2.values()["git_head"]}
    g = R1.TA[R1.TA.run_id == FINAL9]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def shift_part(m):
    """'Part B' -> 'Part C', 'Parts B–H' -> 'Parts C–I', 'Parts G and H' -> 'Parts H and I' (one pass; Part A is kept)."""
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGH" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def main():
    R1.use_snapshot(FINAL9)
    vals = values()
    tpl = (REP / "report_template_round9.md").read_text()
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    R8.main()
    prev = (REP / "final_research_prototype_report.md").read_text()
    cut = prev.index("\n# Parts B–H: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-8 summary", "## R8-0. Round-8 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R8-0.\1 ", own, flags=re.M)
    own = re.sub(r"§0\.(\d)", r"§R8-0.\1", own)
    prev = re.sub(r"\b(Parts? )([B-H])(–| and )?([B-H])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("Part A is round 8", "Part B is round 8", 1)
    full = out + "\n\n# Part B: round-8 report (2026-10-18), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round9.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
