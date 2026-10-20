"""Assemble the final report, revision 11: round-11 part (reports/report_template_round11.md: implementation and
acceptance tests CK-1..CK-12), then the round-10..1 reports as historical records (Parts B–K)."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round10 as R10  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL11 = "final_full_suite_round11"
CK_FILE = REP / "logs/round11_ck.jsonl"
CRIT = [("CK-1", "DET", "Breslow/Efron NLPL vs lifelines and direct formulas, < 1e-8"),
        ("CK-2", "DET", "Breslow baseline vs lifelines, < 1e-6"),
        ("CK-3", "STAT", "linear-edge CoxKAN slopes vs lifelines within 0.05 (seed 11)"),
        ("CK-4", "DET", "functional quadrature with φ(x) = x, < 1e-3"),
        ("CK-5", "STAT", "B2-S Cox recovers S-Add φ₁, φ₂ with RISE < 0.10 (seed 12)"),
        ("CK-6", "STAT", "AFCM recovers a known F with RISE < 0.10 (seed 13)"),
        ("CK-7", "STAT", "PACE sparse: \\|cos\\| > 0.95 and lower ISE than per-subject smoothing (seed 14)"),
        ("CK-8", "STAT", "pruning removes the null channel, keeps both active (seed 15)"),
        ("CK-9", "DET", "landmark builder: no future data, at-risk only, censored at t_L + w"),
        ("CK-10", "DET", "symbolic stages 1–2 recover sin, tanh, x² (R² > 0.99)"),
        ("CK-11", "STAT", "residual-calibrated AFT D-cal p > 0.05 where Weibull p < 0.05 (seed 16)"),
        ("CK-12", "DET", "LME BLUPs vs closed form, < 1e-8")]


def ck_records():
    """Latest record per CK from the final suite, else from the formal acceptance run."""
    recs = [json.loads(l) for l in CK_FILE.read_text().splitlines() if l.strip()]
    out = {}
    for run in ("round11_acceptance", FINAL11):
        for r in recs:
            if r["run_id"] == run:
                out[r["ck"]] = r
    return out


def outcome(ck, run):
    g = R1.TA[(R1.TA.run_id == run) & (R1.TA.criterion == ck)]
    if not len(g):
        return "not run"
    o = set(g.outcome)
    return "PASS" if o == {"passed"} else ("FAIL" if "failed" in o else "/".join(sorted(o)))


def key_numbers(ck, r):
    if r is None:
        return "–"
    f = lambda v: f"{v:.3f}"  # noqa: E731
    if ck == "CK-3":
        return f"max \\|slope − lifelines\\| {f(r['max_abs_err'])} (slopes {', '.join(map(f, r['slopes']))}; lifelines {', '.join(map(f, r['lifelines']))})"
    if ck in ("CK-5", "CK-6"):
        return f"RISE φ₁ {f(r['rise_phi1'])}, φ₂ {f(r['rise_phi2'])}"
    if ck == "CK-7":
        return f"\\|cos\\| {', '.join(map(f, r['abs_cos']))}; mean ISE PACE {f(r['mean_ise_pace'])} vs smoothing {f(r['mean_ise_smooth'])}"
    if ck == "CK-8":
        return f"selected channels {r['selected']} (required [1, 1, 0])"
    if ck == "CK-11":
        return f"D-cal p: residual-calibrated {r['p_residual_calibrated']:.4f}, Weibull {r['p_weibull']:.4f}"
    return "–"


def ck_table():
    recs = ck_records()
    rows = [[ck, kind, crit, outcome(ck, "round11_acceptance"), outcome(ck, FINAL11), key_numbers(ck, recs.get(ck))]
            for ck, kind, crit in CRIT]
    return md_table(["ID", "Type", "Criterion", "Formal run", "Final suite", "Key numbers"], rows)


def ck_details():
    recs = ck_records()
    rows = []
    r = recs.get("CK-5")
    if r:
        rows.append(["CK-5", f"tuning: lr {r['lr']}, wd {r['wd']} selected; members' RISE φ₁ "
                     + ", ".join(f"{m[0]:.3f}" for m in r["member_rise"]) + "; φ₂ "
                     + ", ".join(f"{m[1]:.3f}" for m in r["member_rise"]) + f"; curve smoothing λ {r['smoothing_lambda']:g}"])
    r = recs.get("CK-6")
    if r:
        rows.append(["CK-6", f"selected λ_x {r['lam_x']:g}, λ_t {r['lam_t']:g} (5 × 5 grid); curve smoothing λ {r['smoothing_lambda']:g}"])
    r = recs.get("CK-7")
    if r:
        rows.append(["CK-7", "per-channel ISE PACE " + ", ".join(f"{v:.3f}" for v in r["ise_pace"]) + "; smoothing "
                     + ", ".join(f"{v:.3f}" for v in r["ise_smooth"]) + f" (λ {r['smoothing_lambda']:g})"])
    r = recs.get("CK-8")
    if r:
        best = min(r["grid"], key=lambda g: (round(g["val_loss"], 10), -g["theta"], -g["lam"]))
        npr = {lam: max(g["pruned"] for g in r["grid"] if g["lam"] == lam) for lam in sorted({g["lam"] for g in r["grid"]})}
        rows.append(["CK-8", f"selected λ {best['lam']:g}, θ_rel {best['theta']:g} (validation NLPL {best['val_loss']:.4f}); "
                     "maximum edges pruned per λ: " + ", ".join(f"{k:g}: {v}" for k, v in npr.items())])
    r = recs.get("CK-11")
    if r:
        rows.append(["CK-11", f"σ̂ {r['sigma']:.3f}; curve smoothing λ {r['smoothing_lambda']:g}"])
    return md_table(["ID", "Details from the records"], rows)


def ident_table():
    d = json.loads((REP / "round11/diag_identifiability.json").read_text())
    rows = []
    for ch in ("channel1", "channel2"):
        c = d[ch]
        rows.append([ch.replace("channel", "channel "), f"{c['sd_int_X']:.1e}", f"{c['sd_int_true_phi']:.3f}",
                     f"{c['tensor_10x10']['rank_1e2']} / {c['tensor_10x10']['rank_1e3']} / {c['tensor_10x10']['rank_1e4']} of 100",
                     f"{c['homogeneous_10']['rank_1e2']} / {c['homogeneous_10']['rank_1e3']} / {c['homogeneous_10']['rank_1e4']} of 10"])
    return ("Development seed 902, n 2,000 (`scripts/diag_round11_identifiability.py`). Rank = number of singular values "
            "of the column-centred design above 1e-2 / 1e-3 / 1e-4 of the largest.\n\n"
            + md_table(["S-Add", "SD over subjects of ∫X_d dt", "SD of the true ∫φ_d term",
                        "AFCM 10 × 10 design rank", "Homogeneous design rank"], rows))


def explore_table():
    d = json.loads((REP / "round11/explore_rise.json").read_text())
    rows = []
    for key, lab in (("CK5_b2s_cox_seed12", "B2-S Cox ensemble (CK-5 data, seed 12)"), ("CK6_afcm_seed13", "AFCM (CK-6 data, seed 13)")):
        v = d[key]
        rows.append([lab, f"{v['phi1']['rise']:.3f}", f"{v['phi1']['rise_aff']:.3f}", f"{v['phi2']['rise']:.3f}",
                     f"{v['phi2']['rise_aff']:.3f}"])
    return md_table(["Model (data)", "φ₁ RISE (registered)", "φ₁ RISE-aff", "φ₂ RISE (registered)", "φ₂ RISE-aff"], rows)


def suite_table():
    rows = []
    for run in ("final_full_suite", "final_full_suite_round2_v3", "final_full_suite_round3", "final_full_suite_round4_rerun",
                "final_full_suite_round5", "final_full_suite_round6", "final_full_suite_round6_rerun",
                "final_full_suite_round7", "final_full_suite_round8", "final_full_suite_round9", "final_full_suite_round10",
                FINAL11):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


def decisions_r11():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D(11[6-9]) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


TABLES = {"suite_table": suite_table, "decisions_r11": decisions_r11, "test_runs": R2.test_run_history,
          "ck_table": ck_table, "ck_details": ck_details, "ident_table": ident_table, "explore_table": explore_table}


def values():
    out = {"git_head": R2.values()["git_head"]}
    g = R1.TA[R1.TA.run_id == FINAL11]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def shift_part(m):
    """'Part B' -> 'Part C', 'Parts B–J' -> 'Parts C–K', 'Parts I and J' -> 'Parts J and K' (one pass; Part A kept)."""
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJ" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def main():
    R1.use_snapshot(FINAL11)
    vals = values()
    tpl = (REP / "report_template_round11.md").read_text()
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    R10.main()
    prev = (REP / "final_research_prototype_report.md").read_text()
    cut = prev.index("\n# Parts B–J: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-10 summary", "## R10-0. Round-10 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R10-0.\1 ", own, flags=re.M)
    own = re.sub(r"§0\.(\d)", r"§R10-0.\1", own)
    prev = re.sub(r"\b(Parts? )([B-J])(–| and )?([B-J])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("Part A is round 10", "Part B is round 10", 1)
    full = out + "\n\n# Part B: round-10 report (2026-10-19), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round11.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
