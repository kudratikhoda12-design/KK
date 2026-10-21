"""Assemble the final report, revision 12: round-12 part (reports/report_template_round12.md: owner decisions D120
closing L38, the repaired acceptance tests CK-3b..CK-11b and the Round-12 gate), then the round-11..1 reports as
historical records (Parts B–L)."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round11 as R11  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL12 = "final_full_suite_round12"
ACC = "round12_ck_acceptance"
CK_FILE = REP / "logs/round12_ck.jsonl"
DEV_FILE = REP / "logs/round12/dev_ckb_dryruns.jsonl"
CRIT = [("CK-3b", 19, "converged linear-edge CoxKAN slopes vs lifelines within 0.05 (no early stopping, wd 0)"),
        ("CK-5b", 17, "B2-S Cox ensemble: identifiable RISE φ₁ and φ₂ each < 0.10 (DGP v2)"),
        ("CK-6b", 18, "AFCM (λ_t grid to 1e4): identifiable RISE φ₁ and φ₂ each < 0.10 (DGP v2)"),
        ("CK-8b", 20, "integrated channel pruning, parsimonious selection: channels [1, 1, 0]"),
        ("CK-11b", 21, "mixture AFT errors: residual-calibrated D-cal p > 0.05 and Weibull p < 0.05")]
DEV = {"CK-3b": ("ck3b", 908), "CK-5b": ("ck5b", 913), "CK-6b": ("ck6b", 914), "CK-8b": ("ck8b", 915)}


def recs():
    out = {}
    for run in (ACC, FINAL12):
        for l in CK_FILE.read_text().splitlines():
            r = json.loads(l)
            if r["run_id"] == run:
                out[r["ck"]] = r
    return out


def dev():
    out = {}
    for l in DEV_FILE.read_text().splitlines():
        r = json.loads(l)
        out.setdefault(r["job"], []).append(r)
    return out


def numbers(ck, r):
    if r is None:
        return "–"
    f = lambda v: f"{v:.3f}"  # noqa: E731
    if ck == "CK-3b":
        return f"max \\|slope − lifelines\\| {r['max_abs_err']:.1e} (slopes {', '.join(map(f, r['slopes']))})"
    if ck in ("CK-5b", "CK-6b"):
        s = r["shapes"]
        return (f"identifiable RISE φ₁ {f(s['phi1']['rise_id'])}, φ₂ {f(s['phi2']['rise_id'])} "
                f"(raw {f(s['phi1']['rise_raw'])}, {f(s['phi2']['rise_raw'])})")
    if ck == "CK-8b":
        return f"selected channels {r['selected']}"
    if ck == "CK-11b":
        return f"D-cal p: residual-calibrated {r['p_residual_calibrated']:.4f}, Weibull {r['p_weibull']:.1e}"
    return "–"


def dev_numbers(ck):
    d = dev()
    if ck == "CK-11b":
        rows = d.get("ck11b", [])
        return "seeds 909–912: residual-calibrated p " + ", ".join(f"{r['p_residual_calibrated']:.3f}" for r in rows) + \
            "; Weibull p all < 1e-16"
    job, seed = DEV[ck]
    rr = [r for r in d.get(job, []) if r["seed"] == seed]
    return f"seed {seed}: " + numbers(ck, rr[0]) if rr else "–"


def ck_table():
    R = recs()
    rows = [[ck, str(seed), crit, R11.outcome(ck, ACC), R11.outcome(ck, FINAL12), numbers(ck, R.get(ck)), dev_numbers(ck)]
            for ck, seed, crit in CRIT]
    return md_table(["ID", "Seed", "Criterion", "Formal run", "Final suite", "Key numbers", "Development dry-run (stated in D120 before the formal run)"], rows)


def ck_details():
    R = recs()
    rows = []
    r = R.get("CK-3b")
    if r:
        rows.append(["CK-3b", f"lifelines {', '.join(f'{v:.3f}' for v in r['lifelines'])}; gradient norm at the end "
                     f"{r['grad_norm']:.1e}; censoring {r['censoring']:.3f}"])
    r = R.get("CK-5b")
    if r:
        rows.append(["CK-5b", f"tuning selected lr {r['lr']}, wd {r['wd']}; members' identifiable RISE φ₁ "
                     + ", ".join(f"{m['phi1']['rise_id']:.3f}" for m in r["member_shapes"]) + "; φ₂ "
                     + ", ".join(f"{m['phi2']['rise_id']:.3f}" for m in r["member_shapes"])])
    r = R.get("CK-6b")
    if r:
        rows.append(["CK-6b", f"selected λ_x {r['lam_x']:g} (smallest in the grid), λ_t {r['lam_t']:g}"])
    r = R.get("CK-8b")
    if r:
        ch = [g for g in r["grid"] if g.get("chosen")]
        c = ch[0] if ch else None
        vmin = min(g["val_loss"] for g in r["grid"])
        rows.append(["CK-8b", (f"chosen λ {c['lam']:g}, θ_rel {c['theta']:g} (validation NLPL {c['val_loss']:.4f}; grid "
                               f"minimum {vmin:.4f}); " if c else "") + "normalised channel scores of the selected model "
                     + ", ".join(f"{v:.3f}" for v in r["channel_scores_selected"])])
    r = R.get("CK-11b")
    if r:
        rows.append(["CK-11b", f"σ̂ {r['sigma']:.3f}"])
    return md_table(["ID", "Details from the records"], rows)


def ident_table():
    d = json.loads((REP / "round12/diag_identifiability_v2.json").read_text())
    rows = []
    for dgp in ("v1", "v2"):
        for ch in ("channel1", "channel2", "channel3"):
            c = d[dgp][ch]
            rows.append([dgp, ch.replace("channel", "channel "), f"{c['sd_int_X']:.2g}", f"{c['sd_int_tX']:.3f}",
                         f"{c['homogeneous_10']['rank_1e3']} of 10",
                         f"{c['tensor_10x10']['rank_1e2']} / {c['tensor_10x10']['rank_1e3']} / {c['tensor_10x10']['rank_1e4']}",
                         str(c["n_unidentified_n400"])])
    return md_table(["DGP", "Channel", "SD ∫X dt", "SD ∫t X dt", "Homogeneous design rank (1e-3)",
                     "Tensor design rank (1e-2 / 1e-3 / 1e-4, of 100)", "Unidentified directions (n 400)"], rows)


def floor_table():
    d = json.loads((REP / "round12/explore_afcm_floor.json").read_text())
    rows = [["10 cubic B-splines on the training range (AFCM's x basis)", f"{d['floor_afcm_basis_10']:.4f}"],
            ["15 on the training range", f"{d['floor_basis_15']:.4f}"],
            ["20 on the training range", f"{d['floor_basis_20']:.4f}"],
            ["10 on the 1st–99th percentile range of the test values", f"{d['floor_10_on_p1_p99']:.4f}"]]
    return md_table(["x basis", "Best possible RISE of φ₁ (approximation floor)"], rows)


def suite_table():
    rows = []
    for run in ("final_full_suite", "final_full_suite_round2_v3", "final_full_suite_round3", "final_full_suite_round4_rerun",
                "final_full_suite_round5", "final_full_suite_round6", "final_full_suite_round6_rerun",
                "final_full_suite_round7", "final_full_suite_round8", "final_full_suite_round9", "final_full_suite_round10",
                "final_full_suite_round11", FINAL12):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


def decisions_r12():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D(12[0-9]) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


TABLES = {"suite_table": suite_table, "decisions_r12": decisions_r12, "test_runs": R2.test_run_history,
          "ck_table": ck_table, "ck_details": ck_details, "ident_table": ident_table, "floor_table": floor_table}


def values():
    out = {"git_head": R2.values()["git_head"]}
    g = R1.TA[R1.TA.run_id == FINAL12]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def shift_part(m):
    """'Part B' -> 'Part C', 'Parts B–K' -> 'Parts C–L', 'Parts J and K' -> 'Parts K and L' (one pass; Part A kept)."""
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJK" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def main():
    R1.use_snapshot(FINAL12)
    vals = values()
    tpl = (REP / "report_template_round12.md").read_text()
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    R11.main()
    prev = (REP / "final_research_prototype_report.md").read_text()
    cut = prev.index("\n# Parts B–K: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Round-11 summary", "## R11-0. Round-11 summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R11-0.\1 ", own, flags=re.M)
    own = re.sub(r"§0\.(\d)", r"§R11-0.\1", own)
    prev = re.sub(r"\b(Parts? )([B-K])(–| and )?([B-K])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("Part A is round 11", "Part B is round 11", 1)
    full = out + "\n\n# Part B: round-11 report (2026-10-20), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round12.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
