"""Assemble the final report, revision 13: Part A = owner decisions closing L39 (D123), the AFCM diagnosis on development
seed 914, the repaired tests CK-6c / CK-7b and the Round-12 gate (reports/report_template_round12b.md); Part B =
revision 12's Part A (L38); Parts C–M = rounds 11..1 as historical records."""
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as R1  # noqa: E402
import build_report_round2 as R2  # noqa: E402
import build_report_round11 as R11  # noqa: E402
import build_report_round12 as R12  # noqa: E402
import round2_data as D  # noqa: E402

ROOT, REP = D.ROOT, D.REP
md_table = R1.md_table
FINAL = "final_full_suite_round12b"
ACC = "round12b_ck_acceptance"
CK_FILE = REP / "logs/round12_ck.jsonl"


def recs():
    out = {}
    for l in CK_FILE.read_text().splitlines():
        r = json.loads(l)
        if r["run_id"] in (ACC, FINAL) and r["ck"] in ("CK-6c", "CK-7b"):
            out.setdefault(r["ck"], {})[r["run_id"]] = r
    return {k: v.get(FINAL, v.get(ACC)) for k, v in out.items()}


def ck_table():
    R = recs()
    rows = []
    r = R.get("CK-6c")
    if r:
        s = r["shapes"]
        rows.append(["CK-6c", "22", "repaired AFCM (D123): identifiable RISE φ₂ < 0.10 and φ₁ < 0.15",
                     R11.outcome("CK-6c", ACC), R11.outcome("CK-6c", FINAL),
                     f"identifiable RISE φ₁ {s['phi1']['rise_id']:.3f} (< 0.10: {'yes' if r['phi1_below_010'] else 'no'}), "
                     f"φ₂ {s['phi2']['rise_id']:.3f} (< 0.10: {'yes' if r['phi2_below_010'] else 'no'}); raw "
                     f"{s['phi1']['rise_raw']:.3f}, {s['phi2']['rise_raw']:.3f}; λ_x {r['lam_x']:g}, λ_t {r['lam_t']:g}"])
    r = R.get("CK-7b")
    if r:
        rows.append(["CK-7b", "23", "PACE under DGP v2: \\|cos\\| > 0.90 and ISE below smoothing, every channel",
                     R11.outcome("CK-7b", ACC), R11.outcome("CK-7b", FINAL),
                     "\\|cos\\| " + ", ".join(f"{v:.3f}" for v in r["abs_cos"]) + "; ISE PACE "
                     + ", ".join(f"{v:.3f}" for v in r["ise_pace"]) + " vs smoothing "
                     + ", ".join(f"{v:.3f}" for v in r["ise_smooth"])])
    for ck in ("CK-3b", "CK-5b", "CK-8b", "CK-11b", "CK-6b"):
        rows.append([ck, "–", "earlier repaired test (D121)", "–", R11.outcome(ck, FINAL), "see Part B"])
    return md_table(["ID", "Seed", "Criterion", "Formal run", "Final suite", "Key numbers"], rows)


def afcm_diag_table():
    res = {}
    for f in sorted(glob.glob(str(REP / "round12/diag_afcm_fix_seed914_*.json"))):
        res.update(json.loads(Path(f).read_text()))
    lab = {"V0": "Round-12a AFCM (CK-6b settings)", "V1": "(a) λ_x grid widened to 1e-7..1e1",
           "V2": "(a) + penalties normalised by basis size", "V3": "(b) V2 + first-order t differences (adopted)",
           "V4": "(c) V3 + per-channel λ", "V5": "(c) V3 + quantile input", "V6": "(c) V4 + quantile input"}
    rows = []
    for k in sorted(res):
        v, s = res[k], res[k]["shapes"]
        rows.append([k.split("_")[0], lab[k.split("_")[0]], f"{s['phi1']['rise_id']:.3f}", f"{s['phi2']['rise_id']:.3f}",
                     f"{s['phi1']['rise_raw']:.3f}", f"{s['phi2']['rise_raw']:.3f}",
                     f"{v['lam_x']}", f"{v['lam_t']}", f"{v['val_nll']:.4f}", f"{v['seconds']:.0f}"])
    return md_table(["Variant", "Specification", "φ₁ identifiable RISE", "φ₂ identifiable RISE", "φ₁ raw", "φ₂ raw",
                     "λ_x", "λ_t", "Validation NLPL", "Seconds"], rows)


def lamt_table():
    r = recs().get("CK-6c")
    if not r:
        return "–"
    g = r["grid"]
    best = min(x["val_nll"] for x in g)
    rows = []
    for lt in sorted({x["lam_t"] for x in g}):
        b = min((x for x in g if x["lam_t"] == lt), key=lambda x: x["val_nll"])
        rows.append([f"{lt:g}", f"{b['lam_x']:g}", f"{b['val_nll']:.4f}", f"{b['val_nll'] - best:+.4f}"])
    return md_table(["λ_t", "Best λ_x at this λ_t", "Validation NLPL", "Difference from the selected point"], rows)


def decisions():
    rows = [l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if re.match(r"\| D(12[3-9]) \|", l)]
    return "| ID | Decision | Status | Content | Rationale |\n|---|---|---|---|---|\n" + "\n".join(rows)


def suite_table():
    rows = []
    for run in ("final_full_suite_round9", "final_full_suite_round10", "final_full_suite_round11",
                "final_full_suite_round12", FINAL):
        g = R1.TA[R1.TA.run_id == run]
        if len(g):
            c = [int((g.outcome == o).sum()) for o in ("passed", "failed", "skipped", "xfailed")]
            rows.append([run] + c + [len(g), "; ".join(n.split("::")[-1] + " (" + n.split("::")[0].split("/")[-1] + ")"
                                                      for n in g[g.outcome == "failed"].nodeid) or "–"])
    return md_table(["Full-suite run", "Passed", "Failed", "Skipped", "Expected failures (xfail)", "Total", "Failed tests"], rows)


TABLES = {"ck_table": ck_table, "afcm_diag_table": afcm_diag_table, "lamt_table": lamt_table,
          "decisions": decisions, "suite_table": suite_table, "test_runs": R2.test_run_history}


def values():
    out = {"git_head": R2.values()["git_head"]}
    g = R1.TA[R1.TA.run_id == FINAL]
    g = g if len(g) else R1.TL
    out["n_tests"] = str(len(g))
    for o in ("passed", "failed", "skipped", "xfailed"):
        out[o] = str(int((g.outcome == o).sum()))
    return out


def shift_part(m):
    nxt = lambda c: chr(ord(c) + 1) if c in "BCDEFGHIJKL" else c  # noqa: E731
    return m.group(1) + nxt(m.group(2)) + (m.group(3) or "") + (nxt(m.group(4)) if m.group(4) else "")


def main():
    R1.use_snapshot(FINAL)
    vals = values()
    tpl = (REP / "report_template_round12b.md").read_text()
    out = re.sub(r"\{\{T:([A-Za-z0-9_\-]+)\}\}", lambda m: TABLES[m.group(1)](), tpl)
    out = re.sub(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", lambda m: vals[m.group(1)], out)
    out = re.sub(r"\{\{F:([A-Za-z0-9_/\-.]+)\}\}", lambda m: R2.embed(m.group(1)), out)
    R12.main()
    prev = (REP / "final_research_prototype_report.md").read_text()
    cut = prev.index("\n# Parts B–L: historical records")
    own, rest = prev[:cut], prev[cut:]
    own = re.sub(r"^## 0\. Summary", "## R12-0. Summary", own, flags=re.M)
    own = re.sub(r"^### 0\.(\d) ", r"### R12-0.\1 ", own, flags=re.M)
    own = re.sub(r"§0\.(\d)", r"§R12-0.\1", own)
    prev = re.sub(r"\b(Parts? )([B-L])(–| and )?([B-L])?\b", shift_part, own + rest)                    # one pass
    prev = prev.replace("Part A covers the owner decisions closing L38", "Part B covers the owner decisions closing L38", 1)
    full = out + "\n\n# Part B: revision 12 (L38 decisions and repaired tests, 2026-10-21), historical record\n\n" + prev
    full = "\n".join(re.sub(r"(?<!\\)\|C − C_R2(?<!\\)\|", r"\\|C − C_R2\\|", l) if l.startswith("|") else l
                     for l in full.split("\n"))
    (REP / "final_research_prototype_report.md").write_text(full)
    (REP / "report_values_round12b.json").write_text(json.dumps(vals, indent=1))
    print("report written;", len(full.splitlines()), "lines")


if __name__ == "__main__":
    main()
