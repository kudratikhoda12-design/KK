"""Round 23 (D160-D162): the cumulative manuscript updated to integrate Round 23 (untouched C-MAPSS FD002 and FD004,
renal history thinning, and a landmark / NP-KAN-AFT blend on six clinical cohorts with 20 new partitions). It is the
Supplementary Material of the Round-23 journal main manuscript (docs/manuscript_journal_main_round23.md).

docs/manuscript_round23_template.md -> docs/manuscript_round23.md (+ reports/manuscript_values_round23.json).
scripts/manuscript_round22.py (and through it the Round-21 .. Round-16 builders and the Round-14 resolver) is loaded as a
separate module instance, so the earlier manuscripts, their ledgers and their verifications are untouched; this module
adds the Round-23 tables. The reference list is unchanged (docs/references_round22.md). Every number is read, without
computation, from saved records.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("manuscript_r23_m22", ROOT / "scripts/manuscript_round22.py")
M22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M22)
B = M22.B

B.TEMPLATE = ROOT / "docs/manuscript_round23_template.md"
B.OUT_MD = ROOT / "docs/manuscript_round23.md"
B.LEDGER_PATH = ROOT / "reports/manuscript_values_round23.json"
B.TITLE = M22.B.TITLE
V, md_table, verdict = B.V, B.md_table, M22.verdict
A = "reports/round23/analysis_summary.json"
CM = ("FD002", "FD004")
CLIN = (("pbc_E2", "PBC E2"), ("pbc_E1", "PBC E1"), ("prothro", "prothro"), ("aids", "aids"), ("renal", "renal"),
        ("heart_valve", "heart valve"))
CM_MODELS = (("b2s_npaft", "NP-KAN-AFT"), ("landmark_cox", "Landmark Cox (version chosen on validation)"),
             ("landmark_cox_raw", "Landmark Cox, last raw values"), ("landmark_cox_smooth", "Landmark Cox, last smoothed values"),
             ("stack", "Blend (landmark Cox and NP-KAN-AFT)"), ("sum_aft", "Summary Weibull AFT"),
             ("fpca_aft", "FPCA Weibull AFT"))
CM_CON = (("b2s_npaft - landmark_cox", "NP-KAN-AFT − landmark Cox"), ("b2s_npaft - sum_aft", "NP-KAN-AFT − summary Weibull AFT"),
          ("b2s_npaft - fpca_aft", "NP-KAN-AFT − FPCA Weibull AFT"), ("stack - landmark_cox", "Blend − landmark Cox"),
          ("b2s_npaft - landmark_cox_raw", "NP-KAN-AFT − landmark Cox, raw values"))
NBH = ["Contrast (a − b)", "Splits", "Mean difference", "Nadeau–Bengio 95% CI", "a < b / a > b"]


def yn(src):
    return verdict(src).replace("not supported", "no").replace("supported", "yes")


def nbcells(p):
    return [V(p + "/n", "d"), V(p + "/mean_diff", "+.4f"), V(p + "/ci95", "+.4f"), f"{V(p + '/n_neg', 'd')} / {V(p + '/n_pos', 'd')}"]


def t_r23_hyp():
    a, c = f"{A}/H_R23a", f"{A}/H_R23c"
    rows = [[f"H-R23a ({fd})", "NP-KAN-AFT − landmark Cox, test IBS: mean < 0 and Nadeau–Bengio upper bound < 0 (10 splits)",
             f"{V(a + '/per_dataset/' + fd + '/mean_diff', '+.4f')}, upper {V(a + '/per_dataset/' + fd + '/upper', '+.4f')}",
             yn(f"{a}/per_dataset/{fd}/met")] for fd in CM]
    rows.append(["H-R23a", "both data sets", "", verdict(a + "/supported")])
    rows.append(["H-R23b", "renal: NP-KAN-AFT on the full history − on three visits, conditional IBS: upper bound < 0 "
                 "(20 partitions)", f"{V(A + '/H_R23b/mean_diff', '+.4f')}, upper {V(A + '/H_R23b/ci95/1', '+.4f')}",
                 verdict(A + "/H_R23b/supported")])
    rows.append(["H-R23c (i)", "blend − landmark Cox, conditional IBS: upper bound ≤ +0.005 on every clinical cohort",
                 "upper bounds " + ", ".join(V(f"{c}/part_i/per_cohort/{ds}/upper", "+.4f") for ds, _ in CLIN),
                 verdict(c + "/part_i/supported")])
    rows.append(["H-R23c (ii)", "renal: blend − landmark Cox upper bound < 0",
                 f"{V(c + '/part_ii/mean_diff', '+.4f')}, upper {V(c + '/part_ii/ci95/1', '+.4f')}",
                 verdict(c + "/part_ii/supported")])
    return md_table(["Hypothesis", "Criterion (D160)", "Result", "Verdict"], rows)


def t_r23_cm():
    rows = []
    for fd in CM:
        for m, lab in CM_MODELS:
            b = f"{A}/cm/{fd}/means"
            rows.append([fd, lab, V(f"{b}/ibs/{m}", ".4f"), V(f"{b}/harrell_c/{m}", ".3f"), V(f"{b}/uno_c/{m}", ".3f"),
                         V(f"{A}/cm/{fd}/dcal_pass/{m}", "d")])
    return md_table(["Data set", "Model", "IBS", "Harrell C", "Uno C", "D-cal p ≥ 0.05 (of 10)"], rows)


def t_r23_cm_con():
    rows = []
    for fd in CM:
        rows += [[f"{fd}: {lab}"] + nbcells(f"{A}/cm/{fd}/contrasts/{k}") for k, lab in CM_CON]
        rows.append([f"{fd}: NP-KAN-AFT − landmark Cox, Harrell C"] + nbcells(f"{A}/cm/{fd}/c_contrasts/b2s_npaft - landmark_cox"))
    return md_table(NBH, rows)


def t_r23_clin():
    rows = []
    for ds, lab in CLIN:
        b = f"{A}/clinical/{ds}"
        rows.append([lab, V(b + "/visits_median", ".0f"), V(b + "/means/ibs_cond/b2s_npaft", ".4f"),
                     V(b + "/means/ibs_cond/landmark_cox", ".4f"), V(b + "/means/ibs_cond/stack", ".4f"),
                     f"{V(b + '/contrasts/stack - landmark_cox/mean_diff', '+.4f')} {V(b + '/contrasts/stack - landmark_cox/ci95', '+.4f')}",
                     f"{V(b + '/contrasts/b2s_npaft - landmark_cox/mean_diff', '+.4f')} {V(b + '/contrasts/b2s_npaft - landmark_cox/ci95', '+.4f')}"])
    return md_table(["Cohort", "Median visits", "NP-KAN-AFT", "Landmark Cox", "Blend", "Blend − landmark Cox (NB 95% CI)",
                     "NP-KAN-AFT − landmark Cox (NB 95% CI)"], rows)


def t_r23_w():
    rows = []
    for ds, lab in CLIN:
        b = f"{A}/clinical/{ds}"
        rows.append([lab] + [V(f"{b}/w_counts/{w}", "d") for w in ("0.0", "0.25", "0.5", "0.75", "1.0")] +
                    [V(b + "/means/harrell_c/b2s_npaft", ".3f"), V(b + "/means/harrell_c/landmark_cox", ".3f"),
                     V(b + "/means/harrell_c/stack", ".3f")])
    return md_table(["Cohort", "w = 0 (NP-KAN-AFT only)", "w = 0.25", "w = 0.5", "w = 0.75", "w = 1 (landmark Cox only)",
                     "Harrell C, NP-KAN-AFT", "Landmark Cox", "Blend"], rows)


def t_r23_renal():
    b = f"{A}/clinical/renal"
    rows = [["NP-KAN-AFT, full history", V(b + "/means/ibs_cond/b2s_npaft", ".4f"), V(b + "/means/harrell_c/b2s_npaft", ".3f")],
            ["NP-KAN-AFT, three visits", V(b + "/means/ibs_cond/b2s_npaft_sparse3", ".4f"),
             V(b + "/means/harrell_c/b2s_npaft_sparse3", ".3f")],
            ["Landmark Cox", V(b + "/means/ibs_cond/landmark_cox", ".4f"), V(b + "/means/harrell_c/landmark_cox", ".3f")]]
    out = md_table(["Renal, 20 partitions", "Conditional IBS", "Harrell C"], rows)
    rows = [[lab] + nbcells(f"{b}/contrasts/{k}") for k, lab in
            (("b2s_npaft - b2s_npaft_sparse3", "Full history − three visits"),
             ("b2s_npaft_sparse3 - landmark_cox", "Three visits − landmark Cox"),
             ("b2s_npaft - landmark_cox", "Full history − landmark Cox"))]
    return out + "\n\n" + md_table(NBH, rows)


def t_tests():
    b = "reports/report_values_round22.json"
    return md_table(["Final test suite before this revision (`final_full_suite_round22`)", "Count"],
                    [["Tests", V(b + "/n_tests", "s")], ["Passed", V(b + "/passed", "s")],
                     ["Failed (visible, recorded: CK-6c, the 4f gate's original R1 and CK-6d)", V(b + "/failed", "s")],
                     ["Expected failures (strict xfail; historical records)", V(b + "/xfailed", "s")],
                     ["Skipped", V(b + "/skipped", "s")]])


B.TABLES.update(r23_hyp=t_r23_hyp, r23_cm=t_r23_cm, r23_cm_con=t_r23_cm_con, r23_clin=t_r23_clin, r23_w=t_r23_w,
                r23_renal=t_r23_renal, tests=t_tests)
render, TITLE, LEDGER, CITED, resolve = B.render, B.TITLE, B.LEDGER, B.CITED, B.resolve


def main():
    return B.main()


if __name__ == "__main__":
    sys.exit(main())
