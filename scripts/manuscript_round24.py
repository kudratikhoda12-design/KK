"""Round 24 (D163-D165): the cumulative manuscript updated to integrate Round 24 (a Bayesian joint model, JMbayes2, as the
standard dynamic-prediction comparator on the six clinical cohorts and the Round-23 partitions). It is the Supplementary
Material of the Round-24 journal main manuscript (docs/manuscript_journal_main_round24.md).

docs/manuscript_round24_template.md -> docs/manuscript_round24.md (+ reports/manuscript_values_round24.json).
scripts/manuscript_round23.py (and through it the Round-22 .. Round-16 builders and the Round-14 resolver) is loaded as a
separate module instance, so the earlier manuscripts, their ledgers and their verifications are untouched; this module
adds the Round-24 tables. The reference list is unchanged (docs/references_round22.md). Every number is read, without
computation, from saved records.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("manuscript_r24_m23", ROOT / "scripts/manuscript_round23.py")
M23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M23)
M22 = M23.M22
B = M23.B

B.TEMPLATE = ROOT / "docs/manuscript_round24_template.md"
B.OUT_MD = ROOT / "docs/manuscript_round24.md"
B.LEDGER_PATH = ROOT / "reports/manuscript_values_round24.json"
B.TITLE = M23.B.TITLE
V, md_table, verdict = B.V, B.md_table, M22.verdict
A = "reports/round24/analysis_summary.json"
CLIN = M23.CLIN
NBH, nbcells, yn = M23.NBH, M23.nbcells, M23.yn


def _ci(p):
    return f"{V(p + '/mean_diff', '+.4f')} {V(p + '/ci95', '+.4f')}"


def t_r24_hyp():
    a, b = f"{A}/H_R24a", f"{A}/H_R24b"
    rows = [["H-R24a", "renal: NP-KAN-AFT − joint model, conditional IBS: mean < 0 and Nadeau–Bengio upper bound < 0",
             f"{V(a + '/mean_diff', '+.4f')}, upper {V(a + '/ci95/1', '+.4f')}", verdict(a + "/supported")]]
    rows += [[f"H-R24b ({lab})", "NP-KAN-AFT − joint model, conditional IBS: upper bound ≤ +0.005",
              f"{V(b + '/per_cohort/' + ds + '/mean_diff', '+.4f')}, upper {V(b + '/per_cohort/' + ds + '/upper', '+.4f')}",
              yn(f"{b}/per_cohort/{ds}/met")] for ds, lab in CLIN]
    rows.append(["H-R24b", "all six cohorts", "", verdict(b + "/supported")])
    return md_table(["Hypothesis", "Criterion (D163)", "Result", "Verdict"], rows)


def t_r24_ibs():
    rows = []
    for ds, lab in CLIN:
        b = f"{A}/cohorts/{ds}"
        rows.append([lab, V(b + "/contrasts/b2s_npaft - jm/n", "d"), V(b + "/means/ibs_cond/b2s_npaft", ".4f"),
                     V(b + "/means/ibs_cond/landmark_cox", ".4f"), V(b + "/means/ibs_cond/jm", ".4f"),
                     f"{V(b + '/chosen_form/lin', 'd')} / {V(b + '/chosen_form/ns2', 'd')}",
                     _ci(b + "/contrasts/b2s_npaft - jm"), _ci(b + "/contrasts/landmark_cox - jm")])
    return md_table(["Cohort", "Partitions compared", "NP-KAN-AFT", "Landmark Cox", "Joint model", "Linear / spline form chosen",
                     "NP-KAN-AFT − joint model (NB 95% CI)", "Landmark Cox − joint model (NB 95% CI)"], rows)


def t_r24_c():
    rows = []
    for ds, lab in CLIN:
        b = f"{A}/cohorts/{ds}"
        rows.append([lab, V(b + "/means/harrell_c/b2s_npaft", ".3f"), V(b + "/means/harrell_c/landmark_cox", ".3f"),
                     V(b + "/means/harrell_c/jm", ".3f"),
                     " / ".join(V(f"{b}/dcal_pass/{m}", "d") for m in ("b2s_npaft", "landmark_cox", "jm")),
                     V(b + "/rhat_max_median", ".3f"), V(b + "/n_rhat_above_1_1", "d"), V(b + "/jm_failed_partitions", "d")])
    return md_table(["Cohort", "Harrell C, NP-KAN-AFT", "Landmark Cox", "Joint model",
                     "D-calibration passed: NP-KAN-AFT / landmark Cox / joint model", "Joint model: median max Rhat",
                     "Partitions with Rhat above 1.1", "Joint-model fits failed"], rows)


def t_tests():
    b = "reports/report_values_round23.json"
    return md_table(["Final test suite before this revision (`final_full_suite_round23`)", "Count"],
                    [["Tests", V(b + "/n_tests", "s")], ["Passed", V(b + "/passed", "s")],
                     ["Failed (visible, recorded: CK-6c, the 4f gate's original R1 and CK-6d)", V(b + "/failed", "s")],
                     ["Expected failures (strict xfail; historical records)", V(b + "/xfailed", "s")],
                     ["Skipped", V(b + "/skipped", "s")]])


B.TABLES.update(r24_hyp=t_r24_hyp, r24_ibs=t_r24_ibs, r24_c=t_r24_c, tests=t_tests)
render, TITLE, LEDGER, CITED, resolve = B.render, B.TITLE, B.LEDGER, B.CITED, B.resolve
CM = M23.CM


def main():
    return B.main()


if __name__ == "__main__":
    sys.exit(main())
