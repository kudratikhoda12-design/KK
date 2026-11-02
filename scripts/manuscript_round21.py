"""Round 21 (D153-D155): the standalone manuscript updated to integrate Round 21 (the Round-20 NP-KAN-AFT, the
log-normal KAN-AFT, the two-half cross-confirmed anchor and fixed-mu symbolic distillation on the real benchmarks and
C-MAPSS FD001 Option B).

docs/manuscript_round21_template.md -> docs/manuscript_round21.md (+ reports/manuscript_values_round21.json).
scripts/manuscript_round20.py (and through it the Round-19 .. Round-16 builders and the Round-14 resolver) is loaded as a
separate module instance, so the earlier manuscripts, their ledgers and their verifications are untouched; this module
adds the Round-21 tables. Every number is read, without computation, from saved records.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("manuscript_r21_m20", ROOT / "scripts/manuscript_round20.py")
M20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M20)
B = M20.B

B.TEMPLATE = ROOT / "docs/manuscript_round21_template.md"
B.OUT_MD = ROOT / "docs/manuscript_round21.md"
B.LEDGER_PATH = ROOT / "reports/manuscript_values_round21.json"
B.TITLE = M20.B.TITLE
V, md_table, prow4, PH4, verdict = B.V, B.md_table, M20.prow4, M20.PH4, M20.verdict
A = "reports/round21/analysis_summary.json"
DS = ("pbc_E2", "pbc_E1", "prothro", "aids")
LAB = {"pbc_E2": "PBC E2", "pbc_E1": "PBC E1", "prothro": "prothro", "aids": "aids"}
MODELS = (("b2s_cox", "B2-S Cox"), ("b2s_waft", "B2-S Weibull AFT"), ("b2s_lnaft", "B2-S log-normal AFT"),
          ("b2s_npaft", "NP-KAN-AFT"), ("b2s_anchor_cox_cross", "Cross-confirmed, Cox"),
          ("b2s_anchor_lnaft_cross", "Cross-confirmed, log-normal AFT"), ("b2s_anchor_npaft_cross", "Cross-confirmed, NP-KAN-AFT"),
          ("landmark_cox", "Landmark Cox (Round 13)"), ("r15_b2s_anchor_cox", "Anchored Cox (Round 15)"))
NB = ["Contrast (a − b), conditional IPCW IBS", "Splits", "Mean difference", "Nadeau–Bengio 95% CI", "Plain 95% CI", "a < b / a > b"]


def t_r21_means():
    rows = []
    for ds in DS:
        for m, lab in MODELS:
            b = f"{A}/clinical/{ds}/means/{m}"
            dp = V(f"{A}/clinical/{ds}/dcal_pass/{m}", "d") if not m.startswith(("landmark", "r15")) else "–"
            rows.append([LAB[ds], lab, V(b + "/ibs_cond", ".4f"), V(b + "/harrell_c", ".3f"), V(b + "/uno_c", ".3f"), dp])
    return md_table(["Data set", "Model", "Conditional IPCW IBS", "Harrell C", "Uno C", "D-cal p ≥ 0.05 (of 5)"], rows)


def nbrow(ds, key, label):
    b = f"{A}/clinical/{ds}/contrasts/ibs_cond {key}"
    return [f"{LAB[ds]}: {label}", V(b + "/nb/n", "d"), V(b + "/nb/mean_diff", "+.4f"), V(b + "/nb/ci95", "+.4f"),
            V(b + "/paired/ci95", "+.4f"), f"{V(b + '/nb/n_neg', 'd')} / {V(b + '/nb/n_pos', 'd')}"]


def t_r21_npaft():
    rows = []
    for ds in DS:
        rows += [nbrow(ds, "b2s_npaft - b2s_cox", "NP-KAN-AFT − B2-S Cox"),
                 nbrow(ds, "b2s_npaft - b2s_waft", "NP-KAN-AFT − Weibull AFT"),
                 nbrow(ds, "b2s_npaft - b2s_lnaft", "NP-KAN-AFT − log-normal AFT"),
                 nbrow(ds, "b2s_npaft - landmark_cox", "NP-KAN-AFT − landmark Cox")]
    return md_table(NB, rows)


def t_r21_cross():
    rows = []
    for ds in DS:
        g = f"{A}/clinical/{ds}/gates"
        rows.append([LAB[ds], V(f"{g}/cox/n_confirmed", "d"), V(f"{g}/lnaft/n_confirmed", "d"), V(f"{g}/npaft/n_confirmed", "d"),
                     V(f"{A}/hypotheses/H_R21b/{ds}/cox_mean_shift", "+.4f"), V(f"{A}/hypotheses/H_R21b/{ds}/npaft_mean_shift", "+.4f"),
                     verdict(f"{A}/hypotheses/H_R21b/{ds}/fallback_exact").replace("not supported", "no").replace("supported", "yes")])
    out = md_table(["Data set", "Confirmed of 5: Cox", "log-normal AFT", "NP-KAN-AFT", "Cox: anchored − B2-S, mean",
                    "NP-KAN-AFT: anchored − B2-S, mean", "Exact fallback on every non-confirmed split"], rows)
    rows = []
    for ds in DS:
        rows += [nbrow(ds, "b2s_anchor_cox_cross - r15_b2s_anchor_cox", "cross-confirmed Cox − Round-15 anchored Cox"),
                 nbrow(ds, "b2s_anchor_cox_cross - landmark_cox", "cross-confirmed Cox − landmark Cox")]
    return out + "\n\n" + md_table(NB, rows)


def t_r21_hyp():
    h = f"{A}/hypotheses"
    return md_table(["Hypothesis", "Criterion", "Result", "Verdict"], [
        ["H-R21a (1)", "all clinical NP-KAN-AFT fits finite", V(f"{h}/H_R21a/n_npaft_fits", "d") + " fits",
         verdict(f"{h}/H_R21a/part1_all_finite")],
        ["H-R21a (2)", "PBC E2 mean NP − Cox ≤ +0.005 and NP − Weibull ≤ 0",
         V(f"{h}/H_R21a/part2_mean_vs_cox", "+.4f") + "; " + V(f"{h}/H_R21a/part2_mean_vs_waft", "+.4f"), verdict(f"{h}/H_R21a/part2")],
        ["H-R21a (3)", "PBC E2 D-cal p ≥ 0.05 on ≥ 4 of 5 splits", V(f"{h}/H_R21a/part3_dcal_pass", "d") + " of 5",
         verdict(f"{h}/H_R21a/part3")],
        ["H-R21a", "all three parts", "", verdict(f"{h}/H_R21a/supported")],
        ["H-R21b", "every data set: anchored − B2-S mean ≤ +0.005 (Cox and NP) and exact fallback", "",
         verdict(f"{h}/H_R21b/supported")],
        ["H-R21c", "C-MAPSS: better D-calibration than B2-S Cox with IBS ≤ 0.055 and Harrell C ≥ 0.88", "",
         verdict(f"{h}/H_R21c/supported")],
        ["H-R21c, relative (secondary)", "the same with IBS ≤ Cox + 0.005 and Harrell C ≥ Cox − 0.01", "",
         verdict(f"{h}/H_R21c/secondary_relative_supported")]])


def t_r21_cmapss():
    rows = []
    for m, lab in (("b2s_cox", "B2-S Cox"), ("b2s_waft", "B2-S Weibull AFT"), ("b2s_lnaft", "B2-S log-normal AFT"),
                   ("b2s_npaft", "NP-KAN-AFT"), ("b2s_anchor_npaft_cross", "Cross-confirmed, NP-KAN-AFT")):
        b = f"{A}/cmapss/means/{m}"
        rows.append([lab, V(b + "/ibs", ".4f"), V(b + "/harrell_c", ".3f"), V(b + "/uno_c", ".3f"), V(b + "/dcal_chi2", ".1f"),
                     V(f"{A}/cmapss/dcal_pass/{m}", "d")])
    return md_table(["Model", "IBS", "Harrell C", "Uno C", "D-calibration χ²", "D-cal p ≥ 0.05 (of 5)"], rows)


def t_r21_sym():
    rows = []
    for ds in DS:
        for f, lab in (("npaft", "NP-KAN-AFT"), ("cox", "B2-S Cox")):
            b = f"{A}/clinical/{ds}/symbolic/{f}"
            rows.append([LAB[ds], lab, V(b + "/replacement_rate_mean", ".2f"), V(b + "/n_rollback", "d"),
                         V(b + "/ibs_cond_before_mean", ".4f"), V(b + "/ibs_cond_after_mean", ".4f"),
                         V(b + "/after_minus_before/nb/ci95", "+.4f")])
    return md_table(["Data set", "Model (seed-0 member)", "Edges substituted (mean share)", "Rollbacks (of 5)",
                     "Conditional IBS, spline", "After distillation", "Difference, Nadeau–Bengio 95% CI"], rows)


def t_tests():
    b = "reports/report_values_round20.json"
    return md_table(["Final test suite before this revision (`final_full_suite_round20`)", "Count"],
                    [["Tests", V(b + "/n_tests", "s")], ["Passed", V(b + "/passed", "s")],
                     ["Failed (visible, recorded: CK-6c and the 4f gate's original R1)", V(b + "/failed", "s")],
                     ["Expected failures (strict xfail; historical records)", V(b + "/xfailed", "s")],
                     ["Skipped", V(b + "/skipped", "s")]])


B.TABLES.update(r21_means=t_r21_means, r21_npaft=t_r21_npaft, r21_cross=t_r21_cross, r21_hyp=t_r21_hyp,
                r21_cmapss=t_r21_cmapss, r21_sym=t_r21_sym, tests=t_tests)
render, TITLE, LEDGER, CITED, resolve = B.render, B.TITLE, B.LEDGER, B.CITED, B.resolve


def main():
    return B.main()


if __name__ == "__main__":
    sys.exit(main())
