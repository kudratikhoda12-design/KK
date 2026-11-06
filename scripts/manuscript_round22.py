"""Round 22 (D156-D159): the cumulative manuscript updated to integrate Round 22 (per-channel REML AFCM and CK-6d, the
untouched joineRML cohorts renal and heart.valve with C-MAPSS FD003 Option B, and the three-law NP-KAN-AFT simulation).
It is the Supplementary Material of the journal main manuscript (docs/manuscript_journal_main.md).

docs/manuscript_round22_template.md -> docs/manuscript_round22.md (+ reports/manuscript_values_round22.json).
scripts/manuscript_round21.py (and through it the Round-20 .. Round-16 builders and the Round-14 resolver) is loaded as a
separate module instance, so the earlier manuscripts, their ledgers and their verifications are untouched; this module
adds the Round-22 tables and points the reference list at docs/references_round22.md. Every number is read, without
computation, from saved records.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("manuscript_r22_m21", ROOT / "scripts/manuscript_round21.py")
M21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M21)
B = M21.B

B.TEMPLATE = ROOT / "docs/manuscript_round22_template.md"
B.OUT_MD = ROOT / "docs/manuscript_round22.md"
B.LEDGER_PATH = ROOT / "reports/manuscript_values_round22.json"
B.REFS = "docs/references_round22.md"
B.TITLE = M21.B.TITLE
V, md_table, verdict = B.V, B.md_table, M21.verdict
A = "reports/round22/analysis_summary.json"
G = "reports/round22/afcm_gate.json"
CK = "reports/round22/ck6d.json"
DEV = "reports/round22/dev/results_afcmdev_{s}.jsonl/job_name=afcmdev_{s}"
CO = ("renal", "heart_valve")
LAB = {"renal": "renal", "heart_valve": "heart valve"}
LAWS = (("bimodal", "bimodal normal mixture"), ("studentt3", "Student t, 3 df"), ("logistic05", "logistic"))
NB = M21.NB
MODELS = (("b2s_cox", "B2-S Cox"), ("b2s_waft", "B2-S Weibull AFT"), ("b2s_lnaft", "B2-S log-normal AFT"),
          ("b2s_npaft", "NP-KAN-AFT"), ("b2s_anchor_cox_cross", "Cross-confirmed, Cox"),
          ("b2s_anchor_lnaft_cross", "Cross-confirmed, log-normal AFT"),
          ("b2s_anchor_npaft_cross", "Cross-confirmed, NP-KAN-AFT"), ("landmark_cox", "Landmark Cox"),
          ("fpca_cox", "FPCA Cox"), ("lme2_cox", "Two-stage mixed-effects Cox"), ("sum_cox", "Summary Cox"),
          ("sum_rsf", "Summary RSF"))
BASE_MODELS = {"landmark_cox", "fpca_cox", "lme2_cox", "sum_cox", "sum_rsf"}


def t_r22_gate():
    rows = []
    for s in ("22", "914", "966"):
        r, d = f"{G}/seeds/{s}", DEV.format(s=s)
        rows.append([f"{s} (development)", f"{V(r + '/afcm_reml/phi1', '.3f')} / {V(r + '/afcm_reml/phi2', '.3f')}",
                     V(d + "/afcm_reml/fit/chosen_iter", "d"),
                     " / ".join(V(f"{d}/afcm_reml/fit/edf_channel/{c}", ".1f") for c in range(3)),
                     f"{V(r + '/afcm_reml_ens/phi1', '.3f')} / {V(r + '/afcm_reml_ens/phi2', '.3f')}",
                     verdict(f"{r}/afcm_reml/meets").replace("not supported", "no").replace("supported", "yes")])
    rows.append(["24 (held out, CK-6d)", f"{V(CK + '/shapes/phi1/rise_id', '.3f')} / {V(CK + '/shapes/phi2/rise_id', '.3f')}",
                 "–", "–", "–", verdict(CK + "/passed").replace("not supported", "no").replace("supported", "yes")])
    return md_table(["Seed (identifiable RISE φ₁ / φ₂; rule φ₁ < 0.15 and φ₂ < 0.10)", "Per-channel REML AFCM",
                     "Fellner–Schall iteration chosen", "EDF by channel (1 / 2 / 3)", "Bagged REML AFCM (B = 3)",
                     "Per-channel REML meets the rule"], rows)


def t_r22_ck6d_lams():
    return md_table(["Seed 24, per-channel REML AFCM", "Channel 1", "Channel 2", "Channel 3 (null)"],
                    [["λ_x (trajectory direction)"] + [V(f"{CK}/lam_x/{c}", ".2e") for c in range(3)],
                     ["λ_t (time direction)"] + [V(f"{CK}/lam_t/{c}", ".2e") for c in range(3)]])


def t_r22_hyp():
    h = f"{A}/H_R22b"
    rows = [["H-R22a", "B2-S Cox − REML AFCM in RISE (S-Add, fresh seeds), run only if CK-6d passes",
             "CK-6d φ₁ " + V(CK + "/shapes/phi1/rise_id", ".3f"), "not run"],
            ["H-R22b (1)", "every external-cohort and FD003 fit finite", "", verdict(f"{h}/part1_all_finite")],
            ["H-R22b (2)", "renal and heart valve: anchored − unanchored conditional IBS ≤ +0.005, exact fallback (Cox, "
             "log-normal AFT, NP-KAN-AFT)", "", verdict(f"{h}/part2_ok")],
            ["H-R22b (3)", "distillation finite and change in conditional IBS ≤ +0.002 (NP-KAN-AFT and Cox)", "",
             verdict(f"{h}/part3_ok")],
            ["H-R22b", "all three parts", "", verdict(f"{h}/supported")]]
    c = f"{A}/H_R22c"
    rows += [["H-R22c (1)", "bimodal errors: NP-KAN-AFT − each of Cox, Weibull and log-normal AFT, upper 95% bound < 0", "",
              verdict(f"{c}/part1")],
             ["H-R22c (2)", "t₃ and logistic errors: upper bound < 0 against Cox and Weibull, < +0.005 against log-normal", "",
              verdict(f"{c}/part2")],
             ["H-R22c", "both parts", "", verdict(f"{c}/supported")]]
    return md_table(["Hypothesis", "Criterion", "Result", "Verdict"], rows)


def t_r22_cohorts():
    rows = []
    for ds in CO:
        b = f"{A}/cohorts/{ds}/cohort"
        rows.append([LAB[ds], V(b + "/n_all", "d"), V(b + "/t_L", ".1f"), V(b + "/n", "d"), V(b + "/events", "d"),
                     V(b + "/obs_per_subject_median", ".1f"), " / ".join(V(f"{b}/unseen_per_channel/{c}", "d") for c in range(2))])
    return md_table(["Cohort", "Patients", "Landmark (years)", "At risk at the landmark", "Events after the landmark",
                     "Median visits before the landmark", "Patients with no visit, channel 1 / 2"], rows)


def t_r22_means():
    rows = []
    for ds in CO:
        for m, lab in MODELS:
            b = f"{A}/cohorts/{ds}/means/{m}"
            rows.append([LAB[ds], lab, V(b + "/ibs_cond", ".4f"), V(b + "/harrell_c", ".3f"), V(b + "/uno_c", ".3f"),
                         V(f"{A}/cohorts/{ds}/dcal_pass/{m}", "d")])
    return md_table(["Cohort", "Model", "Conditional IPCW IBS", "Harrell C", "Uno C", "D-cal p ≥ 0.05 (of 5)"], rows)


def nbrow(ds, key, label):
    b = f"{A}/cohorts/{ds}/contrasts/ibs_cond {key}"
    return [f"{LAB[ds]}: {label}", V(b + "/nb/n", "d"), V(b + "/nb/mean_diff", "+.4f"), V(b + "/nb/ci95", "+.4f"),
            V(b + "/paired/ci95", "+.4f"), f"{V(b + '/nb/n_neg', 'd')} / {V(b + '/nb/n_pos', 'd')}"]


def t_r22_contrasts():
    rows = []
    for ds in CO:
        rows += [nbrow(ds, "b2s_npaft - b2s_cox", "NP-KAN-AFT − B2-S Cox"),
                 nbrow(ds, "b2s_npaft - b2s_lnaft", "NP-KAN-AFT − log-normal AFT"),
                 nbrow(ds, "b2s_npaft - landmark_cox", "NP-KAN-AFT − landmark Cox"),
                 nbrow(ds, "b2s_cox - landmark_cox", "B2-S Cox − landmark Cox"),
                 nbrow(ds, "b2s_cox - lme2_cox", "B2-S Cox − two-stage mixed-effects Cox"),
                 nbrow(ds, "b2s_cox - fpca_cox", "B2-S Cox − FPCA Cox")]
    return md_table(NB, rows)


def t_r22_cross():
    rows = []
    for ds in CO:
        for f, lab in (("cox", "Cox"), ("lnaft", "log-normal AFT"), ("npaft", "NP-KAN-AFT")):
            p = f"{A}/H_R22b/part2/{ds}/{f}"
            rows.append([LAB[ds], lab, V(f"{A}/cohorts/{ds}/gates/{f}/n_confirmed", "d"), V(p + "/mean_shift", "+.4f"),
                         verdict(p + "/fallback_exact").replace("not supported", "no").replace("supported", "yes"),
                         verdict(p + "/ok").replace("not supported", "no").replace("supported", "yes")])
    return md_table(["Cohort", "Family", "Splits confirming (of 5)", "Anchored − unanchored, mean conditional IBS",
                     "Exact fallback", "Within +0.005 with exact fallback"], rows)


def t_r22_sym():
    rows = []
    for ds in CO:
        for f, lab in (("npaft", "NP-KAN-AFT"), ("cox", "B2-S Cox")):
            b = f"{A}/cohorts/{ds}/symbolic/{f}"
            rows.append([LAB[ds], lab, V(b + "/replacement_rate_mean", ".2f"), V(b + "/n_rollback", "d"),
                         V(b + "/ibs_cond_before_mean", ".4f"), V(b + "/ibs_cond_after_mean", ".4f"),
                         V(f"{A}/H_R22b/part3/{ds}/{f}/mean_change", "+.4f")])
    return md_table(["Cohort", "Model (seed-0 member)", "Edges substituted (mean share)", "Rollbacks (of 5)",
                     "Conditional IBS, spline", "After distillation", "Mean change"], rows)


def t_r22_fd003():
    rows = []
    for m, lab in (("b2s_cox", "B2-S Cox"), ("b2s_waft", "B2-S Weibull AFT"), ("b2s_lnaft", "B2-S log-normal AFT"),
                   ("b2s_npaft", "NP-KAN-AFT"), ("b2s_anchor_npaft_cross", "Cross-confirmed, NP-KAN-AFT"),
                   ("landmark_cox", "Landmark Cox"), ("sum_aft", "Summary Weibull AFT"), ("fpca_aft", "FPCA Weibull AFT")):
        b = f"{A}/fd003/means/{m}"
        rows.append([lab, V(b + "/ibs", ".4f"), V(b + "/harrell_c", ".3f"), V(b + "/uno_c", ".3f"), V(f"{A}/fd003/dcal_pass/{m}", "d")])
    return md_table(["Model", "IBS", "Harrell C", "Uno C", "D-cal p ≥ 0.05 (of 5)"], rows)


def t_r22_fd003_con():
    rows = []
    for a, la in (("b2s_npaft", "NP-KAN-AFT"), ("b2s_lnaft", "log-normal AFT"), ("b2s_waft", "Weibull AFT")):
        for b, lb in (("b2s_cox", "B2-S Cox"), ("landmark_cox", "landmark Cox")):
            p = f"{A}/fd003/contrasts/ibs {a} - {b}"
            rows.append([f"{la} − {lb}", V(p + "/nb/n", "d"), V(p + "/nb/mean_diff", "+.4f"), V(p + "/nb/ci95", "+.4f"),
                         V(p + "/paired/ci95", "+.4f"), f"{V(p + '/nb/n_neg', 'd')} / {V(p + '/nb/n_pos', 'd')}"])
    return md_table(["Contrast (a − b), IBS, FD003"] + NB[1:], rows)


def t_r22_sims():
    rows = []
    for law, lab in LAWS:
        for m, lm in (("b2s_npaft", "NP-KAN-AFT"), ("b2s_lnaft", "log-normal AFT"), ("b2s_waft", "Weibull AFT"), ("b2s_cox", "Cox")):
            b = f"{A}/sims/{law}/means/{m}"
            rows.append([lab, lm, V(b + "/curve_error", ".4f"), V(b + "/ibs", ".4f"), V(b + "/harrell_c", ".3f"),
                         V(f"{A}/sims/{law}/dcal_pass/{m}", "d")])
    return md_table(["Error law", "B2-S model", "Survival-curve error", "IBS", "Harrell C", "D-cal p ≥ 0.05 (of 20)"], rows)


def t_r22_sims_con():
    rows = []
    for law, lab in LAWS:
        for m, lm in (("b2s_cox", "Cox"), ("b2s_waft", "Weibull AFT"), ("b2s_lnaft", "log-normal AFT")):
            p = f"{A}/sims/{law}/contrasts/b2s_npaft - {m}"
            rows.append([lab, f"NP-KAN-AFT − {lm}", V(p + "/n", "d"), V(p + "/mean_diff", "+.4f"), V(p + "/ci95", "+.4f"),
                         f"{V(p + '/n_neg', 'd')} / {V(p + '/n_pos', 'd')}"])
    return md_table(["Error law", "Contrast, survival-curve error", "Replicates", "Mean difference", "95% CI",
                     "a < b / a > b"], rows)


def t_tests():
    b = "reports/report_values_round21.json"
    return md_table(["Final test suite before this revision (`final_full_suite_round21`)", "Count"],
                    [["Tests", V(b + "/n_tests", "s")], ["Passed", V(b + "/passed", "s")],
                     ["Failed (visible, recorded: CK-6c and the 4f gate's original R1)", V(b + "/failed", "s")],
                     ["Expected failures (strict xfail; historical records)", V(b + "/xfailed", "s")],
                     ["Skipped", V(b + "/skipped", "s")]])


B.TABLES.update(r22_gate=t_r22_gate, r22_ck6d_lams=t_r22_ck6d_lams, r22_hyp=t_r22_hyp, r22_cohorts=t_r22_cohorts,
                r22_means=t_r22_means, r22_contrasts=t_r22_contrasts, r22_cross=t_r22_cross, r22_sym=t_r22_sym,
                r22_fd003=t_r22_fd003, r22_fd003_con=t_r22_fd003_con, r22_sims=t_r22_sims, r22_sims_con=t_r22_sims_con,
                tests=t_tests)
render, TITLE, LEDGER, CITED, resolve = B.render, B.TITLE, B.LEDGER, B.CITED, B.resolve


def main():
    return B.main()


if __name__ == "__main__":
    sys.exit(main())
