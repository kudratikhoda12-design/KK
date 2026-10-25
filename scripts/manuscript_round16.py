"""Round 16 (D138/D139): the standalone manuscript updated to integrate Rounds 13-16.

docs/manuscript_round16_template.md -> docs/manuscript_round16.md (+ reports/manuscript_values_round16.json).
The placeholder syntax, the resolver and the value ledger are those of scripts/manuscript_round14.py, loaded here as an
independent module instance (so the Round-14 manuscript, its ledger and its tests are untouched); this module adds the
Round-15/16 tables. Every number is read, without computation, from saved records.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("manuscript_r16_base", ROOT / "scripts/manuscript_round14.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

B.TEMPLATE = ROOT / "docs/manuscript_round16_template.md"
B.OUT_MD = ROOT / "docs/manuscript_round16.md"
B.LEDGER_PATH = ROOT / "reports/manuscript_values_round16.json"
B.TITLE = ("Functional Kolmogorov–Arnold survival models for longitudinal covariates: pre-registered comparisons, "
           "safer symbolic summaries and a landmark-anchored variant")
V, md_table = B.V, B.md_table
S15, S16 = "reports/round15_summary.json", "reports/round16_summary.json"
DEV15 = "reports/round15/dev_summary.json"
PH = ["Contrast (a − b)", "Replicates", "Mean difference", "95% CI", "Paired-t p", "a < b / a > b"]


def prow(label, b):
    return [label, V(b + "/n", "d"), V(b + "/mean_diff", "+.4f"), V(b + "/ci95", "+.4f"), V(b + "/p_t", ".3g"),
            f"{V(b + '/n_neg', 'd')} / {V(b + '/n_pos', 'd')}"]


def t_distill():
    rows = []
    for kind, lab in (("add", "S-Add"), ("sparse", "S-Sparse")):
        a15, a16 = f"{S15}/family_a/{kind}", f"{S16}/family_a/{kind}"
        rows += [
            [f"{lab}, after pruning", V(a15 + "/rise_means/after_pruning", ".3f"), V(a16 + "/rise_means/after_pruning", ".3f")],
            [f"{lab}, substituted, no refit", V(a15 + "/rise_means/sub_noguard", ".3f"), V(a16 + "/rise_means/sub_noguard", ".3f")],
            [f"{lab}, old unconstrained refit", V(a15 + "/rise_means/refit_old", ".3f"), V(a16 + "/rise_means/refit_old", ".3f")],
            [f"{lab}, distillation (R15: μ by validation; R16: μ = 10, NaN-safe)", V(a15 + "/rise_means/distill", ".3f"),
             V(a16 + "/rise_means/distill_fixed", ".3f")]]
    return md_table(["Model (identifiable RISE, mean over replicates)", "Round 15", "Round 16"], rows)


def t_distill_contrasts():
    rows = []
    for rnd, S, key in (("R15", S15, "distill"), ("R16", S16, "distill_fixed")):
        for kind, lab in (("add", "S-Add"), ("sparse", "S-Sparse")):
            b = f"{S}/family_a/{kind}"
            rows.append(prow(f"{rnd} {lab}: distilled − after pruning (RISE)", b + "/distill_minus_pruned_rise"))
            rows.append(prow(f"{rnd} {lab}: distilled − old refit (RISE)", b + "/distill_minus_oldrefit_rise"))
            rows.append(prow(f"{rnd} {lab}: distilled − no refit (validation NLPL)", b + "/distill_minus_noguard_val_loss"))
    return md_table(PH, rows)


def t_hyp():
    rows = [["H-R15a (μ chosen by validation)", f"{V(S15 + '/family_a/H_R15a/n_finite', 'd')} of {V(S15 + '/family_a/H_R15a/n_cells', 'd')}",
             V(S15 + "/family_a/H_R15a/part2_upper95", "+.4f"), "not supported" if not B.resolve(S15 + "/family_a/H_R15a/supported") else "supported"],
            ["H-R16a (μ = 10 fixed, NaN-safe, degenerate formulas rejected)",
             f"{V(S16 + '/family_a/H_R16a/n_finite', 'd')} of {V(S16 + '/family_a/H_R16a/n_cells', 'd')}",
             V(S16 + "/family_a/H_R16a/part2_upper95", "+.4f"), "supported" if B.resolve(S16 + "/family_a/H_R16a/supported") else "not supported"]]
    out = md_table(["Hypothesis", "Distillation refits finite (rule: all)", "S-Add upper 95% bound (rule: < +0.05)", "Verdict"], rows)
    rows = [["H-R15b (D133 gate)", V(S15 + "/family_b/H_R15b/part1_ci95", "+.4f"), V(S15 + "/family_b/H_R15b/part2_upper95", "+.4f"),
             "supported" if B.resolve(S15 + "/family_b/H_R15b/supported") else "not supported"],
            ["H-R16b (non-cliff gate)", V(S16 + "/family_b/H_R16b/part1_ci95", "+.4f"), V(S16 + "/family_b/H_R16b/part2_upper95", "+.4f"),
             "supported" if B.resolve(S16 + "/family_b/H_R16b/supported") else "not supported"]]
    return out + "\n\n" + md_table(["Hypothesis", "S-Last: anchored − B2-S Cox curve error, 95% CI (rule: below 0)",
                                    "S-Sparse upper 95% bound (rule: < +0.005)", "Verdict"], rows)


def t_anchor():
    rows = []
    for kind, lab in (("last", "S-Last"), ("sparse", "S-Sparse")):
        b15, b16 = f"{S15}/family_b/{kind}/curve_error_means", f"{S16}/family_b/{kind}/curve_error_means"
        rows += [[f"{lab}, B2-S Cox", V(b15 + "/b2s_cox", ".4f"), V(b16 + "/b2s_cox", ".4f")],
                 [f"{lab}, landmark Cox", V(b15 + "/landmark_cox", ".4f"), V(b16 + "/landmark_cox", ".4f")],
                 [f"{lab}, static CoxKAN (last values)", V(b15 + "/static_coxkan", ".4f"), V(b16 + "/static_coxkan", ".4f")],
                 [f"{lab}, anchored, D133 gate", V(b15 + "/b2s_anchor_cox", ".4f"), V(b16 + "/b2s_anchor_cox_d133", ".4f")],
                 [f"{lab}, anchored, non-cliff gate", "–", V(b16 + "/b2s_anchor_cox_noncliff", ".4f")]]
    return md_table(["Model (survival-curve error, mean over replicates)", "Round 15", "Round 16"], rows)


def t_anchor_contrasts():
    rows = []
    for kind, lab in (("last", "S-Last"), ("sparse", "S-Sparse")):
        b = f"{S16}/family_b/{kind}/contrasts"
        for k in ("b2s_anchor_cox_noncliff - b2s_cox", "b2s_anchor_cox_noncliff - landmark_cox",
                  "b2s_anchor_cox_noncliff - b2s_anchor_cox_d133"):
            rows.append(prow(f"R16 {lab}: {k.replace(' - ', ' − ')}", f"{b}/{k}"))
    rows += [[f"R16 {lab}: replicates with every trajectory channel off (non-cliff / D133 gate)",
              f"{V(f'{S16}/family_b/{k}/gate_all_off/b2s_anchor_cox_noncliff', 'd')} / {V(f'{S16}/family_b/{k}/gate_all_off/b2s_anchor_cox_d133', 'd')}",
              "", "", "", ""] for k, lab in (("last", "S-Last"), ("sparse", "S-Sparse"))]
    return md_table(PH, rows)


LAML = {"22": 1, "914": 2, "966": 3}                     # development-check record lane of each seed (D135)


def t_afcm_dev():
    rows = []
    for seed, lane in LAML.items():
        b, L = f"{DEV15}/afcm_{seed}", f"reports/round15/dev/results_check_lane{lane}.jsonl/job_name=afcmlaml_{seed}/laml"
        rows.append([seed] + [f"{V(f'{b}{v}/phi1', '.3f')} / {V(f'{b}{v}/phi2', '.3f')}" for v in ("_v3", "_v3_bag3", "_v3_pc_t", "_v3_pars")] +
                    [f"{V(L + '/shapes/phi1/rise_id', '.3f')} / {V(L + '/shapes/phi2/rise_id', '.3f')}"])
    return md_table(["Development seed (identifiable RISE φ₁ / φ₂; rule: φ₁ < 0.15, φ₂ < 0.10)", "V3 (validation)",
                     "3-resample bag", "per-channel λ_t", "parsimonious λ rule", "LAML"], rows)


def t_real15():
    rows = []
    for ds, lab in B.DS:
        b = f"{S15}/real/{ds}"
        rows.append([lab] + [V(f"{b}/means/{m}/ibs", ".3f") for m in ("b2s_anchor_cox", "b2s_cox", "landmark_cox", "static_coxkan")] +
                    [f"{V(b + '/contrasts/ibs b2s_anchor_cox - ' + m + '/mean_diff', '+.4f')} {V(b + '/contrasts/ibs b2s_anchor_cox - ' + m + '/ci95', '+.4f')}"
                     for m in ("b2s_cox", "landmark_cox")])
    return md_table(["Data set (Round 15, splits 0–4)", "IBS anchored", "IBS B2-S Cox", "IBS landmark Cox", "IBS static CoxKAN",
                     "Anchored − B2-S Cox (NB 95% interval)", "Anchored − landmark Cox (NB 95% interval)"], rows)


def t_tests():
    b = "reports/report_values_round15.json"
    return md_table(["Final test suite before this revision (`final_full_suite_round15`)", "Count"],
                    [["Tests", V(b + "/n_tests", "s")], ["Passed", V(b + "/passed", "s")],
                     ["Failed (visible, recorded: CK-6c and the 4f gate's original R1)", V(b + "/failed", "s")],
                     ["Expected failures (strict xfail; historical records)", V(b + "/xfailed", "s")],
                     ["Skipped", V(b + "/skipped", "s")]])


B.TABLES.update(distill=t_distill, distill_contrasts=t_distill_contrasts, phase2_hyp=t_hyp, anchor=t_anchor,
                anchor_contrasts=t_anchor_contrasts, afcm_dev=t_afcm_dev, tests=t_tests, real15=t_real15)
render, TITLE, LEDGER, CITED, resolve = B.render, B.TITLE, B.LEDGER, B.CITED, B.resolve


def main():
    return B.main()


if __name__ == "__main__":
    sys.exit(main())
