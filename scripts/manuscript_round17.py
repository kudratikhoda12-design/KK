"""Round 17 (D141-D143): the standalone manuscript updated to integrate Round 17 (two-way gated anchoring for Cox and
log-normal KAN-AFT; fixed-mu symbolic distillation for KAN-AFT).

docs/manuscript_round17_template.md -> docs/manuscript_round17.md (+ reports/manuscript_values_round17.json).
scripts/manuscript_round16.py (and through it the Round-14 resolver and ledger) is loaded as a separate module
instance, so the Round-14 and Round-16 manuscripts, their ledgers and their verifications are untouched; this module
adds the Round-17 tables. Every number is read, without computation, from saved records.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("manuscript_r17_m16", ROOT / "scripts/manuscript_round16.py")
M16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M16)
B = M16.B

B.TEMPLATE = ROOT / "docs/manuscript_round17_template.md"
B.OUT_MD = ROOT / "docs/manuscript_round17.md"
B.LEDGER_PATH = ROOT / "reports/manuscript_values_round17.json"
B.TITLE = ("Functional Kolmogorov–Arnold survival models for longitudinal covariates: pre-registered comparisons, "
           "safer symbolic summaries and gated landmark anchoring")
V, md_table, prow, PH = B.V, B.md_table, M16.prow, M16.PH
S17, DEV17 = "reports/round17_summary.json", "reports/round17/dev_summary.json"


def verdict(src):
    return "supported" if B.resolve(src) else "not supported"


def t_r17_hyp():
    h = f"{S17}/H_R17a"
    rows = [["H-R17a (two-way gated b2s_anchor_cox)", V(h + "/part1_ci95", "+.4f"), V(h + "/part2_ci95", "+.4f"),
             verdict(h + "/supported")]]
    out = md_table(["Hypothesis", "S-Last: two-way − B2-S Cox curve error, 95% CI (rule: upper < 0)",
                    "S-Sparse, 95% CI (rule: upper < +0.005)", "Verdict"], rows)
    h = f"{S17}/H_R17b"
    rows = [["H-R17b (log-normal KAN-AFT)", f"{V(h + '/part1_n_finite', 'd')} of {V(h + '/part1_n_cells', 'd')}",
             V(h + "/part1_upper95", "+.4f"), V(h + "/part2_misspec_ci95", "+.4f"), V(h + "/part2_last_ci95", "+.4f"),
             verdict(h + "/supported")]]
    return out + "\n\n" + md_table(["Hypothesis", "AFT distillation refits finite (rule: all)",
                                    "S-Misspec RISE upper bound (rule: < +0.05)",
                                    "S-Misspec two-way − B2-S AFT, 95% CI (rule: upper < +0.005)",
                                    "S-Last two-way − B2-S AFT, 95% CI (rule: upper < 0)", "Verdict"], rows)


LAB = {"last": "S-Last", "sparse": "S-Sparse", "misspec": "S-Misspec"}


def t_r17_twoway():
    rows = []
    for loss, kinds in (("cox", ("last", "sparse")), ("aft", ("last", "sparse", "misspec"))):
        for k in kinds:
            b = f"{S17}/{loss}/{k}"
            m = f"b2s_anchor_{loss}_twoway"
            rows.append([f"{LAB[k]}, {'Cox' if loss == 'cox' else 'log-normal AFT'}", V(f"{b}/curve_error_means/b2s_{loss}", ".4f"),
                         V(f"{b}/curve_error_means/{m}", ".4f"),
                         f"{V(b + '/gates/n_reduced', 'd')} of {V(b + '/gates/n', 'd')}",
                         str([V(f"{b}/gates/last_on_frequency/{d}", ".2f") for d in range(3)]).replace("'", ""),
                         str([V(f"{b}/gates/traj_on_frequency/{d}", ".2f") for d in range(3)]).replace("'", "")])
    out = md_table(["Scenario, model family", "Comparator curve error", "Two-way curve error",
                    "Reduced to the comparator", "Last-value edge kept (channels 1, 2, 3)",
                    "Trajectory channel kept (channels 1, 2, 3)"], rows)
    rows = []
    for loss, kinds in (("cox", ("last", "sparse")), ("aft", ("last", "sparse", "misspec"))):
        for k in kinds:
            b = f"{S17}/{loss}/{k}"
            fam = "Cox" if loss == "cox" else "AFT"
            rows.append(prow(f"{LAB[k]}: two-way − B2-S {fam}", b + "/contrast"))
            rows.append(prow(f"{LAB[k]}: anchored alternative − B2-S {fam}", b + "/alternative_minus_comparator"))
    return out + "\n\n" + md_table(PH, rows)


def t_r17_aftsym():
    a = f"{S17}/aft_symbolic"
    rows = [[lab, V(f"{a}/rise_means/{k}", ".3f")] for k, lab in (
        ("after_pruning", "after pruning"), ("sub_guard", "guarded substitution, no refit"),
        ("refit_plain", "unanchored refit"), ("distill_fixed", "distillation, μ = 10, NaN-safe"))]
    rows.append(["B2-S AFT ensemble (unpruned)", V(f"{a}/b2s_aft_rise_mean", ".3f")])
    out = md_table(["Log-normal KAN-AFT on S-Misspec (identifiable RISE, mean)", "Round 17"], rows)
    rows = [prow("distilled − after pruning (RISE)", a + "/distill_minus_pruned_rise"),
            prow("distilled − unanchored refit (RISE)", a + "/distill_minus_plain_rise"),
            prow("distilled − no refit (curve error)", a + "/distill_minus_sub_curve_error"),
            prow("distilled − no refit (validation loss)", a + "/distill_minus_sub_val_loss")]
    return out + "\n\n" + md_table(PH, rows)


def t_r17_dev():
    rows = []
    for loss, sc, lab in (("cox", "sparse", "Cox, S-Sparse 3900–3919"), ("cox", "last", "Cox, S-Last 4000–4019"),
                          ("aft", "misspec", "AFT, S-Misspec 961–969"), ("aft", "sparse", "AFT, S-Sparse 3900–3919"),
                          ("aft", "last", "AFT, S-Last 4000–4019")):
            lt = B.resolve(f"{DEV17}/last_tol/{loss}")
            for l in ("0.0", str(lt)):
                b = f"{DEV17}/sweep/{loss}/{sc}/{l}"
                rows.append([lab, V(f"{DEV17}/sweep/{loss}/{sc}/{l}/n_anchored", "d") + " of " + V(b + "/n", "d"),
                             B.K(f"{DEV17}/sweep/{loss}/{sc}", l), V(b + "/mean_diff", "+.4f"), V(b + "/ci95", "+.4f")])
    return md_table(["Development replicates", "Anchored", "last_tol", "Two-way − comparator", "95% CI"], rows)


def t_tests():
    b = "reports/report_values_round16.json"
    return md_table(["Final test suite before this revision (`final_full_suite_round16`)", "Count"],
                    [["Tests", V(b + "/n_tests", "s")], ["Passed", V(b + "/passed", "s")],
                     ["Failed (visible, recorded: CK-6c and the 4f gate's original R1)", V(b + "/failed", "s")],
                     ["Expected failures (strict xfail; historical records)", V(b + "/xfailed", "s")],
                     ["Skipped", V(b + "/skipped", "s")]])


B.TABLES.update(r17_hyp=t_r17_hyp, r17_twoway=t_r17_twoway, r17_aftsym=t_r17_aftsym, r17_dev=t_r17_dev, tests=t_tests)
render, TITLE, LEDGER, CITED, resolve = B.render, B.TITLE, B.LEDGER, B.CITED, B.resolve


def main():
    return B.main()


if __name__ == "__main__":
    sys.exit(main())
