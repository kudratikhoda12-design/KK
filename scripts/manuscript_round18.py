"""Round 18 (D144-D146): the standalone manuscript updated to integrate Round 18 (landmark anchoring decided on a
confirmation split).

docs/manuscript_round18_template.md -> docs/manuscript_round18.md (+ reports/manuscript_values_round18.json).
scripts/manuscript_round17.py (and through it the Round-16 builder and the Round-14 resolver) is loaded as a separate
module instance, so the earlier manuscripts, their ledgers and their verifications are untouched; this module adds the
Round-18 tables. Every number is read, without computation, from saved records.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("manuscript_r18_m17", ROOT / "scripts/manuscript_round17.py")
M17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M17)
B = M17.B

B.TEMPLATE = ROOT / "docs/manuscript_round18_template.md"
B.OUT_MD = ROOT / "docs/manuscript_round18.md"
B.LEDGER_PATH = ROOT / "reports/manuscript_values_round18.json"
B.TITLE = ("Functional Kolmogorov–Arnold survival models for longitudinal covariates: pre-registered comparisons, "
           "safer symbolic summaries and confirmed landmark anchoring")
V, md_table, prow, PH, verdict = B.V, B.md_table, M17.prow, M17.PH, M17.verdict
S18, DEV18 = "reports/round18_summary.json", "reports/round18/dev_summary.json"
LAB = {"last": "S-Last", "sparse": "S-Sparse", "misspec": "S-Misspec"}
FAMS = (("cox", ("last", "sparse")), ("aft", ("last", "sparse", "misspec")))
PH4 = ["Contrast (a − b)", "Replicates", "Mean difference", "95% CI", "Paired-t p (two-sided)", "a < b / a > b / ties"]


def prow4(label, b):
    return [label, V(b + "/n", "d"), V(b + "/mean_diff", "+.4f"), V(b + "/ci95", "+.4f"), V(b + "/p_t", ".3g"),
            f"{V(b + '/n_neg', 'd')} / {V(b + '/n_pos', 'd')} / {V(b + '/n_zero', 'd')}"]


def fam_label(loss):
    return "Cox" if loss == "cox" else "log-normal AFT"


def t_r18_hyp():
    h = f"{S18}/H_R18a"
    out = md_table(["Hypothesis", "S-Last: confirm − B2-S Cox curve error, 95% CI (rule: upper < 0)",
                    "S-Sparse, 95% CI (rule: upper < +0.005)", "Verdict"],
                   [["H-R18a (Cox)", V(h + "/part1_ci95", "+.4f"), V(h + "/part2_ci95", "+.4f"), verdict(h + "/supported")]])
    h = f"{S18}/H_R18b"
    return out + "\n\n" + md_table(["Hypothesis", "S-Misspec: confirm − B2-S AFT, 95% CI (rule: upper < +0.005)",
                                    "S-Last, 95% CI (rule: upper < 0)", "Verdict"],
                                   [["H-R18b (log-normal AFT)", V(h + "/part1_misspec_ci95", "+.4f"),
                                     V(h + "/part2_last_ci95", "+.4f"), verdict(h + "/supported")]])


def t_r18_main():
    rows = []
    for loss, kinds in FAMS:
        for k in kinds:
            b = f"{S18}/{loss}/{k}"
            m = f"b2s_anchor_{loss}_confirm"
            rows.append([f"{LAB[k]}, {fam_label(loss)}", V(f"{b}/curve_error_means/b2s_{loss}", ".4f"),
                         V(f"{b}/curve_error_means/b2s_{loss}_fullval", ".4f"), V(f"{b}/curve_error_means/{m}", ".4f"),
                         f"{V(b + '/anchor_rate/n_anchored', 'd')} of {V(b + '/anchor_rate/n', 'd')}"])
    out = md_table(["Scenario, model family", "Comparator (val_select)", "Comparator (whole validation set)",
                    "Confirmation-gated", "Anchored"], rows)
    rows = []
    for loss, kinds in FAMS:
        for k in kinds:
            b = f"{S18}/{loss}/{k}"
            rows.append(prow4(f"{LAB[k]}: confirm − B2-S {fam_label(loss)}", b + "/contrast"))
            rows.append(prow4(f"{LAB[k]}: anchored alternative − B2-S {fam_label(loss)}", b + "/alternative_minus_comparator"))
            rows.append(prow4(f"{LAB[k]}: confirm − B2-S {fam_label(loss)} on the whole validation set", b + "/confirm_minus_fullval"))
    return out + "\n\n" + md_table(PH4, rows)


def t_r18_diag():
    rows = []
    for src, lab in ((DEV18 + "/diagnostics", "development"), (None, "formal")):
        for loss, kinds in FAMS:
            for k in kinds:
                b = f"{src}/{loss}%2F{k}" if src else f"{S18}/{loss}/{k}/diagnostics"
                rows.append([f"{lab}, {LAB[k]}, {fam_label(loss)}", V(b + "/select_gap_mean", "+.4f"),
                             V(b + "/confirm_gap_mean", "+.4f"), V(b + "/alternative_test_diff_mean", "+.4f"),
                             V(b + "/corr_select_gap_test", "+.2f"), V(b + "/corr_confirm_gap_test", "+.2f")])
    return md_table(["Replicates", "Comparator − anchored loss, val_select", "the same on val_confirm",
                     "Anchored alternative − comparator, test curve error", "Correlation with the test difference: val_select",
                     "val_confirm"], rows)


def t_r18_dev():
    rows = []
    for loss, sc in (("cox", "sparse"), ("cox", "last"), ("aft", "misspec"), ("aft", "sparse"), ("aft", "last")):
        lt = B.resolve(f"{DEV18}/margin/{loss}")
        for l in sorted({"0.0", str(lt)}, key=float):
            b = f"{DEV18}/sweep/{loss}/{sc}/{l}"
            rows.append([f"{loss.upper()}, {LAB[sc]}", B.K(f"{DEV18}/sweep/{loss}/{sc}", l),
                         V(b + "/n_anchored", "d") + " of " + V(b + "/n", "d"), V(b + "/mean_diff", "+.4f"), V(b + "/ci95", "+.4f")])
    return md_table(["Development replicates (Round-17 seeds)", "Margin", "Anchored", "Confirm − comparator", "95% CI"], rows)


def t_tests():
    b = "reports/report_values_round17.json"
    return md_table(["Final test suite before this revision (`final_full_suite_round17`)", "Count"],
                    [["Tests", V(b + "/n_tests", "s")], ["Passed", V(b + "/passed", "s")],
                     ["Failed (visible, recorded: CK-6c and the 4f gate's original R1)", V(b + "/failed", "s")],
                     ["Expected failures (strict xfail; historical records)", V(b + "/xfailed", "s")],
                     ["Skipped", V(b + "/skipped", "s")]])


B.TABLES.update(r18_hyp=t_r18_hyp, r18_main=t_r18_main, r18_diag=t_r18_diag, r18_dev=t_r18_dev, tests=t_tests)
render, TITLE, LEDGER, CITED, resolve = B.render, B.TITLE, B.LEDGER, B.CITED, B.resolve


def main():
    return B.main()


if __name__ == "__main__":
    sys.exit(main())
