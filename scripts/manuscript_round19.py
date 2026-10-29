"""Round 19 (D147-D149): the standalone manuscript updated to integrate Round 19 (the confirmation-split decision with
a full-validation estimator, tested against the usual full-validation B2-S).

docs/manuscript_round19_template.md -> docs/manuscript_round19.md (+ reports/manuscript_values_round19.json).
scripts/manuscript_round18.py (and through it the Round-17/16 builders and the Round-14 resolver) is loaded as a separate
module instance, so the earlier manuscripts, their ledgers and their verifications are untouched; this module adds the
Round-19 tables. Every number is read, without computation, from saved records.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("manuscript_r19_m18", ROOT / "scripts/manuscript_round18.py")
M18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M18)
B = M18.B

B.TEMPLATE = ROOT / "docs/manuscript_round19_template.md"
B.OUT_MD = ROOT / "docs/manuscript_round19.md"
B.LEDGER_PATH = ROOT / "reports/manuscript_values_round19.json"
B.TITLE = ("Functional Kolmogorov–Arnold survival models for longitudinal covariates: pre-registered comparisons, "
           "safer symbolic summaries and confirmed landmark anchoring")
V, md_table, prow4, PH4, verdict = B.V, B.md_table, M18.prow4, M18.PH4, M18.verdict
S19, DEV19 = "reports/round19_summary.json", "reports/round19/dev_summary.json"
LAB = {"last": "S-Last", "sparse": "S-Sparse", "misspec": "S-Misspec"}
FAMS = (("cox", ("last", "sparse")), ("aft", ("last", "misspec")))


def fam_label(loss):
    return "Cox" if loss == "cox" else "log-normal AFT"


def t_r19_hyp():
    h = f"{S19}/H_R19a"
    out = md_table(["Hypothesis (against the full-validation B2-S)", "S-Last, 95% CI (rule: upper < 0)",
                    "S-Sparse, 95% CI (rule: upper < +0.005)", "Verdict"],
                   [["H-R19a (Cox)", V(h + "/part1_ci95", "+.4f"), V(h + "/part2_ci95", "+.4f"), verdict(h + "/supported")]])
    h = f"{S19}/H_R19b"
    return out + "\n\n" + md_table(["Hypothesis (against the full-validation B2-S)",
                                    "S-Misspec, 95% CI (rule: upper < +0.005)", "S-Last, 95% CI (rule: upper < 0)", "Verdict"],
                                   [["H-R19b (log-normal AFT)", V(h + "/part1_misspec_ci95", "+.4f"),
                                     V(h + "/part2_last_ci95", "+.4f"), verdict(h + "/supported")]])


def t_r19_main():
    rows = []
    for loss, kinds in FAMS:
        for k in kinds:
            b = f"{S19}/{loss}/{k}"
            rows.append([f"{LAB[k]}, {fam_label(loss)}", V(f"{b}/curve_error_means/b2s_{loss}_fullval", ".4f"),
                         V(f"{b}/curve_error_means/b2s_{loss}_select", ".4f"),
                         V(f"{b}/curve_error_means/b2s_anchor_{loss}_confirm_r18", ".4f"),
                         V(f"{b}/curve_error_means/b2s_anchor_{loss}_refit", ".4f"),
                         f"{V(b + '/anchor_rate/n_anchored', 'd')} of {V(b + '/anchor_rate/n', 'd')}"])
    out = md_table(["Scenario, model family", "B2-S, whole validation set (primary comparator)", "B2-S, val_select",
                    "Round-18 estimator", "Round-19 estimator (refit)", "Anchored"], rows)
    rows = []
    for loss, kinds in FAMS:
        for k in kinds:
            b = f"{S19}/{loss}/{k}"
            F = fam_label(loss)
            rows.append(prow4(f"{LAB[k]}: refit − B2-S {F} (whole validation set; primary)", b + "/contrast"))
            rows.append(prow4(f"{LAB[k]}: refit − B2-S {F} (val_select)", b + "/vs_select"))
            rows.append(prow4(f"{LAB[k]}: Round-18 estimator − B2-S {F} (whole validation set)", b + "/r18_vs_fullval"))
            if loss == "cox":
                rows.append(prow4(f"{LAB[k]}: refit − landmark Cox", b + "/vs_landmark"))
    return out + "\n\n" + md_table(PH4, rows)


def t_r19_dev():
    v = B.resolve(f"{DEV19}/variant")
    rows = []
    for how in ("retrain", "finetune"):
        for loss, sc in (("cox", "sparse"), ("cox", "last"), ("aft", "misspec"), ("aft", "last")):
            lt = B.resolve(f"{DEV19}/choice_by_variant/{how}/{loss}")
            for l in sorted({"0.0", str(lt)}, key=float):
                b = f"{DEV19}/sweep/{how}/{loss}/{sc}/{l}"
                rows.append([f"{how}{' (chosen)' if how == v else ''}", f"{loss.upper()}, {LAB[sc]}",
                             B.K(f"{DEV19}/sweep/{how}/{loss}/{sc}", l),
                             V(b + "/n_anchored", "d") + " of " + V(b + "/n", "d"), V(b + "/mean_diff", "+.4f"),
                             V(b + "/ci95", "+.4f")])
    return md_table(["Refit variant", "Development replicates (Round-18 seeds)", "Margin", "Anchored",
                     "Estimator − full-validation B2-S", "95% CI"], rows)


def t_tests():
    b = "reports/report_values_round18.json"
    return md_table(["Final test suite before this revision (`final_full_suite_round18`)", "Count"],
                    [["Tests", V(b + "/n_tests", "s")], ["Passed", V(b + "/passed", "s")],
                     ["Failed (visible, recorded: CK-6c and the 4f gate's original R1)", V(b + "/failed", "s")],
                     ["Expected failures (strict xfail; historical records)", V(b + "/xfailed", "s")],
                     ["Skipped", V(b + "/skipped", "s")]])


B.TABLES.update(r19_hyp=t_r19_hyp, r19_main=t_r19_main, r19_dev=t_r19_dev, tests=t_tests)
render, TITLE, LEDGER, CITED, resolve = B.render, B.TITLE, B.LEDGER, B.CITED, B.resolve


def main():
    return B.main()


if __name__ == "__main__":
    sys.exit(main())
