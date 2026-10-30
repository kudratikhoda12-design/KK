"""Round 19 (D147-D149): the standalone manuscript updated to integrate Round 19 (the confirmation-split decision with
a full-validation estimator against the usual full-validation B2-S; development check only -- NO-GO, D147/D148).

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
DEV19 = "reports/round19/dev_summary.json"
LAB = {"last": "S-Last", "sparse": "S-Sparse", "misspec": "S-Misspec"}
FAMS = (("cox", ("last", "sparse")), ("aft", ("last", "misspec")))


def fam_label(loss):
    return "Cox" if loss == "cox" else "log-normal AFT"


def t_r19_devcmp():
    rows = []
    for k, lab in (("cox%2Fsparse", "S-Sparse, Cox"), ("cox%2Flast", "S-Last, Cox"), ("aft%2Fmisspec", "S-Misspec, log-normal AFT"),
                   ("aft%2Flast", "S-Last, log-normal AFT")):
        rows.append(prow4(f"{lab}: B2-S on val_select − B2-S on the whole validation set", f"{DEV19}/select_minus_fullval_comparator/{k}"))
        rows.append(prow4(f"{lab}: Round-18 estimator (margin 0) − B2-S on the whole validation set", f"{DEV19}/r18_estimator_minus_fullval/{k}"))
    out = md_table(PH4, rows)
    rows = [[lab, V(f"{DEV19}/diagnostics/{k}/select_gap_mean", "+.4f"), V(f"{DEV19}/diagnostics/{k}/confirm_gap_mean", "+.4f"),
             V(f"{DEV19}/diagnostics/{k}/alternative_test_diff_mean", "+.4f")]
            for k, lab in (("cox%2Fsparse", "S-Sparse, Cox"), ("cox%2Flast", "S-Last, Cox"), ("aft%2Fmisspec", "S-Misspec, log-normal AFT"),
                           ("aft%2Flast", "S-Last, log-normal AFT"))]
    return out + "\n\n" + md_table(["Development replicates (Round-18 seeds)", "Comparator − anchored loss: val_select",
                                    "the same on val_confirm", "Refitted anchored state (retrain) − B2-S on the whole validation set, test"], rows)


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


B.TABLES.update(r19_devcmp=t_r19_devcmp, r19_dev=t_r19_dev, tests=t_tests)
render, TITLE, LEDGER, CITED, resolve = B.render, B.TITLE, B.LEDGER, B.CITED, B.resolve


def main():
    return B.main()


if __name__ == "__main__":
    sys.exit(main())
