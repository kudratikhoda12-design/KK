"""Round 20 (D150-D152): the standalone manuscript updated to integrate Round 20 (a semiparametric functional KAN AFT
with a Kaplan-Meier error law, H-R20a, and the two-half cross-confirmed landmark anchor against the 3-seed B2-S
ensembles, H-R20b).

docs/manuscript_round20_template.md -> docs/manuscript_round20.md (+ reports/manuscript_values_round20.json).
scripts/manuscript_round19.py (and through it the Round-18/17/16 builders and the Round-14 resolver) is loaded as a
separate module instance, so the earlier manuscripts, their ledgers and their verifications are untouched; this module
adds the Round-20 tables. Every number is read, without computation, from saved records.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("manuscript_r20_m19", ROOT / "scripts/manuscript_round19.py")
M19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M19)
B = M19.B

B.TEMPLATE = ROOT / "docs/manuscript_round20_template.md"
B.OUT_MD = ROOT / "docs/manuscript_round20.md"
B.LEDGER_PATH = ROOT / "reports/manuscript_values_round20.json"
B.TITLE = ("Functional Kolmogorov–Arnold survival models for longitudinal covariates: pre-registered comparisons, "
           "safer symbolic summaries, confirmed landmark anchoring and a semiparametric KAN-AFT")
V, md_table, prow4, PH4, verdict = B.V, B.md_table, M19.prow4, M19.PH4, M19.verdict
DA = "reports/round20/devA_summary.json"
DB = "reports/round20/devB_summary.json"
S20 = "reports/round20_summary.json"
LAB = {"last": "S-Last", "sparse": "S-Sparse", "misspec": "S-Misspec"}
FL = {"cox": "Cox", "lnaft": "log-normal AFT", "npaft": "NP-AFT"}
TRAJ = {"cox": "sparse", "lnaft": "misspec", "npaft": "misspec"}
OBJ = {"bj_km": "b2s_npaft_bj_fullval", "rank_km": "b2s_npaft_rank_fullval"}


def t_r20_devA():
    rows = []
    for sc in ("sparse", "misspec"):
        rows.append([LAB[sc]] + [V(f"{DA}/means/{sc}/{m}", ".4f") if (sc == "misspec" or m != "b2s_weibull_fullval") else "–"
                                 for m in ("b2s_cox_fullval", "b2s_weibull_fullval", OBJ["bj_km"], OBJ["rank_km"])])
    out = md_table(["Development replicates (Round-18 seeds)", "B2-S Cox", "B2-S Weibull AFT", "NP-AFT, Buckley–James (bj_km)",
                    "NP-AFT, smoothed Gehan (rank_km)"], rows)
    rows = []
    for o, lab in (("bj_km", "bj_km"), ("rank_km", "rank_km")):
        b = f"{DA}/contrasts/{o}"
        rows += [prow4(f"S-Misspec: NP-AFT ({lab}) − B2-S Cox", b + "/misspec_vs_cox"),
                 prow4(f"S-Misspec: NP-AFT ({lab}) − B2-S Weibull AFT", b + "/misspec_vs_weibull"),
                 prow4(f"S-Sparse: NP-AFT ({lab}) − B2-S Cox", b + "/sparse_vs_cox")]
    rows += [prow4("S-Misspec: bj_km − rank_km", f"{DA}/contrasts/bj_km/misspec_bj_vs_rank"),
             prow4("S-Sparse: bj_km − rank_km", f"{DA}/contrasts/bj_km/sparse_bj_vs_rank")]
    return out + "\n\n" + md_table(PH4, rows)


def t_r20_devB():
    rows = []
    for f in ("cox", "lnaft", "npaft"):
        lm = B.resolve(f"{DB}/margin/{f}/margin")
        for sc in ("last", TRAJ[f]):
            for l in sorted({0.0, float(lm)}):
                b = f"{DB}/sweep/{f}/{sc}/{l}"
                rows.append([FL[f], LAB[sc], B.K(f"{DB}/sweep/{f}/{sc}", str(l)) + (" (chosen)" if l == lm else ""),
                             V(b + "/n_confirmed", "d") + " of " + V(b + "/n", "d"), V(b + "/mean_diff", "+.4f"),
                             V(b + "/ci95", "+.4f")])
    return md_table(["Model family", "Development replicates (Round-18 seeds)", "Margin", "Confirmed",
                     "Cross-confirmed − full-validation B2-S", "95% CI"], rows)


def t_r20_hyp():
    h = f"{S20}/H_R20a"
    out = md_table(["Hypothesis", "S-Misspec: NP-AFT − B2-S Cox, 95% CI (rule: upper < 0)",
                    "S-Misspec: NP-AFT − B2-S Weibull AFT, 95% CI (rule: upper < 0)", "Verdict"],
                   [["H-R20a (NP-AFT)", V(h + "/misspec_vs_cox/ci95", "+.4f"), V(h + "/misspec_vs_weibull/ci95", "+.4f"),
                     verdict(h + "/supported")]])
    rows = []
    for f in ("cox", "lnaft", "npaft"):
        b = f"{S20}/H_R20b/{f}"
        rows.append([f"H-R20b, {FL[f]}", V(b + "/last/contrast/ci95", "+.4f"),
                     f"{LAB[TRAJ[f]]}: " + V(b + "/traj/contrast/ci95", "+.4f"), verdict(b + "/supported")])
    rows.append(["H-R20b (all three families)", "", "", verdict(f"{S20}/H_R20b/supported")])
    return out + "\n\n" + md_table(["Hypothesis", "S-Last: cross-confirmed − full-validation B2-S, 95% CI (rule: upper < 0)",
                                    "Trajectory scenario, 95% CI (rule: upper < +0.005)", "Verdict"], rows)


def t_r20_np():
    rows = []
    for sc, ms in (("misspec", ("b2s_npaft_fullval", "b2s_cox_fullval", "b2s_weibull_fullval", "b2s_lnaft_fullval")),
                   ("sparse", ("b2s_npaft_fullval", "b2s_cox_fullval"))):
        rows.append([LAB[sc]] + [V(f"{S20}/means/{sc}/{m}", ".4f") for m in ms] + ["–"] * (4 - len(ms)))
    out = md_table(["Fresh replicates", "NP-AFT (rank_km)", "B2-S Cox", "B2-S Weibull AFT", "B2-S log-normal AFT"], rows)
    h = f"{S20}/H_R20a"
    rows = [prow4("S-Misspec: NP-AFT − B2-S Cox", h + "/misspec_vs_cox"),
            prow4("S-Misspec: NP-AFT − B2-S Weibull AFT", h + "/misspec_vs_weibull"),
            prow4("S-Sparse: NP-AFT − B2-S Cox (secondary)", h + "/sparse_vs_cox"),
            prow4("S-Misspec: NP-AFT − B2-S log-normal AFT (exploratory)", h + "/misspec_vs_lognormal"),
            prow4("S-Misspec, IBS: NP-AFT − B2-S Cox (exploratory)", h + "/misspec_vs_cox_ibs"),
            prow4("S-Misspec, IBS: NP-AFT − B2-S Weibull AFT (exploratory)", h + "/misspec_vs_weibull_ibs")]
    return out + "\n\n" + md_table(PH4, rows)


def t_r20_cross():
    rows = []
    for f in ("cox", "lnaft", "npaft"):
        for sc in ("last", TRAJ[f]):
            k = "last" if sc == "last" else "traj"
            b = f"{S20}/H_R20b/{f}/{k}"
            rows.append([f"{LAB[sc]}, {FL[f]}", V(f"{S20}/means/{sc}/b2s_{f}_fullval", ".4f"),
                         V(f"{S20}/means/{sc}/b2s_{f}_h1", ".4f"), V(f"{S20}/means/{sc}/b2s_anchor_{f}_cross", ".4f"),
                         V(b + "/n_confirmed", "d") + " of " + V(b + "/n", "d")])
    out = md_table(["Scenario, model family", "B2-S, whole validation set (comparator)", "B2-S, half h1",
                    "Cross-confirmed anchor", "Confirmed"], rows)
    rows = []
    for f in ("cox", "lnaft", "npaft"):
        for sc in ("last", TRAJ[f]):
            k = "last" if sc == "last" else "traj"
            rows.append(prow4(f"{LAB[sc]}, {FL[f]}: cross-confirmed − full-validation B2-S", f"{S20}/H_R20b/{f}/{k}/contrast"))
    return out + "\n\n" + md_table(PH4, rows)


def t_r20_diag():
    rows = []
    for f in ("cox", "lnaft", "npaft"):
        for sc in ("last", TRAJ[f]):
            k = "last" if sc == "last" else "traj"
            b = f"{S20}/H_R20b/{f}/{k}"
            rows.append([f"{LAB[sc]}, {FL[f]}", V(b + "/n_h1_favours", "d"), V(b + "/n_h2_favours", "d"),
                         V(b + "/n_confirmed", "d"), V(b + "/n_r18_rule", "d"), V(b + "/refit_always/mean_diff", "+.4f"),
                         V(b + "/refit_always/ci95", "+.4f")])
    return md_table(["Fresh replicates (of 20)", "h1 direction favours anchoring", "h2 direction", "Both (confirmed)",
                     "Round-18 rule (seed-0 member, one direction)", "Always refit − full-validation B2-S", "95% CI"], rows)


def t_tests():
    b = "reports/report_values_round19.json"
    return md_table(["Final test suite before this revision (`final_full_suite_round19`)", "Count"],
                    [["Tests", V(b + "/n_tests", "s")], ["Passed", V(b + "/passed", "s")],
                     ["Failed (visible, recorded: CK-6c and the 4f gate's original R1)", V(b + "/failed", "s")],
                     ["Expected failures (strict xfail; historical records)", V(b + "/xfailed", "s")],
                     ["Skipped", V(b + "/skipped", "s")]])


B.TABLES.update(r20_devA=t_r20_devA, r20_devB=t_r20_devB, r20_hyp=t_r20_hyp, r20_np=t_r20_np, r20_cross=t_r20_cross,
                r20_diag=t_r20_diag, tests=t_tests)
render, TITLE, LEDGER, CITED, resolve = B.render, B.TITLE, B.LEDGER, B.CITED, B.resolve


def main():
    return B.main()


if __name__ == "__main__":
    sys.exit(main())
