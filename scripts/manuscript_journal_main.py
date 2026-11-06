"""Round 22 (D159): the journal main manuscript, a concise paper built on the cumulative Round-22 manuscript, which is
its Supplementary Material.

docs/manuscript_journal_main_template.md -> docs/manuscript_journal_main.md (+ reports/manuscript_values_journal_main.json).
scripts/manuscript_round22.py (and through it every earlier builder and the Round-14 resolver) is loaded as a separate
module instance, so the Supplementary Material and its ledger are untouched. Every number is read, without computation,
from the same saved records through the same resolver, and scripts/verify_manuscript_journal_main.py requires every
logged (source, format, text) triple to occur in reports/manuscript_values_round22.json, i.e. every number in the main
manuscript is also a verified number of the Supplementary Material. The reference list is docs/references_journal_main.md,
rows copied unchanged from docs/references_round22.md.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("manuscript_journal_m22", ROOT / "scripts/manuscript_round22.py")
M22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M22)
B = M22.B

B.TEMPLATE = ROOT / "docs/manuscript_journal_main_template.md"
B.OUT_MD = ROOT / "docs/manuscript_journal_main.md"
B.LEDGER_PATH = ROOT / "reports/manuscript_values_journal_main.json"
B.REFS = "docs/references_journal_main.md"
B.TITLE = ("Functional Kolmogorov–Arnold networks for survival prediction from longitudinal trajectories: "
           "interpretable effects, safe symbolic distillation, conditional landmark anchoring and semiparametric AFT "
           "robustness")
V, md_table, verdict = B.V, B.md_table, M22.verdict
S13 = "reports/round13_summary.json"
DS13 = (("pbc_E2", "PBC E2"), ("pbc_E1", "PBC E1"), ("prothro", "prothro"), ("aids", "aids"))


def yn(src):
    return verdict(src)


def t_j_hyp():
    rows = [["H1", "framing (b)", "KAN-FAE − MLP-FAE, survival-curve error (simulation)", "KAN-FAE worse"],
            ["H-R12", "Round 12", "B2-S Cox − AFCM V3, identifiable RISE (simulation)", "inconclusive"],
            ["H-R15a / H-R16a", "Rounds 15–16", "symbolic distillation refit finite and shape-preserving",
             f"{yn('reports/round15_summary.json/family_a/H_R15a/supported')} / {yn('reports/round16_summary.json/family_a/H_R16a/supported')}"],
            ["H-R15b / H-R16b", "Rounds 15–16", "landmark-anchored B2-S: gain when the last value carries the risk, "
             "no loss otherwise", f"{yn('reports/round15_summary.json/family_b/H_R15b/supported')} / "
                                  f"{yn('reports/round16_summary.json/family_b/H_R16b/supported')}"],
            ["H-R17a / H-R17b", "Round 17", "two-way gated anchoring (Cox / log-normal KAN-AFT)",
             f"{yn('reports/round17_summary.json/H_R17a/supported')} / {yn('reports/round17_summary.json/H_R17b/supported')}"],
            ["H-R18a / H-R18b", "Round 18", "anchoring decided on a confirmation split (Cox / log-normal KAN-AFT)",
             f"{yn('reports/round18_summary.json/H_R18a/supported')} / {yn('reports/round18_summary.json/H_R18b/supported')}"],
            ["H-R20a", "Round 20", "NP-KAN-AFT under log-normal errors, against Cox and Weibull",
             yn("reports/round20_summary.json/H_R20a/supported")],
            ["H-R20b", "Round 20", "cross-confirmed anchoring against full-validation B2-S (three families)",
             yn("reports/round20_summary.json/H_R20b/supported")],
            ["H-R21a / H-R21b / H-R21c", "Round 21", "Round-20 methods on four clinical cohorts and C-MAPSS FD001",
             " / ".join(yn(f"reports/round21/analysis_summary.json/hypotheses/H_R21{x}/supported") for x in "abc")],
            ["H-R22a", "Round 22", "B2-S Cox − per-channel REML AFCM (run only if CK-6d passed)", "not run (CK-6d failed)"],
            ["H-R22b", "Round 22", "safety of anchoring and distillation on untouched cohorts",
             yn("reports/round22/analysis_summary.json/H_R22b/supported")],
            ["H-R22c", "Round 22", "NP-KAN-AFT under bimodal, t₃ and logistic errors",
             yn("reports/round22/analysis_summary.json/H_R22c/supported")]]
    return md_table(["Hypothesis", "Stage", "Question", "Verdict"], rows)


def t_j_real13():
    rows = []
    for ds, lab in DS13:
        r = [lab]
        for comp in ("landmark_cox", "lme2_cox", "fpca_cox"):
            b = f"{S13}/contrasts/{ds}: ibs b2s_cox - {comp}"
            r.append(f"{V(b + '/mean_diff', '+.3f')} {V(b + '/ci95', '+.3f')}")
        rows.append(r)
    return md_table(["Data set", "B2-S Cox − landmark Cox", "B2-S Cox − two-stage mixed-effects Cox", "B2-S Cox − FPCA Cox"],
                    rows)


def t_j_ext():
    rows = []
    for ds in M22.CO:
        rows += [M22.nbrow(ds, "b2s_npaft - landmark_cox", "NP-KAN-AFT − landmark Cox"),
                 M22.nbrow(ds, "b2s_cox - landmark_cox", "B2-S Cox − landmark Cox"),
                 M22.nbrow(ds, "b2s_cox - lme2_cox", "B2-S Cox − two-stage mixed-effects Cox")]
    return md_table(M22.NB, rows)


B.TABLES.update(j_hyp=t_j_hyp, j_real13=t_j_real13, j_ext=t_j_ext)
render, TITLE, LEDGER, CITED, resolve = B.render, B.TITLE, B.LEDGER, B.CITED, B.resolve


def main():
    return B.main()


if __name__ == "__main__":
    sys.exit(main())
