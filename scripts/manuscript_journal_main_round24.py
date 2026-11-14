"""Round 24 (D165): the journal main manuscript updated with Round 24 (the Bayesian joint-model comparator), built on the
cumulative Round-24 manuscript, which is its Supplementary Material. The Round-23 journal manuscript
(docs/manuscript_journal_main_round23.md) is kept unchanged.

docs/manuscript_journal_main_round24_template.md -> docs/manuscript_journal_main_round24.md
(+ reports/manuscript_values_journal_main_round24.json). scripts/manuscript_round24.py (and through it every earlier
builder and the Round-14 resolver) is loaded as a separate module instance, so the Supplementary Material and its ledger
are untouched. Every number is read, without computation, from the same saved records through the same resolver, and
scripts/verify_manuscript_journal_main_round24.py requires every logged (source, format, text) triple to occur in
reports/manuscript_values_round24.json, i.e. every number in the main manuscript is also a verified number of the
Supplementary Material. The reference list is docs/references_journal_main.md (unchanged).
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("manuscript_journal_m24", ROOT / "scripts/manuscript_round24.py")
M24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M24)
M23 = M24.M23
M22 = M23.M22
B = M24.B

B.TEMPLATE = ROOT / "docs/manuscript_journal_main_round24_template.md"
B.OUT_MD = ROOT / "docs/manuscript_journal_main_round24.md"
B.LEDGER_PATH = ROOT / "reports/manuscript_values_journal_main_round24.json"
B.REFS = "docs/references_journal_main.md"
B.TITLE = ("Functional Kolmogorov–Arnold networks for survival prediction from longitudinal trajectories: "
           "interpretable effects, safe symbolic distillation, conditional landmark anchoring and semiparametric AFT "
           "robustness")
V, md_table, verdict = B.V, B.md_table, M22.verdict
S13 = "reports/round13_summary.json"
A23 = M23.A
A24 = M24.A
DS13 = (("pbc_E2", "PBC E2"), ("pbc_E1", "PBC E1"), ("prothro", "prothro"), ("aids", "aids"))


def yn(src):
    return verdict(src)


def t_j_hyp():
    """Table 1. The 'Topic' column names the part of the study in words (the Supplementary Material is organised by the
    registration labels, which are kept in the first column)."""
    rows = [["H1", "KAN blocks inside a functional autoencoder", "KAN autoencoder − MLP autoencoder, survival-curve error "
             "(simulation)", "KAN autoencoder worse"],
            ["H-R12", "Shape recovery", "B2-S Cox − AFCM, identifiable RISE (simulation)", "inconclusive"],
            ["H-R15a / H-R16a", "Symbolic distillation", "distillation refit finite and shape-preserving "
             "(first version / fixed-penalty version)",
             f"{yn('reports/round15_summary.json/family_a/H_R15a/supported')} / {yn('reports/round16_summary.json/family_a/H_R16a/supported')}"],
            ["H-R15b / H-R16b", "Landmark anchoring", "always-anchored B2-S: gain when the last value carries the risk, "
             "no loss otherwise (two tests)", f"{yn('reports/round15_summary.json/family_b/H_R15b/supported')} / "
                                             f"{yn('reports/round16_summary.json/family_b/H_R16b/supported')}"],
            ["H-R17a / H-R17b", "Landmark anchoring", "anchoring gated on the validation loss (Cox / log-normal KAN-AFT)",
             f"{yn('reports/round17_summary.json/H_R17a/supported')} / {yn('reports/round17_summary.json/H_R17b/supported')}"],
            ["H-R18a / H-R18b", "Landmark anchoring", "anchoring decided on a confirmation split (Cox / log-normal KAN-AFT)",
             f"{yn('reports/round18_summary.json/H_R18a/supported')} / {yn('reports/round18_summary.json/H_R18b/supported')}"],
            ["H-R20a", "Error-law robustness", "NP-KAN-AFT under log-normal errors, against Cox and Weibull",
             yn("reports/round20_summary.json/H_R20a/supported")],
            ["H-R20b", "Landmark anchoring", "cross-confirmed anchoring against B2-S trained on the whole validation set "
             "(three model families)", yn("reports/round20_summary.json/H_R20b/supported")],
            ["H-R21a / H-R21b / H-R21c", "Clinical cohorts", "the fixed methods on four clinical cohorts and C-MAPSS FD001",
             " / ".join(yn(f"reports/round21/analysis_summary.json/hypotheses/H_R21{x}/supported") for x in "abc")],
            ["H-R22a", "Shape recovery", "B2-S Cox − AFCM with per-channel marginal-likelihood smoothing (to be run only "
             "if that AFCM passed its acceptance check)", "not run (acceptance check failed)"],
            ["H-R22b", "Held-out cohorts", "every fit finite, anchoring and distillation within their margins",
             yn("reports/round22/analysis_summary.json/H_R22b/supported")],
            ["H-R22c", "Error-law robustness", "NP-KAN-AFT under bimodal, t₃ and logistic errors",
             yn("reports/round22/analysis_summary.json/H_R22c/supported")],
            ["H-R23a", "Long histories", "NP-KAN-AFT better than landmark Cox on held-out C-MAPSS FD002 and FD004",
             yn(f"{A23}/H_R23a/supported")],
            ["H-R23b", "Long histories", "renal: thinning histories to three visits makes NP-KAN-AFT worse",
             yn(f"{A23}/H_R23b/supported")],
            ["H-R23c (i) / (ii)", "Long histories", "blend with landmark Cox: non-inferior on every clinical cohort / "
             "better on renal", f"{yn(A23 + '/H_R23c/part_i/supported')} / {yn(A23 + '/H_R23c/part_ii/supported')}"],
            ["H-R24a", "Joint-model comparison", "renal: NP-KAN-AFT better than a Bayesian joint model",
             yn(f"{A24}/H_R24a/supported")],
            ["H-R24b", "Joint-model comparison", "NP-KAN-AFT non-inferior to the joint model on every clinical cohort",
             yn(f"{A24}/H_R24b/supported")]]
    return md_table(["Hypothesis", "Topic", "Question", "Verdict"], rows)


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


def t_j_r23_cm():
    rows = []
    for m, lab in (("b2s_npaft", "NP-KAN-AFT"), ("landmark_cox", "Landmark Cox (version chosen on validation)"),
                   ("stack", "Blend (landmark Cox and NP-KAN-AFT)"), ("sum_aft", "Summary Weibull AFT"),
                   ("fpca_aft", "FPCA Weibull AFT")):
        r = [lab]
        for fd in M23.CM:
            b = f"{A23}/cm/{fd}"
            r += [V(f"{b}/means/ibs/{m}", ".4f"), V(f"{b}/means/harrell_c/{m}", ".3f"), V(f"{b}/dcal_pass/{m}", "d")]
        rows.append(r)
    return md_table(["Model", "FD002 IBS", "FD002 Harrell C", "FD002 D-cal p ≥ 0.05 (of 10)", "FD004 IBS", "FD004 Harrell C",
                     "FD004 D-cal p ≥ 0.05 (of 10)"], rows)


def t_j_r23_con():
    rows = []
    for fd in M23.CM:
        rows += [[f"{fd}: {lab}"] + M23.nbcells(f"{A23}/cm/{fd}/contrasts/{k}") for k, lab in
                 (("b2s_npaft - landmark_cox", "NP-KAN-AFT − landmark Cox"),
                  ("b2s_npaft - sum_aft", "NP-KAN-AFT − summary Weibull AFT"),
                  ("b2s_npaft - fpca_aft", "NP-KAN-AFT − FPCA Weibull AFT"))]
    rows += [[f"renal: {lab}"] + M23.nbcells(f"{A23}/clinical/renal/contrasts/{k}") for k, lab in
             (("b2s_npaft - b2s_npaft_sparse3", "NP-KAN-AFT, full history − three visits"),
              ("b2s_npaft_sparse3 - landmark_cox", "NP-KAN-AFT, three visits − landmark Cox"),
              ("b2s_npaft - landmark_cox", "NP-KAN-AFT, full history − landmark Cox"))]
    return md_table(M23.NBH, rows)


def t_j_r24():
    rows = []
    for ds, lab in M23.CLIN:
        b = f"{A24}/cohorts/{ds}"
        rows.append([lab, V(b + "/means/ibs_cond/b2s_npaft", ".4f"), V(b + "/means/ibs_cond/landmark_cox", ".4f"),
                     V(b + "/means/ibs_cond/jm", ".4f"),
                     f"{V(b + '/contrasts/b2s_npaft - jm/mean_diff', '+.4f')} {V(b + '/contrasts/b2s_npaft - jm/ci95', '+.4f')}",
                     f"{V(b + '/means/harrell_c/b2s_npaft', '.3f')} / {V(b + '/means/harrell_c/jm', '.3f')}",
                     V(b + "/dcal_pass/jm", "d"), V(b + "/jm_failed_partitions", "d")])
    return md_table(["Cohort", "NP-KAN-AFT", "Landmark Cox", "Joint model", "NP-KAN-AFT − joint model (NB 95% CI)",
                     "Harrell C, NP-KAN-AFT / joint model", "Joint model: D-calibration passed", "Joint-model fits failed"], rows)


B.TABLES.update(j_hyp=t_j_hyp, j_real13=t_j_real13, j_ext=t_j_ext, j_r23_cm=t_j_r23_cm, j_r23_con=t_j_r23_con, j_r24=t_j_r24)

# Journal wording of labels inside tables inherited from the Supplementary Material's builders: only label text is
# replaced (study-internal stage names -> plain descriptions); no value is touched, and every value is still logged and
# checked against the Supplementary Material's ledger.
RELABEL = (("(Round-12a AFCM)", "(earlier AFCM version)"), ("raw (Round-11) RISE", "RISE without the identifiability projection"),
           ("| Round 15 | Round 16 |", "| Initial test | Follow-up test |"),
           ("(R15: μ by validation; R16: μ = 10, NaN-safe)", "(initial: μ by validation; follow-up: μ = 10, NaN-safe)"),
           ("anchored, D133 gate", "anchored, original gate"), ("(not validated, CK-6c)", "(not validated)"))


def _relabel(fn):
    def g():
        t = fn()
        for a, b in RELABEL:
            t = t.replace(a, b)
        return t
    return g


for _k in ("hr12", "sadd", "distill", "anchor"):
    B.TABLES[_k] = _relabel(B.TABLES[_k])
render, TITLE, LEDGER, CITED, resolve = B.render, B.TITLE, B.LEDGER, B.CITED, B.resolve


def main():
    return B.main()


if __name__ == "__main__":
    sys.exit(main())
