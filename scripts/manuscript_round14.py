"""Round 14 (D131): record-driven manuscript builder.

docs/manuscript_round14_template.md -> docs/manuscript_round14.md (+ the value ledger reports/manuscript_values_round14.json).

Placeholders in the template:
  {{V:<source>|<spec>}}  a value read from a saved record and formatted (spec: a Python format spec, 's' for a string,
                         'd' for an integer, 'pct' for a percentage of a fraction); every rendered value is logged in
                         the ledger with its source, so that scripts/verify_manuscript_round14.py can re-resolve it.
  {{C:<key>}}            narrative citation ("Cui et al. (2021)"), resolved against docs/references_round14.md
  {{CP:<key>;<key>}}     parenthetical citation ("(Cui et al., 2021; Gellar et al., 2015)")
  {{T:<name>}}           a table built below from saved records, every cell through V()
  {{REFLIST}}            the reference list (cited keys only)

A <source> is a repository-relative file path followed by a path inside it, separated by '/':
  reports/round12_summary.json/H_R12/ci95                       (JSON: keys, or list indices)
  reports/tables/r12_summary_by_model.csv/scenario=S-Add,model=b2h_cox,front=bspline/rise_mean_primary_mean
                                                                (CSV: a filter selecting exactly one row, then a column;
                                                                 'col=' selects a missing value)
  reports/logs/round12_ck.jsonl/ck=CK-6c,run_id=round12b_ck_acceptance/shapes/phi1/rise_id
                                                                (JSONL: a filter selecting exactly one record, then keys)
  docs/references_round14.md/cui2021/Year                       (the reference table: key, then column)
A '/' inside a record key is written %2F (e.g. reports/round13_summary.json/stability/pbc_E2%2Fb2s_cox/...).
No number is computed here: values are read as saved, so the manuscript cannot contain a derived number that the
records do not hold.
"""
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs/manuscript_round14_template.md"
OUT_MD = ROOT / "docs/manuscript_round14.md"
LEDGER_PATH = ROOT / "reports/manuscript_values_round14.json"
REFS = "docs/references_round14.md"
TITLE = ("Functional Kolmogorov–Arnold survival models for longitudinal covariates: a pre-registered comparison with "
         "the additive functional Cox model, and why KAN blocks inside a functional autoencoder did not help")

_cache = {}
LEDGER = []
CITED = []


def _split(src):
    parts = src.split("/")
    for i, p in enumerate(parts):
        if p.endswith((".json", ".jsonl", ".csv", ".md")):
            return "/".join(parts[:i + 1]), parts[i + 1:]
    raise ValueError(f"no record file in source {src!r}")


def read_ref_table(path):
    rows = {}
    for line in (ROOT / path).read_text().splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if line.startswith("| ") and len(c) == 5 and c[0] not in ("Key", "---"):
            rows[c[0]] = dict(zip(("Key", "Cite", "Year", "Reference", "Verified"), c))
    return rows


def load(path):
    if path not in _cache:
        p = ROOT / path
        if path.endswith(".json"):
            _cache[path] = json.loads(p.read_text())
        elif path.endswith(".jsonl"):
            _cache[path] = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        elif path.endswith(".csv"):
            _cache[path] = pd.read_csv(p)
        else:
            _cache[path] = read_ref_table(path)
    return _cache[path]


def _native(v):
    if hasattr(v, "item"):
        v = v.item()
    return v


def resolve(src):
    path, rest = _split(src)
    obj = load(path)
    if path.endswith(".csv"):
        filt, col = rest
        g = obj
        for cond in filt.split(","):
            k, v = cond.split("=", 1)
            g = g[g[k].isna()] if v == "" else g[g[k].astype(str) == v]
        if len(g) != 1:
            raise KeyError(f"{src}: {len(g)} rows match")
        return _native(g.iloc[0][col])
    if path.endswith(".jsonl"):
        hits = [r for r in obj if all(str(r.get(k)) == v for k, v in (c.split("=", 1) for c in rest[0].split(",")))]
        if len(hits) != 1:
            raise KeyError(f"{src}: {len(hits)} records match")
        obj, rest = hits[0], rest[1:]
    for seg in rest:
        seg = seg.replace("%2F", "/")                      # a '/' inside a record key is written %2F
        obj = obj[int(seg)] if isinstance(obj, list) else obj[seg]
    return _native(obj) if not isinstance(obj, list) else [_native(x) for x in obj]


def fmt(v, spec):
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x, spec) for x in v) + "]"
    if spec == "s":
        return str(v)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if spec == "d":
        if float(v) != int(v):
            raise ValueError(f"not an integer: {v}")
        s = f"{int(v):,}" if abs(int(v)) >= 10000 else str(int(v))
    elif spec == "pct":
        s = f"{100 * v:.0f}%"
    else:
        s = format(v, spec)
    s = re.sub(r"e([-+])0*(\d)", lambda m: "e" + ("-" if m.group(1) == "-" else "") + m.group(2), s)
    return s.replace("-", "−")


def K(parent, key):
    """A record key used as a label (e.g. a landmark time stored as a JSON key), logged like a value."""
    if key not in resolve(parent):
        raise KeyError(f"{key} not in {parent}")
    LEDGER.append(dict(src=parent, spec="key", value=key, text=key))
    return key


def V(src, spec="s"):
    v = resolve(src)
    t = fmt(v, spec)
    LEDGER.append(dict(src=src, spec=spec, value=v, text=t))
    return t


def _cite(key):
    refs = load(REFS)
    if key not in refs:
        raise KeyError(f"unknown citation key {key}")
    if key not in CITED:
        CITED.append(key)
    return refs[key]["Cite"], V(f"{REFS}/{key}/Year")


def cite(key):
    c, y = _cite(key)
    return f"{c} ({y})"


def citep(keys):
    return "(" + "; ".join("{}, {}".format(*_cite(k.strip())) for k in keys.split(";")) + ")"


def reflist():
    refs = load(REFS)
    out = []
    for k in sorted(CITED, key=lambda k: (refs[k]["Cite"].lower(), refs[k]["Year"])):
        out.append(f"- {refs[k]['Cite']} ({refs[k]['Year']}). {refs[k]['Reference']}")
    return "\n".join(out)


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(r) + " |" for r in rows])


# ----------------------------------------------------------------------------------------------------------------------
# Tables (every number through V)
S12 = "reports/round12_summary.json"
S13 = "reports/round13_summary.json"
S3 = "reports/step3_summary.json"
S9 = "reports/round9_summary.json"
C12 = "reports/tables/r12_summary_by_model.csv"
C13 = "reports/tables/r13_single_by_model.csv"
COH = "reports/round14/cohort_summary.json"


def contrast_cells(base, spec="+.4f", ci="ci95", lo_hi=None, sign=True):
    """Mean difference, 95% CI, p and sign counts for a stored paired contrast."""
    if lo_hi:
        c = f"[{V(base + '/' + lo_hi[0], spec)}, {V(base + '/' + lo_hi[1], spec)}]"
    else:
        c = V(f"{base}/{ci}", spec)
    return [V(base + "/mean_diff", spec), c]


def t_hr12():
    rows = []
    items = [("**H-R12 (confirmatory): B2-S Cox (ensemble) − AFCM V3**", f"{S12}/H_R12"),
             ("Sensitivity: B2-S Cox − AFCM V0 (Round-12a AFCM)", f"{S12}/H_R12_sensitivity/b2s_vs_afcm_v0"),
             ("Sensitivity: single-seed B2-S Cox − AFCM V3", f"{S12}/H_R12_sensitivity/single_seed_b2s_vs_afcm_v3"),
             ("Sensitivity: raw (Round-11) RISE", f"{S12}/H_R12_sensitivity/raw_rise_b2s_vs_afcm_v3"),
             ("Per channel: φ₁ (homogeneous sin effect)", f"{S12}/H_R12_sensitivity/phi1_b2s_vs_afcm_v3"),
             ("Per channel: φ₂ (time-varying tanh effect)", f"{S12}/H_R12_sensitivity/phi2_b2s_vs_afcm_v3")]
    for lab, b in items:
        rows.append([lab, V(b + "/mean_a", ".4f"), V(b + "/mean_b", ".4f")] + contrast_cells(b) +
                    [V(b + "/p_t", ".3g"), V(b + "/p_wilcoxon", ".3g"), f"{V(b + '/n_neg', 'd')} / {V(b + '/n_pos', 'd')}",
                     V(b + "/p_tost", ".3g")])
    return md_table(["Contrast (S-Add, identifiable RISE, 20 replicates)", "Mean a", "Mean b", "Difference a − b",
                     "95% CI", "Paired-t p", "Wilcoxon p", "Replicates a < b / a > b", "TOST p (±0.05)"], rows)


SADD_MODELS = [("b2s_cox", "bspline", "B2-S Cox (ensemble)"), ("b2s_aft", "bspline", "B2-S KAN-AFT"),
               ("b2h_cox", "bspline", "B2-H Cox"), ("b2h_aft", "bspline", "B2-H KAN-AFT"),
               ("b2s_cox_quantile", "bspline", "B2-S Cox, quantile input"),
               ("afcm_v3", "bspline", "AFCM V3 (not validated, CK-6c)"), ("afcm_v0", "bspline", "AFCM V0"),
               ("b1_fpca1_cox", "bspline", "B1-FPCA-1 Cox"), ("b1_fpca1_aft", "bspline", "B1-FPCA-1 KAN-AFT"),
               ("fpca_cox", "bspline", "fpca_cox"), ("b1_sum_cox", "raw", "B1-SUM Cox"),
               ("b1_sum_aft", "raw", "B1-SUM KAN-AFT"), ("sum_cox", "raw", "sum_cox"), ("sum_rsf", "raw", "sum_rsf"),
               ("static_coxkan", "raw", "static CoxKAN (last values)")]


def _r12(scen, model, front, col, spec, sub=""):
    return V(f"{C12}/scenario={scen},sub={sub},model={model},front={front}/{col}", spec)


def t_sadd():
    rows = []
    for m, fr, lab in SADD_MODELS:
        has_shape = m.startswith(("b2", "afcm", "b1_fpca", "fpca"))
        rows.append([lab, _r12("S-Add", m, fr, "rise_mean_primary_mean", ".4f") if has_shape else "–",
                     _r12("S-Add", m, fr, "rise_phi1.rise_primary_mean", ".3f") if has_shape else "–",
                     _r12("S-Add", m, fr, "rise_phi2.rise_primary_mean", ".3f") if has_shape else "–",
                     _r12("S-Add", m, fr, "curve_error_mean", ".4f"), _r12("S-Add", m, fr, "ibs_mean", ".4f"),
                     _r12("S-Add", m, fr, "uno_c_mean", ".4f"), _r12("S-Add", m, fr, "dcal_frac_p_lt_005", "pct")])
    return md_table(["Model", "Mean RISE (φ₁, φ₂)", "RISE φ₁", "RISE φ₂", "Curve error", "IBS", "Uno C",
                     "D-cal p < 0.05"], rows)


def t_secondary():
    items = [("S-Add", "curve error", "B2-S Cox − AFCM V3", "S-Add curve error b2s_cox - afcm_v3"),
             ("S-Add", "RISE", "B2-S Cox − B1-FPCA-1 Cox", "S-Add RISE b2s_cox - b1_fpca1_cox"),
             ("S-Int", "curve error", "B2-deep Cox − B2-S Cox", "S-Int curve error b2deep_cox - b2s_cox"),
             ("S-Int", "curve error", "B1-FPCA-2 − B1-FPCA-1 (Cox)", "S-Int curve error b1_fpca2_cox - b1_fpca1_cox"),
             ("S-Sparse", "RISE", "B2-S Cox: PACE − B-spline front end", "S-Sparse RISE b2s_cox pace - bspline"),
             ("S-Sparse", "curve error", "B2-S Cox: PACE − B-spline front end", "S-Sparse curve error b2s_cox pace - bspline"),
             ("S-Sparse", "RISE", "B1-FPCA-1 Cox: PACE − B-spline front end", "S-Sparse RISE b1_fpca1_cox pace - bspline"),
             ("S-Sparse", "curve error", "flat_ae − B2-S Cox (B-spline)", "S-Sparse curve error flat_ae - b2s_cox (bspline)"),
             ("S-Misspec, log-normal", "curve error", "B2-S KAN-AFT, residual-calibrated − B2-S Cox",
              "S-Misspec-lognormal curve_error b2s_aft_rc - b2s_cox"),
             ("S-Misspec, log-normal", "D-cal p", "B2-S KAN-AFT, residual-calibrated − B2-S Cox",
              "S-Misspec-lognormal dcal_p b2s_aft_rc - b2s_cox"),
             ("S-Misspec, log-logistic", "curve error", "B2-S KAN-AFT, residual-calibrated − B2-S Cox",
              "S-Misspec-loglogistic curve_error b2s_aft_rc - b2s_cox"),
             ("S-Misspec, log-logistic", "D-cal p", "B2-S KAN-AFT, residual-calibrated − B2-S Cox",
              "S-Misspec-loglogistic dcal_p b2s_aft_rc - b2s_cox")]
    rows = []
    for scen, met, lab, key in items:
        b = f"{S12}/secondary/{key}"
        rows.append([scen, met, lab, V(b + "/mean_a", ".4f"), V(b + "/mean_b", ".4f")] + contrast_cells(b) +
                    [V(b + "/p_t", ".3g"), f"{V(b + '/n_neg', 'd')} / {V(b + '/n_pos', 'd')}"])
    return md_table(["Scenario", "Metric", "Contrast a − b", "Mean a", "Mean b", "Difference", "95% CI", "Paired-t p",
                     "a < b / a > b"], rows)


def t_sparse_int():
    rows = []
    for m, fr, lab in [("b2s_cox", "bspline", "B2-S Cox, B-spline"), ("b2s_cox", "pace", "B2-S Cox, PACE"),
                       ("b2h_cox", "bspline", "B2-H Cox, B-spline"), ("b2h_cox", "pace", "B2-H Cox, PACE"),
                       ("afcm_v3", "bspline", "AFCM V3, B-spline"), ("b1_fpca1_cox", "bspline", "B1-FPCA-1 Cox, B-spline"),
                       ("b1_fpca1_cox", "pace", "B1-FPCA-1 Cox, PACE"), ("fpca_cox", "bspline", "fpca_cox, B-spline"),
                       ("sum_cox", "raw", "sum_cox"), ("flat_ae", "raw", "flat_ae")]:
        shape = m != "sum_cox" and m != "flat_ae"
        rows.append([lab, _r12("S-Sparse", m, fr, "rise_mean_primary_mean", ".3f") if shape else "–",
                     _r12("S-Sparse", m, fr, "curve_error_mean", ".4f"), _r12("S-Sparse", m, fr, "ibs_mean", ".4f"),
                     _r12("S-Sparse", m, fr, "uno_c_mean", ".4f")])
    return md_table(["S-Sparse model, front end", "Mean RISE", "Curve error", "IBS", "Uno C"], rows)


def t_int():
    rows = []
    for m, fr, lab in [("b2deep_cox", "bspline", "B2-deep Cox"), ("b2s_cox", "bspline", "B2-S Cox"),
                       ("b2h_cox", "bspline", "B2-H Cox"), ("afcm_v3", "bspline", "AFCM V3"),
                       ("b1_fpca2_cox", "bspline", "B1-FPCA-2 Cox"), ("b1_fpca1_cox", "bspline", "B1-FPCA-1 Cox"),
                       ("fpca_cox", "bspline", "fpca_cox"), ("sum_cox", "raw", "sum_cox"), ("sum_rsf", "raw", "sum_rsf")]:
        rows.append([lab, _r12("S-Int", m, fr, "curve_error_mean", ".4f"), _r12("S-Int", m, fr, "ibs_mean", ".4f"),
                     _r12("S-Int", m, fr, "uno_c_mean", ".4f")])
    return md_table(["S-Int model", "Curve error", "IBS", "Uno C"], rows)


def t_misspec():
    rows = []
    for sub, slab in [("lognormal", "log-normal"), ("loglogistic", "log-logistic")]:
        for m, lab in [("b2s_cox", "B2-S Cox"), ("b2s_aft", "B2-S KAN-AFT (Weibull)"),
                       ("b2s_aft_rc", "B2-S KAN-AFT, residual-calibrated"), ("afcm_v3", "AFCM V3")]:
            rows.append([slab, lab, _r12("S-Misspec", m, "bspline", "curve_error_mean", ".4f", sub),
                         _r12("S-Misspec", m, "bspline", "ibs_mean", ".4f", sub),
                         _r12("S-Misspec", m, "bspline", "dcal_p_mean", ".3f", sub),
                         _r12("S-Misspec", m, "bspline", "dcal_frac_p_lt_005", "pct", sub)])
    return md_table(["Error law", "Model", "Curve error", "IBS", "Mean D-cal p", "Replicates with D-cal p < 0.05"], rows)


def t_selsym():
    rows = []
    for sc in ("S-Add", "S-Sparse"):
        b = f"{S12}/selection_symbolic/{sc}"
        rows.append([sc, V(b + "/recall_mean", ".2f"), V(b + "/fdr_mean", ".3f"), V(b + "/acceptance_rate_mean", ".3f"),
                     V(b + "/r2_val_accepted_mean", ".4f"), V(b + "/rise_after_pruning", ".3f"),
                     V(b + "/rise_after_substitution_before_refit", ".3f"),
                     f"{V(b + '/rise_after_substitution', '.3f')} ({V(b + '/n_after_substitution_ok', 'd')} of {V(b + '/n', 'd')})",
                     V(b + "/n_refit_nonfinite", "d")])
    return md_table(["Scenario", "Recall", "FDR", "Symbolic acceptance", "Mean validation R² (accepted)",
                     "RISE after pruning", "RISE after substitution, no refit", "RISE after refit (finite refits)",
                     "Non-finite refits"], rows)


def t_fae():
    rows = []
    h = f"{S3}/H1/curve_error"
    rows.append(["H1 (confirmatory, Round 8)", "fae_kan − fae_mlp, S2, P0", "curve error", V(h + "/n", "d")] +
                contrast_cells(h, lo_hi=("ci_lo", "ci_hi")) + [V(h + "/p_t", ".2g"), f"{V(h + '/n_pos', 'd')} / {V(h + '/n_neg', 'd')}"])
    h = f"{S3}/H1/ibs"
    rows.append(["H1 secondary metric", "fae_kan − fae_mlp, S2, P0", "IBS", V(h + "/n", "d")] +
                contrast_cells(h, lo_hi=("ci_lo", "ci_hi")) + [V(h + "/p_t", ".2g"), f"{V(h + '/n_pos', 'd')} / {V(h + '/n_neg', 'd')}"])
    for i, lab, arm in [(0, "F1 (Round 9)", "kan_S3 − mlp_S3 (tuned S3)")]:
        for met, idx in (("curve error", 0), ("IBS", 1), ("test reconstruction", 5)):
            b = f"{S9}/contrasts/{idx}"
            assert resolve(b + "/id") == "F1" and resolve(b + "/metric") == {0: "curve_error", 1: "ibs", 5: "reconstruction_test"}[idx]
            rows.append([lab, arm, met, V(b + "/n", "d")] + contrast_cells(b, spec="+.5f" if idx == 0 else "+.4f",
                                                                             lo_hi=("ci_lo", "ci_hi")) +
                        [V(b + "/p_t", ".2g"), f"{V(b + '/n_pos', 'd')} / {V(b + '/n_neg', 'd')}"])
    for met, key in (("curve error", "curve_error"), ("IBS", "ibs")):
        b = f"{S12}/P0/{key}"
        rows.append(["S-Int/P0 continuity (Round 12)", "fae_kan_s3 − fae_mlp_s3, fresh seeds", met, V(b + "/n", "d")] +
                    contrast_cells(b) + [V(b + "/p_t", ".2g"), f"{V(b + '/n_pos', 'd')} / {V(b + '/n_neg', 'd')}"])
    return md_table(["Analysis", "Contrast", "Metric", "Replicates", "Mean difference", "95% CI", "Paired-t p",
                     "KAN worse / better"], rows)


def t_gates():
    rows = [["Round 2 (A3, seed 21)", V("reports/report_values_round2.json/gate_C", "s"),
             V("reports/report_values_round2.json/gate_thr", "s"), V("reports/report_values_round2.json/gate_probe_mean", "s"),
             "–", "not confirmed (R2 failed)"]]
    for tag, lab in [("", "Revision 4 (round 3)"), ("_fa4b", "Revision 4b (round 4)"), ("_fa4c", "Revision 4c (round 5)"),
                     ("_fa4e", "Revision 4e (round 7)"), ("_fa4f", "Revision 4f (round 8)")]:
        b = f"reports/revised_gate_verdict{tag}.json"
        rows.append([f"{lab}, seed {V(b + '/gate_seed', 'd')}", V(b + "/gate_R2_C", ".4f"), V(b + "/gate_R2_threshold", ".4f"),
                     V(b + "/gate_probe_mean", ".3f"),
                     f"{V(b + '/confirmations_passed', 'd')} / {V(b + '/confirmations_total', 'd')}", V(b + "/verdict", "s")])
    return md_table(["MLP-FAE design (learnability gate)", "Gate-seed C", "R2 threshold", "Probe mean",
                     "Fresh confirmations passed", "Verdict"], rows)


def t_kanfae_prune():
    rows = []
    for var, stage, lab in [("plain", "before", "before pruning"), ("plain", "pruned_centred", "after centred pruning"),
                            ("reg", "before", "regularised, before pruning"), ("reg", "pruned_centred", "regularised, after centred pruning")]:
        b = f"reports/tables/step3_kanfae_prune.csv/variant={var},stage={stage}"
        rows.append([lab, V(b + "/harrell_c_mean", ".4f"), V(b + "/curve_error_mean", ".4f"),
                     V(b + "/reconstruction_test_mean", ".3f"), V(b + "/enc_active_risk_mean", ".1f")])
    return md_table(["KAN-FAE (Round 8; replicate means)", "Harrell C", "Curve error", "Test reconstruction",
                     "Active risk-encoder edges (of 512)"], rows)


DS = [("pbc_E2", "PBC, E2 (primary)"), ("pbc_E1", "PBC, E1 (sensitivity)"), ("prothro", "prothro"), ("aids", "aids")]


def t_cohorts():
    rows = []
    for ds, lab in DS:
        b = f"{COH}/{ds}"
        rows.append([lab, f"{V(b + '/landmark', 'g')} ({V(b + '/unit', 's')})", V(b + "/n_all", "d"), V(b + "/n_cohort", "d"),
                     V(b + "/n_events", "d"),
                     f"{V(b + '/visits_median', 'g')} ({V(b + '/visits_q1', 'g')}–{V(b + '/visits_q3', 'g')}; range {V(b + '/visits_min', 'd')}–{V(b + '/visits_max', 'd')})"])
    return md_table(["Data set", "Landmark", "Subjects", "At risk at the landmark (cohort)", "Events after the landmark",
                     "Visits in the window: median (IQR; range)"], rows)


SINGLE_MODELS = ["b2s_cox", "b2s_aft", "b2h_cox", "b2h_aft", "b1_fpca1_cox", "b1_fpca1_aft", "b1_sum_cox", "b1_sum_aft",
                 "static_coxkan", "afcm_v3", "fpca_cox", "lme2_cox", "landmark_cox", "sum_cox", "sum_rsf", "b2s_cox_pace"]


def t_single():
    rows = []
    for m in SINGLE_MODELS:
        r = [f"`{m}`"]
        for ds, _ in DS:
            b = f"{C13}/dataset={ds},model={m}"
            r += [V(b + "/ibs_mean", ".3f"), V(b + "/uno_c_mean", ".3f")]
        rows.append(r)
    return md_table(["Model", "PBC E2 IBS", "PBC E2 Uno C", "PBC E1 IBS", "PBC E1 Uno C", "prothro IBS", "prothro Uno C",
                     "aids IBS", "aids Uno C"], rows)


def t_single_dcal():
    rows = []
    for m in SINGLE_MODELS:
        r = [f"`{m}`"]
        for ds, _ in DS:
            b = f"{C13}/dataset={ds},model={m}"
            r.append(f"{V(b + '/dcal_n_p_lt_005', 'd')} / {V(b + '/n_splits', 'd')}")
        rows.append(r)
    return md_table(["Model: splits with D-cal p < 0.05", "PBC E2", "PBC E1", "prothro", "aids"], rows)


def t_contrasts13():
    rows = []
    for comp in ("afcm_v3", "landmark_cox", "lme2_cox", "sum_cox", "fpca_cox", "b1_fpca1_cox"):
        for ds, lab in DS:
            r = [f"`{comp}`", lab]
            for met in ("ibs", "uno_c"):
                b = f"{S13}/contrasts/{ds}: {met} b2s_cox - {comp}"
                r += [V(b + "/mean_diff", "+.3f"), V(b + "/ci95", "+.3f")]
            rows.append(r)
    return md_table(["Comparator b", "Data set", "IBS: B2-S Cox − b", "95% NB interval", "Uno C: B2-S Cox − b",
                     "95% NB interval"], rows)


def t_prune13():
    rows = []
    for ds, lab in DS:
        b = f"{S13}/prune_symbolic/{ds}"
        chans = resolve(b + "/channels")
        sel = "; ".join(f"{V(f'{b}/channels/{i}', 's')} {V(f'{b}/selection_frequency/{i}', '.1f')}" for i in range(len(chans)))
        rows.append([lab, sel, V(b + "/acceptance_rate", ".2f"), V(b + "/mean_r2_val", ".4f"),
                     V(b + "/fidelity_corr_no_refit", ".4f"), V(b + "/fidelity_corr_refit", ".3f"),
                     f"{V(b + '/ibs_pruned', '.3f')} / {V(b + '/ibs_no_refit', '.3f')} / {V(b + '/ibs_refit', '.3f')}",
                     V(b + "/n_refit_nonfinite", "d")])
    return md_table(["Data set", "Selection frequency over 5 splits", "Symbolic acceptance", "Mean validation R²",
                     "Risk correlation with the pruned model: substituted, no refit", "Same: refitted",
                     "IBS: pruned / substituted / refitted", "Non-finite refits"], rows)


def t_formulas():
    rows = []
    for ds, lab in DS:
        b = f"{S13}/prune_symbolic/{ds}/formulas_split0"
        for i in range(len(resolve(b))):
            e = f"{b}/{i}"
            name = resolve(e).get("channel")
            rows.append([lab, V(e + "/channel", "s") if name else f"baseline: {V(f'{COH}/{ds}/baseline/' + str(resolve(e + '/i')), 's')}",
                         V(e + "/stage", "s"), "`" + V(e + "/formula", "s") + "`", V(e + "/r2_val", ".4f"),
                         V(e + "/share", ".2f")])
    return md_table(["Data set", "Input", "Stage", "Formula (x in the input's original units; log scale for log markers)",
                     "Validation R²", "Share of layer score"], rows)


def t_stab():
    rows = []
    for key in resolve(f"{S13}/stability"):
        ds, model = key.split("/")
        k = key.replace("/", "%2F")
        for ch, v in resolve(f"{S13}/stability/{k}").items():
            b = f"{S13}/stability/{k}/{ch}"
            rows.append([dict(DS)[ds], "B2-S Cox" if model == "b2s_cox" else "B1-FPCA-1 Cox", ch, V(b + "/n_fits", "d"),
                         V(b + "/r2_mean", ".3f"), V(b + "/cos_mean", ".3f"),
                         V(b + "/direction", "s") if "direction" in v else "–",
                         f"{V(b + '/frac_aligned', '.2f')} ({'aligned' if v['aligned'] else '**not aligned**'})"
                         if "frac_aligned" in v else "–"])
    return md_table(["Data set", "Model", "Channel", "Bootstrap fits", "Stability R² (time-averaged effect)",
                     "Mean pairwise cosine", "Stated clinical direction of risk", "Share of fits monotone (aligned if ≥ 80%)"],
                    rows)


def t_b3():
    rows = []
    for ds in resolve(f"{S13}/b3"):
        for lm in resolve(f"{S13}/b3/{ds}/b3_coxkan"):
            a, c = f"{S13}/b3/{ds}/b3_coxkan/{lm}", f"{S13}/b3/{ds}/b3_cox/{lm}"
            rows.append([dict(DS)[ds], f"{K(f'{S13}/b3/{ds}/b3_coxkan', lm)} ({V(f'{COH}/{ds}/unit', 's')})",
                         V(a + "/n_test", ".1f"), V(a + "/brier", ".3f"), V(c + "/brier", ".3f"),
                         V(a + "/harrell_c", ".3f"), V(c + "/harrell_c", ".3f"),
                         f"{V(a + '/td_auc', '.3f')} ({V(a + '/n_auc', 'd')})", f"{V(c + '/td_auc', '.3f')} ({V(c + '/n_auc', 'd')})"])
    return md_table(["Data set", "Landmark (data units)", "Test rows (mean)", "Brier, KAN supermodel",
                     "Brier, linear supermodel", "C, KAN", "C, linear", "tdAUC, KAN (splits)", "tdAUC, linear (splits)"], rows)


def t_cmapss():
    rows = []
    for i, m in enumerate(resolve(f"{S13}/cmapss")):
        b = f"{S13}/cmapss/{i}"
        rows.append([f"`{V(b + '/model', 's')}`", V(b + "/ibs_mean", ".4f"), V(b + "/harrell_c_mean", ".3f"),
                     V(b + "/uno_c_mean", ".3f"), V(b + "/dcal_p_mean", ".2g"),
                     f"{V(b + '/dcal_n_p_lt_005', 'd')} / {V(b + '/n_splits', 'd')}"])
    out = md_table(["Model (C-MAPSS FD001, Option B)", "IBS", "Harrell C", "Uno C", "Mean D-cal p", "Splits with D-cal p < 0.05"],
                   rows)
    crow = []
    for k in resolve(f"{S13}/cmapss_contrasts"):
        b = f"{S13}/cmapss_contrasts/{k}"
        crow.append([k.replace("C-MAPSS: ", "").replace(" - ", " − "), V(b + "/mean_diff", "+.4f"), V(b + "/ci95", "+.4f"),
                     V(b + "/p", ".2g")])
    return out + "\n\n" + md_table(["Contrast (C-MAPSS)", "Mean difference", "95% NB interval", "p"], crow)


def t_tests():
    b = "reports/report_values_round13.json"
    return md_table(["Final test suite before the manuscript (`final_full_suite_round13`)", "Count"],
                    [["Tests", V(b + "/n_tests", "s")], ["Passed", V(b + "/passed", "s")],
                     ["Failed (visible, recorded: CK-6c and the 4f gate's original R1)", V(b + "/failed", "s")],
                     ["Expected failures (strict xfail; historical records)", V(b + "/xfailed", "s")],
                     ["Skipped", V(b + "/skipped", "s")]])


TABLES = {k[2:]: v for k, v in globals().items() if k.startswith("t_") and callable(v)}


def render(template_text):
    LEDGER.clear()
    CITED.clear()

    def sub(m):
        kind, arg = m.group(1), m.group(2)
        if kind == "V":
            src, _, spec = arg.rpartition("|")
            return V(src, spec or "s") if src else V(arg)
        if kind == "C":
            return cite(arg)
        if kind == "CP":
            return citep(arg)
        if kind == "T":
            return TABLES[arg]()
        raise KeyError(kind)

    out = re.sub(r"\{\{(V|CP|C|T):([^{}]+?)\}\}", sub, template_text)
    out = out.replace("{{REFLIST}}", reflist())
    left = re.findall(r"\{\{[^}]*\}\}", out)
    if left:
        raise ValueError(f"unresolved placeholders: {left[:5]}")
    return out


def main():
    md = render(TEMPLATE.read_text())
    OUT_MD.write_text(md)
    LEDGER_PATH.write_text(json.dumps(dict(template=str(TEMPLATE.relative_to(ROOT)), output=str(OUT_MD.relative_to(ROOT)),
                                           n_values=len(LEDGER), cited=CITED, values=LEDGER), indent=1, ensure_ascii=False))
    print(f"manuscript written: {len(md.splitlines())} lines, {len(LEDGER)} logged values, {len(CITED)} references")


if __name__ == "__main__":
    sys.exit(main())
