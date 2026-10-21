"""Round 12 analysis (D113 as amended by D120, D123, D126). Reads reports/round12/results_*.jsonl (fsim scenarios)
and reports/round12/p0_lane*.jsonl (S-Int/P0); writes reports/round12_summary.json and reports/tables/r12_*.csv.

H-R12 (single confirmatory test): S-Add, b2s_cox (3-seed ensemble, B-spline front end) minus afcm_v3 in the mean
identifiable RISE over channels 1-2; paired over the replicates where both fits are ok; two-sided paired t (alpha 0.05),
95% t-CI (df n-1), exact Wilcoxon signed-rank p, sign counts, TOST at +-0.05 (two one-sided t-tests at 0.05, i.e. the
90% CI inside +-0.05). Interpretation fixed in D113: CI entirely below 0 -> B2-S better; entirely above 0 -> worse;
otherwise TOST equivalent -> equivalent within the margin; otherwise inconclusive. Caveat (D126): afcm_v3 failed its
single-seed phi_1 acceptance check CK-6c. Sensitivity rows (not confirmatory): single-seed b2s_cox, afcm_v0, raw RISE.
Every other contrast is exploratory (no multiplicity correction).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R12 = Path(os.environ.get("R12_DIR", ROOT / "reports/round12"))
TAB = ROOT / "reports/tables"


def load():
    recs = []
    for f in sorted(R12.glob("results_*.jsonl")):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return recs


def get(r, path):
    for k in path.split("."):
        if r is None:
            return None
        r = r.get(k) if isinstance(r, dict) else None
    return r


def by_rep(recs, scen, model, front="bspline", path="metrics.shape.mean_id", sub=None):
    out = {}
    for r in recs:
        if (r.get("status") == "ok" and r["scenario"] == scen and r["model"] == model and r["front"] == front
                and r.get("sub") == sub and r["replicate"] not in out):
            v = get(r, path)
            if v is not None:
                out[r["replicate"]] = float(v)
    return out


def paired(a, b, margin=None):
    """a - b over common replicates."""
    reps = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in reps])
    n = len(d)
    if n < 2:
        return dict(n=n)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tq, tq90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    out = dict(n=n, mean_a=float(np.mean([a[k] for k in reps])), mean_b=float(np.mean([b[k] for k in reps])),
               mean_diff=m, sd_diff=sd, ci95=[m - tq * se, m + tq * se], ci90=[m - tq90 * se, m + tq90 * se],
               p_t=float(stats.ttest_1samp(d, 0.0).pvalue) if sd > 0 else float("nan"),
               p_wilcoxon=float(stats.wilcoxon(d, method="exact").pvalue) if np.any(d != 0) else float("nan"),
               n_neg=int((d < 0).sum()), n_pos=int((d > 0).sum()), n_zero=int((d == 0).sum()), reps=reps)
    if margin is not None and sd > 0:
        p_lo = float(stats.t.sf((m + margin) / se, n - 1))           # H0: diff <= -margin
        p_hi = float(stats.t.cdf((m - margin) / se, n - 1))          # H0: diff >= +margin
        out.update(margin=margin, p_tost=max(p_lo, p_hi), equivalent=bool(max(p_lo, p_hi) < 0.05))
    return out


def interpret(p):
    if p.get("n", 0) < 2:
        return "not computable"
    if p["ci95"][1] < 0:
        return "B2-S recovers shapes better than AFCM (95% CI entirely below 0)"
    if p["ci95"][0] > 0:
        return "B2-S recovers shapes worse than AFCM (95% CI entirely above 0)"
    if p.get("equivalent"):
        return "equivalent within +-0.05 (TOST)"
    return "inconclusive"


def summary_table(recs):
    rows = []
    keys = sorted({(r["scenario"], r.get("sub") or "", r["model"], r["front"]) for r in recs if "model" in r})
    for sc, sub, mo, fr in keys:
        g = [r for r in recs if r["scenario"] == sc and (r.get("sub") or "") == sub and r["model"] == mo and r["front"] == fr]
        ok = [r for r in g if r.get("status") == "ok"]
        reps_ok = {r["replicate"] for r in ok}
        row = dict(scenario=sc, sub=sub, model=mo, front=fr, n_ok=len(reps_ok),
                   n_failed=len({r["replicate"] for r in g if r.get("status") == "failed"} - reps_ok),
                   n_dropped=len({r["replicate"] for r in g if r.get("status") == "dropped_budget"} - reps_ok))
        first = {}
        for r in ok:
            first.setdefault(r["replicate"], r)
        for k in ("curve_error", "ibs", "harrell_c", "uno_c", "dcal_p"):
            v = [get(r, f"metrics.{k}") for r in first.values() if get(r, f"metrics.{k}") is not None]
            if v:
                row[f"{k}_mean"], row[f"{k}_sd"] = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        v = [get(r, "metrics.dcal_p") for r in first.values() if get(r, "metrics.dcal_p") is not None]
        if v:
            row["dcal_frac_p_lt_005"] = float(np.mean(np.array(v) < 0.05))
        for k in ("mean_primary", "mean_id", "mean_raw", "phi1.rise_primary", "phi2.rise_primary", "phi1.rise_raw",
                  "phi2.rise_raw"):
            v = [get(r, f"metrics.shape.{k}") for r in first.values() if get(r, f"metrics.shape.{k}") is not None]
            if v:
                row[f"rise_{k}_mean"] = float(np.mean(v))
                row[f"rise_{k}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        v = [get(r, "metrics_seed0.shape.mean_id") for r in first.values() if get(r, "metrics_seed0.shape.mean_id") is not None]
        if v:
            row["rise_mean_id_seed0_mean"] = float(np.mean(v))
        v = [get(r, "metrics_seed0.curve_error") for r in first.values() if get(r, "metrics_seed0.curve_error") is not None]
        if v:
            row["curve_error_seed0_mean"] = float(np.mean(v))
        row["seconds_mean"] = float(np.mean([r.get("seconds", np.nan) for r in first.values()])) if first else None
        rows.append(row)
    return pd.DataFrame(rows)


def selection_symbolic(recs, scen):
    g = {}
    for r in recs:
        if r.get("status") == "ok" and r["scenario"] == scen and r["model"] == "prune_symbolic":
            g.setdefault(r["replicate"], r)
    if not g:
        return None
    R = list(g.values())
    f = lambda path: [get(r, path) for r in R if get(r, path) is not None]  # noqa: E731
    stage_tot = {}
    for r in R:
        for k, v in (get(r, "symbolic.stages") or {}).items():
            stage_tot[k] = stage_tot.get(k, 0) + v
    return dict(n=len(R), recall_mean=float(np.mean(f("recall"))), fdr_mean=float(np.mean(f("fdr"))),
                selected_patterns={str(p): int(sum(str(r["selected"]) == str(p) for r in R))
                                   for p in sorted({str(r["selected"]) for r in R})},
                chosen_theta=dict(pd.Series(f("chosen_theta")).value_counts().sort_index()),
                acceptance_rate_mean=float(np.mean(f("symbolic.acceptance_rate"))),
                r2_val_accepted_mean=float(np.mean(f("symbolic.mean_r2_val_accepted"))) if f("symbolic.mean_r2_val_accepted") else None,
                stages_total=stage_tot, n_substituted_mean=float(np.mean(f("symbolic.n_substituted"))),
                rise_after_pruning=float(np.mean(f("after_pruning.shape.mean_id"))),
                rise_after_substitution=float(np.mean(f("after_substitution.shape.mean_id"))),
                curve_error_after_pruning=float(np.mean(f("after_pruning.curve_error"))),
                curve_error_after_substitution=float(np.mean(f("after_substitution.curve_error"))))


def p0():
    recs = []
    for f in sorted(R12.glob("p0_lane*.jsonl")):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not recs:
        return None
    E = {}
    for r in recs:
        if r.get("phase") == "ensemble" and r.get("status", "ok") == "ok":
            E.setdefault(r["arm"], {}).setdefault(r["replicate"], float(r["curve_error"]))
    if "kan" not in E or "mlp" not in E:
        return dict(records=len(recs))
    out = dict(curve_error=paired(E["kan"], E["mlp"], margin=0.005))
    for k in ("ibs", "harrell_c"):
        a = {r["replicate"]: float(r[k]) for r in recs if r.get("phase") == "ensemble" and r["arm"] == "kan"}
        b = {r["replicate"]: float(r[k]) for r in recs if r.get("phase") == "ensemble" and r["arm"] == "mlp"}
        out[k] = paired(a, b)
    return out


def main():
    recs = load()
    S = {}
    rid = lambda m, fr="bspline", path="metrics.shape.mean_id": by_rep(recs, "S-Add", m, fr, path)  # noqa: E731
    S["H_R12"] = paired(rid("b2s_cox"), rid("afcm_v3"), margin=0.05)
    S["H_R12"]["interpretation"] = interpret(S["H_R12"])
    S["H_R12_sensitivity"] = {
        "single_seed_b2s_vs_afcm_v3": paired(rid("b2s_cox", path="metrics_seed0.shape.mean_id"), rid("afcm_v3"), 0.05),
        "b2s_vs_afcm_v0": paired(rid("b2s_cox"), rid("afcm_v0"), 0.05),
        "raw_rise_b2s_vs_afcm_v3": paired(rid("b2s_cox", path="metrics.shape.mean_raw"),
                                          rid("afcm_v3", path="metrics.shape.mean_raw"), 0.05),
        "phi1_b2s_vs_afcm_v3": paired(rid("b2s_cox", path="metrics.shape.phi1.rise_id"),
                                      rid("afcm_v3", path="metrics.shape.phi1.rise_id"), 0.05),
        "phi2_b2s_vs_afcm_v3": paired(rid("b2s_cox", path="metrics.shape.phi2.rise_id"),
                                      rid("afcm_v3", path="metrics.shape.phi2.rise_id"), 0.05)}
    for k, v in S["H_R12_sensitivity"].items():
        v["interpretation"] = interpret(v)
    ce = "metrics.curve_error"
    S["secondary"] = {
        "S-Add curve error b2s_cox - afcm_v3": paired(rid("b2s_cox", path=ce), rid("afcm_v3", path=ce)),
        "S-Add RISE b2s_cox - b1_fpca1_cox": paired(rid("b2s_cox"), rid("b1_fpca1_cox")),
        "S-Int curve error b2deep_cox - b2s_cox": paired(by_rep(recs, "S-Int", "b2deep_cox", path=ce),
                                                         by_rep(recs, "S-Int", "b2s_cox", path=ce)),
        "S-Int curve error b1_fpca2_cox - b1_fpca1_cox": paired(by_rep(recs, "S-Int", "b1_fpca2_cox", path=ce),
                                                                by_rep(recs, "S-Int", "b1_fpca1_cox", path=ce)),
        "S-Sparse RISE b2s_cox pace - bspline": paired(by_rep(recs, "S-Sparse", "b2s_cox", "pace"),
                                                       by_rep(recs, "S-Sparse", "b2s_cox", "bspline")),
        "S-Sparse curve error b2s_cox pace - bspline": paired(by_rep(recs, "S-Sparse", "b2s_cox", "pace", ce),
                                                              by_rep(recs, "S-Sparse", "b2s_cox", "bspline", ce)),
        "S-Sparse RISE b1_fpca1_cox pace - bspline": paired(by_rep(recs, "S-Sparse", "b1_fpca1_cox", "pace"),
                                                            by_rep(recs, "S-Sparse", "b1_fpca1_cox", "bspline")),
        "S-Sparse curve error flat_ae - b2s_cox (bspline)": paired(by_rep(recs, "S-Sparse", "flat_ae", "raw", ce),
                                                                   by_rep(recs, "S-Sparse", "b2s_cox", "bspline", ce)),
    }
    for sub in ("lognormal", "loglogistic"):
        for metric in ("curve_error", "dcal_p"):
            S["secondary"][f"S-Misspec-{sub} {metric} b2s_aft_rc - b2s_cox"] = paired(
                by_rep(recs, "S-Misspec", "b2s_aft_rc", path=f"metrics.{metric}", sub=sub),
                by_rep(recs, "S-Misspec", "b2s_cox", path=f"metrics.{metric}", sub=sub))
    S["selection_symbolic"] = {sc: selection_symbolic(recs, sc) for sc in ("S-Add", "S-Sparse")}
    S["P0"] = p0()
    T = summary_table(recs)
    S["n_records"], S["n_failed"] = len(recs), int(sum(r.get("status") == "failed" for r in recs))
    S["n_dropped"] = int(sum(r.get("status") == "dropped_budget" for r in recs))
    S["failed"] = [dict(scenario=r["scenario"], sub=r.get("sub"), replicate=r["replicate"], model=r["model"],
                        front=r["front"], error=r.get("error")) for r in recs if r.get("status") == "failed"]
    TAB.mkdir(parents=True, exist_ok=True)
    T.to_csv(TAB / "r12_summary_by_model.csv", index=False)
    (ROOT / "reports/round12_summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps({k: S[k] for k in ("H_R12", "n_records", "n_failed", "n_dropped")}, indent=1, default=float))


if __name__ == "__main__":
    main()
