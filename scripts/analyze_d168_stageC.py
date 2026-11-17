"""D168 Stage C / D170: analysis of the paired comparison (written and committed before any Stage-C seed is used).

Reads reports/d168/stageC/results_lane*.jsonl. Primary hypothesis H-R25a: d = mean identifiable RISE of B2-S Cox minus
that of the reference AFCM, per replicate; equivalence within +-0.05 by the paired TOST (two one-sided t tests at
alpha 0.05; identical to the 90% CI lying inside +-0.05), computed with scripts/analyse_round12.paired, the function
used for H-R12. Reported in every case: n, means, mean difference, SD, 95% and 90% CIs, paired-t and Wilcoxon
signed-rank p, counts of d < 0 and d > 0. Directional statements only if the 95% CI excludes 0 (secondary).
Failures: a method's replicate failed if its record says so (error, non-finite, or reference not fully converged).
Primary analysis on complete pairs. Not evaluable if more than 10% of the 110 replicates failed for either method
(more than 11). Sensitivity: a failed method's replicate is assigned the worst (largest) mean identifiable RISE
observed for that method. Secondary: per-channel identifiable RISE, raw RISE, survival-curve error, IBS, Harrell C,
Uno C (paired, no decision rule); run times.
Outputs: reports/d168/stageC/analysis_summary.json, docs/d168_stageC_results.md.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_round12 import paired  # noqa: E402

BASE = ROOT / "reports/d168/stageC"
OUT_MD = ROOT / "docs/d168_stageC_results.md"
N_PLANNED = 110
MAX_FAIL = 11                      # more than 10% of 110
MARGIN = 0.05
METHODS = ("b2s_cox", "reference")


def load(base=BASE):
    recs = {}
    for f in sorted(base.glob("results_lane*.jsonl")):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                recs[r["seed"]] = r
    return recs


def get(r, method, path):
    x = r.get(method, {})
    for k in path.split("."):
        x = x.get(k, {}) if isinstance(x, dict) else {}
    return x if isinstance(x, (int, float)) else None


def ok(r, method):
    return r.get("status") == "ok" and r.get(method, {}).get("status") == "ok"


def analyse(recs):
    seeds = sorted(recs)
    fails = {m: [s for s in seeds if not ok(recs[s], m)] for m in METHODS}
    pairs = [s for s in seeds if all(ok(recs[s], m) for m in METHODS)]
    val = lambda m, path, ss: {s: get(recs[s], m, path) for s in ss if get(recs[s], m, path) is not None}  # noqa: E731
    a, b = val("b2s_cox", "metrics.shape.mean_id", pairs), val("reference", "metrics.shape.mean_id", pairs)
    prim = paired(a, b, margin=MARGIN)
    prim.pop("reps", None)
    evaluable = all(len(fails[m]) <= MAX_FAIL for m in METHODS) and len(seeds) == N_PLANNED
    s = dict(n_planned=N_PLANNED, n_records=len(seeds), n_complete_pairs=len(pairs),
             failures={m: dict(n=len(fails[m]), seeds=fails[m],
                               errors=sorted({str(recs[x].get(m, {}).get("error") or recs[x].get("error"))[:200]
                                              for x in fails[m]})) for m in METHODS},
             evaluable=bool(evaluable), primary=prim)
    if evaluable and prim.get("n", 0) >= 2:
        eq = bool(prim.get("equivalent"))
        lo, hi = prim["ci95"]
        direction = "B2-S Cox lower (better) than the reference AFCM" if hi < 0 else (
            "B2-S Cox higher (worse) than the reference AFCM" if lo > 0 else "no direction shown (95% CI includes 0)")
        s["H_R25a"] = dict(equivalent=eq, verdict="supported (equivalent within +-0.05)" if eq else "not supported",
                           direction=direction)
    else:
        s["H_R25a"] = dict(equivalent=None, verdict="not evaluable", direction=None)
    # sensitivity: failed replicates set to the method's worst observed value
    worst = {m: max([get(recs[x], m, "metrics.shape.mean_id") for x in seeds if ok(recs[x], m)
                     and get(recs[x], m, "metrics.shape.mean_id") is not None] or [np.nan]) for m in METHODS}
    imp = {m: {x: (get(recs[x], m, "metrics.shape.mean_id") if ok(recs[x], m) else worst[m]) for x in seeds} for m in METHODS}
    sens = paired({k: v for k, v in imp["b2s_cox"].items() if v is not None and np.isfinite(v)},
                  {k: v for k, v in imp["reference"].items() if v is not None and np.isfinite(v)}, margin=MARGIN)
    sens.pop("reps", None)
    s["sensitivity_worst_imputation"] = sens
    sec = {}
    for path in ("metrics.shape.phi1.rise_id", "metrics.shape.phi2.rise_id", "metrics.shape.mean_raw", "metrics.curve_error",
                 "metrics.ibs", "metrics.harrell_c", "metrics.uno_c"):
        p = paired(val("b2s_cox", path, pairs), val("reference", path, pairs))
        p.pop("reps", None)
        sec[path] = p
    s["secondary"] = sec
    s["dcal_pass"] = {m: int(sum(1 for x in pairs if (get(recs[x], m, "metrics.dcal_p") or 0) >= 0.05)) for m in METHODS}
    s["seconds_median"] = {m: float(np.median([recs[x][m].get("seconds", np.nan) for x in seeds if m in recs[x]]))
                           if seeds else None for m in METHODS}
    s["reference_outer_conv"] = sorted({str(recs[x].get("reference", {}).get("info", {}).get("outer_conv")) for x in seeds})
    s["tied_event_times_max"] = max([recs[x].get("tied_event_times_train", 0) for x in seeds] or [0])
    return s


def f(x, spec="+.4f"):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def write_md(s):
    p = s["primary"]
    L = ["# D168 Stage C results: B2-S Cox against the mgcv reference AFCM (H-R25a)", "",
         "Generated by `scripts/analyze_d168_stageC.py` from `reports/d168/stageC/results_lane*.jsonl`. Rules: "
         "`docs/d170_stageC_preregistration.md`.", "",
         f"Replicates recorded: {s['n_records']} of {s['n_planned']}; complete pairs: {s['n_complete_pairs']}; failures: "
         f"B2-S Cox {s['failures']['b2s_cox']['n']}, reference AFCM {s['failures']['reference']['n']}; "
         f"evaluable: {s['evaluable']}.", "",
         "## Primary: mean identifiable RISE, B2-S Cox − reference AFCM", "",
         "| n | Mean B2-S | Mean reference | Mean difference | 95% CI | 90% CI | TOST p (±0.05) | paired-t p | Wilcoxon p "
         "| d < 0 / d > 0 |", "|---|---|---|---|---|---|---|---|---|---|"]
    if p.get("n", 0) >= 2:
        L.append(f"| {p['n']} | {f(p['mean_a'], '.4f')} | {f(p['mean_b'], '.4f')} | {f(p['mean_diff'])} | "
                 f"[{f(p['ci95'][0])}, {f(p['ci95'][1])}] | [{f(p['ci90'][0])}, {f(p['ci90'][1])}] | {f(p.get('p_tost'), '.4g')} | "
                 f"{f(p['p_t'], '.4g')} | {f(p['p_wilcoxon'], '.4g')} | {p['n_neg']} / {p['n_pos']} |")
    h = s["H_R25a"]
    L += ["", f"**H-R25a: {h['verdict']}.** Direction (secondary, 95% CI): {h['direction']}.", "",
          "## Secondary (paired, no decision rule)", "", "| Outcome | n | Mean B2-S | Mean reference | Difference | 95% CI |",
          "|---|---|---|---|---|---|"]
    for k, q in s["secondary"].items():
        if q.get("n", 0) >= 2:
            L.append(f"| {k} | {q['n']} | {f(q['mean_a'], '.4f')} | {f(q['mean_b'], '.4f')} | {f(q['mean_diff'])} | "
                     f"[{f(q['ci95'][0])}, {f(q['ci95'][1])}] |")
    q = s["sensitivity_worst_imputation"]
    L += ["", "## Sensitivity: failed replicates set to the method's worst value", "",
          f"n {q.get('n')}, mean difference {f(q.get('mean_diff'))}, 95% CI "
          f"[{f((q.get('ci95') or [None, None])[0])}, {f((q.get('ci95') or [None, None])[1])}], TOST p {f(q.get('p_tost'), '.4g')}.",
          "", "## Failures and run time", ""]
    for m in METHODS:
        L.append(f"- {m}: {s['failures'][m]['n']} failed ({', '.join(map(str, s['failures'][m]['seeds'])) or 'none'}); "
                 f"median {f(s['seconds_median'][m], '.0f')} s per replicate.")
    L.append(f"- Reference REML convergence messages: {', '.join(s['reference_outer_conv'])}. Tied training event times "
             f"(maximum over replicates): {s['tied_event_times_max']}.")
    L.append(f"- D-calibration p ≥ 0.05 (complete pairs): B2-S {s['dcal_pass']['b2s_cox']}, reference {s['dcal_pass']['reference']}.")
    OUT_MD.write_text("\n".join(L) + "\n")


def main():
    s = analyse(load())
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "analysis_summary.json").write_text(json.dumps(s, indent=1, default=float))
    write_md(s)
    print(json.dumps({k: s[k] for k in ("n_records", "n_complete_pairs", "evaluable", "H_R25a")}, indent=1, default=float))


if __name__ == "__main__":
    main()
