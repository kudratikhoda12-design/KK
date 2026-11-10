"""Round 24 analysis (rules: docs/round24_preregistration.md §4, D163). Reads reports/round24/results_lane*.jsonl (joint
model) and reports/round23/results_lane*.jsonl (NP-KAN-AFT, landmark Cox, blend on the same partitions; not refitted).

Writes reports/round24/analysis_summary.json and docs/round24_results.md. Intervals: Nadeau-Bengio corrected 95% t
(analyse_round13.nb), frac = mean n_test / n_train of the cohort.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_round23 as A23  # noqa: E402

BASE = ROOT / "reports/round24"
OUT_MD = ROOT / "docs/round24_results.md"
CLIN = A23.CLIN
N_SPLITS = 20
MAX_FAIL = 2
NI_MARGIN = 0.005
R23_MODELS = ("b2s_npaft", "landmark_cox", "stack")
JM_MODELS = ("jm", "jm_lin", "jm_ns2")
CONTRASTS = (("b2s_npaft", "jm"), ("landmark_cox", "jm"), ("stack", "jm"))
METRICS = ("ibs_cond", "ibs", "harrell_c", "uno_c")


def merged():
    """{cohort: {split: record}} with the Round-23 models and the Round-24 joint-model forms side by side. A partition is
    paired only if the cohort, test-set size and test events agree in both records."""
    r23 = A23.group(A23.load(ROOT / "reports/round23"), "real", "dataset")
    jm_all = A23.load(BASE)
    r24 = {}
    for r in jm_all:
        if r.get("job") == "real" and r.get("status") in ("ok", "fit_failed"):
            r24.setdefault(r["dataset"], {})[r["split"]] = r
    out, mismatch = {}, []
    for ds in CLIN:
        for k, a in r23.get(ds, {}).items():
            b = r24.get(ds, {}).get(k)
            if b is None:
                continue
            if (a["n"], a["n_test"], a["events_test"]) != (b["n"], b["n_test"], b["events_test"]):
                mismatch.append(f"{ds}_{k}")
                continue
            models = {m: a["models"][m] for m in R23_MODELS if m in a["models"]}
            models.update({m: b["models"][m] for m in JM_MODELS if m in b["models"]})
            out.setdefault(ds, {})[k] = dict(a, models=models, jm_status=b["status"], jm_seconds=b.get("r_seconds_total"),
                                             jm_record=b)
    return out, mismatch, jm_all


def failures(jm_all):
    out = []
    for r in jm_all:
        if r.get("status") == "failed":
            out.append(dict(job=r.get("job_name"), stage="job", error=r.get("error")))
        for m in JM_MODELS:
            mm = r.get("models", {}).get(m, {})
            if mm.get("status") == "failed" or mm.get("finite") is False:
                out.append(dict(job=r.get("job_name"), model=m, error=mm.get("error"), finite=mm.get("finite")))
    return out


def _rhat(Rd, agg):
    """Convergence of the chosen joint model: max Rhat over its association parameters, per partition, aggregated."""
    v = [max(r["models"]["jm"]["rhat_alphas"]) for r in Rd.values()
         if r["models"].get("jm", {}).get("rhat_alphas")]
    return agg(v) if v else None


def main():
    R, mismatch, jm_all = merged()
    s = dict(n_records=len(jm_all), mismatched_partitions=mismatch, failures=failures(jm_all), cohorts={})
    for ds in CLIN:
        Rd = R.get(ds, {})
        c = dict(n_splits=len(Rd), visits_median=float(np.median([r["visits_median"] for r in Rd.values()])) if Rd else None,
                 means={met: A23.means(Rd, met, R23_MODELS + JM_MODELS) for met in METRICS},
                 dcal_pass={m: int(sum(1 for r in Rd.values() if r["models"].get(m, {}).get("dcal_p", -1) >= 0.05))
                            for m in R23_MODELS + JM_MODELS},
                 chosen_form={f: int(sum(1 for r in Rd.values() if r["models"].get("jm", {}).get("chosen") == f))
                              for f in ("lin", "ns2")},
                 jm_seconds_median=float(np.median([r["jm_seconds"] for r in Rd.values() if r["jm_seconds"]])) if Rd else None,
                 rhat_max_median=_rhat(Rd, np.median), n_rhat_above_1_1=_rhat(Rd, lambda v: int(np.sum(np.array(v) > 1.1))),
                 contrasts={}, c_contrasts={})
        for a, b in CONTRASTS:
            c["contrasts"][f"{a} - jm"] = A23.contrast(Rd, a, b, "ibs_cond", N_SPLITS)
            c["c_contrasts"][f"{a} - jm"] = A23.contrast(Rd, a, b, "harrell_c", N_SPLITS)
        s["cohorts"][ds] = c
    # H-R24a: renal, NP-KAN-AFT - JM conditional IBS: mean < 0 and NB upper bound < 0
    ca = s["cohorts"]["renal"]["contrasts"]["b2s_npaft - jm"]
    ev = A23.evaluable(ca, "real")
    s["H_R24a"] = dict(evaluable=ev, mean_diff=ca.get("mean_diff"), ci95=ca.get("ci95"),
                       supported=bool(ev and ca["mean_diff"] < 0 and ca["ci95"][1] < 0) if ev else None)
    # H-R24b: on every cohort NB upper bound of NP-KAN-AFT - JM conditional IBS <= +0.005
    per = {}
    for ds in CLIN:
        cb = s["cohorts"][ds]["contrasts"]["b2s_npaft - jm"]
        e = A23.evaluable(cb, "real")
        per[ds] = dict(evaluable=e, mean_diff=cb.get("mean_diff"), upper=cb["ci95"][1] if "ci95" in cb else None,
                       met=bool(e and cb["ci95"][1] <= NI_MARGIN) if e else None)
    all_ev = all(p["evaluable"] for p in per.values())
    s["H_R24b"] = dict(per_cohort=per, evaluable=all_ev, supported=bool(all(p["met"] for p in per.values())) if all_ev else None)
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "analysis_summary.json").write_text(json.dumps(s, indent=1, default=float))
    write_md(s)
    print(json.dumps({k: s[k] for k in ("H_R24a", "H_R24b", "mismatched_partitions")}, indent=1, default=float))
    print("failures:", len(s["failures"]))


def fmt(x, f="+.4f"):
    return "n/a" if x is None else format(x, f)


def write_md(s):
    L = ["# Round 24 results (D163): a Bayesian joint model as comparator on the clinical cohorts", "",
         "Generated by `scripts/analyze_round24.py` from `reports/round24/results_lane*.jsonl` and the Round-23 records "
         "(same partitions k = 5..24). Rules: `docs/round24_preregistration.md` §4. Intervals: Nadeau–Bengio 95%.", "",
         f"Joint-model records: {s['n_records']}; failed fits or non-finite curves: {len(s['failures'])}; "
         f"partitions not paired: {len(s['mismatched_partitions'])}.", "",
         "## Hypotheses", "",
         "| Hypothesis | Result | Verdict |", "|---|---|---|"]
    a = s["H_R24a"]
    L.append(f"| H-R24a renal: NP-KAN-AFT − JM, conditional IBS upper bound < 0 | {fmt(a['mean_diff'])} "
             f"[{fmt((a['ci95'] or [None, None])[0])}, {fmt((a['ci95'] or [None, None])[1])}] | "
             f"{'not evaluable' if a['supported'] is None else ('supported' if a['supported'] else 'not supported')} |")
    b = s["H_R24b"]
    ups = ", ".join(f"{ds} {fmt(p['upper'])}" for ds, p in b["per_cohort"].items())
    L.append(f"| H-R24b every cohort: NP-KAN-AFT − JM upper bound ≤ +0.005 | {ups} | "
             f"{'not evaluable' if b['supported'] is None else ('supported' if b['supported'] else 'not supported')} |")
    L += ["", "## Conditional IBS (mean over partitions) and contrasts with the joint model", "",
          "| Cohort | Splits | NP-KAN-AFT | Landmark Cox | Blend | JM (chosen) | JM lin | JM ns2 | lin / ns2 chosen |"
          " NP-KAN-AFT − JM | Landmark Cox − JM | Blend − JM |", "|" + "---|" * 12]
    for ds, c in s["cohorts"].items():
        m = c["means"]["ibs_cond"]
        con = [c["contrasts"][f"{x} - jm"] for x in R23_MODELS]
        cells = [f"{fmt(q.get('mean_diff'))} [{fmt((q.get('ci95') or [None, None])[0])}, {fmt((q.get('ci95') or [None, None])[1])}]"
                 for q in con]
        L.append(f"| {ds} | {c['n_splits']} | " + " | ".join(fmt(m.get(x), ".4f") for x in R23_MODELS + JM_MODELS) +
                 f" | {c['chosen_form']['lin']} / {c['chosen_form']['ns2']} | " + " | ".join(cells) + " |")
    L += ["", "## Harrell C and D-calibration", "",
          "| Cohort | C NP-KAN-AFT | C landmark Cox | C JM | NP-KAN-AFT − JM (C) | D-cal p ≥ 0.05: NP / landmark / JM |",
          "|---|---|---|---|---|---|"]
    for ds, c in s["cohorts"].items():
        m = c["means"]["harrell_c"]
        q = c["c_contrasts"]["b2s_npaft - jm"]
        L.append(f"| {ds} | {fmt(m.get('b2s_npaft'), '.3f')} | {fmt(m.get('landmark_cox'), '.3f')} | {fmt(m.get('jm'), '.3f')} | "
                 f"{fmt(q.get('mean_diff'))} [{fmt((q.get('ci95') or [None, None])[0])}, {fmt((q.get('ci95') or [None, None])[1])}] | "
                 f"{c['dcal_pass']['b2s_npaft']} / {c['dcal_pass']['landmark_cox']} / {c['dcal_pass']['jm']} |")
    L += ["", "## Convergence of the chosen joint model (max Rhat of the association parameters)", "",
          "| Cohort | Median over partitions | Partitions with Rhat > 1.1 |", "|---|---|---|"]
    for ds, c in s["cohorts"].items():
        L.append(f"| {ds} | {fmt(c['rhat_max_median'], '.3f')} | {c['n_rhat_above_1_1']} |")
    if s["failures"]:
        L += ["", "## Failed fits (recorded, not refitted)", ""] + [f"- {f}" for f in s["failures"]]
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
