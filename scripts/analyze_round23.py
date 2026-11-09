"""Round 23 analysis (rules: docs/round23_preregistration.md §4, D160). Reads reports/round23/results_lane*.jsonl only.

Writes reports/round23/analysis_summary.json and docs/round23_results.md. Intervals: Nadeau-Bengio corrected 95% t
(analyse_round13.nb), frac = mean n_test / n_train of the data set.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_round13 import nb  # noqa: E402

BASE = ROOT / "reports/round23"
CLIN = ("pbc_E2", "pbc_E1", "prothro", "aids", "renal", "heart_valve")
CM = ("FD002", "FD004")
MAX_FAIL = {"cm": 2, "real": 2}
NI_MARGIN = 0.005


def load(base=BASE):
    recs = []
    for f in sorted(base.glob("results_lane*.jsonl")):
        recs += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return recs


def group(recs, job, key):
    out = {}
    for r in recs:
        if r.get("status") == "ok" and r.get("job") == job:
            out.setdefault(r[key], {})[r["split"]] = r            # the last ok record of a split wins (none expected twice)
    return out


def col(R, model, metric):
    out = {}
    for s, r in R.items():
        m = r["models"].get(model, {})
        v = m.get(metric)
        if m.get("status", "ok") == "ok" and m.get("finite", True) and v is not None and np.isfinite(v):
            out[s] = float(v)
    return out


def contrast(R, a, b, metric, n_expected):
    A, B = col(R, a, metric), col(R, b, metric)
    frac = float(np.mean([r["n_test"] / r["n_train"] for r in R.values()])) if R else float("nan")
    res = nb(A, B, frac)
    res.update(frac=frac, n_missing_a=n_expected - len(A), n_missing_b=n_expected - len(B))
    return res


def evaluable(c, kind):
    return max(c["n_missing_a"], c["n_missing_b"]) <= MAX_FAIL[kind] and c.get("n", 0) >= 2


def means(R, metric, models):
    return {m: (float(np.mean(list(col(R, m, metric).values()))) if col(R, m, metric) else None) for m in models}


def final():
    recs = load()
    failed = [dict(job=r.get("job_name"), error=r.get("error")) for r in recs if r.get("status") != "ok"]
    CMR, CLR = group(recs, "cm", "fd"), group(recs, "real", "dataset")
    out = dict(n_records=len(recs), n_ok=sum(r.get("status") == "ok" for r in recs), failed_records=failed, cm={}, clinical={})
    cm_models = ("b2s_npaft", "landmark_cox", "landmark_cox_raw", "landmark_cox_smooth", "stack", "sum_aft", "fpca_aft")
    for fd in CM:
        R = CMR.get(fd, {})
        d = dict(n_splits=len(R), means={k: means(R, k, cm_models) for k in ("ibs", "harrell_c", "uno_c")},
                 dcal_pass={m: int(sum(v >= 0.05 for v in col(R, m, "dcal_p").values())) for m in cm_models},
                 landmark_chosen={s: r["models"]["landmark_cox"].get("chosen") for s, r in sorted(R.items())},
                 w={s: r["stack"]["w"] for s, r in sorted(R.items())},
                 model_failures={m: [s for s, r in R.items() if r["models"].get(m, {}).get("status", "ok") != "ok"]
                                 for m in cm_models},
                 contrasts={f"{a} - {b}": contrast(R, a, b, "ibs", 10) for a, b in
                            (("b2s_npaft", "landmark_cox"), ("b2s_npaft", "sum_aft"), ("b2s_npaft", "fpca_aft"),
                             ("stack", "landmark_cox"), ("b2s_npaft", "landmark_cox_raw"),
                             ("b2s_npaft", "landmark_cox_smooth"))},
                 c_contrasts={f"{a} - {b}": contrast(R, a, b, "harrell_c", 10) for a, b in
                              (("b2s_npaft", "landmark_cox"), ("stack", "landmark_cox"))})
        out["cm"][fd] = d
    cl_models = ("b2s_npaft", "landmark_cox", "stack", "b2s_npaft_sparse3")
    for ds in CLIN:
        R = CLR.get(ds, {})
        d = dict(n_splits=len(R), visits_median=float(np.median([r["visits_median"] for r in R.values()])) if R else None,
                 means={k: means(R, k, cl_models) for k in ("ibs_cond", "ibs", "harrell_c")},
                 w={s: r["stack"]["w"] for s, r in sorted(R.items())},
                 contrasts={f"{a} - {b}": contrast(R, a, b, "ibs_cond", 20) for a, b in
                            (("stack", "landmark_cox"), ("b2s_npaft", "landmark_cox"))},
                 c_contrasts={"b2s_npaft - landmark_cox": contrast(R, "b2s_npaft", "landmark_cox", "harrell_c", 20)})
        if ds == "renal":
            d["contrasts"].update({f"{a} - {b}": contrast(R, a, b, "ibs_cond", 20) for a, b in
                                   (("b2s_npaft", "b2s_npaft_sparse3"), ("b2s_npaft_sparse3", "landmark_cox"))})
        d["w_counts"] = {str(w): sum(v == w for v in d["w"].values()) for w in (0.0, 0.25, 0.5, 0.75, 1.0)}
        out["clinical"][ds] = d
    # ---- decisions (§4)
    ha = {}
    for fd in CM:
        c = out["cm"][fd]["contrasts"]["b2s_npaft - landmark_cox"]
        ev = evaluable(c, "cm")
        ha[fd] = dict(evaluable=ev, mean_diff=c.get("mean_diff"), upper=c.get("ci95", [None, None])[1],
                      met=bool(ev and c["mean_diff"] < 0 and c["ci95"][1] < 0))
    out["H_R23a"] = dict(per_dataset=ha, evaluable=all(v["evaluable"] for v in ha.values()),
                         supported=all(v["met"] for v in ha.values()))
    cb = out["clinical"]["renal"]["contrasts"].get("b2s_npaft - b2s_npaft_sparse3", dict(n_missing_a=20, n_missing_b=20))
    evb = evaluable(cb, "real")
    out["H_R23b"] = dict(evaluable=evb, mean_diff=cb.get("mean_diff"), ci95=cb.get("ci95"),
                         supported=bool(evb and cb["mean_diff"] < 0 and cb["ci95"][1] < 0))
    ni = {}
    for ds in CLIN:
        c = out["clinical"][ds]["contrasts"]["stack - landmark_cox"]
        ev = evaluable(c, "real")
        ni[ds] = dict(evaluable=ev, mean_diff=c.get("mean_diff"), upper=c.get("ci95", [None, None])[1],
                      met=bool(ev and c["ci95"][1] <= NI_MARGIN))
    cr = out["clinical"]["renal"]["contrasts"]["stack - landmark_cox"]
    out["H_R23c"] = dict(part_i=dict(per_cohort=ni, margin=NI_MARGIN, evaluable=all(v["evaluable"] for v in ni.values()),
                                     supported=all(v["met"] for v in ni.values())),
                         part_ii=dict(cohort="renal", mean_diff=cr.get("mean_diff"), ci95=cr.get("ci95"),
                                      evaluable=evaluable(cr, "real"),
                                      supported=bool(evaluable(cr, "real") and cr["ci95"][1] < 0)))
    (BASE / "analysis_summary.json").write_text(json.dumps(out, indent=1))
    (ROOT / "docs/round23_results.md").write_text(results_md(out))
    print(json.dumps({k: out[k] for k in ("n_records", "n_ok", "H_R23a", "H_R23b", "H_R23c")}, indent=1))
    return out


def f4(x):
    return "–" if x is None else f"{x:+.4f}"


def ci(c):
    return "–" if not c.get("ci95") else f"{c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] ({c['n_neg']}/{c['n']} < 0)"


def results_md(s):
    L = ["# Round 23 results (generated by `scripts/analyze_round23.py`; rules D160)", "",
         f"Records: {s['n_records']} ({s['n_ok']} ok, {len(s['failed_records'])} failed). Intervals: Nadeau–Bengio "
         "corrected 95%; differences are first − second model (negative = first model better for IBS).", "",
         f"**H-R23a (FD002 and FD004, untouched): {'supported' if s['H_R23a']['supported'] else 'not supported'}"
         f"{'' if s['H_R23a']['evaluable'] else ' (not evaluable)'}.**", ""]
    for fd, d in s["cm"].items():
        L += [f"## C-MAPSS {fd} Option B ({d['n_splits']} splits)", "", "| Model | IBS | Harrell C | D-cal passes |",
              "|---|---|---|---|"]
        for m, v in d["means"]["ibs"].items():
            if v is not None:
                L.append(f"| {m} | {v:.4f} | {d['means']['harrell_c'][m]:.4f} | {d['dcal_pass'][m]}/{d['n_splits']} |")
        L += ["", "| Contrast (IBS) | Mean [95% CI] |", "|---|---|"]
        L += [f"| {k} | {ci(c)} |" for k, c in d["contrasts"].items()]
        L += ["", f"Landmark version chosen: {d['landmark_chosen']}; blend weight w: {d['w']}", ""]
    L += [f"**H-R23b (renal history thinning): {'supported' if s['H_R23b']['supported'] else 'not supported'}"
          f"{'' if s['H_R23b']['evaluable'] else ' (not evaluable)'}.** "
          f"**H-R23c (i) non-inferiority of the blend on every cohort: {'supported' if s['H_R23c']['part_i']['supported'] else 'not supported'}; "
          f"(ii) blend better than landmark Cox on renal: {'supported' if s['H_R23c']['part_ii']['supported'] else 'not supported'}.**", "",
          "## Clinical cohorts (new partitions 5–24; previously analysed data)", "",
          "| Cohort | visits | NP-KAN-AFT | landmark Cox | blend | blend − landmark | NP − landmark | w = 1 / 0 |",
          "|---|---|---|---|---|---|---|---|"]
    for ds, d in s["clinical"].items():
        m = d["means"]["ibs_cond"]
        if m["landmark_cox"] is None:
            continue
        L.append(f"| {ds} | {d['visits_median']:.0f} | {m['b2s_npaft']:.4f} | {m['landmark_cox']:.4f} | {m['stack']:.4f} | "
                 f"{ci(d['contrasts']['stack - landmark_cox'])} | {ci(d['contrasts']['b2s_npaft - landmark_cox'])} | "
                 f"{d['w_counts']['1.0']} / {d['w_counts']['0.0']} |")
    r = s["clinical"]["renal"]
    if r["means"]["ibs_cond"].get("b2s_npaft_sparse3") is not None:
        L += ["", f"Renal thinning: NP full {r['means']['ibs_cond']['b2s_npaft']:.4f}, NP thinned to three visits "
                  f"{r['means']['ibs_cond']['b2s_npaft_sparse3']:.4f}; full − thinned "
                  f"{ci(r['contrasts']['b2s_npaft - b2s_npaft_sparse3'])}; thinned − landmark "
                  f"{ci(r['contrasts']['b2s_npaft_sparse3 - landmark_cox'])}."]
    if s["failed_records"]:
        L += ["", "Failed jobs: " + "; ".join(f"{f['job']}: {f['error']}" for f in s["failed_records"])]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    final()
