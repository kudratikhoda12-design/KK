"""Round 21 analysis (D153; rules in docs/round21_preregistration.md §3).

reports/round21/results_lane*.jsonl (+ Round-13 and Round-15 records on the same splits) ->
reports/round21/analysis_summary.json and docs/round21_results.md.
Every contrast: plain paired t (analyse_round12.paired) and Nadeau-Bengio corrected interval (analyse_round13.nb, variance
factor 1/J + mean n_test/n_train). Conditional IPCW IBS is the primary metric for H-R21a/b; C-MAPSS uses the Round-13
metrics (IBS, Harrell C, Uno C, D-cal).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_round12 import paired  # noqa: E402
from analyse_round13 import nb  # noqa: E402

R21 = ROOT / "reports/round21"
DATASETS = ("pbc_E2", "pbc_E1", "prothro", "aids")
FAM = ("cox", "waft", "lnaft", "npaft")
METRICS = ("ibs_cond", "ibs", "harrell_c", "uno_c")
R13_MAP = {"b2s_cox": "r13_b2s_cox", "b2s_aft": "r13_b2s_aft", "landmark_cox": "landmark_cox", "afcm_v3": "afcm_v3",
           "fpca_cox": "fpca_cox", "lme2_cox": "lme2_cox", "sum_cox": "sum_cox", "sum_rsf": "sum_rsf"}


def load(path, pat="results_lane*.jsonl"):
    out = []
    for f in sorted(path.glob(pat)):
        out += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return out


def baselines():
    """Round-13 single-landmark records and Round-15 real records on the same (dataset, split): {ds: {model: {k: m}}}."""
    B = {ds: {} for ds in DATASETS}
    for r in load(ROOT / "reports/round13"):
        if r.get("job") == "single" and r.get("status") == "ok" and r.get("model") in R13_MAP and r["dataset"] in B:
            B[r["dataset"]].setdefault(R13_MAP[r["model"]], {}).setdefault(r["split"], r["metrics"])
    for r in load(ROOT / "reports/round15"):
        if r.get("job") == "real" and r.get("status") == "ok" and r.get("dataset") in B:
            B[r["dataset"]].setdefault("r15_b2s_anchor_cox", {})[r["split"]] = r["models"]["b2s_anchor_cox"]
    return B


def both(a, b, frac):
    return dict(paired=paired(a, b), nb=nb(a, b, frac))


def col(R, model, metric, src="models"):
    return {k: r[src][model][metric] for k, r in R.items() if model in r[src]}


def finite_rec(m):
    return all(np.isfinite(m[k]) for k in ("ibs_cond", "ibs", "harrell_c", "uno_c", "dcal_p")) and m.get("finite", True)


def clinical(recs, B):
    out = {}
    for ds in DATASETS:
        R = {r["split"]: r for r in recs if r.get("job") == "real" and r["dataset"] == ds and r["status"] == "ok"}
        if not R:
            continue
        frac = float(np.mean([r["n_test"] / r["n_train"] for r in R.values()]))
        models = sorted({m for r in R.values() for m in r["models"]})
        means = {m: {k: float(np.mean(list(col(R, m, k).values()))) for k in METRICS + ("dcal_p",)} for m in models}
        dcal_pass = {m: int(sum(v >= 0.05 for v in col(R, m, "dcal_p").values())) for m in models}
        base = {m: {k: {s: v[k] for s, v in B[ds].get(m, {}).items()} for k in METRICS} for m in B[ds]}
        means.update({m: {k: float(np.mean(list(base[m][k].values()))) for k in METRICS if base[m][k]} for m in base})
        con = {}
        for a, refs in (("b2s_npaft", ("b2s_cox", "b2s_waft", "b2s_lnaft", "landmark_cox")),
                        ("b2s_lnaft", ("b2s_cox", "b2s_waft", "landmark_cox")),
                        ("b2s_anchor_npaft_cross", ("b2s_npaft", "landmark_cox", "r15_b2s_anchor_cox")),
                        ("b2s_anchor_cox_cross", ("b2s_cox", "landmark_cox", "r15_b2s_anchor_cox")),
                        ("b2s_anchor_lnaft_cross", ("b2s_lnaft", "landmark_cox")),
                        ("b2s_cox", ("landmark_cox", "afcm_v3", "fpca_cox", "lme2_cox", "sum_cox", "sum_rsf"))):
            for b in refs:
                for k in METRICS:
                    av = col(R, a, k)
                    bv = col(R, b, k) if b in means and any(b in r["models"] for r in R.values()) else base.get(b, {}).get(k, {})
                    if av and bv:
                        con[f"{k} {a} - {b}"] = both(av, bv, frac)
        gates = {}
        for f in ("cox", "lnaft", "npaft"):
            g = {s: r["gates"][f"b2s_anchor_{f}_cross"] for s, r in R.items()}
            nc = [s for s, x in g.items() if not x["confirmed"]]
            gates[f] = dict(n_confirmed=sum(x["confirmed"] for x in g.values()), n=len(g),
                            confirmed_splits=sorted(s for s, x in g.items() if x["confirmed"]),
                            fallback_exact=all(g[s]["fallback_max_abs_diff_S"] == 0.0 and g[s]["fallback_max_abs_diff_risk"] == 0.0
                                               for s in nc),
                            fallback_max_abs_diff=max([max(g[s]["fallback_max_abs_diff_S"], g[s]["fallback_max_abs_diff_risk"])
                                                       for s in nc], default=0.0),
                            n_h1_favours=sum(x["dirs"][0]["loss_anchored"] < x["dirs"][0]["loss_ref"] for x in g.values()),
                            n_h2_favours=sum(x["dirs"][1]["loss_anchored"] < x["dirs"][1]["loss_ref"] for x in g.values()),
                            last_on_frequency=np.mean([x["last_on_h1"] for x in g.values()], 0).tolist(),
                            traj_on_frequency=np.mean([x["traj_on_h1"] for x in g.values()], 0).tolist())
        sym = {}
        for f in ("npaft", "cox"):
            S = {s: r["symbolic"][f"b2s_{f}"] for s, r in R.items() if f"b2s_{f}" in r.get("symbolic", {})}
            if S:
                sym[f] = dict(n=len(S), replacement_rate_mean=float(np.mean([x["replacement_rate"] for x in S.values()])),
                              n_rollback=sum(x["rollback"] for x in S.values()),
                              n_refit_fallback=sum(x["refit_fallback"] for x in S.values()),
                              ibs_cond_before_mean=float(np.mean([x["seed0_before"]["ibs_cond"] for x in S.values()])),
                              ibs_cond_after_mean=float(np.mean([x["after"]["ibs_cond"] for x in S.values()])),
                              after_minus_before=both({s: x["after"]["ibs_cond"] for s, x in S.items()},
                                                      {s: x["seed0_before"]["ibs_cond"] for s, x in S.items()}, frac),
                              formulas={s: [dict(channel_index=e["i"], layer=e["layer"], formula=e["formula"], stage=e["stage"],
                                                 r2_val=e["r2_val"]) for e in x["edges"] if e["accepted"]][:12]
                                        for s, x in S.items()})
        repro = {}
        for new, old in (("b2s_cox", "r13_b2s_cox"), ("b2s_waft", "r13_b2s_aft")):
            d = [abs(R[s]["models"][new][k] - base[old][k][s]) for s in R for k in METRICS if s in base.get(old, {}).get(k, {})]
            repro[f"{new} vs Round-13 {old[4:]}"] = dict(n_values=len(d), max_abs_diff=float(max(d)) if d else None)
        secs = {m: float(np.mean([r["models"][m]["seconds"] for r in R.values()])) for m in models}
        out[ds] = dict(n_splits=len(R), frac=frac, means=means, dcal_pass=dcal_pass, contrasts=con, gates=gates,
                       symbolic=sym, reproduction=repro, seconds_mean=secs,
                       n_val=[R[s]["n_val"] for s in sorted(R)], n_halves=[[R[s]["n_h1"], R[s]["n_h2"]] for s in sorted(R)])
    return out


def cmapss(recs):
    R = {r["split"]: r for r in recs if r.get("job") == "cmapss" and r["status"] == "ok"}
    if not R:
        return {}
    models = sorted({m for r in R.values() for m in r["models"]})
    frac = float(np.mean([r["n_test"] / r["n_train"] for r in R.values()]))
    means = {m: {k: float(np.mean(list(col(R, m, k).values()))) for k in ("ibs", "harrell_c", "uno_c", "dcal_p", "dcal_chi2")}
             for m in models}
    npass = {m: int(sum(v >= 0.05 for v in col(R, m, "dcal_p").values())) for m in models}
    con = {f"{k} {a} - b2s_cox": both(col(R, a, k), col(R, "b2s_cox", k), frac)
           for a in ("b2s_npaft", "b2s_lnaft", "b2s_waft", "b2s_anchor_npaft_cross") for k in ("ibs", "harrell_c", "dcal_chi2")}
    g = {s: r["gates"]["b2s_anchor_npaft_cross"] for s, r in R.items()}
    nc = [s for s, x in g.items() if not x["confirmed"]]
    gate = dict(n_confirmed=sum(x["confirmed"] for x in g.values()), n=len(g),
                fallback_exact=all(g[s]["fallback_max_abs_diff_S"] == 0.0 and g[s]["fallback_max_abs_diff_risk"] == 0.0 for s in nc))
    crit = {}
    c = means["b2s_cox"]
    for m in ("b2s_npaft", "b2s_lnaft"):
        better_cal = bool(means[m]["dcal_chi2"] < c["dcal_chi2"] and npass[m] >= npass["b2s_cox"])
        crit[m] = dict(dcal_improved=better_cal, ibs_le_0055=bool(means[m]["ibs"] <= 0.055),
                       harrell_ge_088=bool(means[m]["harrell_c"] >= 0.88),
                       relative_ibs_ok=bool(means[m]["ibs"] <= c["ibs"] + 0.005),
                       relative_c_ok=bool(means[m]["harrell_c"] >= c["harrell_c"] - 0.01))
        crit[m]["owner_criterion"] = crit[m]["dcal_improved"] and crit[m]["ibs_le_0055"] and crit[m]["harrell_ge_088"]
        crit[m]["relative_criterion"] = crit[m]["dcal_improved"] and crit[m]["relative_ibs_ok"] and crit[m]["relative_c_ok"]
    secs = {m: float(np.mean([r["models"][m]["seconds"] for r in R.values()])) for m in models}
    return dict(n_splits=len(R), means=means, dcal_pass=npass, contrasts=con, gate=gate, criterion=crit, seconds_mean=secs,
                n_train=[R[s]["n_train"] for s in sorted(R)], n_test=[R[s]["n_test"] for s in sorted(R)])


def hypotheses(C, CM):
    h = {}
    E2 = C.get("pbc_E2", {})
    finite = {ds: all(finite_rec(r["models"]["b2s_npaft"]) for r in LOADED if r.get("job") == "real" and r["dataset"] == ds
                      and r["status"] == "ok") for ds in DATASETS}
    n_np = sum(1 for r in LOADED if r.get("job") == "real" and r["status"] == "ok" and "b2s_npaft" in r["models"])
    a = dict(part1_all_finite=bool(all(finite.values()) and n_np == 20), n_npaft_fits=n_np)
    if E2:
        d_cox = E2["contrasts"]["ibs_cond b2s_npaft - b2s_cox"]["paired"].get("mean_diff", float("nan"))
        d_waft = E2["contrasts"]["ibs_cond b2s_npaft - b2s_waft"]["paired"].get("mean_diff", float("nan"))
        a.update(part2_mean_vs_cox=d_cox, part2_mean_vs_waft=d_waft, part2=bool(d_cox <= 0.005 and d_waft <= 0.0),
                 part3_dcal_pass=E2["dcal_pass"]["b2s_npaft"], part3=bool(E2["dcal_pass"]["b2s_npaft"] >= 4))
        a["supported"] = bool(a["part1_all_finite"] and a["part2"] and a["part3"])
    h["H_R21a"] = a
    b = {}
    for ds, X in C.items():
        dn = X["contrasts"]["ibs_cond b2s_anchor_npaft_cross - b2s_npaft"]["paired"].get("mean_diff", float("nan"))
        dc = X["contrasts"]["ibs_cond b2s_anchor_cox_cross - b2s_cox"]["paired"].get("mean_diff", float("nan"))
        b[ds] = dict(npaft_mean_shift=dn, cox_mean_shift=dc, npaft_ok=bool(dn <= 0.005), cox_ok=bool(dc <= 0.005),
                     fallback_exact=bool(X["gates"]["npaft"]["fallback_exact"] and X["gates"]["cox"]["fallback_exact"]),
                     confirmed={f: f"{X['gates'][f]['n_confirmed']}/{X['gates'][f]['n']}" for f in ("cox", "lnaft", "npaft")})
    b["supported"] = bool(len(C) == 4 and all(v["npaft_ok"] and v["cox_ok"] and v["fallback_exact"] for v in b.values()
                                              if isinstance(v, dict)))
    h["H_R21b"] = b
    if CM:
        h["H_R21c"] = dict(criterion=CM["criterion"],
                           supported=bool(any(v["owner_criterion"] for v in CM["criterion"].values())),
                           secondary_relative_supported=bool(any(v["relative_criterion"] for v in CM["criterion"].values())))
    return h


def fmt(p):
    if p.get("n", 0) < 2:
        return "–"
    return f"{p['mean_diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"


def results_md(s):
    L = ["# Round 21 results (generated by scripts/analyze_round21.py from reports/round21/results_lane*.jsonl)", "",
         f"Records: {s['n_records']} ({s['n_failed']} failed). Rules: docs/round21_preregistration.md (D153).", ""]
    h = s["hypotheses"]
    a = h["H_R21a"]
    L += ["## H-R21a (NP-KAN-AFT, clinical)", "",
          f"- (1) all {a['n_npaft_fits']} clinical b2s_npaft fits finite: {a['part1_all_finite']}",
          f"- (2) pbc_E2 mean conditional IBS b2s_npaft − b2s_cox {a.get('part2_mean_vs_cox', float('nan')):+.4f} (≤ +0.005), "
          f"− b2s_waft {a.get('part2_mean_vs_waft', float('nan')):+.4f} (≤ 0): {a.get('part2')}",
          f"- (3) pbc_E2 D-cal p ≥ 0.05 on {a.get('part3_dcal_pass')} of 5 splits (≥ 4): {a.get('part3')}",
          f"- **{'Supported' if a.get('supported') else 'Not supported'}**", ""]
    L += ["## H-R21b (cross-confirmed anchoring, clinical)", "", "| Data set | NP-AFT shift | Cox shift | Exact fallback | Confirmed (cox / lnaft / npaft) |", "|---|---|---|---|---|"]
    for ds in DATASETS:
        v = h["H_R21b"].get(ds)
        if v:
            L.append(f"| {ds} | {v['npaft_mean_shift']:+.4f} | {v['cox_mean_shift']:+.4f} | {v['fallback_exact']} | "
                     f"{v['confirmed']['cox']} / {v['confirmed']['lnaft']} / {v['confirmed']['npaft']} |")
    L += ["", f"**{'Supported' if h['H_R21b']['supported'] else 'Not supported'}**", ""]
    if "H_R21c" in h:
        L += ["## H-R21c (C-MAPSS calibration)", ""]
        for m, v in h["H_R21c"]["criterion"].items():
            L.append(f"- {m}: {json.dumps(v)}")
        L += ["", f"**Owner criterion: {'Supported' if h['H_R21c']['supported'] else 'Not supported'}**; secondary relative: "
              f"{h['H_R21c']['secondary_relative_supported']}", ""]
    for ds, X in s["clinical"].items():
        L += [f"## {ds}: means over {X['n_splits']} splits", "", "| Model | Cond. IBS | Marg. IBS | Harrell C | Uno C | D-cal pass |",
              "|---|---|---|---|---|---|"]
        for m, v in sorted(X["means"].items()):
            L.append(f"| {m} | {v.get('ibs_cond', float('nan')):.4f} | {v.get('ibs', float('nan')):.4f} | "
                     f"{v.get('harrell_c', float('nan')):.4f} | {v.get('uno_c', float('nan')):.4f} | {X['dcal_pass'].get(m, '–')} |")
        L += ["", "| Contrast (conditional IBS) | Paired 95% CI | Nadeau–Bengio 95% CI | wins/losses |", "|---|---|---|---|"]
        for k, v in X["contrasts"].items():
            if k.startswith("ibs_cond"):
                p = v["paired"]
                L.append(f"| {k[9:]} | {fmt(p)} | {fmt(v['nb'])} | {p.get('n_neg', '–')}/{p.get('n_pos', '–')} |")
        L.append("")
    if s.get("cmapss"):
        X = s["cmapss"]
        L += ["## C-MAPSS FD001 Option B: means over 5 splits", "", "| Model | IBS | Harrell C | Uno C | D-cal χ² | D-cal pass |",
              "|---|---|---|---|---|---|"]
        for m, v in sorted(X["means"].items()):
            L.append(f"| {m} | {v['ibs']:.4f} | {v['harrell_c']:.4f} | {v['uno_c']:.4f} | {v['dcal_chi2']:.1f} | {X['dcal_pass'][m]} |")
        L.append("")
    return "\n".join(L) + "\n"


LOADED = []


def main():
    recs = load(R21)
    LOADED[:] = recs
    B = baselines()
    C = clinical(recs, B)
    CM = cmapss(recs)
    s = dict(n_records=len(recs), n_failed=sum(r.get("status") != "ok" for r in recs),
             failed=[dict(job=r.get("job_name"), error=r.get("error")) for r in recs if r.get("status") != "ok"],
             clinical=C, cmapss=CM)
    s["hypotheses"] = hypotheses(C, CM)
    (R21 / "analysis_summary.json").write_text(json.dumps(s, indent=1, default=float))
    (ROOT / "docs/round21_results.md").write_text(results_md(s))
    print(json.dumps(dict(n=s["n_records"], failed=s["n_failed"], hypotheses=s["hypotheses"]), indent=1, default=float))


if __name__ == "__main__":
    main()
