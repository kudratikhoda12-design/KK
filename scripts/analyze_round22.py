"""Round 22 analysis (D156; rules in docs/round22_preregistration.md §3). Written and committed before any registered
Round-22 job ran.

  python scripts/analyze_round22.py gate    reports/round22/dev/results_*.jsonl (afcmdev_22, _914, _966)
                                            -> reports/round22/afcm_gate.json (the D156 go rule)
  python scripts/analyze_round22.py final   reports/round22/results_*.jsonl (+ Round-12 b2s_cox records, CK-6d)
                                            -> reports/round22/analysis_summary.json, docs/round22_results.md

Statistics: plain paired t (analyse_round12.paired; replicates are independent data sets in H-R22a and H-R22c) and,
for the outer splits of one cohort, the Nadeau-Bengio corrected interval as well (analyse_round13.nb; factor
1/J + mean n_test/n_train). H-R22a uses the H-R12 rule (analyse_round12.interpret: 95% CI below 0 better, above 0
worse, otherwise TOST +-0.05 equivalent, otherwise inconclusive).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_round12 import by_rep, interpret, paired  # noqa: E402
from analyse_round12 import load as load12  # noqa: E402
from analyse_round13 import nb  # noqa: E402

BASE = ROOT / "reports/round22"
DEV_SEEDS = (22, 914, 966)
VARIANTS = ("afcm_reml", "afcm_reml_ens")             # fixed priority order (D156)
COHORTS = ("renal", "heart_valve")
METRICS = ("ibs_cond", "ibs", "harrell_c", "uno_c")
LAWS = ("bimodal", "studentt3", "logistic05")
CROSS = {"cox": "b2s_cox", "npaft": "b2s_npaft", "lnaft": "b2s_lnaft"}
BASELINES = ("landmark_cox", "fpca_cox", "lme2_cox", "sum_cox", "sum_rsf")


def load(path, pat="results_*.jsonl"):
    out = []
    for f in sorted(path.glob(pat)):
        out += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return out


def ok(recs, job):
    return [r for r in recs if r.get("job") == job and r.get("status") == "ok"]


# --------------------------------------------------------------------------------------------------- gate (Task 1)
def r15_laml():
    """The Round-15 C2 LAML choice per development seed (reproduction check of the REML pilot)."""
    out = {}
    for f in sorted((ROOT / "reports/round15/dev").glob("results_check_lane*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("job") == "afcmlaml":
                out[r["seed"]] = dict(lam_x=r["laml"]["lam_x"], lam_t=r["laml"]["lam_t"], shapes=r["laml"]["shapes"])
    return out


def gate():
    R = {r["seed"]: r for r in ok(load(BASE / "dev"), "afcmdev")}
    ref = r15_laml()
    rows = {}
    for s in DEV_SEEDS:
        if s not in R:
            continue
        rows[s] = {v: dict(phi1=R[s][v]["shapes"]["phi1"]["rise_id"], phi2=R[s][v]["shapes"]["phi2"]["rise_id"],
                           meets=R[s][v]["meets_ck6c_criterion"],
                           converged=R[s][v]["fit"]["converged"] if isinstance(R[s][v]["fit"], dict) else None)
                   for v in VARIANTS}
        p = R[s]["afcm_reml"]["fit"]["pilot"]
        rows[s]["pilot_equals_round15_laml"] = bool(s in ref and p["lam_x"] == ref[s]["lam_x"] and p["lam_t"] == ref[s]["lam_t"])
    complete = all(s in rows for s in DEV_SEEDS)
    passes = {v: bool(complete and all(rows[s][v]["meets"] for s in DEV_SEEDS)) for v in VARIANTS}
    go_variant = next((v for v in VARIANTS if passes[v]), None)
    out = dict(rule="GO iff one pre-specified variant (priority afcm_reml, then afcm_reml_ens) has identifiable RISE "
                    "phi_1 < 0.15 and phi_2 < 0.10 on all of seeds 22, 914, 966 (D156)",
               complete=complete, seeds=rows, passes=passes, go=go_variant is not None, go_variant=go_variant)
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "afcm_gate.json").write_text(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------------------------------- H-R22a
def h_r22a(recs):
    g = json.loads((BASE / "afcm_gate.json").read_text()) if (BASE / "afcm_gate.json").exists() else None
    ck = json.loads((BASE / "ck6d.json").read_text()) if (BASE / "ck6d.json").exists() else None
    out = dict(gate=g, ck6d=ck)
    if not g or not g["go"]:
        out.update(status="not run: development gate NO-GO; seed 24 unused; H-R12 stays inconclusive (D127)")
        return out
    if not ck or not ck.get("passed"):
        out.update(status="not run: CK-6d failed on seed 24; H-R12 stays inconclusive (D127)")
        return out
    A = {r["seed"] - 3000: float(r["shape"]["mean_id"]) for r in ok(recs, "hr22a")}
    B = by_rep(load12(), "S-Add", "b2s_cox")
    p = paired(B, A, margin=0.05)
    rep = [r for r in ok(recs, "hr22a_repro")]
    out.update(status="run", variant=g["go_variant"], n_afcm=len(A), b2s_minus_afcm_reml=p, interpretation=interpret(p),
               afcm_reml_mean=float(np.mean(list(A.values()))) if A else None,
               phi=dict(phi1=paired(by_rep(load12(), "S-Add", "b2s_cox", path="metrics.shape.phi1.rise_id"),
                                    {r["seed"] - 3000: r["shape"]["phi1"]["rise_id"] for r in ok(recs, "hr22a")}),
                        phi2=paired(by_rep(load12(), "S-Add", "b2s_cox", path="metrics.shape.phi2.rise_id"),
                                    {r["seed"] - 3000: r["shape"]["phi2"]["rise_id"] for r in ok(recs, "hr22a")})),
               harness_reproduction=rep[0] if rep else None)
    return out


# --------------------------------------------------------------------------------------------------- H-R22b
def both(a, b, frac):
    return dict(paired=paired(a, b), nb=nb(a, b, frac))


def finite_metrics(m, keys):
    return all(k in m and m[k] is not None and np.isfinite(m[k]) for k in keys)


def cohort(recs, ds):
    R = {r["split"]: r for r in ok(recs, "real") if r["dataset"] == ds}
    if not R:
        return None
    frac = float(np.mean([r["n_test"] / r["n_train"] for r in R.values()]))

    def col(m, k):
        out = {}
        for s, r in R.items():
            if m in r["models"]:
                out[s] = r["models"][m][k]
            elif m in r.get("baselines", {}) and r["baselines"][m].get("status") == "ok":
                out[s] = r["baselines"][m]["metrics"][k]
        return out
    models = sorted({m for r in R.values() for m in r["models"]}) + list(BASELINES)
    means = {m: {k: float(np.mean(list(col(m, k).values()))) for k in METRICS + ("dcal_p",) if col(m, k)} for m in models}
    dcal_pass = {m: int(sum(v >= 0.05 for v in col(m, "dcal_p").values())) for m in models}
    con = {}
    for a in ("b2s_npaft", "b2s_cox"):
        for b in ("landmark_cox", "lme2_cox", "fpca_cox", "sum_rsf") + (("b2s_cox", "b2s_waft", "b2s_lnaft") if a == "b2s_npaft" else ()):
            for k in METRICS:
                av, bv = col(a, k), col(b, k)
                if av and bv:
                    con[f"{k} {a} - {b}"] = both(av, bv, frac)
    for f, ref in CROSS.items():
        for k in METRICS:
            con[f"{k} b2s_anchor_{f}_cross - {ref}"] = both(col(f"b2s_anchor_{f}_cross", k), col(ref, k), frac)
    gates = {}
    for f in CROSS:
        g = {s: r["gates"][f"b2s_anchor_{f}_cross"] for s, r in R.items()}
        nc = [s for s, x in g.items() if not x["confirmed"]]
        gates[f] = dict(n_confirmed=sum(x["confirmed"] for x in g.values()), n=len(g),
                        confirmed_splits=sorted(s for s, x in g.items() if x["confirmed"]),
                        fallback_exact=all(g[s]["fallback_max_abs_diff_S"] == 0.0 and g[s]["fallback_max_abs_diff_risk"] == 0.0
                                           for s in nc),
                        fallback_max_abs_diff=max([max(g[s]["fallback_max_abs_diff_S"], g[s]["fallback_max_abs_diff_risk"])
                                                   for s in nc], default=0.0))
    sym = {}
    for f in ("npaft", "cox"):
        S = {s: r["symbolic"][f"b2s_{f}"] for s, r in R.items() if f"b2s_{f}" in r.get("symbolic", {})}
        if S:
            sym[f] = dict(n=len(S), replacement_rate_mean=float(np.mean([x["replacement_rate"] for x in S.values()])),
                          n_rollback=sum(bool(x["rollback"]) for x in S.values()),
                          all_finite=all(finite_metrics(x["after"], METRICS + ("dcal_p",)) for x in S.values()),
                          ibs_cond_before_mean=float(np.mean([x["seed0_before"]["ibs_cond"] for x in S.values()])),
                          ibs_cond_after_mean=float(np.mean([x["after"]["ibs_cond"] for x in S.values()])),
                          after_minus_before=both({s: x["after"]["ibs_cond"] for s, x in S.items()},
                                                  {s: x["seed0_before"]["ibs_cond"] for s, x in S.items()}, frac),
                          formulas={s: [dict(channel_index=e["i"], layer=e["layer"], formula=e["formula"], stage=e["stage"],
                                             r2_val=e["r2_val"]) for e in x["edges"] if e["accepted"]][:12]
                                    for s, x in S.items()})
    fin = all(finite_metrics(m, METRICS + ("dcal_p",)) and m.get("finite", True) for r in R.values() for m in r["models"].values())
    fin_b = all(r["baselines"][b].get("status") == "ok" and finite_metrics(r["baselines"][b]["metrics"], METRICS + ("dcal_p",))
                for r in R.values() for b in BASELINES)
    return dict(n_splits=len(R), frac=frac, cohort=R[min(R)]["cohort"], means=means, dcal_pass=dcal_pass, contrasts=con,
                gates=gates, symbolic=sym, all_finite_models=fin, all_finite_baselines=fin_b,
                n_val=[R[s]["n_val"] for s in sorted(R)], n_halves=[[R[s]["n_h1"], R[s]["n_h2"]] for s in sorted(R)],
                seconds_mean={m: float(np.mean([r["models"][m]["seconds"] for r in R.values()]))
                              for m in sorted({m for r in R.values() for m in r["models"]})})


def fd003(recs):
    R = {}
    for r in ok(recs, "fd003"):
        R.setdefault(r["split"], {"models": {}, "gates": {}, "n_test": r["n_test"], "n_train": r["n_train"]})
        R[r["split"]]["models"].update(r["models"])
        R[r["split"]]["gates"].update(r["gates"])
    if not R:
        return None
    frac = float(np.mean([r["n_test"] / r["n_train"] for r in R.values()]))
    models = sorted({m for r in R.values() for m in r["models"]})

    def col(m, k):
        return {s: r["models"][m][k] for s, r in R.items() if m in r["models"] and k in r["models"][m]}
    means = {m: {k: float(np.mean(list(col(m, k).values()))) for k in ("ibs", "harrell_c", "uno_c", "dcal_p", "dcal_chi2")
                 if col(m, k)} for m in models}
    npass = {m: int(sum(v >= 0.05 for v in col(m, "dcal_p").values())) for m in models}
    con = {f"{k} {a} - {b}": both(col(a, k), col(b, k), frac)
           for a in ("b2s_npaft", "b2s_lnaft", "b2s_waft", "b2s_anchor_npaft_cross") for b in ("b2s_cox", "landmark_cox")
           for k in ("ibs", "harrell_c")}
    g = {s: r["gates"].get("b2s_anchor_npaft_cross") for s, r in R.items() if "b2s_anchor_npaft_cross" in r["gates"]}
    nc = [s for s, x in g.items() if not x["confirmed"]]
    gate = dict(n_confirmed=sum(x["confirmed"] for x in g.values()), n=len(g),
                fallback_exact=all(g[s]["fallback_max_abs_diff_S"] == 0.0 and g[s]["fallback_max_abs_diff_risk"] == 0.0 for s in nc))
    fin = all(finite_metrics(m, ("ibs", "harrell_c", "uno_c", "dcal_p")) and m.get("finite", True)
              for r in R.values() for m in r["models"].values())
    n_parts = sum(1 for r in ok(recs, "fd003"))
    return dict(n_splits=len(R), n_part_records=n_parts, means=means, dcal_pass=npass, contrasts=con, gate=gate,
                all_finite=fin, n_train=[R[s]["n_train"] for s in sorted(R)], n_test=[R[s]["n_test"] for s in sorted(R)])


def h_r22b(C, F, recs):
    expected = {f"real_{ds}_{k}" for ds in COHORTS for k in range(5)} | {f"fd003_{k}_{p}" for k in range(5)
                                                                          for p in ("npaft", "lnaft", "waft", "cox", "base")}
    done = {r["job_name"] for r in recs if r.get("status") == "ok"}
    failed = sorted({r["job_name"] for r in recs if r.get("status") == "failed"} & expected)
    p1 = bool(expected <= done and all(C[ds] and C[ds]["all_finite_models"] and C[ds]["all_finite_baselines"] for ds in COHORTS)
              and F and F["all_finite"])
    p2 = {}
    for ds in COHORTS:
        X = C[ds]
        if not X:
            continue
        p2[ds] = {f: dict(mean_shift=X["contrasts"][f"ibs_cond b2s_anchor_{f}_cross - {ref}"]["paired"].get("mean_diff", float("nan")),
                          fallback_exact=X["gates"][f]["fallback_exact"], confirmed=f"{X['gates'][f]['n_confirmed']}/{X['gates'][f]['n']}")
                  for f, ref in CROSS.items()}
        for f in CROSS:
            p2[ds][f]["ok"] = bool(p2[ds][f]["mean_shift"] <= 0.005 and p2[ds][f]["fallback_exact"])
    p2_ok = bool(len(p2) == 2 and all(v["ok"] for d in p2.values() for v in d.values()))
    p3 = {}
    for ds in COHORTS:
        X = C[ds]
        if not X:
            continue
        p3[ds] = {f: dict(mean_change=s["after_minus_before"]["paired"].get("mean_diff", float("nan")), all_finite=s["all_finite"],
                          n=s["n"]) for f, s in X["symbolic"].items()}
        for f in p3[ds]:
            p3[ds][f]["ok"] = bool(p3[ds][f]["all_finite"] and p3[ds][f]["n"] == 5 and p3[ds][f]["mean_change"] <= 0.002)
    p3_ok = bool(len(p3) == 2 and all(len(d) == 2 for d in p3.values()) and all(v["ok"] for d in p3.values() for v in d.values()))
    return dict(part1_all_finite=p1, failed_jobs=failed, missing_jobs=sorted(expected - done), part2=p2, part2_ok=p2_ok,
                part3=p3, part3_ok=p3_ok, supported=bool(p1 and p2_ok and p3_ok))


# --------------------------------------------------------------------------------------------------- H-R22c
def sims(recs):
    out = {}
    for law in LAWS:
        R = {r["seed"]: r for r in ok(recs, "sim") if r["law"] == law}
        if not R:
            continue
        col = {m: {s: r["models"][m]["curve_error"] for s, r in R.items()} for m in ("b2s_cox", "b2s_waft", "b2s_lnaft", "b2s_npaft")}
        means = {m: {k: float(np.mean([r["models"][m][k] for r in R.values()])) for k in ("curve_error", "ibs", "harrell_c", "uno_c")}
                 for m in col}
        dpass = {m: int(sum(r["models"][m]["dcal_p"] >= 0.05 for r in R.values())) for m in col}
        con = {f"b2s_npaft - {b}": paired(col["b2s_npaft"], col[b]) for b in ("b2s_cox", "b2s_waft", "b2s_lnaft")}
        out[law] = dict(n=len(R), means=means, dcal_pass=dpass, contrasts=con,
                        censored_frac_mean=float(np.mean([r["censored_frac"] for r in R.values()])),
                        all_finite=all(m["finite"] for r in R.values() for m in r["models"].values()))
    return out


def h_r22c(S):
    def up(law, b):
        p = S[law]["contrasts"][f"b2s_npaft - {b}"]
        return p["ci95"][1] if p.get("n", 0) >= 2 else float("nan")
    if any(law not in S or S[law]["n"] != 20 for law in LAWS):
        return dict(supported=False, complete=False)
    c1 = {b: up("bimodal", b) for b in ("b2s_cox", "b2s_waft", "b2s_lnaft")}
    c2 = {law: {b: up(law, b) for b in ("b2s_cox", "b2s_waft", "b2s_lnaft")} for law in ("studentt3", "logistic05")}
    part1 = bool(all(v < 0 for v in c1.values()))
    part2 = bool(all(c2[law]["b2s_cox"] < 0 and c2[law]["b2s_waft"] < 0 and c2[law]["b2s_lnaft"] < 0.005 for law in c2))
    return dict(complete=True, upper_bounds_bimodal=c1, upper_bounds_other=c2, part1=part1, part2=part2,
                supported=bool(part1 and part2))


# --------------------------------------------------------------------------------------------------- output
def fmt(p):
    if p.get("n", 0) < 2:
        return "–"
    return f"{p['mean_diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"


def results_md(s):
    L = ["# Round 22 results (generated by `scripts/analyze_round22.py`; rules: `docs/round22_preregistration.md`)", ""]
    a = s["H_R22a"]
    L += ["## Task 1: per-channel REML AFCM (CK-6d, H-R22a)", ""]
    g = a.get("gate")
    if g:
        L += ["| Seed | afcm_reml φ1 / φ2 | afcm_reml_ens φ1 / φ2 | REML pilot = Round-15 LAML |", "|---|---|---|---|"]
        for sd, r in g["seeds"].items():
            L.append(f"| {sd} | {r['afcm_reml']['phi1']:.3f} / {r['afcm_reml']['phi2']:.3f} | "
                     f"{r['afcm_reml_ens']['phi1']:.3f} / {r['afcm_reml_ens']['phi2']:.3f} | {r['pilot_equals_round15_laml']} |")
        L += ["", f"Gate: **{'GO (' + g['go_variant'] + ')' if g['go'] else 'NO-GO'}**.", ""]
    L += [f"H-R22a: {a['status']}" + (f"; {a['interpretation']}: {fmt(a['b2s_minus_afcm_reml'])}" if a.get("status") == "run" else ""), ""]
    L += ["## Task 2: external cohorts (H-R22b)", ""]
    for ds in COHORTS:
        X = s["cohorts"].get(ds)
        if not X:
            continue
        L += [f"### {ds} (n {X['cohort']['n']}, events {X['cohort']['events']}, t_L {X['cohort']['t_L']})", "",
              "| Model | Cond. IBS | Harrell C | Uno C | D-cal p ≥ 0.05 |", "|---|---|---|---|---|"]
        for m, v in X["means"].items():
            if "ibs_cond" in v:
                L.append(f"| {m} | {v['ibs_cond']:.4f} | {v['harrell_c']:.3f} | {v['uno_c']:.3f} | {X['dcal_pass'].get(m, 0)} |")
        L += ["", "| Contrast (conditional IBS) | paired 95% CI | NB 95% CI |", "|---|---|---|"]
        for k, v in X["contrasts"].items():
            if k.startswith("ibs_cond "):
                nbp = v["nb"]
                L.append(f"| {k[9:]} | {fmt(v['paired'])} | " + (f"[{nbp['ci95'][0]:+.4f}, {nbp['ci95'][1]:+.4f}]" if nbp.get("n", 0) >= 2 else "–") + " |")
        L.append("")
    b = s["H_R22b"]
    L += [f"H-R22b: part 1 {b['part1_all_finite']}, part 2 {b['part2_ok']}, part 3 {b['part3_ok']} -> "
          f"**{'supported' if b['supported'] else 'not supported'}**", ""]
    F = s.get("fd003")
    if F:
        L += ["### C-MAPSS FD003 Option B (analysed in Round 3; not untouched)", "", "| Model | IBS | Harrell C | D-cal χ² | D-cal p ≥ 0.05 |",
              "|---|---|---|---|---|"]
        for m, v in F["means"].items():
            L.append(f"| {m} | {v['ibs']:.4f} | {v['harrell_c']:.3f} | {v.get('dcal_chi2', float('nan')):.1f} | {F['dcal_pass'][m]} |")
        L.append("")
    L += ["## Task 3: multi-law simulation (H-R22c)", "", "| Law | Contrast (curve error) | mean [95% CI] | a < b / a > b |", "|---|---|---|---|"]
    for law, X in s["sims"].items():
        for k, p in X["contrasts"].items():
            L.append(f"| {law} | {k} | {fmt(p)} | {p.get('n_neg', '–')} / {p.get('n_pos', '–')} |")
    c = s["H_R22c"]
    L += ["", f"H-R22c: **{'supported' if c.get('supported') else 'not supported'}**", ""]
    return "\n".join(L) + "\n"


def final():
    recs = load(BASE)
    C = {ds: cohort(recs, ds) for ds in COHORTS}
    F = fd003(recs)
    S = sims(recs)
    out = dict(H_R22a=h_r22a(recs), cohorts=C, fd003=F, sims=S)
    out["H_R22b"] = h_r22b(C, F, recs)
    out["H_R22c"] = h_r22c(S)
    out["jobs"] = dict(ok=sum(r.get("status") == "ok" for r in recs), failed=sum(r.get("status") == "failed" for r in recs))
    (BASE / "analysis_summary.json").write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    (ROOT / "docs/round22_results.md").write_text(results_md(out))
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "final"
    r = gate() if mode == "gate" else final()
    print(json.dumps({k: v for k, v in r.items() if k in ("go", "go_variant", "passes", "H_R22b", "H_R22c", "jobs")},
                     indent=1, default=str)[:3000])
