"""Round 15 analysis (D133/D134/D135): reports/round15/results_lane*.jsonl -> reports/round15_summary.json.

H-R15a (symbolic_distill; confirmatory): (i) every distillation refit finite (S-Add and S-Sparse, 40 cells) AND
  (ii) S-Add, distilled - after-pruning mean identifiable RISE, paired over replicates: upper 95% t-bound < +0.05.
H-R15b (b2s_anchor_cox; confirmatory): (i) S-Last, anchored - B2-S Cox survival-curve error: 95% CI entirely below 0
  AND (ii) S-Sparse, the same difference: upper 95% bound < +0.005.
Secondary and exploratory contrasts (no multiplicity correction) and the real-data lane (Nadeau-Bengio intervals over
the Round-13 splits, against the saved Round-13 records after the split-0 reproduction check of landmark_cox).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_round12 import paired  # noqa: E402
from analyse_round13 import first_ok, nb  # noqa: E402
from analyse_round13 import load as load13  # noqa: E402

R15 = ROOT / "reports/round15"
CLINICAL = {"log bili": "increasing", "albumin": "decreasing", "log protime": "increasing",
            "prothrombin index": "decreasing", "sqrt CD4": "decreasing"}        # D114 directions (risk)


def load():
    recs = []
    for f in sorted(R15.glob("results_lane*.jsonl")):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return recs


def sim(recs, kind):
    out = {}
    for r in recs:
        if r.get("status") == "ok" and r["job_name"].startswith(kind + "_") and r["seed"] not in out:
            out[r["seed"]] = r
    return out


def val(v, *path):
    for p in path:
        if not isinstance(v, dict) or p not in v:
            return None
        v = v[p]
    return v


def per(R, *path):
    out = {}
    for s, r in R.items():
        v = val(r, *path)
        if isinstance(v, (int, float)) and np.isfinite(v):
            out[s] = float(v)
    return out


def finite(v):
    return isinstance(v, dict) and not v.get("status")


def family_a(recs):
    res = {}
    for kind in ("add", "sparse"):
        R = sim(recs, kind)
        n_fin = sum(finite(val(r, "variants", "distill")) for r in R.values())
        n_old_fin = sum(finite(val(r, "variants", "refit_old")) for r in R.values())
        rise = lambda var: per(R, "variants", var, "shape", "mean_id")  # noqa: E731
        chosen = [g["mu"] for r in R.values() for g in r.get("distill_grid", []) if g.get("chosen")]
        res[kind] = dict(
            n=len(R), n_distill_finite=n_fin, n_refit_old_finite=n_old_fin,
            n_guarded_total=sum(r["symbolic"].get("n_guarded", 0) for r in R.values()),
            n_substituted_mean=float(np.mean([r["symbolic"]["n_substituted"] for r in R.values()])) if R else None,
            mu_chosen={str(m): chosen.count(m) for m in sorted(set(chosen))},
            recall_mean=float(np.mean([r["recall"] for r in R.values()])) if R else None,
            fdr_mean=float(np.mean([r["fdr"] for r in R.values()])) if R else None,
            rise_means={v: float(np.mean(list(rise(v).values()))) if rise(v) else None
                        for v in ("after_pruning", "sub_noguard", "refit_old", "sub_guard", "distill_mu0.01", "distill_mu0.1",
                                  "distill_mu1", "distill_mu10", "distill")},
            distill_minus_pruned_rise=paired(rise("distill"), rise("after_pruning"), margin=0.05),
            distill_minus_oldrefit_rise=paired(rise("distill"), rise("refit_old")),
            **{f"distill_minus_noguard_{m}": paired(per(R, "variants", "distill", m), per(R, "variants", "sub_noguard", m))
               for m in ("curve_error", "ibs", "uno_c", "val_loss")},
            distill_minus_oldrefit_curve=paired(per(R, "variants", "distill", "curve_error"),
                                                per(R, "variants", "refit_old", "curve_error")))
    a = res["add"]["distill_minus_pruned_rise"]
    i = res["add"]["n_distill_finite"] + res["sparse"]["n_distill_finite"]
    n = res["add"]["n"] + res["sparse"]["n"]
    res["H_R15a"] = dict(part1_all_finite=bool(i == n and n == 40), n_finite=i, n_cells=n,
                         part2_upper95=a.get("ci95", [None, None])[1],
                         part2_noninferior=bool(a.get("n", 0) >= 2 and a["ci95"][1] < 0.05))
    res["H_R15a"]["supported"] = res["H_R15a"]["part1_all_finite"] and res["H_R15a"]["part2_noninferior"]
    return res


def family_b(recs):
    res = {}
    for kind in ("last", "sparse"):
        R = sim(recs, kind)
        ce = {m: per(R, "family_b", m, "curve_error") for m in ("static_coxkan", "landmark_cox", "b2s_anchor_cox", "b2s_res_cox")}
        ce["b2s_cox"] = per(R, "b2s_cox", "curve_error")
        ib = {m: per(R, "family_b", m, "ibs") for m in ("static_coxkan", "landmark_cox", "b2s_anchor_cox", "b2s_res_cox")}
        ib["b2s_cox"] = per(R, "b2s_cox", "ibs")
        on = {m: [r["gates"][m]["residual_on"] for r in R.values() if m in r.get("gates", {})] for m in ("b2s_anchor_cox", "b2s_res_cox")}
        res[kind] = dict(
            n=len(R), curve_error_means={m: float(np.mean(list(v.values()))) if v else None for m, v in ce.items()},
            ibs_means={m: float(np.mean(list(v.values()))) if v else None for m, v in ib.items()},
            gate_on_frequency={m: (np.mean(v, 0).tolist() if v else None) for m, v in on.items()},
            gate_all_off={m: int(sum(sum(x) == 0 for x in v)) for m, v in on.items()},
            contrasts={f"{a} - {b}": paired(ce[a], ce[b]) for a, b in (
                ("b2s_anchor_cox", "b2s_cox"), ("b2s_anchor_cox", "static_coxkan"), ("b2s_anchor_cox", "landmark_cox"),
                ("b2s_res_cox", "b2s_cox"), ("b2s_res_cox", "b2s_anchor_cox"))},
            contrasts_ibs={f"{a} - {b}": paired(ib[a], ib[b]) for a, b in (
                ("b2s_anchor_cox", "b2s_cox"), ("b2s_anchor_cox", "landmark_cox"))})
    p1 = res["last"]["contrasts"]["b2s_anchor_cox - b2s_cox"]
    p2 = res["sparse"]["contrasts"]["b2s_anchor_cox - b2s_cox"]
    res["H_R15b"] = dict(part1_ci95=p1.get("ci95"), part1_better=bool(p1.get("n", 0) >= 2 and p1["ci95"][1] < 0),
                         part2_upper95=p2.get("ci95", [None, None])[1],
                         part2_noninferior=bool(p2.get("n", 0) >= 2 and p2["ci95"][1] < 0.005))
    res["H_R15b"]["supported"] = res["H_R15b"]["part1_better"] and res["H_R15b"]["part2_noninferior"]
    return res


def monotone_share(curves, direction):
    ok = []
    for c in curves:
        dif = np.diff(np.asarray(c))
        ok.append(bool(np.all(dif >= -1e-12) if direction == "increasing" else np.all(dif <= 1e-12)))
    return float(np.mean(ok)) if ok else None


def real(recs):
    R13 = load13()
    out, repro = {}, {}
    for ds in ("pbc_E2", "pbc_E1", "prothro", "aids"):
        new = {r["split"]: r for r in recs if r.get("status") == "ok" and r.get("job") == "real" and r["dataset"] == ds}
        old = {m: first_ok(R13, job="single", dataset=ds, model=m) for m in ("b2s_cox", "landmark_cox", "static_coxkan")}
        if 0 in new and 0 in old["landmark_cox"]:
            a, b = new[0]["models"]["landmark_cox_repro"], old["landmark_cox"][0]["metrics"]
            repro[ds] = dict(max_abs_diff=max(abs(a[k] - b[k]) for k in ("ibs", "ibs_cond", "harrell_c", "uno_c", "dcal_p")))
        frac = np.mean([r["n_test"] / r["n_train"] for r in old["b2s_cox"].values()]) if old["b2s_cox"] else 1 / 3
        mets = {m: {k: r["models"][m] for k, r in new.items() if m in r["models"]} for m in ("b2s_anchor_cox", "b2s_res_cox")}
        mets.update({m: {k: r["metrics"] for k, r in old[m].items()} for m in old})
        means = {m: {k: float(np.mean([v[k] for v in d.values()])) for k in ("ibs", "ibs_cond", "harrell_c", "uno_c")}
                 for m, d in mets.items() if d}
        con = {}
        for a in ("b2s_anchor_cox", "b2s_res_cox"):
            for b in ("b2s_cox", "landmark_cox", "static_coxkan"):
                for k in ("ibs", "uno_c"):
                    con[f"{k} {a} - {b}"] = nb({s: v[k] for s, v in mets[a].items()}, {s: v[k] for s, v in mets[b].items()}, frac)
        gates = {m: [r["gates"][m]["residual_on"] for r in new.values() if m in r["gates"]] for m in ("b2s_anchor_cox", "b2s_res_cox")}
        align = {}
        for c, direc in CLINICAL.items():
            curves = [r["gates"]["b2s_anchor_cox"]["last_value_effects"][c] for r in new.values()
                      if c in r["gates"]["b2s_anchor_cox"]["last_value_effects"]]
            if curves:
                align[c] = dict(direction=direc, share_monotone=monotone_share(curves, direc), n_splits=len(curves))
        out[ds] = dict(n_splits=len(new), means=means, contrasts=con, gate_on={m: np.mean(v, 0).tolist() if v else None
                                                                                  for m, v in gates.items()},
                       last_value_alignment=align)
    return out, repro


def main():
    recs = load()
    s = dict(n_records=len(recs), n_failed=sum(r["status"] != "ok" for r in recs),
             failed=[dict(job=r["job_name"], error=r.get("error")) for r in recs if r["status"] != "ok"],
             family_a=family_a(recs), family_b=family_b(recs))
    s["real"], s["real_repro_landmark_cox_split0"] = real(recs)
    (ROOT / "reports/round15_summary.json").write_text(json.dumps(s, indent=1, default=float))
    a, b = s["family_a"]["H_R15a"], s["family_b"]["H_R15b"]
    print(json.dumps(dict(H_R15a=a, H_R15b=b, n_records=s["n_records"], n_failed=s["n_failed"]), indent=1, default=float))


if __name__ == "__main__":
    main()
