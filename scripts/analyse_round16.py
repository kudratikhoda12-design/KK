"""Round 16 analysis (D138): reports/round16/results_lane*.jsonl -> reports/round16_summary.json.

H-R16a: (i) all 40 fixed-mu distillation refits finite (a fallback counts as finite; fallbacks reported) AND
  (ii) S-Add distilled − after-pruning mean identifiable RISE, paired over replicates: upper 95% t-bound < +0.05.
H-R16b: (i) S-Last b2s_anchor_cox (non-cliff gate) − B2-S Cox survival-curve error: 95% CI entirely below 0 AND
  (ii) S-Sparse, the same difference: upper 95% bound < +0.005.
Everything else exploratory (no multiplicity correction).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_round12 import paired  # noqa: E402
from analyse_round15 import finite, per, sim  # noqa: E402

R16 = ROOT / "reports/round16"


def load():
    recs = []
    for f in sorted(R16.glob("results_lane*.jsonl")):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return recs


def family_a(recs):
    res = {}
    for kind in ("add", "sparse"):
        R = sim(recs, kind)
        rise = lambda var: per(R, "variants", var, "shape", "mean_id")  # noqa: E731
        res[kind] = dict(
            n=len(R), n_distill_finite=sum(finite(r["variants"].get("distill_fixed")) for r in R.values()),
            n_refit_old_finite=sum(finite(r["variants"].get("refit_old")) for r in R.values()),
            n_fallback=sum(bool(r["symbolic"]["refit_fallback"]) for r in R.values()),
            n_guarded_total=sum(r["symbolic"]["n_guarded"] for r in R.values()),
            n_rejected_total=sum(len(r["symbolic"]["formulas_r15_rejected"]) for r in R.values()),
            n_substituted_mean_r15=float(np.mean([r["symbolic"]["n_substituted_r15"] for r in R.values()])) if R else None,
            n_substituted_mean_r16=float(np.mean([r["symbolic"]["n_substituted_r16"] for r in R.values()])) if R else None,
            recall_mean=float(np.mean([r["recall"] for r in R.values()])) if R else None,
            fdr_mean=float(np.mean([r["fdr"] for r in R.values()])) if R else None,
            rise_means={v: float(np.mean(list(rise(v).values()))) if rise(v) else None
                        for v in ("after_pruning", "sub_noguard", "refit_old", "sub_guard_r16", "distill_fixed")},
            distill_minus_pruned_rise=paired(rise("distill_fixed"), rise("after_pruning"), margin=0.05),
            distill_minus_oldrefit_rise=paired(rise("distill_fixed"), rise("refit_old")),
            distill_minus_oldrefit_curve=paired(per(R, "variants", "distill_fixed", "curve_error"),
                                                per(R, "variants", "refit_old", "curve_error")),
            **{f"distill_minus_noguard_{m}": paired(per(R, "variants", "distill_fixed", m), per(R, "variants", "sub_noguard", m))
               for m in ("curve_error", "ibs", "uno_c", "val_loss")})
    a = res["add"]["distill_minus_pruned_rise"]
    i = res["add"]["n_distill_finite"] + res["sparse"]["n_distill_finite"]
    n = res["add"]["n"] + res["sparse"]["n"]
    res["H_R16a"] = dict(part1_all_finite=bool(i == n and n == 40), n_finite=i, n_cells=n,
                         part2_upper95=a.get("ci95", [None, None])[1],
                         part2_noninferior=bool(a.get("n", 0) >= 2 and a["ci95"][1] < 0.05))
    res["H_R16a"]["supported"] = res["H_R16a"]["part1_all_finite"] and res["H_R16a"]["part2_noninferior"]
    return res


def family_b(recs):
    res = {}
    ms = ("b2s_anchor_cox_noncliff", "b2s_anchor_cox_d133", "static_coxkan", "landmark_cox")
    for kind in ("last", "sparse"):
        R = sim(recs, kind)
        ce = {m: per(R, "family_b", m, "curve_error") for m in ms}
        ce["b2s_cox"] = per(R, "b2s_cox", "curve_error")
        ib = {m: per(R, "family_b", m, "ibs") for m in ms}
        ib["b2s_cox"] = per(R, "b2s_cox", "ibs")
        on = {m: [r["gates"][m]["residual_on"] for r in R.values()] for m in ("b2s_anchor_cox_noncliff", "b2s_anchor_cox_d133")}
        res[kind] = dict(
            n=len(R), curve_error_means={m: float(np.mean(list(v.values()))) if v else None for m, v in ce.items()},
            ibs_means={m: float(np.mean(list(v.values()))) if v else None for m, v in ib.items()},
            gate_on_frequency={m: (np.mean(v, 0).tolist() if v else None) for m, v in on.items()},
            gate_all_off={m: int(sum(sum(x) == 0 for x in v)) for m, v in on.items()},
            gate_decisions_differ=int(sum(a != b for a, b in zip(on["b2s_anchor_cox_noncliff"], on["b2s_anchor_cox_d133"]))),
            contrasts={f"{a} - {b}": paired(ce[a], ce[b]) for a, b in (
                ("b2s_anchor_cox_noncliff", "b2s_cox"), ("b2s_anchor_cox_noncliff", "static_coxkan"),
                ("b2s_anchor_cox_noncliff", "landmark_cox"), ("b2s_anchor_cox_d133", "b2s_cox"),
                ("b2s_anchor_cox_noncliff", "b2s_anchor_cox_d133"))},
            contrasts_ibs={f"{a} - {b}": paired(ib[a], ib[b]) for a, b in (
                ("b2s_anchor_cox_noncliff", "b2s_cox"), ("b2s_anchor_cox_noncliff", "landmark_cox"))})
    p1 = res["last"]["contrasts"]["b2s_anchor_cox_noncliff - b2s_cox"]
    p2 = res["sparse"]["contrasts"]["b2s_anchor_cox_noncliff - b2s_cox"]
    res["H_R16b"] = dict(part1_ci95=p1.get("ci95"), part1_better=bool(p1.get("n", 0) >= 2 and p1["ci95"][1] < 0),
                         part2_upper95=p2.get("ci95", [None, None])[1],
                         part2_noninferior=bool(p2.get("n", 0) >= 2 and p2["ci95"][1] < 0.005))
    res["H_R16b"]["supported"] = res["H_R16b"]["part1_better"] and res["H_R16b"]["part2_noninferior"]
    return res


def main():
    recs = load()
    s = dict(n_records=len(recs), n_failed=sum(r["status"] != "ok" for r in recs),
             failed=[dict(job=r["job_name"], error=r.get("error")) for r in recs if r["status"] != "ok"],
             family_a=family_a(recs), family_b=family_b(recs))
    dev = R16 / "dev/results_dev3515.jsonl"
    if dev.exists():
        d = json.loads(dev.read_text().splitlines()[0])
        s["dev_3515"] = dict(status=d["status"], distill_fixed=d["variants"]["distill_fixed"],
                             after_pruning=d["variants"]["after_pruning"], refit_fallback=d["symbolic"]["refit_fallback"],
                             rejected=d["symbolic"]["formulas_r15_rejected"])
    g = R16 / "dev/gate_rule_on_round15_records.json"
    if g.exists():
        s["dev_gate_rule_on_round15"] = json.loads(g.read_text())
    (ROOT / "reports/round16_summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(json.dumps(dict(H_R16a=s["family_a"]["H_R16a"], H_R16b=s["family_b"]["H_R16b"], n_records=s["n_records"],
                          n_failed=s["n_failed"]), indent=1, default=float))


if __name__ == "__main__":
    main()
