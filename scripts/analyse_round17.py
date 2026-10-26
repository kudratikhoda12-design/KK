"""Round 17 analysis (D141): reports/round17/results_lane*.jsonl -> reports/round17_summary.json.

H-R17a (b2s_anchor_cox_twoway): (i) S-Last 4100-4119 upper 95% bound of (two-way - b2s_cox) survival-curve error < 0
  AND (ii) S-Sparse 4200-4219 upper bound < +0.005.
H-R17b (log-normal KAN-AFT): (i) S-Misspec 4300-4319: all 20 AFT distillation refits finite (a fallback counts as
  finite) AND upper bound of (distilled - after pruning) identifiable RISE < +0.05; (ii) S-Misspec upper bound of
  (b2s_anchor_aft_twoway - b2s_aft) curve error < +0.005 AND S-Last upper bound < 0.
Everything else exploratory (no multiplicity correction).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_round12 import paired  # noqa: E402
from analyse_round17_dev import by_kind, load  # noqa: E402

R17 = ROOT / "reports/round17"


def ce(R, model, metric="curve_error", src="models"):
    return {s: r[src][model][metric] for s, r in R.items() if model in r[src]}


def gate_stats(R, m):
    g = [r["gates"][m] for r in R.values() if m in r["gates"]]
    if not g:
        return None
    return dict(n=len(g), n_reduced=sum(x["reduced"] for x in g),
                traj_on_frequency=np.mean([x["traj_on"] for x in g], 0).tolist(),
                last_on_frequency=np.mean([x["last_on"] for x in g], 0).tolist(),
                anchored_kind_counts={k: sum(x["anchored_choice"]["kind"] == k for x in g) for k in ("joint", "static")},
                val_gap_mean=float(np.mean([x["ref_val_loss"] - x["anchored_val_loss"] for x in g])))


def family(R, loss):
    m, b = f"b2s_anchor_{loss}_twoway", f"b2s_{loss}"
    out = dict(n=len(R), curve_error_means={k: float(np.mean(list(ce(R, k).values()))) for k in (m, b, "landmark_cox")
                                            if ce(R, k)},
               ibs_means={k: float(np.mean(list(ce(R, k, "ibs").values()))) for k in (m, b, "landmark_cox") if ce(R, k, "ibs")},
               contrast=paired(ce(R, m), ce(R, b)), contrast_ibs=paired(ce(R, m, "ibs"), ce(R, b, "ibs")),
               alternative_minus_comparator=paired(ce(R, m, src="anchored_alternative"), ce(R, b)),
               gates=gate_stats(R, m))
    if ce(R, "landmark_cox"):
        out["twoway_minus_landmark"] = paired(ce(R, m), ce(R, "landmark_cox"))
    return out


def main():
    recs = load(R17)
    LA, SP, MI = by_kind(recs, "last"), by_kind(recs, "sparse"), by_kind(recs, "misspec")
    s = dict(n_records=len(recs), n_failed=sum(r["status"] != "ok" for r in recs),
             failed=[dict(job=r["job_name"], error=r.get("error")) for r in recs if r["status"] != "ok"],
             last_tol={loss: next((r["gates"][f"b2s_anchor_{loss}_twoway"]["last_tol"] for r in list(LA.values()) +
                                   list(MI.values()) if f"b2s_anchor_{loss}_twoway" in r["gates"]), None)
                       for loss in ("cox", "aft")})
    s["cox"] = {"last": family(LA, "cox"), "sparse": family(SP, "cox")}
    s["aft"] = {"last": family(LA, "aft"), "sparse": family(SP, "aft"), "misspec": family(MI, "aft")}
    p1, p2 = s["cox"]["last"]["contrast"], s["cox"]["sparse"]["contrast"]
    up = lambda p: p.get("ci95", [None, None])[1]  # noqa: E731
    s["H_R17a"] = dict(part1_ci95=p1.get("ci95"), part1_better=bool(p1.get("n", 0) >= 2 and up(p1) < 0),
                       part2_ci95=p2.get("ci95"), part2_upper95=up(p2),
                       part2_noninferior=bool(p2.get("n", 0) >= 2 and up(p2) < 0.005))
    s["H_R17a"]["supported"] = s["H_R17a"]["part1_better"] and s["H_R17a"]["part2_noninferior"]
    rise = lambda v: {sd: r["variants"][v]["shape"]["mean_id"] for sd, r in MI.items() if "shape" in r["variants"][v]}  # noqa: E731
    sym = dict(n=len(MI), n_distill_finite=sum("shape" in r["variants"]["distill_fixed"] for r in MI.values()),
               n_fallback=sum(bool(r["symbolic"]["refit_fallback"]) for r in MI.values()),
               n_rejected=sum(len(r["symbolic"]["rejected"]) for r in MI.values()),
               n_guarded=sum(r["symbolic"]["n_guarded"] for r in MI.values()),
               n_substituted_mean=float(np.mean([r["symbolic"]["n_substituted"] for r in MI.values()])) if MI else None,
               recall_mean=float(np.mean([r["recall"] for r in MI.values()])) if MI else None,
               fdr_mean=float(np.mean([r["fdr"] for r in MI.values()])) if MI else None,
               rise_means={v: float(np.mean(list(rise(v).values()))) if rise(v) else None
                           for v in ("after_pruning", "sub_guard", "refit_plain", "distill_fixed")},
               b2s_aft_rise_mean=float(np.mean([r["models"]["b2s_aft"]["shape"]["mean_id"] for r in MI.values()])) if MI else None,
               distill_minus_pruned_rise=paired(rise("distill_fixed"), rise("after_pruning"), margin=0.05),
               distill_minus_plain_rise=paired(rise("distill_fixed"), rise("refit_plain")),
               **{f"distill_minus_sub_{m}": paired({sd: r["variants"]["distill_fixed"][m] for sd, r in MI.items() if m in r["variants"]["distill_fixed"]},
                                                   {sd: r["variants"]["sub_guard"][m] for sd, r in MI.items() if m in r["variants"]["sub_guard"]})
                  for m in ("curve_error", "ibs", "uno_c", "val_loss")})
    s["aft_symbolic"] = sym
    pa = sym["distill_minus_pruned_rise"]
    pm, pl = s["aft"]["misspec"]["contrast"], s["aft"]["last"]["contrast"]
    s["H_R17b"] = dict(part1_n_finite=sym["n_distill_finite"], part1_n_cells=len(MI),
                       part1_all_finite=bool(sym["n_distill_finite"] == len(MI) == 20), part1_upper95=up(pa),
                       part1_noninferior=bool(pa.get("n", 0) >= 2 and up(pa) < 0.05),
                       part2_misspec_ci95=pm.get("ci95"), part2_misspec_upper95=up(pm),
                       part2_misspec_noninferior=bool(pm.get("n", 0) >= 2 and up(pm) < 0.005),
                       part2_last_ci95=pl.get("ci95"), part2_last_better=bool(pl.get("n", 0) >= 2 and up(pl) < 0))
    h = s["H_R17b"]
    h["supported"] = h["part1_all_finite"] and h["part1_noninferior"] and h["part2_misspec_noninferior"] and h["part2_last_better"]
    (ROOT / "reports/round17_summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(json.dumps(dict(H_R17a=s["H_R17a"], H_R17b=s["H_R17b"], n_records=s["n_records"], n_failed=s["n_failed"]),
                     indent=1, default=float))


if __name__ == "__main__":
    main()
