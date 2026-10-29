"""Round 19 analysis (D147): reports/round19/results_lane*.jsonl -> reports/round19_summary.json.

H-R19a (Cox): primary contrast b2s_anchor_cox_refit - b2s_cox_fullval (survival-curve error): S-Last 4700-4719 upper
  95% bound < 0 AND S-Sparse 4800-4819 upper bound < +0.005.
H-R19b (log-normal KAN-AFT): b2s_anchor_aft_refit - b2s_aft_fullval: S-Misspec-lognormal 4900-4919 upper bound < +0.005
  AND S-Last 4700-4719 upper bound < 0.
Secondary (exploratory): the same estimator against the val_select B2-S and landmark Cox; the Round-18 estimator from
the same fit against the full-validation B2-S. Every contrast: paired mean, 95% t-interval, two-sided paired-t p,
wins / losses / ties; anchor rates.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_round12 import paired  # noqa: E402
from analyse_round17_dev import by_kind, load  # noqa: E402
from analyse_round18_dev import diagnostics  # noqa: E402
from analyse_round19_dev import _diag_view  # noqa: E402

R19 = ROOT / "reports/round19"


def ce(R, model, metric="curve_error"):
    return {s: r["models"][model][metric] for s, r in R.items() if model in r["models"]}


def family(R, loss):
    m, f, sel = f"b2s_anchor_{loss}_refit", f"b2s_{loss}_fullval", f"b2s_{loss}_select"
    r18 = f"b2s_anchor_{loss}_confirm_r18"
    g = [r["gates"][m] for r in R.values()]
    out = dict(n=len(R), curve_error_means={k: float(np.mean(list(ce(R, k).values()))) for k in (m, f, sel, r18, "landmark_cox")
                                            if ce(R, k)},
               contrast=paired(ce(R, m), ce(R, f)), contrast_ibs=paired(ce(R, m, "ibs"), ce(R, f, "ibs")),
               vs_select=paired(ce(R, m), ce(R, sel)), r18_vs_fullval=paired(ce(R, r18), ce(R, f)),
               select_vs_fullval=paired(ce(R, sel), ce(R, f)),
               anchor_rate=dict(n_anchored=sum(not x["reduced"] for x in g), n=len(g)),
               last_on_frequency=np.mean([x["last_on"] for x in g], 0).tolist() if g else None,
               traj_on_frequency=np.mean([x["traj_on"] for x in g], 0).tolist() if g else None,
               diagnostics=diagnostics(_diag_view(R, loss), loss) if R else None)
    if ce(R, "landmark_cox"):
        out["vs_landmark"] = paired(ce(R, m), ce(R, "landmark_cox"))
    return out


def main():
    recs = load(R19)
    LA, SP, MI = by_kind(recs, "last"), by_kind(recs, "sparse"), by_kind(recs, "misspec")
    s = dict(n_records=len(recs), n_failed=sum(r["status"] != "ok" for r in recs),
             failed=[dict(job=r["job_name"], error=r.get("error")) for r in recs if r["status"] != "ok"],
             settings={loss: dict(margin=next((r["gates"][f"b2s_anchor_{loss}_refit"]["margin"] for r in LA.values()), None),
                                  refit=next((r["gates"][f"b2s_anchor_{loss}_refit"]["refit"] for r in LA.values()), None))
                       for loss in ("cox", "aft")})
    s["cox"] = {"last": family(LA, "cox"), "sparse": family(SP, "cox")}
    s["aft"] = {"last": family(LA, "aft"), "misspec": family(MI, "aft")}
    up = lambda p: p.get("ci95", [None, None])[1]  # noqa: E731
    p1, p2 = s["cox"]["last"]["contrast"], s["cox"]["sparse"]["contrast"]
    s["H_R19a"] = dict(part1_ci95=p1.get("ci95"), part1_better=bool(p1.get("n", 0) >= 2 and up(p1) < 0),
                       part2_ci95=p2.get("ci95"), part2_upper95=up(p2),
                       part2_noninferior=bool(p2.get("n", 0) >= 2 and up(p2) < 0.005))
    s["H_R19a"]["supported"] = s["H_R19a"]["part1_better"] and s["H_R19a"]["part2_noninferior"]
    pm, pl = s["aft"]["misspec"]["contrast"], s["aft"]["last"]["contrast"]
    s["H_R19b"] = dict(part1_misspec_ci95=pm.get("ci95"), part1_misspec_upper95=up(pm),
                       part1_noninferior=bool(pm.get("n", 0) >= 2 and up(pm) < 0.005),
                       part2_last_ci95=pl.get("ci95"), part2_last_better=bool(pl.get("n", 0) >= 2 and up(pl) < 0))
    s["H_R19b"]["supported"] = s["H_R19b"]["part1_noninferior"] and s["H_R19b"]["part2_last_better"]
    (ROOT / "reports/round19_summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(json.dumps(dict(H_R19a=s["H_R19a"], H_R19b=s["H_R19b"], n_records=s["n_records"], n_failed=s["n_failed"]),
                     indent=1, default=float))


if __name__ == "__main__":
    main()
