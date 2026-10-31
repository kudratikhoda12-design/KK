"""Round 20 development analysis (docs/round20_devplan.md §3-5; rules fixed before the development runs).

  stage A: reports/round20/devA/results_lane*.jsonl -> reports/round20/devA_summary.json (NP objective, GO-a)
  stage B: reports/round20/devB/results_lane*.jsonl -> reports/round20/devB_summary.json (margins, GO-b)
Usage: analyse_round20_dev.py A|B
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from analyse_round12 import paired  # noqa: E402
from analyse_round17_dev import by_kind, load, upper  # noqa: E402
from kanfae.models.fres import cross_confirm_decision  # noqa: E402

import numpy as np  # noqa: E402

GRID = (0.0, 0.0025, 0.005, 0.01, 0.02)
OBJ = {"bj_km": "b2s_npaft_bj_fullval", "rank_km": "b2s_npaft_rank_fullval"}
TRAJ = {"cox": "sparse", "lnaft": "misspec", "npaft": "misspec"}
RELATIVE = {"cox": False, "lnaft": False, "npaft": True}


def ce(R, model):
    return {s: r["models"][model]["curve_error"] for s, r in R.items() if model in r["models"]}


def stage_a():
    recs = load(ROOT / "reports/round20/devA")
    s = dict(n_records=len(recs), n_failed=sum(r["status"] != "ok" for r in recs),
             failed=[dict(job=r["job_name"], error=r.get("error")) for r in recs if r["status"] != "ok"])
    SP, MI = by_kind(recs, "sparse"), by_kind(recs, "misspec")
    s["n"] = dict(sparse=len(SP), misspec=len(MI))
    s["means"] = {sc: {m: float(np.mean(list(ce(R, m).values()))) for m in
                       ("b2s_cox_fullval", "b2s_weibull_fullval", *OBJ.values()) if ce(R, m)}
                  for sc, R in (("sparse", SP), ("misspec", MI))}
    score = {o: 0.5 * (s["means"]["misspec"][m] + s["means"]["sparse"][m]) for o, m in OBJ.items()}
    s["objective_score"] = score
    s["objective"] = "rank_km" if score["rank_km"] < score["bj_km"] else "bj_km"           # ties -> bj_km
    s["contrasts"] = {o: dict(misspec_vs_cox=paired(ce(MI, m), ce(MI, "b2s_cox_fullval")),
                              misspec_vs_weibull=paired(ce(MI, m), ce(MI, "b2s_weibull_fullval")),
                              sparse_vs_cox=paired(ce(SP, m), ce(SP, "b2s_cox_fullval")),
                              misspec_bj_vs_rank=paired(ce(MI, OBJ["bj_km"]), ce(MI, OBJ["rank_km"])),
                              sparse_bj_vs_rank=paired(ce(SP, OBJ["bj_km"]), ce(SP, OBJ["rank_km"])))
                      for o, m in OBJ.items()}
    c = s["contrasts"][s["objective"]]
    g = dict(misspec_vs_cox_upper=upper(c["misspec_vs_cox"]), misspec_vs_weibull_upper=upper(c["misspec_vs_weibull"]),
             sparse_vs_cox_upper=upper(c["sparse_vs_cox"]))
    g["go_a"] = bool(g["misspec_vs_cox_upper"] < 0 and g["misspec_vs_weibull_upper"] < 0 and g["sparse_vs_cox_upper"] < 0.005)
    s["go_a"] = g
    (ROOT / "reports/round20/devA_summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(json.dumps(dict(n=s["n"], failed=s["n_failed"], means=s["means"], score=score, objective=s["objective"], go_a=g),
                     indent=1, default=float))


def estimator(r, f, l):
    m = f"b2s_anchor_{f}_cross"
    use = cross_confirm_decision(r["gates"][m]["dirs"], l, RELATIVE[f])
    return (r["alternatives"][m]["refit"] if use else r["models"][f"b2s_{f}_fullval"])["curve_error"], use


def sweep(R, f):
    out = {}
    for l in GRID:
        a, n = {}, 0
        for s, r in R.items():
            a[s], u = estimator(r, f, l)
            n += u
        p = paired(a, ce(R, f"b2s_{f}_fullval"))
        p["n_confirmed"] = n
        out[str(l)] = p
    return out


def choose(sw_traj, sw_last, cap=0.0025):
    """docs/round20_devplan.md §4: smallest l with U(traj) < cap and mean(S-Last) < 0; else the l with the smallest
    U(traj) among those with mean(S-Last) < 0 (ties: smaller l); else 0."""
    ok = [l for l in GRID if sw_last[str(l)].get("mean_diff", 0.0) < 0]
    good = [l for l in ok if upper(sw_traj[str(l)]) < cap]
    if good:
        return min(good), "rule"
    if ok:
        return min(ok, key=lambda l: (upper(sw_traj[str(l)]), l)), "fallback"
    return 0.0, "fallback_zero"


def stage_b():
    recs = load(ROOT / "reports/round20/devB")
    s = dict(n_records=len(recs), n_failed=sum(r["status"] != "ok" for r in recs),
             failed=[dict(job=r["job_name"], error=r.get("error")) for r in recs if r["status"] != "ok"],
             np_objective=next((r.get("np_objective") for r in recs if r["status"] == "ok"), None))
    K = {k: by_kind(recs, k) for k in ("last", "sparse", "misspec")}
    s["n"] = {k: len(v) for k, v in K.items()}
    s["sweep"], s["margin"], s["go_b"] = {}, {}, {}
    for f in ("cox", "lnaft", "npaft"):
        sw = {"last": sweep(K["last"], f), TRAJ[f]: sweep(K[TRAJ[f]], f)}
        l, why = choose(sw[TRAJ[f]], sw["last"])
        s["sweep"][f], s["margin"][f] = sw, dict(margin=l, reason=why)
        s["go_b"][f] = dict(last_mean=sw["last"][str(l)]["mean_diff"], last_upper=upper(sw["last"][str(l)]),
                            traj_upper=upper(sw[TRAJ[f]][str(l)]), traj=TRAJ[f])
        s["go_b"][f]["pass"] = bool(s["go_b"][f]["last_mean"] < 0 and s["go_b"][f]["traj_upper"] < 0.005)
    s["go_b"]["go"] = all(s["go_b"][f]["pass"] for f in ("cox", "lnaft", "npaft"))
    # diagnostics: always-refit contrast, h1 B2-S vs full-validation B2-S, one-direction seed-0 (Round-18 rule) rate
    s["diag"] = {}
    for f in ("cox", "lnaft", "npaft"):
        for k in ("last", TRAJ[f]):
            R = K[k]
            m = f"b2s_anchor_{f}_cross"
            dirs = {sd: r["gates"][m]["dirs"] for sd, r in R.items()}
            s["diag"][f"{f}/{k}"] = dict(
                refit_always=paired({sd: r["alternatives"][m]["refit"]["curve_error"] for sd, r in R.items()},
                                    ce(R, f"b2s_{f}_fullval")),
                h1_b2s_vs_fullval=paired(ce(R, f"b2s_{f}_h1"), ce(R, f"b2s_{f}_fullval")),
                n_h1_favours=sum(d[0]["loss_anchored"] < d[0]["loss_ref"] for d in dirs.values()),
                n_h2_favours=sum(d[1]["loss_anchored"] < d[1]["loss_ref"] for d in dirs.values()),
                n_both_favour=sum(cross_confirm_decision(d, 0.0) for d in dirs.values()),
                n_r18_rule=sum(d[0]["r18_loss_anchored"] < d[0]["r18_loss_ref"] for d in dirs.values()),
                vs_landmark=paired(ce(R, m), ce(R, "landmark_cox")) if ce(R, "landmark_cox") else None)
    (ROOT / "reports/round20/devB_summary.json").write_text(json.dumps(s, indent=1, default=float))
    tab = {f"{f}/{k}": {l: (round(p.get("mean_diff", float("nan")), 4), round(upper(p), 4), p["n_confirmed"])
                        for l, p in d.items()} for f, x in s["sweep"].items() for k, d in x.items()}
    print(json.dumps(dict(n=s["n"], failed=s["n_failed"], margin=s["margin"], go_b=s["go_b"], sweep=tab), indent=1,
                     default=float))


if __name__ == "__main__":
    {"A": stage_a, "B": stage_b}[sys.argv[1]]()
