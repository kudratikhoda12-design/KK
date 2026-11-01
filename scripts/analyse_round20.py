"""Round 20 formal analysis (D150): reports/round20/results_lane*.jsonl -> reports/round20_summary.json.

H-R20a (if GO-a): S-Misspec-lognormal 4900-4919, survival-curve error, upper 95% bound of
  b2s_npaft_fullval - b2s_cox_fullval < 0 AND of b2s_npaft_fullval - b2s_weibull_fullval < 0.
  Registered secondary: S-Sparse 4800-4819 NP-AFT - Cox (upper bound against +0.005); exploratory: NP-AFT - log-normal
  AFT on S-Misspec.
H-R20b (if GO-b): per family f in (cox, lnaft, npaft), b2s_anchor_f_cross - b2s_f_fullval: S-Last 4700-4719 upper bound
  < 0 AND trajectory scenario (Cox: S-Sparse; AFT families: S-Misspec) upper bound < +0.005; supported iff all three.
Every contrast: paired mean, 95% t-interval, two-sided paired-t p, wins / losses / ties. Secondary: confirmation rates,
contrasts against the h1 B2-S and landmark Cox, the Round-18-style one-vs-one losses. No multiplicity correction.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_round12 import paired  # noqa: E402
from analyse_round17_dev import by_kind, load, upper  # noqa: E402
from run_round20 import SETTINGS  # noqa: E402

R20 = ROOT / "reports/round20"
TRAJ = {"cox": "sparse", "lnaft": "misspec", "npaft": "misspec"}


def ce(R, model, metric="curve_error"):
    return {s: r["models"][model][metric] for s, r in R.items() if model in r["models"]}


def means(R, metric="curve_error"):
    ks = sorted({k for r in R.values() for k in r["models"]})
    return {k: float(np.mean(list(ce(R, k, metric).values()))) for k in ks}


def cross(R, f):
    m, fv = f"b2s_anchor_{f}_cross", f"b2s_{f}_fullval"
    if not R or m not in next(iter(R.values()))["models"]:
        return None
    g = [r["gates"][m] for r in R.values()]
    dirs = [x["dirs"] for x in g]
    out = dict(n=len(R), contrast=paired(ce(R, m), ce(R, fv)), contrast_ibs=paired(ce(R, m, "ibs"), ce(R, fv, "ibs")),
               vs_h1=paired(ce(R, m), ce(R, f"b2s_{f}_h1")),
               refit_always=paired({s: r["alternatives"][m]["refit"]["curve_error"] for s, r in R.items()}, ce(R, fv)),
               h1_anchored_vs_fullval=paired({s: r["alternatives"][m]["h1_anchored"]["curve_error"] for s, r in R.items()},
                                             ce(R, fv)),
               n_confirmed=sum(x["confirmed"] for x in g), margin=g[0]["margin"], relative_margin=g[0]["relative_margin"],
               n_h1_favours=sum(d[0]["loss_anchored"] < d[0]["loss_ref"] for d in dirs),
               n_h2_favours=sum(d[1]["loss_anchored"] < d[1]["loss_ref"] for d in dirs),
               n_r18_rule=sum(d[0]["r18_loss_anchored"] < d[0]["r18_loss_ref"] for d in dirs),
               last_on_frequency=np.mean([x["last_on_h1"] for x in g], 0).tolist(),
               traj_on_frequency=np.mean([x["traj_on_h1"] for x in g], 0).tolist())
    if ce(R, "landmark_cox"):
        out["vs_landmark"] = paired(ce(R, m), ce(R, "landmark_cox"))
    return out


def main():
    st = SETTINGS["formal"]
    recs = load(R20)
    K = {k: by_kind(recs, k) for k in ("last", "sparse", "misspec")}
    s = dict(n_records=len(recs), n_failed=sum(r["status"] != "ok" for r in recs),
             failed=[dict(job=r["job_name"], error=r.get("error")) for r in recs if r["status"] != "ok"],
             n={k: len(v) for k, v in K.items()}, settings=st,
             means={k: means(v) for k, v in K.items()}, ibs_means={k: means(v, "ibs") for k, v in K.items()})
    MI, SP = K["misspec"], K["sparse"]
    if st["go_a"]:
        a = dict(misspec_vs_cox=paired(ce(MI, "b2s_npaft_fullval"), ce(MI, "b2s_cox_fullval")),
                 misspec_vs_weibull=paired(ce(MI, "b2s_npaft_fullval"), ce(MI, "b2s_weibull_fullval")),
                 sparse_vs_cox=paired(ce(SP, "b2s_npaft_fullval"), ce(SP, "b2s_cox_fullval")),
                 misspec_vs_lognormal=paired(ce(MI, "b2s_npaft_fullval"), ce(MI, "b2s_lnaft_fullval")),
                 misspec_vs_cox_ibs=paired(ce(MI, "b2s_npaft_fullval", "ibs"), ce(MI, "b2s_cox_fullval", "ibs")),
                 misspec_vs_weibull_ibs=paired(ce(MI, "b2s_npaft_fullval", "ibs"), ce(MI, "b2s_weibull_fullval", "ibs")))
        a["upper_vs_cox"], a["upper_vs_weibull"] = upper(a["misspec_vs_cox"]), upper(a["misspec_vs_weibull"])
        a["sparse_upper"] = upper(a["sparse_vs_cox"])
        a["supported"] = bool(a["misspec_vs_cox"]["n"] >= 2 and a["upper_vs_cox"] < 0 and a["upper_vs_weibull"] < 0)
        a["secondary_sparse_noninferior"] = bool(a["sparse_upper"] < 0.005)
        s["H_R20a"] = a
    else:
        s["H_R20a"] = "not tested (GO-a failed)"
    if st["go_b"]:
        b = {}
        for f in ("cox", "lnaft", "npaft"):
            cl, ct = cross(K["last"], f), cross(K[TRAJ[f]], f)
            b[f] = dict(last=cl, traj=ct, traj_scenario=TRAJ[f], last_upper=upper(cl["contrast"]),
                        traj_upper=upper(ct["contrast"]))
            b[f]["supported"] = bool(b[f]["last_upper"] < 0 and b[f]["traj_upper"] < 0.005)
        b["supported"] = all(b[f]["supported"] for f in ("cox", "lnaft", "npaft"))
        s["H_R20b"] = b
    else:
        s["H_R20b"] = "not tested (GO-b failed)"
    (ROOT / "reports/round20_summary.json").write_text(json.dumps(s, indent=1, default=float))
    brief = dict(n_records=s["n_records"], n_failed=s["n_failed"], n=s["n"])
    if isinstance(s["H_R20a"], dict):
        brief["H_R20a"] = {k: (v if not isinstance(v, dict) else dict(mean=v.get("mean_diff"), ci95=v.get("ci95")))
                           for k, v in s["H_R20a"].items()}
    if isinstance(s["H_R20b"], dict):
        brief["H_R20b"] = {f: (dict(last=s["H_R20b"][f]["last"]["contrast"].get("ci95"),
                                    traj=s["H_R20b"][f]["traj"]["contrast"].get("ci95"),
                                    supported=s["H_R20b"][f]["supported"]) if f != "supported" else s["H_R20b"][f])
                           for f in s["H_R20b"]}
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    main()
