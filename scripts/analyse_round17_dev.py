"""Round 17 development analysis (docs/round17_devplan.md §4-5; rules fixed before the development runs):
reports/round17/dev/results_lane*.jsonl -> reports/round17/dev_summary.json.

For each last_tol l in the grid, the two-way model of a replicate is its anchored alternative if
anchored_val_loss < ref_val_loss - l, else the comparator; U = upper 95% paired-t bound of (two-way - comparator)
survival-curve error. Also: the b2s_cox re-fit against the Round-16 records (reproduction), the AFT distillation
refits (finiteness, RISE) on S-Misspec, and the go / no-go rule.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_round12 import paired  # noqa: E402

DEV = ROOT / "reports/round17/dev"
GRID = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04)


def load(path=DEV):
    recs = []
    for f in sorted(path.glob("results_lane*.jsonl")):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return recs


def by_kind(recs, kind):
    return {r["seed"]: r for r in recs if r["status"] == "ok" and r["job"] == kind}


def twoway_ce(r, loss, l):
    m = f"b2s_anchor_{loss}_twoway"
    g = r["gates"][m]
    use_anchor = g["anchored_val_loss"] < g["ref_val_loss"] - l
    return (r["anchored_alternative"][m] if use_anchor else r["models"][f"b2s_{loss}"])["curve_error"], use_anchor


def sweep(R, loss):
    out = {}
    for l in GRID:
        a, b, n_anch = {}, {}, 0
        for s, r in R.items():
            a[s], u = twoway_ce(r, loss, l)
            b[s] = r["models"][f"b2s_{loss}"]["curve_error"]
            n_anch += u
        p = paired(a, b)
        p["n_anchored"] = n_anch
        out[str(l)] = p
    return out


def upper(p):
    ci = p.get("ci95")
    if ci is None or ci[1] is None or (isinstance(ci[1], float) and np.isnan(ci[1])):
        return 0.0 if p.get("n", 0) and abs(p.get("mean_diff", 0.0)) < 1e-15 else float("inf")
    return ci[1]


def choose(sw_traj, sw_last, cap=0.0025):
    """Smallest l with every trajectory-scenario U < cap and U(last) < 0; else the l with the smallest worst
    trajectory U among those with U(last) < 0; else 0 (docs/round17_devplan.md §4)."""
    ok_last = [l for l in GRID if upper(sw_last[str(l)]) < 0]
    good = [l for l in ok_last if all(upper(s[str(l)]) < cap for s in sw_traj)]
    if good:
        return min(good), "rule"
    if ok_last:
        return min(ok_last, key=lambda l: max(upper(s[str(l)]) for s in sw_traj)), "fallback: no l met the cap"
    return 0.0, "fallback: no l kept the S-Last gain"


def main():
    recs = load()
    s = dict(n_records=len(recs), n_failed=sum(r["status"] != "ok" for r in recs),
             failed=[dict(job=r["job_name"], error=r.get("error")) for r in recs if r["status"] != "ok"])
    SP, LA, MI = by_kind(recs, "sparse"), by_kind(recs, "last"), by_kind(recs, "misspec")
    s["n"] = dict(sparse=len(SP), last=len(LA), misspec=len(MI))
    sw = {"cox": {"sparse": sweep(SP, "cox"), "last": sweep(LA, "cox")},
          "aft": {"sparse": sweep(SP, "aft"), "last": sweep(LA, "aft"), "misspec": sweep(MI, "aft")}}
    s["sweep"] = sw
    lc, why_c = choose([sw["cox"]["sparse"]], sw["cox"]["last"])
    la, why_a = choose([sw["aft"]["misspec"], sw["aft"]["sparse"]], sw["aft"]["last"])
    s["last_tol"] = dict(cox=lc, cox_reason=why_c, aft=la, aft_reason=why_a)
    # Round-16 reproduction of b2s_cox (same seeds, same code path)
    r16 = {}
    for f in sorted((ROOT / "reports/round16").glob("results_lane*.jsonl")):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if r.get("status") == "ok" and r["job"] in ("sparse", "last"):
                r16[(r["job"], r["seed"])] = r["b2s_cox"]["curve_error"]
    diffs = [abs(r["models"]["b2s_cox"]["curve_error"] - r16[(k, sd)]) for k, R in (("sparse", SP), ("last", LA))
             for sd, r in R.items() if (k, sd) in r16]
    s["r16_reproduction_b2s_cox"] = dict(n=len(diffs), max_abs_diff=max(diffs) if diffs else None)
    # AFT symbolic on S-Misspec
    rise = lambda v: {sd: r["variants"][v]["shape"]["mean_id"] for sd, r in MI.items() if "shape" in r["variants"][v]}  # noqa: E731
    s["aft_symbolic"] = dict(
        n=len(MI), n_distill_finite=sum("shape" in r["variants"]["distill_fixed"] for r in MI.values()),
        n_fallback=sum(bool(r["symbolic"]["refit_fallback"]) for r in MI.values()),
        n_rejected=sum(len(r["symbolic"]["rejected"]) for r in MI.values()),
        rise_means={v: float(np.mean(list(rise(v).values()))) if rise(v) else None
                    for v in ("after_pruning", "sub_guard", "refit_plain", "distill_fixed")},
        distill_minus_pruned_rise=paired(rise("distill_fixed"), rise("after_pruning"), margin=0.05),
        b2s_aft_rise_mean=float(np.mean([r["models"]["b2s_aft"]["shape"]["mean_id"] for r in MI.values()])) if MI else None)
    # go / no-go (devplan §5)
    s["go"] = dict(cox_sparse_upper=upper(sw["cox"]["sparse"][str(lc)]), cox_last_upper=upper(sw["cox"]["last"][str(lc)]),
                   aft_all_finite=s["aft_symbolic"]["n_distill_finite"] == len(MI) == 9)
    s["go"]["go"] = bool(s["go"]["cox_sparse_upper"] < 0.005 and s["go"]["cox_last_upper"] < 0 and s["go"]["aft_all_finite"])
    (ROOT / "reports/round17/dev_summary.json").write_text(json.dumps(s, indent=1, default=float))
    tab = {f"{loss}/{sc}": {l: (round(p.get("mean_diff", float("nan")), 4), round(upper(p), 4), p["n_anchored"])
                            for l, p in d.items()} for loss, x in sw.items() for sc, d in x.items()}
    print(json.dumps(dict(n=s["n"], failed=s["n_failed"], last_tol=s["last_tol"], go=s["go"], sweep=tab,
                          repro=s["r16_reproduction_b2s_cox"], aft_symbolic={k: v for k, v in s["aft_symbolic"].items()
                                                                            if k != "distill_minus_pruned_rise"}),
                     indent=1, default=float))


if __name__ == "__main__":
    main()
