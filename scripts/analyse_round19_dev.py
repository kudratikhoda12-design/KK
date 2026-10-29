"""Round 19 development analysis (docs/round19_devplan.md §4-5; rules fixed before the development runs):
reports/round19/dev/results_lane*.jsonl -> reports/round19/dev_summary.json.

For variant v and margin l, the estimator of a replicate is the refitted anchored model (variant v) if
loss_confirm(anchored) < loss_confirm(val_select B2-S) - l, else the full-validation B2-S; U = upper 95% paired-t bound
of (estimator - full-validation B2-S) survival-curve error.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyse_round17_dev as A17  # noqa: E402
from analyse_round12 import paired  # noqa: E402
from analyse_round17_dev import by_kind, choose, load, upper  # noqa: E402
from analyse_round18_dev import diagnostics  # noqa: E402  (works on records with gates/<model> confirm losses)

DEV = ROOT / "reports/round19/dev"
GRID = (0.0, 0.0025, 0.005, 0.01, 0.02)
VARIANTS = ("retrain", "finetune")


def estimator_ce(r, loss, v, l):
    m = f"b2s_anchor_{loss}_refit"
    g = r["gates"][m]
    use = g["confirm_loss_anchored"] < g["confirm_loss_ref"] - l
    return (r["alternatives"][m][v] if use else r["models"][f"b2s_{loss}_fullval"])["curve_error"], use


def sweep(R, loss, v):
    out = {}
    for l in GRID:
        a, b, n = {}, {}, 0
        for s, r in R.items():
            a[s], u = estimator_ce(r, loss, v, l)
            b[s] = r["models"][f"b2s_{loss}_fullval"]["curve_error"]
            n += u
        p = paired(a, b)
        p["n_anchored"] = n
        out[str(l)] = p
    return out


def _diag_view(R, loss):
    """Round-18-style records for analyse_round18_dev.diagnostics (anchored alternative = the retrain refit)."""
    m = f"b2s_anchor_{loss}_refit"
    V = {}
    for s, r in R.items():
        V[s] = dict(gates={f"b2s_anchor_{loss}_confirm": r["gates"][m]},
                    anchored_alternative={f"b2s_anchor_{loss}_confirm": r["alternatives"][m]["retrain"]},
                    models={f"b2s_{loss}": r["models"][f"b2s_{loss}_fullval"]})
    return V


def main():
    recs = load(DEV)
    s = dict(n_records=len(recs), n_failed=sum(r["status"] != "ok" for r in recs),
             failed=[dict(job=r["job_name"], error=r.get("error")) for r in recs if r["status"] != "ok"])
    LA, SP, MI = by_kind(recs, "last"), by_kind(recs, "sparse"), by_kind(recs, "misspec")
    s["n"] = dict(last=len(LA), sparse=len(SP), misspec=len(MI))
    A17.GRID = GRID
    s["sweep"], choice = {}, {}
    for v in VARIANTS:
        sw = {"cox": {"sparse": sweep(SP, "cox", v), "last": sweep(LA, "cox", v)},
              "aft": {"misspec": sweep(MI, "aft", v), "last": sweep(LA, "aft", v)}}
        s["sweep"][v] = sw
        lc, wc = choose([sw["cox"]["sparse"]], sw["cox"]["last"])
        la, wa = choose([sw["aft"]["misspec"]], sw["aft"]["last"])
        choice[v] = dict(cox=lc, cox_reason=wc, aft=la, aft_reason=wa,
                         worst_traj_upper=max(upper(sw["cox"]["sparse"][str(lc)]), upper(sw["aft"]["misspec"][str(la)])),
                         both_rule=wc == "rule" and wa == "rule")
    ok = [v for v in VARIANTS if choice[v]["both_rule"]]
    if len(ok) == 1:
        v = ok[0]
    elif len(ok) == 2 and choice["retrain"]["cox"] + choice["retrain"]["aft"] != choice["finetune"]["cox"] + choice["finetune"]["aft"]:
        v = min(ok, key=lambda x: choice[x]["cox"] + choice[x]["aft"])
    else:
        cands = ok or list(VARIANTS)
        v = min(cands, key=lambda x: (choice[x]["worst_traj_upper"], VARIANTS.index(x)))
    s["choice_by_variant"], s["variant"] = choice, v
    s["margin"] = dict(cox=choice[v]["cox"], aft=choice[v]["aft"])
    sw = s["sweep"][v]
    g = dict(cox_sparse_upper=upper(sw["cox"]["sparse"][str(s["margin"]["cox"])]),
             cox_last_upper=upper(sw["cox"]["last"][str(s["margin"]["cox"])]),
             aft_misspec_upper=upper(sw["aft"]["misspec"][str(s["margin"]["aft"])]),
             aft_last_upper=upper(sw["aft"]["last"][str(s["margin"]["aft"])]))
    g["go"] = bool(g["cox_sparse_upper"] < 0.005 and g["cox_last_upper"] < 0 and g["aft_misspec_upper"] < 0.005
                   and g["aft_last_upper"] < 0)
    s["go"] = g
    s["diagnostics"] = {f"{loss}/{k}": diagnostics(_diag_view(R, loss), loss) for loss, k, R in
                        (("cox", "sparse", SP), ("cox", "last", LA), ("aft", "last", LA), ("aft", "misspec", MI))}
    s["select_minus_fullval_comparator"] = {f"{loss}/{k}": paired(
        {sd: r["models"][f"b2s_{loss}_select"]["curve_error"] for sd, r in R.items()},
        {sd: r["models"][f"b2s_{loss}_fullval"]["curve_error"] for sd, r in R.items()})
        for loss, k, R in (("cox", "sparse", SP), ("cox", "last", LA), ("aft", "last", LA), ("aft", "misspec", MI))}
    s["r18_estimator_minus_fullval"] = {f"{loss}/{k}": paired(
        {sd: r["models"][f"b2s_anchor_{loss}_confirm_r18"]["curve_error"] for sd, r in R.items()},
        {sd: r["models"][f"b2s_{loss}_fullval"]["curve_error"] for sd, r in R.items()})
        for loss, k, R in (("cox", "sparse", SP), ("cox", "last", LA), ("aft", "last", LA), ("aft", "misspec", MI))}
    (ROOT / "reports/round19/dev_summary.json").write_text(json.dumps(s, indent=1, default=float))
    tab = {f"{vv}/{loss}/{sc}": {l: (round(p.get("mean_diff", float("nan")), 4), round(upper(p), 4), p["n_anchored"])
                                 for l, p in d.items()}
           for vv, x in s["sweep"].items() for loss, y in x.items() for sc, d in y.items()}
    print(json.dumps(dict(n=s["n"], failed=s["n_failed"], variant=v, margin=s["margin"], choice=choice, go=g, sweep=tab),
                     indent=1, default=float))


if __name__ == "__main__":
    main()
