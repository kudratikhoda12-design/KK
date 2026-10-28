"""Round 18 development analysis (docs/round18_devplan.md §4-5; rules fixed before the development runs):
reports/round18/dev/results_lane*.jsonl -> reports/round18/dev_summary.json.

For each margin in the grid, the confirm model of a replicate is the anchored alternative if
loss_confirm(anchored) < loss_confirm(comparator) - margin, else the comparator; U = upper 95% paired-t bound of
(confirm model - comparator) survival-curve error. Diagnostics: the mean (comparator - anchored) loss on val_select and on
val_confirm and their correlation with the test difference of the anchored alternative.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_round12 import paired  # noqa: E402
from analyse_round17_dev import by_kind, choose, load, upper  # noqa: E402

DEV = ROOT / "reports/round18/dev"
GRID = (0.0, 0.0025, 0.005, 0.01, 0.02)


def confirm_ce(r, loss, l):
    m = f"b2s_anchor_{loss}_confirm"
    g = r["gates"][m]
    use = g["confirm_loss_anchored"] < g["confirm_loss_ref"] - l
    return (r["anchored_alternative"][m] if use else r["models"][f"b2s_{loss}"])["curve_error"], use


def sweep(R, loss):
    out = {}
    for l in GRID:
        a, b, n = {}, {}, 0
        for s, r in R.items():
            a[s], u = confirm_ce(r, loss, l)
            b[s] = r["models"][f"b2s_{loss}"]["curve_error"]
            n += u
        p = paired(a, b)
        p["n_anchored"] = n
        out[str(l)] = p
    return out


def diagnostics(R, loss):
    m = f"b2s_anchor_{loss}_confirm"
    sel = [r["gates"][m]["select_loss_ref"] - r["gates"][m]["select_loss_anchored"] for r in R.values()]
    con = [r["gates"][m]["confirm_loss_ref"] - r["gates"][m]["confirm_loss_anchored"] for r in R.values()]
    dif = [r["anchored_alternative"][m]["curve_error"] - r["models"][f"b2s_{loss}"]["curve_error"] for r in R.values()]
    cor = lambda a: float(np.corrcoef(a, dif)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(dif) > 0 else None  # noqa: E731
    return dict(n=len(sel), select_gap_mean=float(np.mean(sel)), confirm_gap_mean=float(np.mean(con)),
                alternative_test_diff_mean=float(np.mean(dif)), corr_select_gap_test=cor(sel), corr_confirm_gap_test=cor(con),
                n_confirm_gap_positive=int(sum(c > 0 for c in con)), n_select_gap_positive=int(sum(c > 0 for c in sel)))


def main():
    recs = load(DEV)
    s = dict(n_records=len(recs), n_failed=sum(r["status"] != "ok" for r in recs),
             failed=[dict(job=r["job_name"], error=r.get("error")) for r in recs if r["status"] != "ok"])
    LA, SP, MI = by_kind(recs, "last"), by_kind(recs, "sparse"), by_kind(recs, "misspec")
    s["n"] = dict(last=len(LA), sparse=len(SP), misspec=len(MI))
    import analyse_round17_dev as A17
    A17.GRID = GRID                                            # choose() iterates over the Round-18 grid
    sw = {"cox": {"sparse": sweep(SP, "cox"), "last": sweep(LA, "cox")},
          "aft": {"sparse": sweep(SP, "aft"), "last": sweep(LA, "aft"), "misspec": sweep(MI, "aft")}}
    s["sweep"] = sw
    lc, why_c = choose([sw["cox"]["sparse"]], sw["cox"]["last"])
    la, why_a = choose([sw["aft"]["misspec"], sw["aft"]["sparse"]], sw["aft"]["last"])
    s["margin"] = dict(cox=lc, cox_reason=why_c, aft=la, aft_reason=why_a)
    s["diagnostics"] = {f"{loss}/{k}": diagnostics(R, loss) for loss, k, R in
                        (("cox", "sparse", SP), ("cox", "last", LA), ("aft", "sparse", SP), ("aft", "last", LA), ("aft", "misspec", MI))}
    s["fullval_minus_select_comparator"] = {f"{loss}/{k}": paired(
        {sd: r["models"][f"b2s_{loss}_fullval"]["curve_error"] for sd, r in R.items()},
        {sd: r["models"][f"b2s_{loss}"]["curve_error"] for sd, r in R.items()})
        for loss, k, R in (("cox", "sparse", SP), ("cox", "last", LA), ("aft", "sparse", SP), ("aft", "last", LA), ("aft", "misspec", MI))}
    s["go"] = dict(cox_sparse_upper=upper(sw["cox"]["sparse"][str(lc)]), cox_last_upper=upper(sw["cox"]["last"][str(lc)]),
                   aft_misspec_upper=upper(sw["aft"]["misspec"][str(la)]), aft_last_upper=upper(sw["aft"]["last"][str(la)]))
    g = s["go"]
    g["go"] = bool(g["cox_sparse_upper"] < 0.005 and g["cox_last_upper"] < 0 and g["aft_misspec_upper"] < 0.005
                   and g["aft_last_upper"] < 0)
    (ROOT / "reports/round18/dev_summary.json").write_text(json.dumps(s, indent=1, default=float))
    tab = {f"{loss}/{sc}": {l: (round(p.get("mean_diff", float("nan")), 4), round(upper(p), 4), p["n_anchored"])
                            for l, p in d.items()} for loss, x in sw.items() for sc, d in x.items()}
    print(json.dumps(dict(n=s["n"], failed=s["n_failed"], margin=s["margin"], go=s["go"], sweep=tab,
                          diagnostics=s["diagnostics"]), indent=1, default=float))


if __name__ == "__main__":
    main()
