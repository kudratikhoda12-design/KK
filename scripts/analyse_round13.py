"""Round 13 analysis (D129; exploratory throughout). Reads reports/round13/results_lane*.jsonl and the stability npz files;
writes reports/round13_summary.json, reports/tables/r13_*.csv and reports/figures/r13_effects_<dataset>.png."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R13 = ROOT / "reports/round13"
TAB, FIG = ROOT / "reports/tables", ROOT / "reports/figures"
DIRECTIONS = {"pbc_E2": {"log bili": +1, "albumin": -1, "log protime": +1},
              "prothro": {"prothrombin index": -1}, "aids": {"sqrt CD4": -1}}
CONTRAST_REFS = ("afcm_v3", "landmark_cox", "lme2_cox", "sum_cox", "fpca_cox", "b1_fpca1_cox")


def load():
    recs = []
    for f in sorted(R13.glob("results_lane*.jsonl")):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return recs


def first_ok(recs, **kw):
    out = {}
    for r in recs:
        if r.get("status") == "ok" and all(r.get(k) == v for k, v in kw.items()) and r["split"] not in out:
            out[r["split"]] = r
    return out


def nb(a, b, frac):
    """Nadeau-Bengio corrected t-interval for the mean of a - b over the common splits."""
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    J = len(d)
    if J < 2:
        return dict(n=J)
    se = np.sqrt((1 / J + frac) * d.var(ddof=1))
    tq = stats.t.ppf(0.975, J - 1)
    return dict(n=J, mean_a=float(np.mean([a[k] for k in ks])), mean_b=float(np.mean([b[k] for k in ks])),
                mean_diff=float(d.mean()), ci95=[float(d.mean() - tq * se), float(d.mean() + tq * se)],
                p=float(2 * stats.t.sf(abs(d.mean() / se), J - 1)) if se > 0 else float("nan"),
                n_neg=int((d < 0).sum()), n_pos=int((d > 0).sum()))


def single_tables(recs):
    rows, contrasts = [], {}
    for ds in ("pbc_E2", "pbc_E1", "prothro", "aids"):
        models = sorted({r["model"] for r in recs if r.get("job") == "single" and r["dataset"] == ds})
        M = {m: first_ok(recs, job="single", dataset=ds, model=m) for m in models}
        for m, R in M.items():
            if m == "prune_symbolic":
                continue
            row = dict(dataset=ds, model=m, n_splits=len(R))
            for k in ("ibs", "ibs_cond", "harrell_c", "uno_c", "dcal_p"):
                v = [r["metrics"][k] for r in R.values()]
                if v:
                    row[f"{k}_mean"], row[f"{k}_sd"] = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            v = [r["metrics"]["dcal_p"] for r in R.values()]
            row["dcal_n_p_lt_005"] = int(sum(x < 0.05 for x in v))
            v = [r["metrics_seed0"]["uno_c"] for r in R.values() if "metrics_seed0" in r]
            if v:
                row["uno_c_seed0_mean"] = float(np.mean(v))
            rows.append(row)
        b2 = M.get("b2s_cox", {})
        if b2:
            frac = np.mean([r["n_test"] / r["n_train"] for r in b2.values()])
            for ref in CONTRAST_REFS:
                for k in ("ibs", "ibs_cond", "uno_c"):
                    contrasts[f"{ds}: {k} b2s_cox - {ref}"] = nb({s: r["metrics"][k] for s, r in b2.items()},
                                                                 {s: r["metrics"][k] for s, r in M.get(ref, {}).items()}, frac)
    return pd.DataFrame(rows), contrasts


def prune_symbolic(recs):
    out = {}
    for ds in ("pbc_E2", "pbc_E1", "prothro", "aids"):
        R = first_ok(recs, job="single", dataset=ds, model="prune_symbolic")
        if not R:
            continue
        ch = next(iter(R.values()))["channels"]
        sel = np.array([r["selected"] for r in R.values()])
        g = lambda r, k, m: (r.get(k) or {}).get(m)  # noqa: E731
        out[ds] = dict(n_splits=len(R), channels=ch, selection_frequency=sel.mean(0).tolist(),
                       acceptance_rate=float(np.mean([r["symbolic"]["acceptance_rate"] for r in R.values()])),
                       mean_r2_val=float(np.nanmean([r["symbolic"]["mean_r2_val_accepted"] or np.nan for r in R.values()])),
                       fidelity_corr_no_refit=float(np.nanmean([r.get("fidelity_corr_risk") or np.nan for r in R.values()])),
                       fidelity_corr_refit=float(np.nanmean([r.get("fidelity_corr_risk_refit") or np.nan for r in R.values()])),
                       uno_c_pruned=float(np.nanmean([g(r, "after_pruning", "uno_c") or np.nan for r in R.values()])),
                       uno_c_no_refit=float(np.nanmean([g(r, "substituted_no_refit", "uno_c") or np.nan for r in R.values()])),
                       uno_c_refit=float(np.nanmean([g(r, "substituted_refit", "uno_c") or np.nan for r in R.values()])),
                       ibs_pruned=float(np.nanmean([g(r, "after_pruning", "ibs") or np.nan for r in R.values()])),
                       ibs_no_refit=float(np.nanmean([g(r, "substituted_no_refit", "ibs") or np.nan for r in R.values()])),
                       ibs_refit=float(np.nanmean([g(r, "substituted_refit", "ibs") or np.nan for r in R.values()])),
                       n_refit_nonfinite=int(sum((r.get("substituted_refit") or {}).get("status") == "refit_nonfinite"
                                                 for r in R.values())),
                       formulas_split0=[e for e in R.get(0, next(iter(R.values())))["symbolic"]["edges"]
                                        if e["accepted"] and e["layer"] in ("g", "stat") and e["share"] >= 0.05])
    return out


def b3(recs):
    out = {}
    for ds in ("pbc_E2", "prothro", "aids"):
        res = {}
        for m in ("b3_coxkan", "b3_cox"):
            R = first_ok(recs, job="b3", dataset=ds, model=m)
            per = {}
            for s, r in R.items():
                for x in r["per_landmark"]:
                    per.setdefault(x["landmark"], []).append(x)
            res[m] = {lm: dict(n_splits=len(v), n_test=float(np.mean([x["n_test"] for x in v])),
                               brier=float(np.mean([x["brier"] for x in v if x.get("brier") is not None])),
                               harrell_c=float(np.mean([x["harrell_c"] for x in v if x.get("harrell_c") is not None])),
                               td_auc=(float(np.mean([x["td_auc"] for x in v if x.get("td_auc") is not None]))
                                       if any(x.get("td_auc") is not None for x in v) else None),
                               n_auc=int(sum(x.get("td_auc") is not None for x in v)))
                       for lm, v in sorted(per.items())}
        out[ds] = res
    return out


def cmapss(recs):
    M = {m: first_ok(recs, job="cmapss", model=m) for m in ("b2s_aft", "b2s_aft_rc", "sum_aft", "fpca_aft")}
    rows = []
    for m, R in M.items():
        if R:
            rows.append(dict(model=m, n_splits=len(R), **{f"{k}_mean": float(np.mean([r["metrics"][k] for r in R.values()]))
                                                          for k in ("ibs", "harrell_c", "uno_c", "dcal_p")},
                             dcal_n_p_lt_005=int(sum(r["metrics"]["dcal_p"] < 0.05 for r in R.values()))))
    con = {}
    rc = M["b2s_aft_rc"]
    if rc:
        frac = np.mean([r["n_test"] / r["n_train"] for r in rc.values()])
        for ref in ("b2s_aft", "sum_aft", "fpca_aft"):
            for k in ("ibs", "dcal_p"):
                con[f"C-MAPSS: {k} b2s_aft_rc - {ref}"] = nb({s: r["metrics"][k] for s, r in rc.items()},
                                                            {s: r["metrics"][k] for s, r in M[ref].items()}, frac)
    return rows, con


def stability():
    out = {}
    for ds in ("pbc_E2", "prothro", "aids"):
        for model in ("b2s_cox", "b1_fpca1_cox"):
            files = sorted((R13 / "stability").glob(f"{ds}_{model}_split*.npz"))
            if not files:
                continue
            Z = [np.load(f) for f in files]
            eff = np.concatenate([z["effects"] for z in Z])                    # (fits, D, X, T)
            xg, ch = Z[0]["x_grid"], [str(c) for c in Z[0]["channels"]]
            res = {}
            for d, c in enumerate(ch):
                avg = eff[:, d].mean(2)                                           # time-averaged (fits, X)
                avg = avg - avg.mean(1, keepdims=True)
                full = eff[:, d].reshape(len(eff), -1)
                full = full - full.mean(1, keepdims=True)
                res[c] = dict(n_fits=int(len(eff)), r2_mean=_pairwise(avg, "r2"), cos_mean=_pairwise(avg, "cos"),
                              r2_mean_surface=_pairwise(full, "r2"), cos_mean_surface=_pairwise(full, "cos"))
                sgn = DIRECTIONS.get(ds, {}).get(c)
                if sgn is not None:
                    diffs = np.diff(avg, axis=1)
                    rng_ = np.ptp(avg, axis=1, keepdims=True)
                    ok = np.all(sgn * diffs >= -0.01 * rng_, axis=1) & (rng_[:, 0] > 0)
                    res[c].update(direction="increasing" if sgn > 0 else "decreasing",
                                  frac_aligned=float(ok.mean()), aligned=bool(ok.mean() >= 0.8))
            out[f"{ds}/{model}"] = res
            plot_effects(ds, model, eff, xg, ch)
    return out


def _pairwise(A, kind):
    n = len(A)
    nrm = np.linalg.norm(A, axis=1)
    C = (A @ A.T) / np.outer(np.where(nrm > 0, nrm, 1), np.where(nrm > 0, nrm, 1))       # cosine (centred vectors)
    iu = np.triu_indices(n, 1)
    vals = C[iu] ** 2 if kind == "r2" else C[iu]
    return float(np.mean(vals))


def plot_effects(ds, model, eff, xg, ch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = len(ch)
    fig, axes = plt.subplots(1, D, figsize=(3.0 * D, 2.6), squeeze=False)
    for d, c in enumerate(ch):
        avg = eff[:, d].mean(2)
        avg = avg - avg.mean(1, keepdims=True)
        x = np.exp(xg[d]) if c.startswith("log ") else xg[d]
        lo, med, hi = np.percentile(avg, [5, 50, 95], axis=0)
        ax = axes[0, d]
        ax.fill_between(x, lo, hi, alpha=0.3, color="#4C72B0", linewidth=0)
        ax.plot(x, med, color="#4C72B0", linewidth=1.5)
        ax.axhline(0, color="0.6", linewidth=0.6)
        ax.set_title(c.replace("log ", ""), fontsize=9)
        ax.set_xlabel(c.replace("log ", "") + (" (original units)" if c.startswith("log ") else ""), fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("centred log-hazard effect", fontsize=8)
    fig.suptitle(f"{ds}, {model}: time-averaged effect, median and 90% band over {len(eff)} bootstrap fits", fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"r13_effects_{ds}_{model}.png", dpi=130)
    plt.close(fig)


def main():
    recs = load()
    T, con = single_tables(recs)
    cm_rows, cm_con = cmapss(recs)
    S = dict(n_records=len(recs), n_failed=int(sum(r.get("status") == "failed" for r in recs)),
             failed=[dict(job=r["job"], dataset=r.get("dataset"), split=r.get("split"), model=r["model"],
                          error=r.get("error")) for r in recs if r.get("status") == "failed"],
             contrasts=con, prune_symbolic=prune_symbolic(recs), b3=b3(recs), cmapss=cm_rows, cmapss_contrasts=cm_con,
             stability=stability())
    TAB.mkdir(parents=True, exist_ok=True)
    T.to_csv(TAB / "r13_single_by_model.csv", index=False)
    (ROOT / "reports/round13_summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps({k: S[k] for k in ("n_records", "n_failed")}, indent=1))


if __name__ == "__main__":
    main()
