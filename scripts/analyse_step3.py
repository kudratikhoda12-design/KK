"""D104 Step-3 analysis: from reports/step3/*.jsonl (primary = phase "ensemble" records) to reports/step3_summary.json
and reports/tables/step3_*.csv. H1 per protocol section 6: paired t-interval (df 4, 95% CI) and sign count for
fae_kan - fae_mlp in survival-curve error; also reported: exact Wilcoxon signed-rank and the same summaries for IBS,
Uno C, Harrell C and D-calibration p. Every run is counted, including failed ones."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
METRICS = ["curve_error", "ibs", "uno_c", "harrell_c", "dcal_p"]


def load():
    rows = []
    for f in sorted(glob.glob(str(REP / "step3/*.jsonl"))):
        if Path(f).name.startswith("kanfae_"):
            continue
        for l in Path(f).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                r["group_file"] = Path(f).name
                rows.append(r)
    return pd.DataFrame(rows)


def paired(a, b):
    """b - a over matched replicates: mean, SD, t-interval (df n-1), t p, sign count, exact Wilcoxon p."""
    d = np.asarray(b, float) - np.asarray(a, float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else float("nan")
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")
    p_t = float(stats.ttest_1samp(d, 0.0).pvalue) if n > 1 and sd > 0 else float("nan")
    try:
        p_w = float(stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue) if n > 1 and np.any(d != 0) else float("nan")
    except TypeError:                                     # older scipy
        p_w = float(stats.wilcoxon(d, alternative="two-sided", mode="exact").pvalue)
    return dict(n=n, mean_diff=m, sd_diff=sd, ci_lo=m - h, ci_hi=m + h, p_t=p_t, n_pos=int((d > 0).sum()),
                n_neg=int((d < 0).sum()), p_wilcoxon=p_w)


def main():
    df = load()
    out = {"n_records": int(len(df))}
    if not len(df):
        (REP / "step3_summary.json").write_text(json.dumps(out, indent=1))
        return
    df["variant"] = df["variant"].fillna("{}")
    (REP / "tables").mkdir(exist_ok=True)
    status = df.groupby(["analysis", "phase", "status"]).size().reset_index(name="n")
    status.to_csv(REP / "tables/step3_status.csv", index=False)
    out["status"] = status.to_dict("records")
    E = df[(df.phase == "ensemble") & (df.status == "ok")]
    cells = (E.groupby(["analysis", "setting", "model", "variant"])[METRICS]
             .agg(["mean", "std", "count"]).reset_index())
    cells.columns = ["_".join(c).strip("_") for c in cells.columns]
    cells.to_csv(REP / "tables/step3_cells.csv", index=False)
    F = df[(df.phase == "final") & (df.status == "ok")]
    seedm = F.groupby(["analysis", "setting", "model", "variant"])[["harrell_c", "curve_error", "ibs"]].mean().reset_index()
    seedm.to_csv(REP / "tables/step3_cells_single_seed_mean.csv", index=False)
    tune = df[df.phase == "tune"]
    if len(tune):
        sel = (df[df.phase == "ensemble"].assign(cfg=lambda x: x.config.map(lambda c: {k: v for k, v in json.loads(c).items()
                                                                                     if k in ("lr", "wd")}))
               .groupby(["analysis", "model"])["cfg"].apply(lambda s: [json.dumps(v) for v in s]).to_dict())
        out["tuning_selected_lr_wd"] = {f"{a}|{m}": v for (a, m), v in sel.items()}
    # H1 (primary) and H2/H3 contrasts
    def contrast(an, setting, ma, mb, va="{}", vb="{}"):
        A = E[(E.analysis == an) & (E.setting == setting) & (E.model == ma) & (E.variant == va)].set_index("replicate")
        B = E[(E.analysis == an) & (E.setting == setting) & (E.model == mb) & (E.variant == vb)].set_index("replicate")
        reps = sorted(set(A.index) & set(B.index))
        if not reps:
            return None
        return {"replicates": reps, **{mt: paired(A.loc[reps, mt], B.loc[reps, mt]) for mt in METRICS}}
    out["H1"] = contrast("neural4f_H1", "P0", "fae_mlp", "fae_kan")
    out["H2"] = {m: contrast("neural4f_H2", "P0", m, m, va='{"alpha": 0.0}', vb="{}") for m in ("fae_mlp", "fae_kan")}
    out["H3"] = {m: contrast("neural4f_H3", "P-sparse", m, "flat_ae") for m in ("fae_mlp", "fae_kan")}
    rows = []
    for name, c in [("H1: fae_kan − fae_mlp (P0)", out["H1"])] + \
                   [(f"H2: α 1 − α 0, {m} (P0)", out["H2"][m]) for m in out["H2"]] + \
                   [(f"H3: flat_ae − {m} (P-sparse)", out["H3"][m]) for m in out["H3"]]:
        if c:
            for mt in METRICS:
                rows.append(dict(contrast=name, metric=mt, **c[mt]))
    pd.DataFrame(rows).to_csv(REP / "tables/step3_contrasts.csv", index=False)
    # real data (D105): Nadeau-Bengio corrected resampled t-interval for fae_kan - fae_mlp over the 5 outer splits
    RM = {"neural4f_pbc_E2": ["ibs", "ibs_conditional_ipcw", "uno_c", "harrell_c", "dcal_p"],
          "neural4f_pbc_E1": ["ibs", "ibs_conditional_ipcw", "uno_c", "harrell_c", "dcal_p"],
          "neural4f_cmapss_B": ["ibs", "uno_c", "harrell_c", "rul_test_rmse", "rul_official_rmse"]}
    rrows, rcells = [], []
    for an, mts in RM.items():
        A = E[(E.analysis == an) & (E.model == "fae_mlp")].set_index("replicate")
        B = E[(E.analysis == an) & (E.model == "fae_kan")].set_index("replicate")
        for mname, X in (("fae_mlp", A), ("fae_kan", B)):
            if len(X):
                rcells.append(dict(analysis=an, model=mname, n=len(X), **{f"{m}_mean": float(X[m].mean()) for m in mts if m in X},
                                   **{f"{m}_sd": float(X[m].std(ddof=1)) for m in mts if m in X}))
        reps = sorted(set(A.index) & set(B.index))
        if not reps:
            continue
        frac = float(np.mean(A.loc[reps, "n_test"] / A.loc[reps, "n_train"]))
        for m in mts:
            if m not in A or m not in B:
                continue
            d = (B.loc[reps, m] - A.loc[reps, m]).to_numpy(float)
            J = len(d)
            se = np.sqrt((1 / J + frac) * d.var(ddof=1)) if J > 1 else float("nan")
            h = stats.t.ppf(0.975, J - 1) * se if J > 1 else float("nan")
            rrows.append(dict(analysis=an, metric=m, n_splits=J, mean_diff=float(d.mean()), ci_lo=float(d.mean() - h),
                              ci_hi=float(d.mean() + h), n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()),
                              nb_factor=1 / J + frac))
    if rrows:
        pd.DataFrame(rrows).to_csv(REP / "tables/step3_real_contrasts.csv", index=False)
        pd.DataFrame(rcells).to_csv(REP / "tables/step3_real_cells.csv", index=False)
        out["real"] = rrows
    r1 = REP / "experiment_results.jsonl"
    if r1.exists():
        R = pd.DataFrame([json.loads(l) for l in r1.read_text().splitlines() if l.strip()])
        R = R[R.analysis.isin(["pbc_E2", "pbc_E1", "cmapss_B"]) & (R.phase == "final") & (R.status == "ok")
              & R.dataset.isin(["pbc", "cmapss_FD001"])]
        if "code_version" in R:      # C-MAPSS: the latest code version only (round 1-3 records)
            latest = R[R.analysis == "cmapss_B"].code_version.dropna().max()
            R = R[(R.analysis != "cmapss_B") | (R.code_version == latest)]
        cols = [c for c in ("ibs", "uno_c", "harrell_c", "rul_test_rmse") if c in R]
        comp = R.groupby(["analysis", "model", "replicate"])[cols].mean().groupby(["analysis", "model"])[cols].mean().reset_index()
        comp.to_csv(REP / "tables/step3_real_round1_comparators.csv", index=False)
    # KAN-FAE pruning / symbolic
    p = REP / "step3/kanfae_prune_symbolic.jsonl"
    if p.exists():
        K = pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
        K["enc_active_risk"] = K.apply(lambda r: r["prune_info"][0]["risk_encoder_active_after"]       # member 0
                                       if isinstance(r.get("prune_info"), list) else r["active_edges"]["risk_encoder"], axis=1)
        # symbolic rows exist for replicate 1000 only: report them against that replicate's own earlier stages
        unc = K[K.stage == "pruned_uncentred"].set_index(["variant", "replicate"])["enc_active_risk"]
        sym = K.stage.str.startswith("symbolic")
        K.loc[sym, "enc_active_risk"] = [unc.get((v, r), np.nan) for v, r in zip(K.loc[sym, "variant"], K.loc[sym, "replicate"])]
        r0 = K[K.replicate == 0][["variant", "stage", "harrell_c", "uno_c", "ibs", "curve_error", "reconstruction_test",
                                  "enc_active_risk", "edges_accepted", "edges_fitted", "pysr_accepted"]]
        r0.to_csv(REP / "tables/step3_kanfae_rep1000.csv", index=False)
        K = K[~sym]
        ks = (K.groupby(["variant", "stage"])[["harrell_c", "uno_c", "ibs", "curve_error", "reconstruction_test",
                                               "enc_active_risk"]].agg(["mean", "std", "count"]).reset_index())
        ks.columns = ["_".join(c).strip("_") for c in ks.columns]
        ks.to_csv(REP / "tables/step3_kanfae_prune.csv", index=False)
        out["kanfae_prune"] = ks.to_dict("records")
    e = REP / "step3/kanfae_symbolic_edges.jsonl"
    if e.exists():
        S = pd.DataFrame([json.loads(l) for l in e.read_text().splitlines() if l.strip()])
        sym = S.groupby(["variant", "stages", "member"]).agg(edges=("accepted", "size"), accepted=("accepted", "sum")).reset_index()
        sym["stage_counts"] = [json.dumps(S[(S.variant == r.variant) & (S.stages == r.stages) & (S.member == r.member)
                                            & S.accepted].stage.value_counts().to_dict()) for r in sym.itertuples()]
        sym.to_csv(REP / "tables/step3_symbolic.csv", index=False)
        out["symbolic"] = sym.to_dict("records")
    (REP / "step3_summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("n_records", "H1") if k in out}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
