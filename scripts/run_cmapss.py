"""Stage F: C-MAPSS FD001 with the validated non-neural baselines (neural models BLOCKED, D33).

Analyses (each labelled; D3b-R unresolved, neither option is the approved design):
  cmapss_A            Option A, single landmark s0 = (min training-split life) - 1, full history window [1, s0]
  cmapss_B            Option B, trailing 30-cycle windows every 10 cycles, s/100 as covariate
  cmapss_A_artificial Option A with ARTIFICIAL independent censoring of training/validation targets
RUL metrics (median residual life) are reported separately from survival-framing metrics.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kanfae.data.cmapss import informative_sensors, official_test_life, option_a, option_b, read_fd  # noqa: E402
from kanfae.data.schema import Observed  # noqa: E402
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.evaluation import metrics as M  # noqa: E402
from kanfae.experiments.runner import median_from_curve, tune_fit_evaluate  # noqa: E402

MODELS = ["fpca_aft", "fpca_cox", "sum_aft", "sum_cox", "sum_rsf", "km_null"]
BANDS = [(30, 100), (100, 200), (200, 10 ** 6)]


def cluster_bootstrap(groups, stat, n_boot=1000, seed=0):
    """D62: percentile 95% CI of stat(idx) under resampling of whole engines (clusters) with replacement."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    members = {g: np.nonzero(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(ug, len(ug), replace=True)])
        vals.append(stat(idx))
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if len(vals) else (np.nan, np.nan)


def rul_metrics(model, obs, grid, prefix):
    S = model.predict_survival(obs, grid)
    med, flag = median_from_curve(S, grid)
    err = med - obs.y_obs
    out = {f"{prefix}_rmse": float(np.sqrt(np.mean(err ** 2))), f"{prefix}_mae": float(np.mean(np.abs(err))),
           f"{prefix}_n": int(len(obs)), f"{prefix}_median_not_reached": int(flag.sum()),
           f"{prefix}_harrell_c": M.harrell_c(obs.y_obs, obs.delta, model.predict_risk(obs))}
    if "landmark" in obs.meta and obs.meta.get("option") == "B" and prefix == "rul_test":
        lm, grp = obs.meta["landmark"], obs.meta["group"]
        w = np.array([1.0 / np.sum(grp == g) for g in grp])
        out["rul_test_rmse_engine_weighted"] = float(np.sqrt(np.sum(w * err ** 2) / w.sum()))
        risk = model.predict_risk(obs)
        out["rul_test_n_engines"] = int(len(np.unique(grp)))
        stats = {"rmse": lambda i: np.sqrt(np.mean(err[i] ** 2)), "mae": lambda i: np.mean(np.abs(err[i])),
                 "harrell_c": lambda i: M.harrell_c(obs.y_obs[i], obs.delta[i], risk[i])}
        for k, fn in stats.items():                       # D62: engine-cluster bootstrap (all windows)
            out[f"rul_test_{k}_cb_lo"], out[f"rul_test_{k}_cb_hi"] = cluster_bootstrap(grp, fn)
        for lo, hi in BANDS:
            sel = (lm >= lo) & (lm < hi)
            if sel.sum() >= 5:                            # D62: per landmark band, engines resampled within band
                sb = np.nonzero(sel)[0]
                for k in ("rmse", "harrell_c"):
                    out[f"band_{lo}_{min(hi, 999)}_{k}_cb_lo"], out[f"band_{lo}_{min(hi, 999)}_{k}_cb_hi"] = \
                        cluster_bootstrap(grp[sb], lambda i, sb=sb, k=k: stats[k](sb[i]))
                out[f"band_{lo}_{min(hi, 999)}_n_engines"] = int(len(np.unique(grp[sb])))
            if sel.sum() >= 5:
                out[f"band_{lo}_{min(hi, 999)}_rmse"] = float(np.sqrt(np.mean(err[sel] ** 2)))
                out[f"band_{lo}_{min(hi, 999)}_harrell_c"] = M.harrell_c(obs.y_obs[sel], obs.delta[sel],
                                                                         model.predict_risk(obs.subset(np.nonzero(sel)[0])))
                out[f"band_{lo}_{min(hi, 999)}_n"] = int(sel.sum())
    return out


def censor_artificially(obs, rng, cmax):
    C = rng.uniform(0, cmax, len(obs))
    y = np.minimum(obs.y_obs, C)
    d = obs.y_obs <= C
    return Observed(obs.ids, obs.times, obs.values, obs.mask, obs.n_obs, y, d, obs.u, obs.channel_names,
                    {**obs.meta, "artificial_censoring": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--analyses", default="cmapss_A,cmapss_B,cmapss_A_artificial")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"],
                    help="single-operating-condition subsets only; FD002/FD004 need condition normalisation (D51)")
    ap.add_argument("--code-version", default="v1", help="recorded in every record; distinguishes re-runs after fixes")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    tr_df, te_df, rul = read_fd(ROOT / "data" / "raw" / "cmapss", args.subset)
    units = np.array(sorted(tr_df.unit.unique()))
    life_test = official_test_life(te_df, rul)
    life_tr = tr_df.groupby("unit").cycle.max().to_dict()
    (ROOT / "configs" / "cmapss").mkdir(parents=True, exist_ok=True)
    for analysis in args.analyses.split(","):
        (ROOT / "configs" / "cmapss" / (f"{analysis}.json" if args.subset == "FD001"
                                         else f"{analysis}_{args.subset}.json")).write_text(json.dumps(
            {"analysis": analysis, "subset": args.subset, "split_seeds": list(range(args.splits)), "models": MODELS,
             "option_B": {"w": 30, "step": 10}, "artificial_censoring": "C ~ U(0, max training residual)"
             if "artificial" in analysis else None}, indent=2))
        for sp in range(args.splits):
            t0 = time.time()
            a, b, c = split_groups(units, sp)
            u_tr, u_va, u_te = units[a], units[b], units[c]
            sens = informative_sensors(tr_df, u_tr)
            extra_info = {}
            if analysis.startswith("cmapss_A"):
                s0 = int(min(life_tr[u] for u in u_tr)) - 1
                o_tr, _ = option_a(tr_df, u_tr, s0, sens)
                o_va, ex_va = option_a(tr_df, u_va, s0, sens)
                o_te, ex_te = option_a(tr_df, u_te, s0, sens)
                o_off, ex_off = option_a(te_df, sorted(te_df.unit.unique()), s0, sens, life=life_test)
                extra_info = dict(s0=s0, excluded_val=len(ex_va), excluded_test=len(ex_te),
                                  excluded_official=len(ex_off))
                if analysis.endswith("artificial"):
                    rng = np.random.default_rng(100 + sp)
                    cmax = float(o_tr.y_obs.max())
                    o_tr, o_va = censor_artificially(o_tr, rng, cmax), censor_artificially(o_va, rng, cmax)
                    extra_info["artificial_censored_frac_train"] = float(1 - o_tr.delta.mean())
            else:
                o_tr, o_va, o_te = (option_b(tr_df, u, sens) for u in (u_tr, u_va, u_te))
                o_off = option_b(te_df, sorted(te_df.unit.unique()), sens, life=life_test, last_only=True)
                extra_info = dict(n_windows_train=len(o_tr), n_windows_test=len(o_te))
            grid = np.linspace(0, 2.0 * float(np.max(o_tr.y_obs)), 2001)
            base = dict(dataset=f"cmapss_{args.subset}", analysis=analysis, setting=analysis, replicate=sp, split_seed=sp,
                        n_train=len(o_tr), n_val=len(o_va), n_test=len(o_te), n_sensors=len(sens),
                        data_kind="REAL (simulated turbofan; RUL benchmark)" + (" + ARTIFICIAL CENSORING"
                                                                                if "artificial" in analysis else ""),
                        censored_frac_test=float(1 - o_te.delta.mean()), code_version=args.code_version, **extra_info)

            def extra(m, seed, o_te=o_te, o_off=o_off, grid=grid):
                out = rul_metrics(m, o_te, grid, "rul_test")
                if len(o_off):
                    out.update(rul_metrics(m, o_off, grid, "rul_official"))
                return out

            for name in args.models.split(","):
                tune_fit_evaluate(name, o_tr, o_va, o_te, base, grid=[{}] if name == "km_null" else None,
                                  extra_eval=extra)
            print(f"[{analysis} split {sp}] done in {time.time() - t0:.1f}s {extra_info}", flush=True)


if __name__ == "__main__":
    main()
