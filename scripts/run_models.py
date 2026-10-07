#!/usr/bin/env python3
"""Stages 5-7: baselines, ML models, walk-forward validation, tuning and feature ablation (DEVELOPMENT ONLY).

Experiment log (every run is listed in reports/tables/experiment_registry.csv, so the number of model
configurations tried is visible for the data-snooping discussion):
  1. LightGBM grid (8 configurations) on feature set E, h = 15 -> chosen by mean validation log loss.
  2. Ablation at h = 15: feature sets A..E x {logit, lgbm(tuned), rf}.
  3. Horizons 5/30/60: sets A, C, E x {logit, lgbm}.
  4. Rule baselines scored as classifiers: momentum (ret_15), order-flow rule (ofi_15_z_1440), prior-only.
  5. Alternative targets on set E, h = 15: 3-class LightGBM, LightGBM regression.
Out-of-fold (validation-block) predictions are stored in data/processed/oof/ for the signal/backtest stage.

    python scripts/run_models.py [--only tuning|ablation|horizons|alt]
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ethof import evaluation as EV  # noqa: E402
from ethof import models as M  # noqa: E402
from ethof import validation as V  # noqa: E402
from ethof.config import Paths, load_config, setup_logging  # noqa: E402
from ethof.features import FEATURE_SETS  # noqa: E402

TAB = ROOT / "reports" / "tables"
log = logging.getLogger("models")


class Ctx:
    def __init__(self, symbol: str = "ETHUSDT"):
        self.cfg = load_config()
        self.rs = self.cfg["research"]
        self.paths = Paths(self.cfg["data_dir"])
        self.sym = symbol
        df = pl.read_parquet(self.paths.processed / "research" / f"{symbol}-research.parquet")
        self.dev = V.development_only(df, self.rs["holdout"]["start"])
        wf = self.rs["walk_forward"]
        self.folds = V.walk_forward_folds(wf["start"], wf["first_validation"], wf["end_exclusive"],
                                          wf["block_months"], timedelta(days=wf["embargo_days"]))
        self.stride = wf["train_stride"]
        self.oof_dir = self.paths.processed / "oof"
        self.oof_dir.mkdir(parents=True, exist_ok=True)


def run_config(ctx: Ctx, name: str, model: str, fset: str, h: int, params: dict | None = None,
               stride: int | None = None, task: str = "binary") -> pl.DataFrame:
    """Walk-forward over all development folds. Returns per-fold metrics; writes OOF predictions."""
    oof_path = ctx.oof_dir / f"{name}.parquet"
    met_path = TAB / "folds" / f"{name}.csv"
    if oof_path.exists() and met_path.exists():
        return pl.read_csv(met_path)
    feats = FEATURE_SETS[fset]
    rows, preds = [], []
    t0 = time.time()
    for fo in ctx.folds:
        tr, va = V.split(ctx.dev, fo, h, stride=stride or ctx.stride)
        tr = tr.filter(pl.col(f"fwd_ret_{h}").is_not_null())
        va = va.filter(pl.col(f"fwd_ret_{h}").is_not_null())
        Xtr = tr.select(feats).to_numpy().astype(np.float32)
        Xva = va.select(feats).to_numpy().astype(np.float32)
        fwd = va[f"fwd_ret_{h}"].to_numpy()
        if task == "binary":
            ytr, yva = tr[f"up_{h}"].to_numpy(), va[f"up_{h}"].to_numpy()
            if model == "lgbm":
                est = M.lightgbm(**(params or {}))
            elif model == "rf":
                est = M.random_forest(**(params or {}))
            else:
                est = M.logistic(**(params or {}))
            est.fit(Xtr, ytr)
            p = est.predict_proba(Xva)[:, 1]
            m = EV.classification(yva, p, fwd)
            preds.append(pl.DataFrame({"ts": va["ts"], "fold": fo.name, "p": p, "y": yva, "fwd": fwd}))
        elif task == "cls3":
            import lightgbm as lgb
            from sklearn.metrics import log_loss, roc_auc_score
            ytr, yva = tr[f"cls3_{h}"].to_numpy(), va[f"cls3_{h}"].to_numpy()
            est = lgb.LGBMClassifier(objective="multiclass", random_state=M.SEED, n_jobs=4, verbose=-1,
                                     **{**M.LGBM_DEFAULT, **(params or {})})
            est.fit(Xtr, ytr)
            P = est.predict_proba(Xva)
            score = P[:, 2] - P[:, 0]
            from scipy.stats import spearmanr
            m = {"n": len(yva), "auc_ovr_macro": float(roc_auc_score(yva, P, multi_class="ovr", average="macro")),
                 "log_loss": float(log_loss(yva, P, labels=[0, 1, 2])),
                 "log_loss_baseline": float(log_loss(yva, np.tile(np.bincount(ytr, minlength=3) / len(ytr), (len(yva), 1)), labels=[0, 1, 2])),
                 "share_neutral": float((yva == 1).mean()),
                 "auc_up_vs_down_excl_neutral": float(roc_auc_score(yva[yva != 1] == 2, score[yva != 1])),
                 "ic_spearman": float(spearmanr(score, fwd).statistic)}
            preds.append(pl.DataFrame({"ts": va["ts"], "fold": fo.name, "p_down": P[:, 0], "p_neutral": P[:, 1],
                                       "p_up": P[:, 2], "y": yva, "fwd": fwd}))
        elif task == "reg":
            from scipy.stats import spearmanr
            ytr = tr[f"fwd_ret_{h}"].to_numpy() * 1e4
            est = M.lightgbm_regressor(**(params or {}))
            est.fit(Xtr, ytr)
            yhat = est.predict(Xva)
            yb = fwd * 1e4
            m = {"n": len(yb), "mae_bps": float(np.abs(yhat - yb).mean()),
                 "mae_zero_bps": float(np.abs(yb).mean()),
                 "rmse_bps": float(np.sqrt(((yhat - yb) ** 2).mean())),
                 "rmse_mean_bps": float(np.sqrt(((ytr.mean() - yb) ** 2).mean())),
                 "ic_spearman": float(spearmanr(yhat, yb).statistic),
                 "hit_rate_sign": float((np.sign(yhat) == np.sign(yb)).mean())}
            preds.append(pl.DataFrame({"ts": va["ts"], "fold": fo.name, "yhat_bps": yhat, "fwd": fwd}))
        rows.append({"config": name, "model": model, "fset": fset, "h": h, "task": task, "fold": fo.name,
                     "train_start": fo.train_start.date().isoformat(), "train_end": fo.train_end.date().isoformat(),
                     "valid_start": fo.valid_start.date().isoformat(), "valid_end": fo.valid_end.date().isoformat(),
                     "n_train": tr.height, "n_features": len(feats), **m})
    res = pl.DataFrame(rows)
    met_path.parent.mkdir(parents=True, exist_ok=True)
    res.write_csv(met_path)
    pl.concat(preds).write_parquet(oof_path)
    log.info("%-38s %5.0fs  %s", name, time.time() - t0,
             {k: round(float(res[k].mean()), 4) for k in ("auc", "log_loss", "auc_ovr_macro", "ic_spearman") if k in res.columns})
    with open(TAB / "experiment_registry.csv", "a") as f:
        f.write(f"{name},{model},{fset},{h},{task},{json.dumps(params or {}).replace(',', ';')}\n")
    return res


def rule_baselines(ctx: Ctx) -> pl.DataFrame:
    """Score simple rules as classifiers on the same validation blocks (no fitting)."""
    rows = []
    for h in ctx.rs["horizons"]:
        for fo in ctx.folds:
            _, va = V.split(ctx.dev, fo, h)
            va = va.filter(pl.col(f"fwd_ret_{h}").is_not_null())
            y, fwd = va[f"up_{h}"].to_numpy(), va[f"fwd_ret_{h}"].to_numpy()
            tr, _ = V.split(ctx.dev, fo, h)
            prior = float(tr[f"up_{h}"].drop_nulls().mean())
            from sklearn.metrics import log_loss, roc_auc_score
            for nm, col in (("momentum_ret15", "ret_15"), ("flow_rule_ofi15z", "ofi_15_z_1440"),
                            ("flow_ofi1", "ofi_1")):
                s = va[col].to_numpy().astype(float)
                ok = np.isfinite(s)
                rows.append({"rule": nm, "h": h, "fold": fo.name, "n": int(ok.sum()),
                             "auc": float(roc_auc_score(y[ok], s[ok]))})
            rows.append({"rule": "prior_only", "h": h, "fold": fo.name, "n": len(y), "auc": 0.5,
                         "log_loss": float(log_loss(y, np.full(len(y), prior), labels=[0, 1]))})
    out = pl.DataFrame(rows, infer_schema_length=None)
    out.write_csv(TAB / "baseline_rules_folds.csv")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="tuning,ablation,horizons,alt,rules")
    a = ap.parse_args()
    steps = a.only.split(",")
    ctx = Ctx()
    setup_logging(ctx.paths.logs / "models.log")
    TAB.mkdir(parents=True, exist_ok=True)
    reg = TAB / "experiment_registry.csv"
    if not reg.exists():
        reg.write_text("config,model,fset,h,task,params\n")
    mcfg = ctx.rs["models"]
    H = ctx.rs["primary_horizon"]
    folds_tbl = pl.DataFrame([f.as_dict() for f in ctx.folds])
    folds_tbl.write_csv(TAB / "walk_forward_folds.csv")

    # ---------------------------------------------------------------- 1. tuning
    grid = mcfg["lgbm_grid"]
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    tune_rows = []
    if "tuning" in steps or "ablation" in steps or "horizons" in steps or "alt" in steps:
        for i, p in enumerate(combos):
            r = run_config(ctx, f"tune_lgbm_E_h{H}_g{i}", "lgbm", "E_all", H, p)
            tune_rows.append({"grid_id": i, **p, "mean_log_loss": r["log_loss"].mean(), "sd_log_loss": r["log_loss"].std(),
                              "mean_auc": r["auc"].mean(), "sd_auc": r["auc"].std(), "min_auc": r["auc"].min(),
                              "mean_ll_improvement": (r["log_loss_baseline"] - r["log_loss"]).mean()})
        tune = pl.DataFrame(tune_rows).sort("mean_log_loss")
        tune.write_csv(TAB / "lgbm_tuning.csv")
        best = {k: tune.row(0, named=True)[k] for k in grid}
        (TAB / "lgbm_selected_params.json").write_text(json.dumps(best, indent=1))
        log.info("selected LightGBM params: %s", best)

    # ---------------------------------------------------------------- 2. ablation at h = 15
    if "ablation" in steps:
        for fset in FEATURE_SETS:
            run_config(ctx, f"logit_{fset}_h{H}", "logit", fset, H, {"C": mcfg["logit_C"]})
            run_config(ctx, f"lgbm_{fset}_h{H}", "lgbm", fset, H, best)
            rf = {k: v for k, v in mcfg["rf"].items() if k != "train_stride"}
            run_config(ctx, f"rf_{fset}_h{H}", "rf", fset, H, rf, stride=mcfg["rf"]["train_stride"])
    # ---------------------------------------------------------------- 3. horizons
    if "horizons" in steps:
        for h in [x for x in ctx.rs["horizons"] if x != H]:
            for fset in ("A_price", "C_plus_flow", "E_all"):
                run_config(ctx, f"logit_{fset}_h{h}", "logit", fset, h, {"C": mcfg["logit_C"]})
                run_config(ctx, f"lgbm_{fset}_h{h}", "lgbm", fset, h, best)
    # ---------------------------------------------------------------- 4. alternative targets
    if "alt" in steps:
        run_config(ctx, f"lgbm3_E_all_h{H}", "lgbm", "E_all", H, best, task="cls3")
        run_config(ctx, f"lgbmreg_E_all_h{H}", "lgbm", "E_all", H, best, task="reg")
    if "rules" in steps:
        rule_baselines(ctx)

    # ---------------------------------------------------------------- summary
    fold_files = sorted((TAB / "folds").glob("*.csv"))
    allm = pl.concat([pl.read_csv(f, infer_schema_length=None) for f in fold_files], how="diagonal_relaxed")
    allm.write_csv(TAB / "model_folds_all.csv")
    binm = allm.filter(pl.col("task") == "binary")
    summ = binm.group_by("config", "model", "fset", "h").agg(
        folds=pl.len(), auc_mean=pl.col("auc").mean(), auc_sd=pl.col("auc").std(), auc_min=pl.col("auc").min(),
        auc_max=pl.col("auc").max(), folds_auc_gt_05=(pl.col("auc") > 0.5).sum(),
        log_loss_mean=pl.col("log_loss").mean(),
        ll_improvement_mean=(pl.col("log_loss_baseline") - pl.col("log_loss")).mean(),
        brier_mean=pl.col("brier").mean(), accuracy_mean=pl.col("accuracy").mean(),
        precision_mean=pl.col("precision").mean(), recall_mean=pl.col("recall").mean(), f1_mean=pl.col("f1").mean(),
        ece_mean=pl.col("ece").mean(), ic_mean=pl.col("ic_spearman").mean(), ic_sd=pl.col("ic_spearman").std(),
    ).sort("h", "fset", "model")
    summ.write_csv(TAB / "model_summary.csv")
    with pl.Config(tbl_rows=80, tbl_cols=12, tbl_width_chars=220):
        print(summ.select("config", "auc_mean", "auc_sd", "auc_min", "folds_auc_gt_05", "ll_improvement_mean", "ic_mean"))


if __name__ == "__main__":
    main()
