#!/usr/bin/env python3
"""Uncertainty and data-snooping accounting for the final results (post-hoc INFERENCE on frozen predictions;
nothing here changes a model, threshold or rule).

  * holdout AUC of the frozen primary model with a moving-block bootstrap over days (5-day blocks)
  * gross and net daily Sharpe of the frozen ML strategy with the same bootstrap
  * number of model configurations tried in development (experiment registry) and a Bonferroni-style check of
    the holdout AUC against that count (one-sided z from the bootstrap standard error)
  * deflated-Sharpe style hurdle: expected maximum Sharpe of N independent zero-skill strategies
    (approximation E[max] ~ sqrt(2 ln N) / sqrt(years))
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ethof.config import Paths, load_config  # noqa: E402

TAB = ROOT / "reports" / "tables"


def block_boot(days: np.ndarray, fn, n_boot=500, block=5, seed=11):
    ud = np.unique(days)
    nd = len(ud)
    rng = np.random.default_rng(seed)
    idx_by_day = {d: np.where(days == d)[0] for d in ud}
    out = []
    for _ in range(n_boot):
        st = rng.integers(0, nd - block + 1, size=int(np.ceil(nd / block)))
        sel = (st[:, None] + np.arange(block)).ravel()[:nd]
        rows = np.concatenate([idx_by_day[ud[i]] for i in sel])
        out.append(fn(rows))
    return np.array(out)


def main() -> None:
    cfg = load_config()
    paths = Paths(cfg["data_dir"])
    res = {}
    pr = pl.read_parquet(paths.processed / "holdout_preds_ETHUSDT_primary.parquet")
    y, p = pr["y"].to_numpy(), pr["p"].to_numpy()
    days = pr["ts"].dt.date().to_numpy()
    auc = roc_auc_score(y, p)
    b = block_boot(days, lambda r: roc_auc_score(y[r], p[r]))
    res["holdout_auc"] = {"auc": auc, "boot_se": float(b.std(ddof=1)), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))],
                          "z_vs_0.5": float((auc - 0.5) / b.std(ddof=1))}
    for name, f in (("ml_base", "holdout_bt_ETHUSDT_ml.parquet"),):
        bt = pl.read_parquet(paths.processed / f)
        d = bt.group_by(pl.col("ts").dt.date().alias("d")).agg(pl.col("gross").sum(), pl.col("net").sum(), pl.col("funding").sum()).sort("d")
        for col in ("gross", "net"):
            x = d[col].to_numpy() + (d["funding"].to_numpy() if col == "gross" else 0)
            sr = x.mean() / x.std(ddof=1) * np.sqrt(365)
            dd = np.arange(len(x))
            bs = block_boot(dd, lambda r: x[r].mean() / x[r].std(ddof=1) * np.sqrt(365))
            res[f"sharpe_{col}"] = {"sharpe": float(sr), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                    "days": int(len(x))}
    reg = pl.read_csv(TAB / "experiment_registry.csv").unique(subset=["config"])
    n_cfg = reg.height
    n_dev_cls = reg.filter(pl.col("task") == "binary").height
    res["n_configs_development"] = n_cfg
    res["n_binary_classifier_configs_development"] = n_dev_cls
    res["n_signal_thresholds"] = len(cfg["research"]["signal"]["prob_deltas"])
    res["n_univariate_stat_tests"] = 36
    z_needed = stats.norm.isf(0.05 / n_cfg)
    res["bonferroni"] = {"alpha": 0.05, "n_tests": n_cfg, "one_sided_z_needed": float(z_needed),
                         "holdout_auc_z": res["holdout_auc"]["z_vs_0.5"],
                         "passes": bool(res["holdout_auc"]["z_vs_0.5"] > z_needed)}
    n_strat = n_cfg * res["n_signal_thresholds"]
    years = res["sharpe_gross"]["days"] / 365
    res["sharpe_hurdle_max_of_noise"] = {"n_strategy_variants": n_strat,
                                         "expected_max_sharpe_noise": float(np.sqrt(2 * np.log(n_strat)) / np.sqrt(years)),
                                         "years": years}
    (TAB / "holdout" / "significance.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
