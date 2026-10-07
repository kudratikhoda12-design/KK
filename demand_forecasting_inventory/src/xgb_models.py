"""Phases 14, 16, 17 - direct multi-horizon XGBoost with modest, chronologically safe tuning.

One *global* (pooled over the 18 series) model per horizon H in {3, 7, 14, 28}; target = sum of demand over t+1..t+H, in
raw units (so SHAP values are in units of demand).  Objective: squared error (the conditional mean is what inventory
planning needs; it is also what RMSE rewards).

Chronology
----------
* Tuning:   fit on rows whose target window ends <= train_end-1; score on validation origins whose window is complete
            inside the validation period; early stopping also uses the validation rows only.
* Val phase forecasts: the tuned train-only model.
* Test phase forecasts: the chosen configuration refit on train + validation (targets ending <= val_end-1) with
            n_estimators = best_iteration + 1 from tuning.  The test period is never used for any choice.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C
from . import metrics as M
from .features import FEATURES, dataset_for_horizon, rows_for_fit, rows_for_origins
from .forecast_core import Context, ForecastBook, H_INDEX

SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
    "min_child_weight": [1, 5, 10, 20, 40],
}
DEFAULT_PARAMS = {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5}


def _model(params: dict, n_estimators: int, early_stopping: int | None = None) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist", n_estimators=n_estimators, random_state=C.SEED,
        n_jobs=C.N_JOBS, early_stopping_rounds=early_stopping, eval_metric="mae", **params)


def sample_configs(n: int, seed: int) -> list[dict]:
    """Default configuration + (n-1) random draws from the search space (fixed seed, no repeats)."""
    rng = np.random.default_rng(seed)
    seen = {tuple(sorted(DEFAULT_PARAMS.items()))}
    out = [dict(DEFAULT_PARAMS)]
    while len(out) < n:
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in SEARCH_SPACE.items()}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def tune_horizon(ds: pd.DataFrame, ctx: Context, H: int, n_iter: int = C.XGB_SEARCH_ITER, verbose: bool = True) -> dict:
    sp = ctx.split
    tr = rows_for_fit(ds, H, sp.train_end)
    va = rows_for_origins(ds, sp.eval_origins("validation", H))
    assert va["target"].notna().all() and (va["origin"] + H).max() <= sp.val_end - 1       # validation targets only
    assert (tr["origin"] + H).max() <= sp.train_end - 1                                     # no overlap with validation
    Xtr, ytr, Xva, yva = tr[FEATURES], tr["target"], va[FEATURES], va["target"]
    rows, best = [], None
    for j, cfg in enumerate(sample_configs(n_iter, C.SEED + H)):
        t0 = time.time()
        m = _model(cfg, C.XGB_MAX_ESTIMATORS, C.XGB_EARLY_STOPPING)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        bi = int(m.best_iteration)
        pred = np.clip(m.predict(Xva, iteration_range=(0, bi + 1)), 0, None)
        r = M.all_metrics(yva.to_numpy(), pred)
        row = {"H": H, "config_id": j, "is_default": j == 0, **cfg, "best_iteration": bi, "n_estimators": bi + 1,
               "val_MAE": r["MAE"], "val_RMSE": r["RMSE"], "val_WAPE": r["WAPE"], "val_sMAPE": r["sMAPE"], "seconds": time.time() - t0}
        rows.append(row)
        if best is None or r["MAE"] < best[0]:
            best = (r["MAE"], cfg, bi + 1, m)
    log = pd.DataFrame(rows)
    if verbose:
        d = log[log["is_default"]].iloc[0]
        print(f"[xgb H={H}] best val MAE {best[0]:.3f} (default {d['val_MAE']:.3f}); params={best[1]}; n_trees={best[2]}")
    return {"H": H, "params": best[1], "n_estimators": best[2], "log": log, "train_only_model": best[3]}


def _predict_rows(model: xgb.XGBRegressor, rows: pd.DataFrame, ctx: Context, n_origins: int, n_trees: int | None) -> np.ndarray:
    kw = {} if n_trees is None else {"iteration_range": (0, n_trees)}
    pred = np.clip(model.predict(rows[FEATURES], **kw), 0, None)
    return pred.reshape(ctx.n_series, n_origins)


def run_xgboost(book: ForecastBook, n_iter: int = C.XGB_SEARCH_ITER, save_models: bool = True, verbose: bool = True) -> dict:
    """Tune per horizon on validation, produce val-phase and test-phase forecasts and register 'XGBoost' in the book."""
    ctx, sp = book.ctx, book.ctx.split
    out_val = np.zeros((ctx.n_series, len(book.origins("val")), len(C.HORIZONS)))
    out_test = np.zeros((ctx.n_series, len(book.origins("test")), len(C.HORIZONS)))
    logs, models, chosen = [], {}, []
    for H in C.HORIZONS:
        ds = dataset_for_horizon(ctx, H)
        tuned = tune_horizon(ds, ctx, H, n_iter, verbose)
        logs.append(tuned["log"])
        m_tr = tuned["train_only_model"]
        rows_v = rows_for_origins(ds, book.origins("val"))
        out_val[:, :, H_INDEX[H]] = _predict_rows(m_tr, rows_v, ctx, len(book.origins("val")), tuned["n_estimators"])
        fit_rows = rows_for_fit(ds, H, sp.val_end)
        m_fin = _model(tuned["params"], tuned["n_estimators"])
        m_fin.fit(fit_rows[FEATURES], fit_rows["target"], verbose=False)
        rows_t = rows_for_origins(ds, book.origins("test"))
        out_test[:, :, H_INDEX[H]] = _predict_rows(m_fin, rows_t, ctx, len(book.origins("test")), None)
        models[H] = {"train_only": m_tr, "final": m_fin, "n_trees_train_only": tuned["n_estimators"]}
        chosen.append({"H": H, **tuned["params"], "n_estimators": tuned["n_estimators"], "train_rows": len(rows_for_fit(ds, H, sp.train_end)),
                       "final_fit_rows": len(fit_rows)})
        if save_models:
            C.OUT_MOD.mkdir(parents=True, exist_ok=True)
            m_tr.save_model(C.OUT_MOD / f"xgb_H{H}_train_only.json")
            m_fin.save_model(C.OUT_MOD / f"xgb_H{H}_final_train_val.json")
    book.add_sums("XGBoost", "val", out_val)
    book.add_sums("XGBoost", "test", out_test)
    return {"models": models, "log": pd.concat(logs, ignore_index=True), "chosen": pd.DataFrame(chosen)}
