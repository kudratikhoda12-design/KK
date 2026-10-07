"""Phase 20 - residual analysis.

Statistical models (SARIMA as selected on validation, Holt-Winters as selected by AICc): one-step in-sample residuals on the
TRAIN window - mean, variance, autocorrelation (Ljung-Box), distribution (skewness, excess kurtosis, Jarque-Bera).
XGBoost: out-of-sample residuals of the H-day forecast on the TEST period - bias, error for high- vs low-demand windows,
temporal error patterns.  These are diagnostics only; no model is changed after looking at them.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from . import stat_models as sm
from .forecast_core import Context, ForecastBook


def _resid_stats(r: np.ndarray, lags=(7, 14), model_df: int = 0) -> dict:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    n = len(r)
    lb = acorr_ljungbox(r, lags=list(lags), model_df=model_df, return_df=True)
    jb = stats.jarque_bera(r)
    a = acf(r, nlags=14, fft=True)
    band = 1.96 / np.sqrt(n)
    return {"n": n, "mean": float(r.mean()), "std": float(r.std(ddof=1)), "skew": float(stats.skew(r)),
            "excess_kurtosis": float(stats.kurtosis(r)), "jarque_bera_p": float(jb.pvalue),
            **{f"ljung_box_p_lag{l}": float(lb.loc[l, "lb_pvalue"]) for l in lags},
            "acf_lag1": float(a[1]), "acf_lag7": float(a[7]), "share_acf_lags1_14_outside_95band": float(np.mean(np.abs(a[1:15]) > band))}


def stat_model_residuals(ctx: Context, sarima_sel: pd.DataFrame, ets_meta: pd.DataFrame, burn_in: int = 28):
    """Refit the selected SARIMA structure and the selected Holt-Winters variant on TRAIN and analyse their residuals."""
    a = ctx.split.train_end
    rows, arrays = [], {}
    hw_kind = (ets_meta[(ets_meta["phase"] == "val") & (ets_meta["model"] == "HoltWinters")]
               .set_index("series_idx")["estimated_kind"].to_dict())
    for i, sid in enumerate(ctx.ids):
        y = ctx.Y[i, :a]
        arrays[sid] = {}
        # --- SARIMA
        name = sarima_sel.loc[sarima_sel["series_id"] == sid, "chosen"].iloc[0]
        if isinstance(name, str) and name in sm.SARIMA_CANDIDATES:
            spec = sm.SARIMA_CANDIDATES[name]
            res, ok = sm._fit_sarima(y, spec)
            p, q = spec["order"][0], spec["order"][2]
            P, Q = spec["seasonal_order"][0], spec["seasonal_order"][2]
            r = np.asarray(res.resid)[burn_in:]
            rows.append({"series_id": sid, "tier": ctx.sel.loc[i, "tier"], "model": f"SARIMA {name}", "converged": ok,
                         **_resid_stats(r, model_df=p + q + P + Q)})
            arrays[sid]["SARIMA"] = r
        # --- Holt-Winters (selected variant)
        kind = hw_kind.get(i, "HW_N")
        fit = sm._fit_ets(y, kind)
        if fit is not None:
            level, trend, season, phi = sm._run_ets_filter(y, fit)
            paths1 = sm.ets_paths_from_states(np.asarray(level), np.asarray(trend), None if season is None else np.asarray(season), phi, max_h=1)
            fitted = np.full(len(y), np.nan)
            fitted[1:] = paths1[:-1, 0]                              # one-step forecast of y[t] made at t-1
            r = (y - fitted)[burn_in:]
            rows.append({"series_id": sid, "tier": ctx.sel.loc[i, "tier"], "model": f"HoltWinters {kind}", "converged": True,
                         **_resid_stats(r, model_df=0)})
            arrays[sid]["HoltWinters"] = r
    return pd.DataFrame(rows), arrays


def stat_residual_summary(tab: pd.DataFrame) -> pd.DataFrame:
    tab = tab.copy()
    tab["family"] = tab["model"].str.split().str[0]
    g = tab.groupby("family")
    return g.agg(series=("series_id", "nunique"), mean_resid=("mean", "mean"), median_std=("std", "median"),
                 share_ljung_box_lag14_p_lt_0_05=("ljung_box_p_lag14", lambda s: float((s < 0.05).mean())),
                 share_jarque_bera_p_lt_0_05=("jarque_bera_p", lambda s: float((s < 0.05).mean())),
                 median_skew=("skew", "median"), median_excess_kurtosis=("excess_kurtosis", "median"),
                 median_acf_lag7=("acf_lag7", "median")).reset_index()


def error_concentration(book: ForecastBook, models: list[str], horizons, worst_share: float = 0.05) -> pd.DataFrame:
    """How concentrated are the forecast errors?  Per model / horizon / period (validation, test):

    * RMSE / MAE            - 1.25 for Gaussian errors; larger = heavier tails (a few large misses dominate RMSE),
    * share of the squared error produced by the worst 5% of (series, origin) windows,
    * excess kurtosis of the errors.
    Used to explain why test RMSE grew much faster than MAE relative to validation (diagnostic only; nothing is re-selected)."""
    rows = []
    for m in models:
        for H in horizons:
            for ph in ("val", "test"):
                _, y, f = book.eval_arrays(m, ph, H)
                e = (f - y).ravel()
                se = np.sort(e ** 2)[::-1]
                k = max(1, int(round(worst_share * len(se))))
                rows.append({"model": m, "H": H, "phase": ph, "n": len(e), "RMSE_over_MAE": float(np.sqrt(np.mean(e ** 2)) / np.mean(np.abs(e))),
                             "worst5pct_share_of_squared_error": float(se[:k].sum() / max(se.sum(), 1e-12)),
                             "excess_kurtosis": float(stats.kurtosis(e))})
    return pd.DataFrame(rows)


def zero_forecast_shares(book: ForecastBook, models: list[str], horizons, phase: str = "test") -> pd.DataFrame:
    """Share of H-day windows whose forecast sum is exactly zero.  sMAPE (0/0 := 0) scores a zero forecast of a zero actual as a perfect 0,
    so models that output exact zeros collect 'free' perfect scores on intermittent data (diagnostic for reading the sMAPE column)."""
    rows = []
    for m in models:
        for H in horizons:
            _, y, f = book.eval_arrays(m, phase, H)
            zf, zy = (f == 0), (y == 0)
            rows.append({"model": m, "H": H, "phase": phase, "share_zero_forecast": float(zf.mean()), "share_actual_zero": float(zy.mean()),
                         "share_both_zero": float((zf & zy).mean())})
    return pd.DataFrame(rows)


def xgboost_residuals(book: ForecastBook, H: int, model: str = "XGBoost", phase: str = "test") -> dict:
    ctx = book.ctx
    t, y, f = book.eval_arrays(model, phase, H)
    e = f - y                                             # positive = over-forecast
    dates = ctx.dates[t]
    out = {"H": H}
    out["overall"] = {"bias(mean f-y)": float(e.mean()), "MAE": float(np.abs(e).mean()), "mean_actual": float(y.mean()),
                      "bias_pct_of_mean_actual": float(100 * e.mean() / y.mean()) if y.mean() > 0 else np.nan}
    # error by demand level (windows ranked on the ACTUAL H-day sum, pooled): decile-style bins
    flat_y, flat_e = y.ravel(), e.ravel()
    q = np.quantile(flat_y, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    bins = pd.cut(flat_y, bins=np.unique(q), include_lowest=True, duplicates="drop")
    by_level = (pd.DataFrame({"bin": bins.astype(str), "e": flat_e, "y": flat_y})
                .groupby("bin", observed=True).agg(n=("e", "size"), mean_actual=("y", "mean"), bias=("e", "mean"),
                                                   MAE=("e", lambda s: float(np.abs(s).mean()))).reset_index())
    by_level["bias_pct_of_actual"] = 100 * by_level["bias"] / by_level["mean_actual"].replace(0, np.nan)
    out["by_level"] = by_level
    # error by tier
    tiers = ctx.sel["tier"].to_numpy()
    out["by_tier"] = pd.DataFrame([{"tier": k, "bias": float(e[tiers == k].mean()), "MAE": float(np.abs(e[tiers == k]).mean()),
                                    "mean_actual": float(y[tiers == k].mean()),
                                    "WAPE": float(100 * np.abs(e[tiers == k]).sum() / y[tiers == k].sum())} for k in ("high", "medium", "low")])
    # temporal pattern: by calendar month of the origin and by weekday of the origin
    df = pd.DataFrame({"date": np.tile(dates, ctx.n_series), "e": flat_e, "ae": np.abs(flat_e), "y": flat_y})
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["weekday"] = df["date"].dt.day_name()
    out["by_month"] = df.groupby("month").agg(bias=("e", "mean"), MAE=("ae", "mean"), mean_actual=("y", "mean")).reset_index()
    out["by_weekday"] = df.groupby("weekday").agg(bias=("e", "mean"), MAE=("ae", "mean")).reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reset_index()
    out["error_matrix"] = e
    out["origins"] = t
    return out
