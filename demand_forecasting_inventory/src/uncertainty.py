"""Phase 22 - forecast uncertainty from VALIDATION residuals.

* sigma_L (per model, horizon, series) = RMSE of validation errors of the H-day demand sum (RMSE rather than the centred
  standard deviation so that forecast bias also inflates the safety stock).  This is the sigma used by the inventory model.
* Prediction intervals: (i) Gaussian  y_hat +- z * sigma ; (ii) empirical  y_hat + q_lo, y_hat + q_hi  with q the quantiles of
  the validation residuals of the same series.  The method is chosen by a *split-half calibration check inside the validation
  period* (quantiles from the first half, coverage measured on the second half); the test period is only used afterwards to
  report honest coverage.
* Normality is not assumed: skewness, excess kurtosis and Shapiro-Wilk tests of the standardised residuals are reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .forecast_core import ForecastBook


def val_residuals(book: ForecastBook, model: str, H: int) -> np.ndarray:
    """(n_series, n_val_origins) residuals y - y_hat of the H-day sum on complete validation windows."""
    _, y, f = book.eval_arrays(model, "val", H)
    return y - f


def sigma_table(book: ForecastBook, models: list[str], horizons=C.HORIZONS) -> pd.DataFrame:
    rows = []
    for m in models:
        for H in horizons:
            e = val_residuals(book, m, H)
            for s, sid in enumerate(book.ctx.ids):
                r = e[s]
                row = {"model": m, "H": H, "series_idx": s, "series_id": sid, "n": len(r), "bias": float(-r.mean()),
                       "sigma_rmse": float(np.sqrt(np.mean(r ** 2))), "sigma_std": float(r.std(ddof=1))}
                for a in C.SERVICE_LEVELS:
                    row[f"q{int(round(a * 100))}"] = float(np.quantile(r, a))
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Residual assumption checks
# --------------------------------------------------------------------------------------
def normality_table(book: ForecastBook, models: list[str], horizons=C.PRIMARY_HORIZONS, max_n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(C.SEED)
    rows = []
    for m in models:
        for H in horizons:
            e = val_residuals(book, m, H)
            sd = np.sqrt(np.mean(e ** 2, axis=1, keepdims=True))
            z = (e / np.where(sd > 0, sd, 1.0))
            pooled = z.ravel()
            sample = pooled if len(pooled) <= max_n else rng.choice(pooled, max_n, replace=False)
            per_series_p = [stats.shapiro(e[s])[1] if np.ptp(e[s]) > 0 else np.nan for s in range(e.shape[0])]
            lag1 = np.mean([np.corrcoef(e[s, :-1], e[s, 1:])[0, 1] for s in range(e.shape[0]) if np.ptp(e[s]) > 0])
            rows.append({
                "model": m, "H": H, "pooled_skew": float(stats.skew(pooled)), "pooled_excess_kurtosis": float(stats.kurtosis(pooled)),
                "shapiro_p_pooled": float(stats.shapiro(sample)[1]),
                "share_series_shapiro_p<0.05": float(np.nanmean(np.array(per_series_p) < 0.05)),
                "mean_residual_lag1_autocorr(overlapping windows)": float(lag1),
                "mean_residual(bias=-mean)": float(-e.mean()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Prediction intervals
# --------------------------------------------------------------------------------------
def _interval(pred: np.ndarray, resid: np.ndarray, method: str, level: float) -> tuple[np.ndarray, np.ndarray]:
    """pred (n_series, n); resid (n_series, m) calibration residuals of the same series."""
    lo_q, hi_q = (1 - level) / 2, (1 + level) / 2
    if method == "gaussian":
        sd = np.sqrt(np.mean(resid ** 2, axis=1, keepdims=True))
        z = stats.norm.ppf(hi_q)
        lo, hi = pred - z * sd, pred + z * sd
    elif method == "empirical":
        qlo = np.quantile(resid, lo_q, axis=1, keepdims=True)
        qhi = np.quantile(resid, hi_q, axis=1, keepdims=True)
        lo, hi = pred + qlo, pred + qhi
    else:
        raise ValueError(method)
    return np.clip(lo, 0, None), np.clip(hi, 0, None)


def split_half_calibration(book: ForecastBook, models: list[str], horizons=C.PRIMARY_HORIZONS) -> pd.DataFrame:
    """Inside VALIDATION: fit residual quantiles on the first half of origins, measure coverage on the second half."""
    rows = []
    for m in models:
        for H in horizons:
            _, y, f = book.eval_arrays(m, "val", H)
            e = y - f
            h = e.shape[1] // 2
            gap = H                                              # skip H origins: windows would overlap the calibration half
            cal, ev = e[:, :h], slice(h + gap, None)
            for method in ("gaussian", "empirical"):
                for level in C.INTERVAL_LEVELS:
                    lo, hi = _interval(f[:, ev], cal, method, level)
                    cov = float(np.mean((y[:, ev] >= lo) & (y[:, ev] <= hi)))
                    rows.append({"model": m, "H": H, "method": method, "level": level, "coverage": cov,
                                 "abs_coverage_error": abs(cov - level), "mean_width": float(np.mean(hi - lo))})
    return pd.DataFrame(rows)


def choose_interval_method(calib: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    summ = calib.groupby("method")[["abs_coverage_error", "mean_width"]].mean()
    return str(summ["abs_coverage_error"].idxmin()), summ.reset_index()


def test_coverage(book: ForecastBook, models: list[str], method: str, horizons=C.PRIMARY_HORIZONS) -> pd.DataFrame:
    """Coverage / width on the TEST phase of intervals calibrated on ALL validation residuals."""
    rows = []
    for m in models:
        for H in horizons:
            e = val_residuals(book, m, H)
            _, y, f = book.eval_arrays(m, "test", H)
            for meth in {method, "gaussian", "empirical"}:
                for level in C.INTERVAL_LEVELS:
                    lo, hi = _interval(f, e, meth, level)
                    inside = (y >= lo) & (y <= hi)
                    rows.append({"model": m, "H": H, "method": meth, "level": level, "coverage": float(inside.mean()),
                                 "abs_coverage_error": float(abs(inside.mean() - level)), "mean_width": float(np.mean(hi - lo)),
                                 "chosen_method": meth == method})
    return pd.DataFrame(rows)


def interval_arrays(book: ForecastBook, model: str, H: int, method: str, level: float, phase: str = "test"):
    """(origins, y_true, y_pred, lo, hi) for plotting."""
    e = val_residuals(book, model, H)
    t, y, f = book.eval_arrays(model, phase, H)
    lo, hi = _interval(f, e, method, level)
    return t, y, f, lo, hi
