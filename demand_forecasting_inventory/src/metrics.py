"""Forecast-accuracy metrics and the Diebold-Mariano test.

Definitions (y = actual, f = forecast; all on the H-day demand sum):

* MAE   = mean |y - f|
* RMSE  = sqrt(mean (y - f)^2)
* sMAPE = mean( 200 |y - f| / (|y| + |f|) )  in %, with the convention 0/0 := 0 (both zero => perfect)
* WAPE  = 100 * sum|y - f| / sum|y|          (volume-weighted; = MAE / mean demand)
* bias  = mean(f - y)                          (positive = over-forecast)

MAPE is deliberately *not* used: it is undefined when y = 0 (frequent in intermittent retail demand) and is
asymmetric (over-forecasts are penalised without bound, under-forecasts are capped at 100 %).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def mae(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.mean(np.abs(y - f)))


def rmse(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.sqrt(np.mean((y - f) ** 2)))


def smape(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    denom = np.abs(y) + np.abs(f)
    num = 2.0 * np.abs(y - f)
    ratio = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
    return float(100.0 * np.mean(ratio))


def wape(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    s = np.sum(np.abs(y))
    return float(100.0 * np.sum(np.abs(y - f)) / s) if s > 0 else float("nan")


def bias(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.mean(f - y))


def all_metrics(y, f) -> dict:
    y, f = np.asarray(y, float).ravel(), np.asarray(f, float).ravel()
    ok = ~(np.isnan(y) | np.isnan(f))
    y, f = y[ok], f[ok]
    return {"n": int(len(y)), "MAE": mae(y, f), "RMSE": rmse(y, f), "sMAPE": smape(y, f), "WAPE": wape(y, f), "bias": bias(y, f),
            "mean_actual": float(np.mean(y)) if len(y) else float("nan")}


# --------------------------------------------------------------------------------------
# Diebold-Mariano
# --------------------------------------------------------------------------------------
def _long_run_variance(d: np.ndarray, max_lag: int, kernel: str) -> float:
    n = len(d)
    dc = d - d.mean()
    gamma0 = float(np.dot(dc, dc) / n)
    total = gamma0
    for k in range(1, max_lag + 1):
        g = float(np.dot(dc[k:], dc[:-k]) / n)
        w = 1.0 if kernel == "rectangular" else 1.0 - k / (max_lag + 1)
        total += 2.0 * w * g
    return total


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int, hln: bool = True) -> dict:
    """Diebold-Mariano test of equal predictive accuracy on a loss differential d = loss_a - loss_b.

    * H0: E[d] = 0.  Negative statistic (mean_d < 0) => model A has the smaller loss.
    * Long-run variance of d with lag h-1 (forecast errors of h-step forecasts from consecutive daily origins
      overlap, so d is MA(h-1)); rectangular kernel (Diebold-Mariano 1995), falling back to the Bartlett kernel when the
      rectangular estimate is not positive.
    * Harvey-Leybourne-Newbold (1997) small-sample correction and Student-t(n-1) reference distribution when ``hln``.

    Caveat reported alongside every use: with daily origins the *effective* number of independent windows is about
    n / h, so power is limited, especially at h = 28.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return {"n": n, "mean_d": float(d.mean()) if n else np.nan, "dm_stat": np.nan, "p_value": np.nan, "kernel": "n/a"}
    kernel = "rectangular"
    var = _long_run_variance(d, h - 1, kernel)
    if var <= 0:
        kernel = "bartlett"
        var = _long_run_variance(d, h - 1, kernel)
    dm = d.mean() / np.sqrt(var / n)
    if hln:
        dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        p = 2 * stats.t.sf(abs(dm), df=n - 1)
    else:
        p = 2 * stats.norm.sf(abs(dm))
    return {"n": n, "mean_d": float(d.mean()), "dm_stat": float(dm), "p_value": float(p), "kernel": kernel}
