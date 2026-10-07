"""Phase 11 - baseline forecasters.

Each function returns the *daily forecast path* from every origin: ``paths[s, t, k-1]`` = forecast of y[t+k] made with
information up to and including day t.  H-day demand forecasts are the sums of the first H path entries.

* Naive             : y_hat(t+k) = y_t
* Seasonal naive    : y_hat(t+k) = y at the same weekday in the latest 7 observed days  (lag 7*ceil(k/7) from t+k)
* Moving average(w) : y_hat(t+k) = mean(y[t-w+1..t])
* Croston-SBA       : intermittent-demand baseline; rate = (1 - alpha/2) * z / p, with z = smoothed non-zero size and
                      p = smoothed inter-demand interval (Syntetos-Boylan bias correction)
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from .forecast_core import ForecastBook


def naive_paths(Y: np.ndarray, max_h: int = C.MAX_HORIZON) -> np.ndarray:
    return np.repeat(Y[:, :, None], max_h, axis=2).astype(float)


def seasonal_naive_paths(Y: np.ndarray, m: int = C.SEASONAL_PERIOD, max_h: int = C.MAX_HORIZON) -> np.ndarray:
    n_s, N = Y.shape
    out = np.full((n_s, N, max_h), np.nan)
    for k in range(1, max_h + 1):
        off = k - m * math.ceil(k / m)               # in [-(m-1), 0]
        # origin t uses observation y[t + off]; valid for t >= m-1
        out[:, m - 1:, k - 1] = Y[:, m - 1 + off: N + off]
    return out


def moving_average_paths(Y: np.ndarray, w: int, max_h: int = C.MAX_HORIZON) -> np.ndarray:
    n_s, N = Y.shape
    c = np.concatenate([np.zeros((n_s, 1)), np.cumsum(Y, axis=1)], axis=1)
    ma = np.full((n_s, N), np.nan)
    ma[:, w - 1:] = (c[:, w:] - c[:, : N - w + 1]) / w
    return np.repeat(ma[:, :, None], max_h, axis=2)


def croston_sba_rate(y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Croston-SBA demand-rate estimate available at the end of each day (0 until the first sale)."""
    N = len(y)
    rate = np.zeros(N)
    z = p = None
    q = 1
    for t in range(N):
        if y[t] > 0:
            if z is None:
                z, p = float(y[t]), float(q)
            else:
                z += alpha * (y[t] - z)
                p += alpha * (q - p)
            q = 1
        else:
            q += 1
        rate[t] = (1.0 - alpha / 2.0) * z / p if z is not None else 0.0
    return rate


def croston_paths(Y: np.ndarray, alpha: float = 0.1, max_h: int = C.MAX_HORIZON) -> np.ndarray:
    rate = np.vstack([croston_sba_rate(Y[i], alpha) for i in range(Y.shape[0])])
    return np.repeat(rate[:, :, None], max_h, axis=2)


BASELINE_NAMES = ["Naive", "SeasonalNaive", "MA7", "MA14", "MA28", "Croston"]


def build_baseline_paths(Y: np.ndarray) -> dict[str, np.ndarray]:
    out = {"Naive": naive_paths(Y), "SeasonalNaive": seasonal_naive_paths(Y)}
    for w in C.MA_WINDOWS:
        out[f"MA{w}"] = moving_average_paths(Y, w)
    out["Croston"] = croston_paths(Y, alpha=0.1)
    return out


def add_baselines(book: ForecastBook) -> dict[str, np.ndarray]:
    """Register all baselines for both phases (they have no fitted parameters, so val/test use the same recursion)."""
    paths = build_baseline_paths(book.ctx.Y)
    for name, p in paths.items():
        for phase in ("val", "test"):
            book.add_paths(name, phase, p)
    return paths
