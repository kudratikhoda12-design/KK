"""Phases 12-13 - exponential smoothing (SES / Holt / Holt-Winters) and SARIMA with weekly seasonality (s = 7).

Protocol (identical for every model family)
-------------------------------------------
* *Validation phase*: parameters are estimated on ``y[:train_end]`` only; the fitted model is then run through the
  validation period day by day with **fixed parameters** (state updating, no re-estimation).
* *Test phase*: parameters are re-estimated on ``y[:val_end]`` (train + validation) using the structure selected on
  validation; the model is run through the test period with fixed parameters.
* A forecast made at origin t uses only y[0..t] (state recursion is causal; verified by a perturbation test).

ETS family
----------
SES  : level only.               Holt : additive *damped* trend (an undamped trend extrapolated for up to 28 daily steps
is unstable on noisy retail data; damped trend is the standard safer variant).
HoltWinters : additive weekly seasonality (multiplicative is impossible with zero sales), with or without damped trend,
chosen by AICc on the fitting window.   Fallback hierarchy on any failure: HoltWinters -> Holt -> SES.

SARIMA
------
Small candidate set (below).  ADF/KPSS and seasonal strength motivate the d / D choices; the final candidate per series
is the one with the smallest validation error (AIC is *not* comparable across different differencing orders and is only
reported).  If every candidate fails the series falls back to HoltWinters.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import config as C
from .forecast_core import Context, ForecastBook, paths_to_sums, truth_sums

M = C.SEASONAL_PERIOD
MAXH = C.MAX_HORIZON

# --------------------------------------------------------------------------------------
# ETS
# --------------------------------------------------------------------------------------
ETS_KINDS = ("SES", "Holt", "HW_N", "HW_D")
_ETS_REQUIRED = {
    "SES": ("smoothing_level", "initial_level"),
    "Holt": ("smoothing_level", "smoothing_trend", "damping_trend", "initial_level", "initial_trend"),
    "HW_N": ("smoothing_level", "smoothing_seasonal", "initial_level", "initial_seasons"),
    "HW_D": ("smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend", "initial_level",
             "initial_trend", "initial_seasons"),
}


def _fit_ets(y: np.ndarray, kind: str) -> dict | None:
    """Estimate one ETS variant on y.  Returns None when estimation fails or yields non-finite output."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kind == "SES":
                r = SimpleExpSmoothing(y, initialization_method="estimated").fit()
            elif kind == "Holt":
                r = ExponentialSmoothing(y, trend="add", damped_trend=True, initialization_method="estimated").fit()
            elif kind == "HW_N":
                r = ExponentialSmoothing(y, seasonal="add", seasonal_periods=M, initialization_method="estimated").fit()
            elif kind == "HW_D":
                r = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=M,
                                         initialization_method="estimated").fit()
            else:
                raise ValueError(kind)
        p = {k: v for k, v in r.params.items() if v is not None}
        # statsmodels reports parameters that a variant does not use as NaN, so only the *relevant* ones are checked
        flat = np.concatenate([np.atleast_1d(np.asarray(p[k], float)) for k in _ETS_REQUIRED[kind]])
        if not np.all(np.isfinite(flat)) or not np.isfinite(r.aicc):
            return None
        return {"kind": kind, "params": p, "aicc": float(r.aicc), "sse": float(r.sse)}
    except Exception:                                                   # noqa: BLE001 - any failure => fallback
        return None


def _run_ets_filter(y_all: np.ndarray, fit: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    """Re-run the ETS recursion over the whole series with the fitted parameters and fitted initial states."""
    kind, p = fit["kind"], fit["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "SES":
            r = SimpleExpSmoothing(y_all, initialization_method="known", initial_level=p["initial_level"]).fit(
                smoothing_level=p["smoothing_level"], optimized=False)
            return r.level, np.zeros_like(r.level), None, 1.0
        if kind == "Holt":
            r = ExponentialSmoothing(y_all, trend="add", damped_trend=True, initialization_method="known",
                                     initial_level=p["initial_level"], initial_trend=p["initial_trend"]).fit(
                smoothing_level=p["smoothing_level"], smoothing_trend=p["smoothing_trend"],
                damping_trend=p["damping_trend"], optimized=False)
            return r.level, r.trend, None, float(p["damping_trend"])
        if kind == "HW_N":
            r = ExponentialSmoothing(y_all, seasonal="add", seasonal_periods=M, initialization_method="known",
                                     initial_level=p["initial_level"], initial_seasonal=p["initial_seasons"]).fit(
                smoothing_level=p["smoothing_level"], smoothing_seasonal=p["smoothing_seasonal"], optimized=False)
            return r.level, np.zeros_like(r.level), r.season, 1.0
        if kind == "HW_D":
            r = ExponentialSmoothing(y_all, trend="add", damped_trend=True, seasonal="add", seasonal_periods=M,
                                     initialization_method="known", initial_level=p["initial_level"],
                                     initial_trend=p["initial_trend"], initial_seasonal=p["initial_seasons"]).fit(
                smoothing_level=p["smoothing_level"], smoothing_trend=p["smoothing_trend"],
                smoothing_seasonal=p["smoothing_seasonal"], damping_trend=p["damping_trend"], optimized=False)
            return r.level, r.trend, r.season, float(p["damping_trend"])
    raise ValueError(kind)


def ets_paths_from_states(level, trend, season, phi: float, max_h: int = MAXH, m: int = M) -> np.ndarray:
    """h-step forecasts from every origin t:  l_t + (phi + ... + phi^h) b_t + s_{t + h - m(k+1)},  k = floor((h-1)/m)."""
    N = len(level)
    out = np.full((N, max_h), np.nan)
    cum = 0.0
    for h in range(1, max_h + 1):
        cum += phi ** h
        v = level + cum * trend
        if season is not None:
            off = h - m * ((h - 1) // m + 1)                            # in [-(m-1), 0]
            sh = np.full(N, np.nan)
            sh[m - 1:] = season[m - 1 + off: N + off]
            v = v + sh
        out[:, h - 1] = v
    return out


def ets_rolling_paths(y_all: np.ndarray, fit: dict) -> np.ndarray:
    level, trend, season, phi = _run_ets_filter(y_all, fit)
    return ets_paths_from_states(np.asarray(level), np.asarray(trend), None if season is None else np.asarray(season), phi)


def choose_ets_models(y_fit: np.ndarray) -> tuple[dict, list[dict]]:
    """Fit all ETS variants on y_fit and apply the fallback hierarchy.  Returns ({name: fit}, log rows)."""
    fits = {k: _fit_ets(y_fit, k) for k in ETS_KINDS}
    log = []
    ses = fits["SES"]
    if ses is None:                                   # SES failing means the series is unusable - flat zero-level fallback
        ses = {"kind": "SES", "params": {"smoothing_level": 0.1, "initial_level": float(np.mean(y_fit[:M]))}, "aicc": np.nan, "sse": np.nan}
        log.append({"model": "SES", "event": "estimation failed -> fixed alpha=0.1 level model"})
    holt = fits["Holt"]
    if holt is None:
        holt = ses
        log.append({"model": "Holt", "event": "failed -> fallback SES"})
    hw_cands = [f for f in (fits["HW_N"], fits["HW_D"]) if f is not None]
    if hw_cands:
        hw = min(hw_cands, key=lambda f: f["aicc"])
    else:
        hw = holt
        log.append({"model": "HoltWinters", "event": "failed -> fallback Holt/SES"})
    chosen = {"SES": ses, "Holt": holt, "HoltWinters": hw}
    return chosen, log


def _ets_one_series(i: int, y: np.ndarray, train_end: int, val_end: int) -> dict:
    out = {"i": i, "paths": {}, "meta": []}
    for phase, fit_end in (("val", train_end), ("test", val_end)):
        chosen, log = choose_ets_models(y[:fit_end])
        for name, f in chosen.items():
            out["paths"][(phase, name)] = ets_rolling_paths(y, f)
            p = f["params"]
            out["meta"].append({
                "series_idx": i, "phase": phase, "model": name, "estimated_kind": f["kind"], "aicc": f["aicc"],
                "alpha": p.get("smoothing_level"), "beta": p.get("smoothing_trend"), "gamma": p.get("smoothing_seasonal"),
                "phi": p.get("damping_trend")})
        for l in log:
            out["meta"].append({"series_idx": i, "phase": phase, **l})
    return out


def run_ets(book: ForecastBook, n_jobs: int = C.N_JOBS) -> pd.DataFrame:
    ctx = book.ctx
    sp = ctx.split
    res = Parallel(n_jobs=n_jobs)(delayed(_ets_one_series)(i, ctx.Y[i], sp.train_end, sp.val_end) for i in range(ctx.n_series))
    meta = []
    for name in ("SES", "Holt", "HoltWinters"):
        for phase in ("val", "test"):
            P = np.stack([r["paths"][(phase, name)] for r in sorted(res, key=lambda r: r["i"])])
            book.add_paths(name, phase, P)
    for r in res:
        meta.extend(r["meta"])
    df = pd.DataFrame(meta)
    df["series_id"] = df["series_idx"].map(dict(enumerate(ctx.ids)))
    return df


# --------------------------------------------------------------------------------------
# SARIMA
# --------------------------------------------------------------------------------------
SARIMA_CANDIDATES = {
    "S1:(1,0,1)(1,0,1)7+c": dict(order=(1, 0, 1), seasonal_order=(1, 0, 1, M), trend="c"),
    "S2:(1,0,0)(1,0,0)7+c": dict(order=(1, 0, 0), seasonal_order=(1, 0, 0, M), trend="c"),
    "S3:(1,0,1)(0,1,1)7": dict(order=(1, 0, 1), seasonal_order=(0, 1, 1, M), trend="n"),
    "S4:(0,1,1)(0,1,1)7": dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, M), trend="n"),
    "S5:(1,1,1)(0,1,1)7": dict(order=(1, 1, 1), seasonal_order=(0, 1, 1, M), trend="n"),
}


def _fit_sarima(y_fit: np.ndarray, spec: dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y_fit, enforce_stationarity=True, enforce_invertibility=True, **spec).fit(disp=False, maxiter=200)
    ok = bool(res.mle_retvals.get("converged", False)) and np.isfinite(res.aic) and np.all(np.isfinite(res.params))
    return res, ok


def roll_sarima(res, y: np.ndarray, start_origin: int, end_origin: int, max_h: int = MAXH) -> np.ndarray:
    """Forecast paths for origins start..end.  ``res`` must have been fit on y[:start_origin+1]; parameters stay fixed and the
    Kalman state is advanced one observation at a time (``extend``), so the forecast at origin t uses y[0..t] only."""
    out = np.full((end_origin - start_origin + 1, max_h), np.nan)
    cur = res
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j, t in enumerate(range(start_origin, end_origin + 1)):
            out[j] = cur.forecast(max_h)
            if t < end_origin:
                cur = cur.extend(y[t + 1: t + 2])
    return out


def _val_score(y: np.ndarray, paths_o: np.ndarray, start_origin: int, val_end: int) -> float:
    """Selection score on validation: sum over H in {7,14,28} of MAE_H / mean(y_H) (complete windows only)."""
    tot = 0.0
    Yrow = y[None, :]
    for H in C.PRIMARY_HORIZONS:
        T = truth_sums(Yrow, H)[0]
        t_eval = np.arange(start_origin, val_end - H)
        f = np.clip(paths_o[: len(t_eval), :H], 0, None).sum(axis=1)
        yt = T[t_eval]
        tot += np.mean(np.abs(yt - f)) / max(float(np.mean(yt)), 1e-6)
    return float(tot)


def _sarima_one_series(i: int, y: np.ndarray, train_end: int, val_end: int, N: int) -> dict:
    start_v, end_v = train_end - 1, val_end - 1
    rows, best = [], None
    for name, spec in SARIMA_CANDIDATES.items():
        row = {"series_idx": i, "candidate": name}
        try:
            res, ok = _fit_sarima(y[:train_end], spec)
            row.update(aic=float(res.aic), bic=float(res.bic), converged=bool(ok))
            if ok:
                paths = roll_sarima(res, y, start_v, end_v)
                row["val_score"] = _val_score(y, paths, start_v, val_end)
                if best is None or row["val_score"] < best[0]:
                    best = (row["val_score"], name, paths)
            else:
                row["val_score"] = np.nan
        except Exception as exc:                                        # noqa: BLE001
            row.update(aic=np.nan, bic=np.nan, converged=False, val_score=np.nan, error=type(exc).__name__)
        rows.append(row)
    out = {"i": i, "cands": rows, "chosen": None, "paths_val": None, "paths_test": None}
    if best is None:
        return out
    _, name, paths_val = best
    out["chosen"] = name
    out["paths_val"] = paths_val
    try:                                                                # test phase: refit chosen structure on train + val
        res2, ok2 = _fit_sarima(y[:val_end], SARIMA_CANDIDATES[name])
        if ok2:
            out["paths_test"] = roll_sarima(res2, y, val_end - 1, N - 1)
        else:
            out["test_note"] = "refit on train+val did not converge"
    except Exception as exc:                                            # noqa: BLE001
        out["test_note"] = f"refit failed: {type(exc).__name__}"
    return out


def run_sarima(book: ForecastBook, fallback: str = "HoltWinters", n_jobs: int = C.N_JOBS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit/select/roll SARIMA for every series and register the model as ``SARIMA`` in the book.

    Returns (candidate table with AIC/BIC/validation score, per-series selection table).  Series whose SARIMA is unusable
    in a phase use the ``fallback`` model's forecasts for that phase (flagged in the selection table).
    """
    ctx = book.ctx
    sp = ctx.split
    res = Parallel(n_jobs=n_jobs)(
        delayed(_sarima_one_series)(i, ctx.Y[i], sp.train_end, sp.val_end, ctx.N) for i in range(ctx.n_series))
    res = sorted(res, key=lambda r: r["i"])
    sums = {ph: np.zeros((ctx.n_series, len(book.origins(ph)), len(C.HORIZONS))) for ph in ("val", "test")}
    sel_rows = []
    for r in res:
        i = r["i"]
        row = {"series_idx": i, "series_id": ctx.ids[i], "chosen": r["chosen"], "note": r.get("test_note", "")}
        for phase, key in (("val", "paths_val"), ("test", "paths_test")):
            paths = r[key]            # (n_origins(phase), MAXH): origin order identical to ForecastBook.origins(phase)
            if paths is not None:
                sums[phase][i] = paths_to_sums(np.clip(paths, 0.0, None))
                row[f"fallback_{phase}"] = ""
            else:
                sums[phase][i] = book.F[phase][fallback][i]
                row[f"fallback_{phase}"] = fallback
        sel_rows.append(row)
    for phase in ("val", "test"):
        book.add_sums("SARIMA", phase, sums[phase])
    cands = pd.DataFrame([row for r in res for row in r["cands"]])
    cands["series_id"] = cands["series_idx"].map(dict(enumerate(ctx.ids)))
    return cands, pd.DataFrame(sel_rows)


# --------------------------------------------------------------------------------------
# Stationarity / seasonality diagnostics (TRAIN data only)
# --------------------------------------------------------------------------------------
def _adf_p(x: np.ndarray) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(adfuller(x, autolag="AIC")[1])
    except Exception:                                                   # noqa: BLE001
        return float("nan")


def _kpss_p(x: np.ndarray) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(kpss(x, regression="c", nlags="auto")[1])
    except Exception:                                                   # noqa: BLE001
        return float("nan")


def seasonal_strength(y: np.ndarray, period: int = M) -> float:
    """F_s = max(0, 1 - Var(remainder) / Var(seasonal + remainder))  from a robust STL decomposition (Wang, Smith & Hyndman 2006)."""
    try:
        r = STL(y, period=period, robust=True).fit()
        v = np.var(r.resid) / np.var(r.seasonal + r.resid)
        return float(max(0.0, 1.0 - v))
    except Exception:                                                   # noqa: BLE001
        return float("nan")


def stationarity_table(ctx: Context) -> pd.DataFrame:
    a = ctx.split.train_end
    rows = []
    for i, sid in enumerate(ctx.ids):
        y = ctx.Y[i, :a]
        d1 = np.diff(y)
        d7 = y[M:] - y[:-M]
        rows.append({
            "series_id": sid, "tier": ctx.sel.loc[i, "tier"],
            "adf_p_level": _adf_p(y), "kpss_p_level": _kpss_p(y),
            "adf_p_diff1": _adf_p(d1), "kpss_p_diff1": _kpss_p(d1),
            "adf_p_seasdiff7": _adf_p(d7), "kpss_p_seasdiff7": _kpss_p(d7),
            "seasonal_strength_Fs": seasonal_strength(y),
            "acf_lag7": float(acf(y, nlags=7, fft=True)[7]),
        })
    df = pd.DataFrame(rows)
    df["level_conclusion"] = np.select(
        [(df["adf_p_level"] < 0.05) & (df["kpss_p_level"] >= 0.05),
         (df["adf_p_level"] < 0.05) & (df["kpss_p_level"] < 0.05),
         (df["adf_p_level"] >= 0.05) & (df["kpss_p_level"] < 0.05)],
        ["stationary (both tests agree)", "conflict: ADF stationary / KPSS non-stationary", "unit root (both tests agree)"],
        default="inconclusive (neither test decisive)")
    df["seasonal_diff_suggested(Fs>0.64)"] = df["seasonal_strength_Fs"] > 0.64
    return df
