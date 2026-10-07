"""Phases 18-19 - forecast evaluation tables, validation-based selections and Diebold-Mariano comparisons.

All models are scored on identical origins (``ForecastBook.eval_arrays``).  Selections (moving-average window, best model
per horizon *for deployment*) use the VALIDATION phase only; test-phase numbers are reported, never used to choose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from .forecast_core import ForecastBook, H_INDEX, truth_sums

MODELS_FINAL = ["Naive", "SeasonalNaive", "MovingAverage", "Croston", "SES", "Holt", "HoltWinters", "SARIMA", "XGBoost"]
MODEL_FAMILY = {"Naive": "baseline", "SeasonalNaive": "baseline", "MovingAverage": "baseline", "Croston": "baseline",
                "SES": "exponential smoothing", "Holt": "exponential smoothing", "HoltWinters": "exponential smoothing",
                "SARIMA": "SARIMA", "XGBoost": "machine learning"}


# --------------------------------------------------------------------------------------
# Moving-average window chosen on validation
# --------------------------------------------------------------------------------------
def select_ma_alias(book: ForecastBook) -> pd.DataFrame:
    """Pick the MA window per horizon with the lowest pooled VALIDATION WAPE and register the alias 'MovingAverage'."""
    rows, chosen = [], {}
    for H in C.HORIZONS:
        scores = {}
        for w in C.MA_WINDOWS:
            _, y, f = book.eval_arrays(f"MA{w}", "val", H)
            scores[w] = M.wape(y.ravel(), f.ravel())
            rows.append({"H": H, "window": w, "val_WAPE": scores[w]})
        chosen[H] = min(scores, key=scores.get)
    for ph in ("val", "test"):
        arr = np.stack([book.F[ph][f"MA{chosen[H]}"][:, :, H_INDEX[H]] for H in C.HORIZONS], axis=-1)
        book.add_sums("MovingAverage", ph, arr)
    tab = pd.DataFrame(rows)
    tab["selected"] = tab.apply(lambda r: chosen[int(r["H"])] == int(r["window"]), axis=1)
    return tab


# --------------------------------------------------------------------------------------
# Metric tables
# --------------------------------------------------------------------------------------
def metrics_by_series(book: ForecastBook, phase: str, models: list[str] | None = None, horizons=C.HORIZONS) -> pd.DataFrame:
    ctx = book.ctx
    rows = []
    for m in models or MODELS_FINAL:
        for H in horizons:
            t, y, f = book.eval_arrays(m, phase, H)
            for s in range(ctx.n_series):
                r = M.all_metrics(y[s], f[s])
                rows.append({"model": m, "phase": phase, "H": H, "series_id": ctx.ids[s], "tier": ctx.sel.loc[s, "tier"],
                             "sb_class": ctx.sel.loc[s, "train_sb_class"], "intermittency": ctx.sel.loc[s, "intermittency"],
                             "cat_id": ctx.sel.loc[s, "cat_id"], **r})
    return pd.DataFrame(rows)


def metrics_pooled(book: ForecastBook, phase: str, models: list[str] | None = None, horizons=C.HORIZONS,
                   series_mask: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    for m in models or MODELS_FINAL:
        for H in horizons:
            t, y, f = book.eval_arrays(m, phase, H)
            if series_mask is not None:
                y, f = y[series_mask], f[series_mask]
            r = M.all_metrics(y, f)
            rows.append({"model": m, "phase": phase, "H": H, **r})
    return pd.DataFrame(rows)


def best_by_horizon(pooled: pd.DataFrame, metric: str = "WAPE") -> pd.DataFrame:
    """Lowest-error model per horizon among the final models."""
    rows = []
    for H, g in pooled.groupby("H"):
        g = g.sort_values(metric)
        rows.append({"H": H, "best_model": g.iloc[0]["model"], metric: g.iloc[0][metric],
                     "runner_up": g.iloc[1]["model"], f"runner_up_{metric}": g.iloc[1][metric]})
    return pd.DataFrame(rows)


def rank_table(pooled: pd.DataFrame, metric: str = "WAPE") -> pd.DataFrame:
    piv = pooled.pivot(index="model", columns="H", values=metric)
    return piv.rank(axis=0, method="min").astype(int)


# --------------------------------------------------------------------------------------
# Diebold-Mariano
# --------------------------------------------------------------------------------------
def series_scale(book: ForecastBook, H: int) -> np.ndarray:
    """Per-series scale = mean H-day demand sum on the TRAIN period (known before validation/test)."""
    ctx = book.ctx
    T = truth_sums(ctx.Y, H)[:, : ctx.split.train_end - H]
    sc = np.nanmean(T, axis=1)
    return np.where(sc > 0, sc, 1.0)


def dm_panel(book: ForecastBook, phase: str, model_a: str, model_b: str, H: int, loss: str = "abs") -> dict:
    """Panel DM: loss differential averaged over series at every origin (this handles the cross-sectional dependence caused
    by shared calendar shocks), losses scaled by the per-series TRAIN mean H-day demand so large series do not dominate."""
    t, y, fa = book.eval_arrays(model_a, phase, H)
    _, _, fb = book.eval_arrays(model_b, phase, H)
    sc = series_scale(book, H)[:, None]
    if loss == "abs":
        la, lb = np.abs(y - fa) / sc, np.abs(y - fb) / sc
    else:
        la, lb = ((y - fa) / sc) ** 2, ((y - fb) / sc) ** 2
    out = M.dm_test(la.mean(axis=0), lb.mean(axis=0), h=H)
    out.update({"model_a": model_a, "model_b": model_b, "H": H, "loss": loss, "phase": phase,
                "mean_scaled_loss_a": float(la.mean()), "mean_scaled_loss_b": float(lb.mean())})
    return out


def dm_per_series(book: ForecastBook, phase: str, model_a: str, model_b: str, H: int, loss: str = "abs") -> pd.DataFrame:
    t, y, fa = book.eval_arrays(model_a, phase, H)
    _, _, fb = book.eval_arrays(model_b, phase, H)
    rows = []
    for s, sid in enumerate(book.ctx.ids):
        la, lb = (np.abs(y[s] - fa[s]), np.abs(y[s] - fb[s])) if loss == "abs" else ((y[s] - fa[s]) ** 2, (y[s] - fb[s]) ** 2)
        r = M.dm_test(la, lb, h=H)
        rows.append({"series_id": sid, **r, "model_a": model_a, "model_b": model_b, "H": H, "loss": loss})
    return pd.DataFrame(rows)


def dm_summary(book: ForecastBook, phase: str = "test", ref: str = "XGBoost", others: list[str] | None = None,
               horizons=C.PRIMARY_HORIZONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference model (A) versus every other final model (B) at each primary horizon.

    mean_d < 0 -> the reference has the smaller loss.  Returns (panel table, per-series table)."""
    others = [m for m in (others or MODELS_FINAL) if m != ref]
    panel, per = [], []
    for H in horizons:
        for m in others:
            for loss in ("abs", "sq"):
                panel.append(dm_panel(book, phase, ref, m, H, loss))
            per.append(dm_per_series(book, phase, ref, m, H, "abs"))
    panel = pd.DataFrame(panel)
    per = pd.concat(per, ignore_index=True)
    # Holm correction within each (horizon, loss) family of comparisons
    panel["p_holm"] = np.nan
    for (H, loss), g in panel.groupby(["H", "loss"]):
        p = g["p_value"].to_numpy()
        order = np.argsort(p)
        m_ = len(p)
        adj = np.empty(m_)
        run = 0.0
        for rank, idx in enumerate(order):
            run = max(run, (m_ - rank) * p[idx])
            adj[idx] = min(1.0, run)
        panel.loc[g.index, "p_holm"] = adj
    return panel, per
