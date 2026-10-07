"""Phase 31 - robustness: results by demand tier (high / medium / low) and by demand regularity (regular vs intermittent).

Groups are defined from TRAIN-period information only (selection tiers; ADI >= 1.32 => intermittent).  With 18 series (6 per
tier, 4 regular vs 14 intermittent) these are descriptive subgroup results, not statistically powered claims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import evaluation as ev
from .forecast_core import ForecastBook
from .inventory_experiments import POLICY_A, aggregate


def group_masks(book: ForecastBook) -> dict[str, np.ndarray]:
    sel = book.ctx.sel
    masks = {f"tier={t}": (sel["tier"] == t).to_numpy() for t in C.TIER_ORDER}
    masks.update({f"{k}": (sel["intermittency"] == k).to_numpy() for k in ("regular", "intermittent")})
    return masks


def forecast_by_group(book: ForecastBook, phase: str = "test", models=None, horizons=C.PRIMARY_HORIZONS) -> pd.DataFrame:
    rows = []
    for g, mask in group_masks(book).items():
        pooled = ev.metrics_pooled(book, phase, models or ev.MODELS_FINAL, horizons, series_mask=mask)
        pooled["group"] = g
        pooled["n_series"] = int(mask.sum())
        rows.append(pooled)
    return pd.concat(rows, ignore_index=True)


def best_by_group(tab: pd.DataFrame, metric: str = "WAPE") -> pd.DataFrame:
    rows = []
    for (g, H), d in tab.groupby(["group", "H"]):
        d = d.sort_values(metric)
        rows.append({"group": g, "H": H, "n_series": int(d["n_series"].iloc[0]), "best_model": d.iloc[0]["model"], metric: d.iloc[0][metric],
                     "SeasonalNaive_WAPE": float(d.loc[d["model"] == "SeasonalNaive", metric].iloc[0]),
                     "MovingAverage_WAPE": float(d.loc[d["model"] == "MovingAverage", metric].iloc[0]),
                     "XGBoost_WAPE": float(d.loc[d["model"] == "XGBoost", metric].iloc[0]),
                     "SARIMA_WAPE": float(d.loc[d["model"] == "SARIMA", metric].iloc[0])})
    return pd.DataFrame(rows)


def inventory_by_group(runs: pd.DataFrame, best_B: dict[int, str], scenario: str = "MEDIUM", SL: float = 0.95, L: int = 7,
                       method: str = "gaussian") -> pd.DataFrame:
    """A vs validation-selected B for each tier / regularity group at one headline configuration."""
    sub = runs[(runs["SL"] == SL) & (runs["L"] == L)]
    rows = []
    for key, col in (("tier", "tier"), ("regularity", "intermittency")):
        for val, g in sub.groupby(col):
            a = g[g["policy"] == POLICY_A]
            b = g[(g["policy"] == best_B[L]) & (g["method"] == method)]
            ra = aggregate(a, ["policy"], scenario).iloc[0]
            rb = aggregate(b, ["policy"], scenario).iloc[0]
            rows.append({"group_type": key, "group": val, "n_series": int(a["series_idx"].nunique()), "L": L, "SL": SL, "scenario": scenario,
                         "B_model": best_B[L], "A_total_cost": ra["total_cost"], "B_total_cost": rb["total_cost"],
                         "B_vs_A_total_cost_pct": 100 * (rb["total_cost"] / ra["total_cost"] - 1),
                         "A_fill_rate": ra["fill_rate"], "B_fill_rate": rb["fill_rate"],
                         "A_holding_cost": ra["holding_cost"], "B_holding_cost": rb["holding_cost"],
                         "A_stockout_cost": ra["stockout_cost"], "B_stockout_cost": rb["stockout_cost"]})
    return pd.DataFrame(rows)
