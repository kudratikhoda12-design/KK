"""Phases 28-30 - summaries of the simulation: headline tables, sensitivity, paired comparisons, accuracy vs business value.

Headline Policy B uses, for each lead time L, the forecasting model with the lowest VALIDATION WAPE at horizon H = L (the
choice a practitioner could have made before the test period).  Whether that choice also minimises TEST inventory cost is
exactly the question of Phase 30 and is answered, not assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .inventory_experiments import POLICY_A, aggregate

RNG_SEED = C.SEED


def best_models_by_validation(pooled_val: pd.DataFrame, models: list[str]) -> dict[int, str]:
    out = {}
    for L in C.LEAD_TIMES:
        g = pooled_val[(pooled_val["H"] == L) & (pooled_val["model"].isin(models))].sort_values("WAPE")
        out[L] = str(g.iloc[0]["model"])
    return out


def headline_runs(runs: pd.DataFrame, best_B: dict[int, str], method: str = "gaussian") -> pd.DataFrame:
    """Rows for Policy A and for the validation-selected Policy B model of each lead time."""
    a = runs[(runs["policy"] == POLICY_A)].copy()
    a["policy_group"] = "A_historical"
    b = runs[(runs["method"] == method) & (runs["policy"] != POLICY_A) &
             (runs.apply(lambda r: best_B.get(int(r["L"])) == r["policy"], axis=1))].copy()
    b["policy_group"] = "B_forecast_driven"
    return pd.concat([a, b], ignore_index=True)


def final_inventory_table(runs: pd.DataFrame, best_B: dict[int, str]) -> pd.DataFrame:
    hr = headline_runs(runs, best_B)
    parts = []
    for scn in C.SCENARIOS:
        agg = aggregate(hr, ["policy_group", "policy", "L", "SL"], scenario=scn)
        parts.append(agg)
    t = pd.concat(parts, ignore_index=True)
    t = t.rename(columns={"policy_group": "Policy", "policy": "Forecast_Model", "L": "Lead_Time_days", "SL": "Service_Level",
                          "stockout_days": "Stockouts", "stockout_units": "Stockout_Units", "fill_rate": "Fill_Rate",
                          "avg_inventory_units": "Average_Inventory", "holding_cost": "Holding_Cost",
                          "ordering_cost": "Ordering_Cost", "stockout_cost": "Stockout_Cost", "total_cost": "Total_Cost",
                          "scenario": "Cost_Scenario"})
    t["Forecast_Model"] = t["Forecast_Model"].replace({POLICY_A: "none (28-day trailing mean/std)"})
    t = t.rename(columns={"cycle_service_level": "Cycle_Service_Level", "in_stock_rate": "In_Stock_Day_Rate", "turnover": "Inventory_Turnover",
                          "avg_inventory_value": "Average_Inventory_Value_USD", "n_orders": "Orders_Placed"})
    cols = ["Policy", "Forecast_Model", "Cost_Scenario", "Lead_Time_days", "Service_Level", "Stockouts", "Stockout_Units", "Fill_Rate",
            "Cycle_Service_Level", "In_Stock_Day_Rate", "Average_Inventory", "Average_Inventory_Value_USD", "Inventory_Turnover", "Orders_Placed",
            "Holding_Cost", "Ordering_Cost", "Stockout_Cost", "Total_Cost"]
    return t[cols].sort_values(["Cost_Scenario", "Lead_Time_days", "Service_Level", "Policy"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Paired comparison B vs A (bootstrap over series)
# --------------------------------------------------------------------------------------
def paired_bootstrap(runs: pd.DataFrame, policy_b: str, L: int, SL: float, scenario: str, method: str = "gaussian",
                     n_boot: int = 5000, metric: str = "total_cost") -> dict:
    col = {"total_cost": f"total_cost_{scenario}", "holding_cost": "holding_cost", "stockout_units": "stockout_units",
           "ordering_cost": "ordering_cost"}[metric]
    a = runs[(runs["policy"] == POLICY_A) & (runs["L"] == L) & (runs["SL"] == SL)].sort_values("series_idx")[col].to_numpy()
    b = runs[(runs["policy"] == policy_b) & (runs["method"] == method) & (runs["L"] == L) & (runs["SL"] == SL)].sort_values("series_idx")[col].to_numpy()
    n = len(a)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, n, size=(n_boot, n))
    rel = (b[idx].sum(axis=1) - a[idx].sum(axis=1)) / a[idx].sum(axis=1)
    wins = int((b < a).sum())
    ties = int((b == a).sum())
    pval = float(stats.binomtest(wins, n - ties, 0.5).pvalue) if n - ties > 0 else np.nan
    return {"policy_B": policy_b, "L": L, "SL": SL, "scenario": scenario, "metric": metric, "sum_A": float(a.sum()), "sum_B": float(b.sum()),
            "rel_diff_pooled": float((b.sum() - a.sum()) / a.sum()), "boot_ci_low": float(np.quantile(rel, 0.025)),
            "boot_ci_high": float(np.quantile(rel, 0.975)), "series_B_cheaper": wins, "series_total": n, "sign_test_p": pval}


def paired_table(runs: pd.DataFrame, best_B: dict[int, str], method: str = "gaussian") -> pd.DataFrame:
    rows = []
    for scn in C.SCENARIOS:
        for L in C.LEAD_TIMES:
            for SL in C.SERVICE_LEVELS:
                rows.append(paired_bootstrap(runs, best_B[L], L, SL, scn, method))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Sensitivity
# --------------------------------------------------------------------------------------
def sensitivity_table(runs: pd.DataFrame, best_B: dict[int, str], method: str = "gaussian") -> pd.DataFrame:
    hr = headline_runs(runs, best_B, method)
    parts = [aggregate(hr, ["policy_group", "L", "SL"], scenario=s) for s in C.SCENARIOS]
    return pd.concat(parts, ignore_index=True)


def optimal_service_level(sens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (grp, L, scn), g in sens.groupby(["policy_group", "L", "scenario"]):
        g = g.sort_values("total_cost")
        rows.append({"policy_group": grp, "L": L, "scenario": scn, "cost_minimising_SL": float(g.iloc[0]["SL"]),
                     "min_total_cost": float(g.iloc[0]["total_cost"]),
                     **{f"total_cost_SL{int(round(sl * 100))}": float(g[g["SL"] == sl]["total_cost"].iloc[0]) for sl in C.SERVICE_LEVELS}})
    return pd.DataFrame(rows)


def factor_importance(sens: pd.DataFrame) -> pd.DataFrame:
    """How much does pooled total cost move when ONE factor changes over its range (others averaged)?
    Reported as (max - min of the factor-level means) / grand mean, for each policy group."""
    rows = []
    for grp, g in sens.groupby("policy_group"):
        grand = g["total_cost"].mean()
        for factor in ("L", "SL", "scenario"):
            m = g.groupby(factor)["total_cost"].mean()
            rows.append({"policy_group": grp, "factor": {"L": "lead time (3/7/14 d)", "SL": "service level (90/95/98 %)",
                                                          "scenario": "stockout-penalty scenario (LOW/MEDIUM/HIGH)"}[factor],
                         "relative_range_of_mean_total_cost": float((m.max() - m.min()) / grand),
                         "levels": ", ".join(f"{k}: {v:,.0f}" for k, v in m.items())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Accuracy vs business value
# --------------------------------------------------------------------------------------
def accuracy_vs_cost(runs: pd.DataFrame, pooled_val: pd.DataFrame, pooled_test: pd.DataFrame, models: list[str],
                     method: str = "gaussian") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (L, SL, scenario): rank correlation between forecast error (WAPE/RMSE at H = L) and simulated total cost."""
    rows, summ = [], []
    for scn in C.SCENARIOS:
        agg = aggregate(runs[(runs["method"] == method) & runs["policy"].isin(models)], ["policy", "L", "SL"], scenario=scn)
        for L in C.LEAD_TIMES:
            acc_test = pooled_test[(pooled_test["H"] == L) & pooled_test["model"].isin(models)].set_index("model")
            acc_val = pooled_val[(pooled_val["H"] == L) & pooled_val["model"].isin(models)].set_index("model")
            for SL in C.SERVICE_LEVELS:
                g = agg[(agg["L"] == L) & (agg["SL"] == SL)].set_index("policy")
                df = pd.DataFrame({"test_WAPE": acc_test["WAPE"], "test_RMSE": acc_test["RMSE"], "val_WAPE": acc_val["WAPE"],
                                   "total_cost": g["total_cost"], "fill_rate": g["fill_rate"], "holding_cost": g["holding_cost"],
                                   "stockout_cost": g["stockout_cost"]}).dropna()
                df["cost_rank"] = df["total_cost"].rank(method="min")
                df["test_wape_rank"] = df["test_WAPE"].rank(method="min")
                df["val_wape_rank"] = df["val_WAPE"].rank(method="min")
                for m, r in df.iterrows():
                    rows.append({"scenario": scn, "L": L, "SL": SL, "model": m, **r.to_dict()})
                rho_t = stats.spearmanr(df["test_WAPE"], df["total_cost"])
                rho_v = stats.spearmanr(df["val_WAPE"], df["total_cost"])
                rho_r = stats.spearmanr(df["test_RMSE"], df["total_cost"])
                dn = df.drop(index="Naive", errors="ignore")             # Naive is worst on both axes and inflates the correlation
                rho_n = stats.spearmanr(dn["test_WAPE"], dn["total_cost"])
                summ.append({"scenario": scn, "L": L, "SL": SL, "n_models": len(df),
                             "best_by_val_WAPE": df["val_WAPE"].idxmin(), "best_by_test_WAPE": df["test_WAPE"].idxmin(),
                             "best_by_total_cost": df["total_cost"].idxmin(),
                             "same_best_test_accuracy_and_cost": df["test_WAPE"].idxmin() == df["total_cost"].idxmin(),
                             "spearman_testWAPE_vs_cost": float(rho_t.statistic), "spearman_p": float(rho_t.pvalue),
                             "spearman_testWAPE_vs_cost_excl_Naive": float(rho_n.statistic), "spearman_excl_Naive_p": float(rho_n.pvalue),
                             "spearman_valWAPE_vs_cost": float(rho_v.statistic), "spearman_testRMSE_vs_cost": float(rho_r.statistic),
                             "cost_of_best_by_val_WAPE": float(df.loc[df["val_WAPE"].idxmin(), "total_cost"]),
                             "min_cost": float(df["total_cost"].min()),
                             "regret_of_val_choice_pct": float(100 * (df.loc[df["val_WAPE"].idxmin(), "total_cost"] / df["total_cost"].min() - 1))})
    return pd.DataFrame(rows), pd.DataFrame(summ)
