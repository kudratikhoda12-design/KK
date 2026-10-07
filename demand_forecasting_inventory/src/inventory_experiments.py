"""Phases 26-31 - run the inventory simulation for every (policy, lead time, service level, series) combination.

Policy A  ("historical baseline", no forecasting model): mu_L = L x mean of the last 28 days, sigma_L = sqrt(L) x std of the
          last 28 days (classical textbook formula, i.i.d. daily demand), D = 365 x trailing mean.
Policy B  ("forecast-driven"): mu_L and D from a forecasting model of the ForecastBook (H = L for the lead-time demand,
          H = 28 for the annual-demand rate used by EOQ); sigma_L from that model's VALIDATION residuals of the L-day sum.
          Safety stock is Gaussian (z * sigma) by default; an empirical-quantile variant is run as a sensitivity.

A and B share lead times, cost parameters, simulation period (the TEST period), initial inventory and decision timing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import inventory as inv
from .features import trailing_stat
from .forecast_core import ForecastBook, H_INDEX, Context

POLICY_A = "A_historical"


def prepare_inputs(ctx: Context) -> dict:
    sp = ctx.split
    t_days = np.arange(sp.val_end, ctx.N)                           # simulated days = test period
    demand = ctx.Y[:, t_days]
    price_ref = ctx.price[:, sp.val_end - 1]                        # last price known before the test period
    mean28 = np.vstack([trailing_stat(ctx.Y[s], 28, "mean") for s in range(ctx.n_series)])
    std28 = np.vstack([trailing_stat(ctx.Y[s], 28, "std") for s in range(ctx.n_series)])
    return {
        "t_days": t_days, "demand": demand, "price_ref": price_ref,
        "mean28_origin": mean28[:, t_days], "std28_origin": std28[:, t_days],      # known at the END of each simulated day
        "mean28_start": mean28[:, sp.val_end - 1],                                  # known before the first simulated day
        "cost_params": [inv.cost_parameters(float(p)) for p in price_ref],
    }


def initial_inventory(inputs: dict, L: int) -> np.ndarray:
    """Identical for every policy: cover of (L + 14) days at the pre-test 28-day mean demand."""
    return np.ceil(inputs["mean28_start"] * (L + 14))


def policy_arrays(ctx: Context, book: ForecastBook, inputs: dict, policy: str, L: int, SL: float, sigma_df: pd.DataFrame | None,
                  method: str = "gaussian") -> tuple[np.ndarray, np.ndarray]:
    """ROP and order-up-to level for every series and simulated day (arrays of shape (n_series, T))."""
    n_s, T = inputs["demand"].shape
    z = C.Z_VALUES[SL]
    rop = np.zeros((n_s, T))
    S = np.zeros((n_s, T))
    for s in range(n_s):
        cp = inputs["cost_params"][s]
        if policy == POLICY_A:
            m28, sd28 = inputs["mean28_origin"][s], inputs["std28_origin"][s]
            mu_L, ann = L * m28, C.DAYS_PER_YEAR * m28
            r = mu_L + z * np.sqrt(L) * sd28
        else:
            F = book.F["test"][policy]                            # (n_series, n_origins, n_H); origin index o = j + 1
            mu_L = F[s, 1:T + 1, H_INDEX[L]]
            ann = C.DAYS_PER_YEAR * F[s, 1:T + 1, H_INDEX[28]] / 28.0
            row = sigma_df[(sigma_df["model"] == policy) & (sigma_df["H"] == L) & (sigma_df["series_idx"] == s)].iloc[0]
            if method == "gaussian":
                r = mu_L + z * row["sigma_rmse"]
            elif method == "empirical":
                r = mu_L + row[f"q{int(round(SL * 100))}"]
            else:
                raise ValueError(method)
        rop[s] = np.clip(r, 0.0, None)
        Q = inv.order_quantity_array(ann, cp["order_cost"], cp["h_year"])
        S[s] = rop[s] + Q
    return rop, S


def run_experiments(ctx: Context, book: ForecastBook, models: list[str], sigma_df: pd.DataFrame,
                    methods=("gaussian", "empirical"), lead_times=C.LEAD_TIMES, service_levels=C.SERVICE_LEVELS,
                    keep_ledgers_for: set | None = None) -> tuple[pd.DataFrame, dict]:
    """Simulate everything.  ``keep_ledgers_for``: set of (policy, L, SL, method) whose per-series ledgers are returned."""
    inputs = prepare_inputs(ctx)
    keep_ledgers_for = keep_ledgers_for or set()
    rows, ledgers = [], {}
    policies = [POLICY_A] + list(models)
    for policy in policies:
        for method in (("gaussian",) if policy == POLICY_A else methods):
            for L in lead_times:
                I0 = initial_inventory(inputs, L)
                for SL in service_levels:
                    rop, S = policy_arrays(ctx, book, inputs, policy, L, SL, sigma_df, method)
                    for s in range(ctx.n_series):
                        led = inv.simulate(inputs["demand"][s], rop[s], S[s], L, I0[s])
                        met = inv.add_costs(inv.ledger_metrics(led, L), inputs["cost_params"][s])
                        rows.append({"series_idx": s, "series_id": ctx.ids[s], "tier": ctx.sel.loc[s, "tier"],
                                     "intermittency": ctx.sel.loc[s, "intermittency"], "policy": policy, "method": method,
                                     "L": L, "SL": SL, **met})
                        if (policy, L, SL, method) in keep_ledgers_for:
                            ledgers[(policy, L, SL, method, s)] = led
    runs = pd.DataFrame(rows)
    runs["policy_label"] = np.where(runs["policy"] == POLICY_A, "A_historical", "B_" + runs["policy"].astype(str))
    return runs, ledgers


# --------------------------------------------------------------------------------------
# Aggregation helpers
# --------------------------------------------------------------------------------------
def aggregate(runs: pd.DataFrame, by: list[str], scenario: str = "MEDIUM") -> pd.DataFrame:
    """Sum over series.  fill_rate is volume-weighted; in_stock_rate is the mean over series-days."""
    g = runs.groupby(by)
    out = g.agg(series=("series_idx", "nunique"), demand_units=("demand_units", "sum"), fulfilled_units=("fulfilled_units", "sum"),
                stockout_units=("stockout_units", "sum"), stockout_days=("stockout_days", "sum"), series_days=("T", "sum"),
                avg_inventory_units=("avg_inventory", "sum"), avg_inventory_value=("avg_inventory_value", "sum"),
                n_orders=("n_orders", "sum"), cycles_evaluated=("cycles_evaluated", "sum"), cycles_with_stockout=("cycles_with_stockout", "sum"),
                holding_cost=("holding_cost", "sum"), ordering_cost=("ordering_cost", "sum"),
                stockout_cost=(f"stockout_cost_{scenario}", "sum"), total_cost=(f"total_cost_{scenario}", "sum")).reset_index()
    out["fill_rate"] = out["fulfilled_units"] / out["demand_units"]
    out["in_stock_rate"] = 1.0 - out["stockout_days"] / out["series_days"]
    out["cycle_service_level"] = 1.0 - out["cycles_with_stockout"] / out["cycles_evaluated"].replace(0, np.nan)
    days_per_series = out["series_days"] / out["series"]
    out["turnover"] = (out["fulfilled_units"] * C.DAYS_PER_YEAR / days_per_series) / out["avg_inventory_units"]
    out["scenario"] = scenario
    return out


def eoq_cycle_sanity(ctx: Context) -> pd.DataFrame:
    """Plausibility check of the HYPOTHETICAL cost parameters using TRAIN data only (no validation/test information).

    EOQ and its cycle length for every series, and the newsvendor-style critical ratio per replenishment cycle
    CR = p / (p + h_day x cycle_days)  (underage = penalty per lost unit, overage = holding cost of a leftover unit for one cycle).
    """
    sp = ctx.split
    rows = []
    for s, sid in enumerate(ctx.ids):
        mean_tr = float(ctx.Y[s, : sp.train_end].mean())
        cp = inv.cost_parameters(float(ctx.price[s, sp.train_end - 1]))
        q = inv.eoq(mean_tr * C.DAYS_PER_YEAR, cp["order_cost"], cp["h_year"])
        cyc = q / mean_tr if mean_tr > 0 else np.nan
        row = {"series_id": sid, "tier": ctx.sel.loc[s, "tier"], "train_mean_per_day": mean_tr, "last_train_price": float(ctx.price[s, sp.train_end - 1]),
               "eoq_units": q, "eoq_cycle_days": cyc}
        for scn in C.SCENARIOS:
            p_ = cp[f"p_{scn}"]
            row[f"critical_ratio_{scn}"] = p_ / (p_ + cp["h_day"] * cyc)
        rows.append(row)
    return pd.DataFrame(rows)
