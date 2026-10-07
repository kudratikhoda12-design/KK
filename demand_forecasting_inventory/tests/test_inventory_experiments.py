"""Aggregation and comparison helpers of the inventory experiments."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import inventory_analysis as ia
from src import inventory_experiments as ie


def _fake_runs():
    rows = []
    rng = np.random.default_rng(0)
    for policy in (ie.POLICY_A, "XGBoost"):
        for L in C.LEAD_TIMES:
            for SL in C.SERVICE_LEVELS:
                for s in range(6):
                    dem = 100.0
                    ful = dem * (0.9 if policy == ie.POLICY_A else 0.95)
                    hold = 10.0 + rng.random()
                    row = {"series_idx": s, "series_id": f"S{s}", "tier": "high", "intermittency": "regular", "policy": policy, "method": "gaussian",
                           "L": L, "SL": SL, "T": 292, "demand_units": dem, "fulfilled_units": ful, "stockout_units": dem - ful,
                           "stockout_days": 10 if policy == ie.POLICY_A else 5, "avg_inventory": 20.0, "avg_inventory_value": 40.0, "n_orders": 5,
                           "cycles_evaluated": 5, "cycles_with_stockout": 2 if policy == ie.POLICY_A else 1,
                           "holding_cost": hold, "ordering_cost": 5.0}
                    for scn, k in (("LOW", 0.1), ("MEDIUM", 1.0), ("HIGH", 3.0)):
                        row[f"stockout_cost_{scn}"] = k * (dem - ful)
                        row[f"total_cost_{scn}"] = hold + 5.0 + k * (dem - ful)
                    rows.append(row)
    runs = pd.DataFrame(rows)
    runs["policy_label"] = np.where(runs["policy"] == ie.POLICY_A, "A_historical", "B_" + runs["policy"])
    return runs


def test_aggregate_sums_and_volume_weighted_fill_rate():
    runs = _fake_runs()
    agg = ie.aggregate(runs, ["policy", "L", "SL"], scenario="MEDIUM")
    a = agg[(agg["policy"] == ie.POLICY_A) & (agg["L"] == 7) & (agg["SL"] == 0.95)].iloc[0]
    assert a["demand_units"] == 600 and a["fill_rate"] == pytest.approx(0.9)
    assert a["stockout_days"] == 60 and a["in_stock_rate"] == pytest.approx(1 - 60 / (6 * 292))
    assert a["total_cost"] == pytest.approx(a["holding_cost"] + a["ordering_cost"] + a["stockout_cost"])
    assert a["series"] == 6
    assert a["cycles_evaluated"] == 30 and a["cycle_service_level"] == pytest.approx(1 - 12 / 30)               # 6 series x 2 stockout cycles of 5
    assert a["turnover"] == pytest.approx((540 * 365 / 292) / (6 * 20.0))                                      # annualised FULFILLED (6 x 90) / avg inventory


def test_paired_bootstrap_reports_direction_and_counts():
    runs = _fake_runs()
    r = ia.paired_bootstrap(runs, "XGBoost", 7, 0.95, "MEDIUM", n_boot=500)
    assert r["rel_diff_pooled"] < 0 and r["series_B_cheaper"] == 6 and r["boot_ci_high"] < 0       # B has fewer lost units in this fake set
    assert r["sum_B"] < r["sum_A"]


def test_headline_runs_uses_validation_selected_model_per_lead_time():
    runs = _fake_runs()
    hr = ia.headline_runs(runs, {3: "XGBoost", 7: "XGBoost", 14: "XGBoost"})
    assert set(hr["policy_group"]) == {"A_historical", "B_forecast_driven"}
    assert (hr[hr["policy_group"] == "B_forecast_driven"]["policy"] == "XGBoost").all()


def test_final_table_has_required_columns():
    runs = _fake_runs()
    t = ia.final_inventory_table(runs, {3: "XGBoost", 7: "XGBoost", 14: "XGBoost"})
    required = {"Policy", "Lead_Time_days", "Service_Level", "Stockouts", "Fill_Rate", "Average_Inventory", "Holding_Cost", "Ordering_Cost",
                "Stockout_Cost", "Total_Cost", "Cycle_Service_Level", "Inventory_Turnover"}
    assert required <= set(t.columns)
    assert len(t) == 2 * 3 * 3 * 3                                                                   # policies x L x SL x cost scenarios


def test_optimal_service_level_picks_cheapest():
    sens = pd.DataFrame({"policy_group": "B_forecast_driven", "L": 7, "scenario": "MEDIUM", "SL": [0.90, 0.95, 0.98],
                         "total_cost": [120.0, 100.0, 110.0]})
    out = ia.optimal_service_level(sens)
    assert out["cost_minimising_SL"].iloc[0] == 0.95 and out["min_total_cost"].iloc[0] == 100.0
