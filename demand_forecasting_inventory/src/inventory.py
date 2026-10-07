"""Phases 23-28 - inventory model: safety stock, reorder point, EOQ, (s, S) policy and the daily simulation.

Policy (reorder point + order-up-to level, "ROP + EOQ")
-------------------------------------------------------
At the END of every day t (after that day's demand is known) the inventory position IP_t = on-hand + on-order is checked::

    mu_L(t)  = forecast of demand over the next L days (days t+1 ... t+L) made with information up to t
    sigma_L  = standard deviation of the L-day forecast error
    SS       = z * sigma_L                      (z = 1.282 / 1.645 / 2.054 for 90 / 95 / 98 % target service)
    ROP_t    = mu_L(t) + SS
    Q_t      = ceil( EOQ ),  EOQ = sqrt(2 * D * S_o / H),  D = annualised 28-day forecast, S_o = order cost,
               H = holding cost per unit per year
    if IP_t <= ROP_t:  order  S_t - IP_t  units, where S_t = ROP_t + Q_t   (order-up-to level)

Timing convention (lead time L days): an order placed at the end of day t is received at the START of day t+L+1, so the
L full days t+1 ... t+L are exactly the protection interval that mu_L and sigma_L describe.  Unmet demand is lost
(retail), and is penalised per unit.  Nothing in the policy can see demand of day t+1 or later (tested).

All costs are HYPOTHETICAL (config.COSTS); none of them is an observed business cost.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------------------
# Closed-form building blocks
# --------------------------------------------------------------------------------------
def safety_stock(z: float, sigma_L: float) -> float:
    return float(z * sigma_L)


def reorder_point(mu_L: float, ss: float) -> float:
    return float(mu_L + ss)


def eoq(annual_demand: float, order_cost: float, holding_cost_per_unit_year: float) -> float:
    """Wilson's economic order quantity  Q* = sqrt(2 D S / H)."""
    if annual_demand <= 0 or holding_cost_per_unit_year <= 0:
        return 0.0
    return math.sqrt(2.0 * annual_demand * order_cost / holding_cost_per_unit_year)


def order_quantity_array(annual_demand: np.ndarray, order_cost: float, h_year: float) -> np.ndarray:
    d = np.clip(annual_demand, 0.0, None)
    q = np.sqrt(2.0 * d * order_cost / h_year)
    return np.maximum(1.0, np.ceil(q))


# --------------------------------------------------------------------------------------
# Daily simulation
# --------------------------------------------------------------------------------------
@dataclass
class Ledger:
    begin_inventory: np.ndarray
    demand: np.ndarray
    fulfilled: np.ndarray
    stockout: np.ndarray          # units of unmet (lost) demand
    orders_placed: np.ndarray     # quantity ordered at the end of the day (0 = no order)
    orders_received: np.ndarray   # quantity received at the start of the day
    ending_inventory: np.ndarray
    inventory_position: np.ndarray   # on-hand + on-order at the decision point: after the day's demand, BEFORE that day's order
    rop: np.ndarray
    order_up_to: np.ndarray

    def to_frame(self, dates=None, h_day: float = 0.0, order_cost: float = 0.0, stockout_cost_unit: float = 0.0) -> pd.DataFrame:
        n = len(self.demand)
        df = pd.DataFrame({
            "date": dates if dates is not None else np.arange(n),
            "begin_inventory": self.begin_inventory, "demand": self.demand, "fulfilled_demand": self.fulfilled,
            "stockout": self.stockout, "orders_placed": self.orders_placed, "orders_received": self.orders_received,
            "ending_inventory": self.ending_inventory, "inventory_position": self.inventory_position,
            "reorder_point": self.rop, "order_up_to": self.order_up_to})
        df["holding_cost"] = h_day * df["ending_inventory"]
        df["ordering_cost"] = order_cost * (df["orders_placed"] > 0)
        df["stockout_cost"] = stockout_cost_unit * df["stockout"]
        df["total_cost"] = df["holding_cost"] + df["ordering_cost"] + df["stockout_cost"]
        return df


def simulate(demand: np.ndarray, rop: np.ndarray, order_up_to: np.ndarray, L: int, initial_inventory: float) -> Ledger:
    """Chronological (s, S) simulation with lost sales.

    ``rop[t]`` and ``order_up_to[t]`` are the decision parameters computed at the END of day t from information up to t.
    """
    T = len(demand)
    pipeline = np.zeros(T + L + 3)                    # arrivals by day index (start-of-day receipts)
    on_hand, on_order = float(initial_inventory), 0.0
    led = {k: np.zeros(T) for k in ("begin", "ful", "so", "ord", "rec", "end", "ip")}
    for t in range(T):
        arr = pipeline[t]                              # 1. receive arriving orders
        on_hand += arr
        on_order -= arr
        led["rec"][t] = arr
        led["begin"][t] = on_hand                      # 2. beginning inventory
        d = demand[t]                                  # 3. observe actual demand
        f = d if d <= on_hand else on_hand             # 4. fulfil from stock
        led["ful"][t] = f
        led["so"][t] = d - f                           # 5. record stockout (lost sales)
        on_hand -= f                                   # 6. update inventory
        led["end"][t] = on_hand
        ip = on_hand + on_order                        # 7. inventory position
        led["ip"][t] = ip
        if ip <= rop[t]:                               # 8. check ROP
            q = math.ceil(order_up_to[t] - ip)         # 9. order up to S
            if q < 1:
                q = 1
            pipeline[t + L + 1] += q                   # 10. schedule arrival (start of day t+L+1)
            on_order += q
            led["ord"][t] = q
    return Ledger(led["begin"], np.asarray(demand, float), led["ful"], led["so"], led["ord"], led["rec"], led["end"], led["ip"],
                  np.asarray(rop, float), np.asarray(order_up_to, float))


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def cycle_stats(led: Ledger, L: int) -> tuple[int, int]:
    """(replenishment cycles evaluated, cycles with a stockout).

    A *cycle* starts when an order is placed at the end of day t; its protection interval is days t+1 ... t+L (the L days before
    the order arrives).  It counts as a stockout cycle if any unmet demand occurs in that interval.  Only orders whose complete
    protection interval lies inside the simulated horizon are evaluated.
    """
    T = len(led.demand)
    t_orders = np.flatnonzero(led.orders_placed > 0)
    t_orders = t_orders[t_orders + L <= T - 1]
    cs = np.concatenate([[0.0], np.cumsum(led.stockout > 0)])             # cs[j] = number of stockout days among days 0..j-1
    so_in = cs[t_orders + L + 1] - cs[t_orders + 1]
    return int(len(t_orders)), int((so_in > 0).sum())


def ledger_metrics(led: Ledger, L: int | None = None) -> dict:
    """Metrics (all definitions in reports/final_report.md)::

    fill_rate        = fulfilled units / demanded units                      (Type-2 service)
    in_stock_rate    = share of days without any stockout                    (day-based Type-1 service level)
    stockout_freq    = share of days with a stockout
    stockout_units   = unmet (lost) units
    avg_inventory    = mean ending inventory (units);  max_inventory = peak ending inventory
    turnover         = annualised fulfilled units / average inventory
    n_orders, order_units = number / total size of replenishment orders placed
    cycle_service_level = share of replenishment cycles without a stockout in the protection interval (Type-1 service level, the
                          quantity the z-based safety stock is designed to hit); needs the lead time L
    """
    T = len(led.demand)
    dem = float(led.demand.sum())
    ful = float(led.fulfilled.sum())
    so_days = int((led.stockout > 0).sum())
    avg_inv = float(led.ending_inventory.mean())
    n_cyc, n_cyc_so = cycle_stats(led, L) if L is not None else (0, 0)
    return {
        "cycles_evaluated": n_cyc, "cycles_with_stockout": n_cyc_so,
        "T": T, "demand_units": dem, "fulfilled_units": ful, "stockout_units": float(led.stockout.sum()),
        "fill_rate": ful / dem if dem > 0 else float("nan"),
        "stockout_days": so_days, "stockout_freq": so_days / T, "in_stock_rate": 1.0 - so_days / T,
        "avg_inventory": avg_inv, "max_inventory": float(led.ending_inventory.max()),
        "turnover": (ful / T * C.DAYS_PER_YEAR) / avg_inv if avg_inv > 0 else float("nan"),
        "n_orders": int((led.orders_placed > 0).sum()), "order_units": float(led.orders_placed.sum()),
        "holding_unit_days": float(led.ending_inventory.sum()),
    }


def cost_parameters(price: float, costs: C.CostAssumptions = C.COSTS) -> dict:
    """Per-series (hypothetical) unit economics from the last observed selling price before the test period."""
    unit_cost = costs.unit_cost_ratio * price
    margin = price - unit_cost
    h_year = costs.holding_rate * unit_cost
    return {"price": price, "unit_cost": unit_cost, "unit_margin": margin, "h_year": h_year,
            "h_day": h_year / C.DAYS_PER_YEAR, "order_cost": costs.order_cost,
            **{f"p_{k}": v * margin for k, v in costs.multipliers().items()}}


def add_costs(m: dict, cp: dict) -> dict:
    """Add holding, ordering and (per scenario) stockout and total costs to a metrics dict."""
    out = dict(m)
    out["holding_cost"] = cp["h_day"] * m["holding_unit_days"]
    out["ordering_cost"] = cp["order_cost"] * m["n_orders"]
    out["avg_inventory_value"] = m["avg_inventory"] * cp["unit_cost"]
    for scn in C.SCENARIOS:
        out[f"stockout_cost_{scn}"] = cp[f"p_{scn}"] * m["stockout_units"]
        out[f"total_cost_{scn}"] = out["holding_cost"] + out["ordering_cost"] + out[f"stockout_cost_{scn}"]
    return out
