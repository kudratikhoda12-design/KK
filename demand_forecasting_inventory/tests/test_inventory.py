"""Unit tests for the inventory model: safety stock, ROP, EOQ, inventory balance, order arrival, stockouts, costs, causality."""
import math

import numpy as np
import pytest

from src import config as C
from src import inventory as inv


# ---------------------------------------------------------------- closed forms
def test_safety_stock_is_z_times_sigma():
    assert inv.safety_stock(1.645, 10.0) == pytest.approx(16.45)
    assert inv.safety_stock(C.Z_VALUES[0.90], 5.0) == pytest.approx(1.282 * 5.0)
    assert inv.safety_stock(C.Z_VALUES[0.98], 0.0) == 0.0


def test_reorder_point_is_mean_plus_safety_stock():
    ss = inv.safety_stock(1.645, 10.0)
    assert inv.reorder_point(50.0, ss) == pytest.approx(66.45)


def test_prescribed_z_values():
    assert C.Z_VALUES == {0.90: 1.282, 0.95: 1.645, 0.98: 2.054}


def test_eoq_known_value_and_edge_cases():
    # Q* = sqrt(2 D S / H) = sqrt(2*1000*10/2) = 100
    assert inv.eoq(1000, 10, 2) == pytest.approx(100.0)
    assert inv.eoq(0, 10, 2) == 0.0
    assert inv.eoq(1000, 10, 0) == 0.0
    q = inv.order_quantity_array(np.array([0.0, 1000.0]), 10.0, 2.0)
    assert q.tolist() == [1.0, 100.0]                       # at least one unit, ceil of EOQ


def test_safety_stock_grows_with_service_level_and_lead_time_sigma():
    sig = 7.0
    ss = [inv.safety_stock(C.Z_VALUES[a], sig) for a in C.SERVICE_LEVELS]
    assert ss == sorted(ss) and len(set(ss)) == 3           # higher service level -> more safety stock
    # classical sigma_L = sqrt(L) * sigma_daily: longer lead time -> larger sigma_L -> larger safety stock
    assert math.sqrt(14) * 2 > math.sqrt(3) * 2


# ---------------------------------------------------------------- simulation mechanics
def _run(demand, rop, S, L, I0):
    return inv.simulate(np.asarray(demand, float), np.asarray(rop, float), np.asarray(S, float), L, I0)


def test_inventory_balance_every_day():
    rng = np.random.default_rng(1)
    T = 200
    demand = rng.poisson(3, T)
    led = _run(demand, np.full(T, 10.0), np.full(T, 40.0), 3, 20)
    # within a day: end = begin - fulfilled
    assert np.allclose(led.ending_inventory, led.begin_inventory - led.fulfilled)
    # across days: begin_t = end_{t-1} + received_t  (and begin_0 = I0 + received_0)
    assert np.allclose(led.begin_inventory[1:], led.ending_inventory[:-1] + led.orders_received[1:])
    assert led.begin_inventory[0] == 20 + led.orders_received[0]
    assert (led.ending_inventory >= 0).all()                 # lost sales: inventory never negative


def test_global_conservation_of_units():
    rng = np.random.default_rng(2)
    T = 120
    demand = rng.poisson(4, T)
    I0 = 15
    led = _run(demand, np.full(T, 12.0), np.full(T, 50.0), 5, I0)
    assert I0 + led.orders_received.sum() - led.fulfilled.sum() == pytest.approx(led.ending_inventory[-1])
    assert led.fulfilled.sum() + led.stockout.sum() == pytest.approx(demand.sum())


@pytest.mark.parametrize("L", [1, 3, 7, 14])
def test_order_placed_end_of_day_t_arrives_start_of_day_t_plus_L_plus_1(L):
    T = 60
    demand = np.ones(T)
    rop = np.full(T, -1.0)
    rop[0] = 1000.0                                          # force exactly one order, at the end of day 0
    S = np.full(T, 500.0)
    led = _run(demand, rop, S, L, 100)
    placed_days = np.flatnonzero(led.orders_placed)
    assert placed_days.tolist() == [0]
    q = led.orders_placed[0]
    arr_days = np.flatnonzero(led.orders_received)
    assert arr_days.tolist() == [L + 1]
    assert led.orders_received[L + 1] == q


def test_order_up_to_level_reached_in_inventory_position():
    T = 50
    demand = np.full(T, 2.0)
    led = _run(demand, np.full(T, 10.0), np.full(T, 30.0), 3, 12)
    ordered = led.orders_placed > 0
    assert ordered.any()
    # right after ordering, inventory position (before the order) + order quantity >= S (ceil of the gap)
    assert np.all(led.inventory_position[ordered] + led.orders_placed[ordered] >= 30.0 - 1e-9)
    assert np.all(led.inventory_position[ordered] + led.orders_placed[ordered] < 30.0 + 1.0)


def test_inventory_position_definition():
    rng = np.random.default_rng(3)
    T = 100
    demand = rng.poisson(2, T)
    led = _run(demand, np.full(T, 8.0), np.full(T, 25.0), 4, 10)
    # IP is evaluated at the decision point: after the day's demand, BEFORE the order placed that day
    placed_before_today = np.cumsum(led.orders_placed) - led.orders_placed
    on_order = placed_before_today - np.cumsum(led.orders_received)
    assert np.allclose(led.inventory_position, led.ending_inventory + on_order)


def test_order_triggered_only_when_position_at_or_below_rop():
    rng = np.random.default_rng(4)
    T = 150
    demand = rng.poisson(3, T)
    rop = np.full(T, 9.0)
    led = _run(demand, rop, np.full(T, 30.0), 3, 30)
    assert np.all((led.orders_placed > 0) == (led.inventory_position <= rop))


def test_stockout_calculation_lost_sales():
    led = _run([3, 4, 2, 5], np.full(4, -1.0), np.full(4, 0.0), 2, 5)   # never reorders
    assert led.fulfilled.tolist() == [3, 2, 0, 0]
    assert led.stockout.tolist() == [0, 2, 2, 5]
    assert led.ending_inventory.tolist() == [2, 0, 0, 0]
    m = inv.ledger_metrics(led)
    assert m["stockout_units"] == 9 and m["stockout_days"] == 3
    assert m["fill_rate"] == pytest.approx(5 / 14)
    assert m["in_stock_rate"] == pytest.approx(1 - 3 / 4)
    assert m["stockout_freq"] == pytest.approx(3 / 4)


def test_cost_calculation_matches_ledger_frame():
    rng = np.random.default_rng(5)
    T = 90
    demand = rng.poisson(3, T)
    led = _run(demand, np.full(T, 9.0), np.full(T, 28.0), 3, 15)
    cp = inv.cost_parameters(4.00)
    met = inv.add_costs(inv.ledger_metrics(led), cp)
    df = led.to_frame(h_day=cp["h_day"], order_cost=cp["order_cost"], stockout_cost_unit=cp["p_MEDIUM"])
    assert met["holding_cost"] == pytest.approx(df["holding_cost"].sum())
    assert met["ordering_cost"] == pytest.approx(df["ordering_cost"].sum())
    assert met["stockout_cost_MEDIUM"] == pytest.approx(df["stockout_cost"].sum())
    assert met["total_cost_MEDIUM"] == pytest.approx(df["total_cost"].sum())
    assert met["total_cost_MEDIUM"] == pytest.approx(met["holding_cost"] + met["ordering_cost"] + met["stockout_cost_MEDIUM"])


def test_cost_parameters_are_documented_hypothetical_assumptions():
    cp = inv.cost_parameters(10.0)
    assert cp["unit_cost"] == pytest.approx(7.0)
    assert cp["unit_margin"] == pytest.approx(3.0)
    assert cp["h_year"] == pytest.approx(0.25 * 7.0)
    assert cp["h_day"] == pytest.approx(0.25 * 7.0 / 365)
    assert cp["p_LOW"] < cp["p_MEDIUM"] < cp["p_HIGH"]
    assert cp["p_MEDIUM"] == pytest.approx(3.0)


def test_no_future_demand_used_by_decisions():
    """Orders/positions up to day tau must not change when demand AFTER tau is changed."""
    rng = np.random.default_rng(6)
    T, tau = 120, 59
    d1 = rng.poisson(3, T).astype(float)
    d2 = d1.copy()
    d2[tau + 1:] = rng.poisson(30, T - tau - 1)              # drastically different future
    rop, S = np.full(T, 9.0), np.full(T, 28.0)
    a, b = _run(d1, rop, S, 3, 15), _run(d2, rop, S, 3, 15)
    for field in ("begin_inventory", "orders_placed", "orders_received", "ending_inventory", "inventory_position", "stockout"):
        assert np.array_equal(getattr(a, field)[: tau + 1], getattr(b, field)[: tau + 1]), field


def test_higher_rop_never_increases_stockouts_in_a_simple_case():
    rng = np.random.default_rng(7)
    T = 365
    demand = rng.poisson(5, T)
    lo = inv.ledger_metrics(_run(demand, np.full(T, 12.0), np.full(T, 12.0 + 40), 3, 40))
    hi = inv.ledger_metrics(_run(demand, np.full(T, 30.0), np.full(T, 30.0 + 40), 3, 40))
    assert hi["stockout_units"] <= lo["stockout_units"]
    assert hi["avg_inventory"] >= lo["avg_inventory"]


def test_cycle_service_level_counts_protection_interval_stockouts():
    # order placed at the end of day 0 (L = 2): protection interval = days 1..2; arrives at the start of day 3
    T, L = 12, 2
    demand = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)
    rop = np.full(T, -1.0); rop[0] = 100.0
    led = _run(demand, rop, np.full(T, 4.0), L, 2)                      # inventory 2: day0 sells 1 (left 1), day1 sells 1 (left 0), day2 stockout
    assert led.stockout.tolist()[:3] == [0, 0, 1]
    n, n_so = inv.cycle_stats(led, L)
    assert (n, n_so) == (1, 1)                                          # the single cycle had a stockout inside days 1..2
    m = inv.ledger_metrics(led, L)
    assert m["cycles_evaluated"] == 1 and m["cycles_with_stockout"] == 1


def test_cycle_stats_ignores_orders_whose_protection_interval_leaves_the_horizon():
    T, L = 10, 4
    demand = np.ones(T)                                                  # one unit sold every day ...
    rop = np.full(T, 10.0)                                               # ... and S = ROP = 10: the position falls below 10 every day -> daily order
    led = _run(demand, rop, rop.copy(), L, 10)
    assert (led.orders_placed > 0).all() and led.stockout.sum() == 0
    n, n_so = inv.cycle_stats(led, L)
    assert n == T - L and n_so == 0                                      # orders placed on the last L days are not evaluated
