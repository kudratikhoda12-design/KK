import numpy as np
import pandas as pd

from src.backtest import positions_from_probs, run_backtest


def _events(n=6):
    t = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"entry_time": t + pd.Timedelta("1min"), "exit_time": t + pd.Timedelta("61min"),
                         "entry_price": 100.0, "exit_price": 101.0, "fwd_ret": 0.01}, index=t)


def test_constant_long_pays_costs_only_at_entry_and_exit():
    e = _events()
    bt = run_backtest(e, pd.Series(1.0, index=e.index), cost_per_side=0.001)
    assert np.isclose(bt["turnover"].sum(), 2.0)              # buy once, sell once
    assert np.isclose(bt["gross_ret"].sum(), 0.06)
    assert np.isclose(bt["cost"].sum(), 0.002)


def test_flip_long_to_short_is_two_units_of_turnover():
    e = _events(2)
    bt = run_backtest(e, pd.Series([1.0, -1.0], index=e.index), cost_per_side=0.001)
    assert list(bt["turnover"]) == [1.0, 3.0]                 # open 1; flip 2 + final close 1
    assert np.isclose(bt["gross_ret"].iloc[1], -0.01)


def test_gap_forces_close_and_reopen():
    e = _events(3).drop(index=_events(3).index[1])            # remove the middle hour
    bt = run_backtest(e, pd.Series(1.0, index=e.index), cost_per_side=0.001)
    assert list(bt["turnover"]) == [2.0, 2.0]                 # not contiguous: each is open+close


def test_threshold_creates_no_trade_region():
    p = pd.Series([0.40, 0.49, 0.50, 0.51, 0.60])
    pos = positions_from_probs(p, scale=0.02, k=1.0, mode="long_short")
    assert list(pos) == [-1.0, 0.0, 0.0, 0.0, 1.0]
    pos_lf = positions_from_probs(p, scale=0.02, k=1.0, mode="long_flat")
    assert list(pos_lf) == [0.0, 0.0, 0.0, 0.0, 1.0]
