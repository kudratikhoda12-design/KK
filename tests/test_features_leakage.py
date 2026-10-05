"""Leakage and timing tests for features and target."""

import numpy as np
import pandas as pd

from src.features import FEATURES, build_dataset, decision_times, minute_features, perturbation_leakage_check


def test_decision_grid_hourly(small_grid):
    t = decision_times(small_grid.index, 60, 0)
    assert (t.minute == 0).all()
    t15 = decision_times(small_grid.index, 15, 0)
    assert set(t15.minute) == {0, 15, 30, 45}
    t_off = decision_times(small_grid.index, 60, 30)
    assert (t_off.minute == 30).all()


def test_feature_alignment_uses_only_closed_candles(small_grid):
    d = build_dataset(small_grid, horizon=60, lag=1)
    t = d.index[200]
    c = small_grid["close"]
    # ret_60m at t = ln close(t-1m) - ln close(t-61m): the last candle used closes exactly at t
    expected = np.log(c.loc[t - pd.Timedelta("1min")]) - np.log(c.loc[t - pd.Timedelta("61min")])
    assert np.isclose(d.loc[t, "ret_60m"], expected)
    expected_1m = np.log(c.loc[t - pd.Timedelta("1min")]) - np.log(c.loc[t - pd.Timedelta("2min")])
    assert np.isclose(d.loc[t, "ret_1m"], expected_1m)


def test_target_uses_execution_lag(small_grid):
    d = build_dataset(small_grid, horizon=60, lag=1)
    t = d.index[200]
    o = small_grid["open"]
    entry, exit_ = o.loc[t + pd.Timedelta("1min")], o.loc[t + pd.Timedelta("61min")]
    assert np.isclose(d.loc[t, "fwd_logret"], np.log(exit_ / entry))
    assert d.loc[t, "entry_time"] == t + pd.Timedelta("1min")
    assert d.loc[t, "exit_time"] == t + pd.Timedelta("61min")
    assert d.loc[t, "y"] == float(exit_ > entry)


def test_perturbing_the_future_never_changes_features(small_grid):
    times = build_dataset(small_grid).dropna(subset=FEATURES).index[[30, 100, 200]]
    res = perturbation_leakage_check(small_grid, times, lookback_days=8, lookahead_days=1)
    assert len(res) == 3
    assert (res["features_changed"] == 0).all(), res
    assert res["target_changed"].all()        # sanity: the perturbation did reach the target


def test_missing_minutes_are_not_filled(small_grid):
    g = small_grid.copy()
    hole = g.index[5000:5030]                  # 30 missing minutes
    g.loc[hole, ["open", "high", "low", "close", "volume", "taker_buy_base"]] = np.nan
    f = minute_features(g)
    # a 60-minute window containing 30 missing minutes (<90% coverage) must be NaN
    assert np.isnan(f.loc[g.index[5040], "log_rv_60m"])
    # a return whose endpoint is missing must be NaN
    assert np.isnan(f.loc[g.index[5010], "ret_1m"])


def test_rolling_windows_are_backward_looking(small_grid):
    f1 = minute_features(small_grid)
    g = small_grid.copy()
    g.iloc[-500:, g.columns.get_loc("close")] *= 3.0     # change only the last 500 minutes
    f2 = minute_features(g)
    early = f1.index[:-600]
    pd.testing.assert_frame_equal(f1.loc[early], f2.loc[early])
