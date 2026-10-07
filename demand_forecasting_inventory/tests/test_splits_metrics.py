"""Chronological splitting, forecast origins, error metrics and the Diebold-Mariano test."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import metrics as M
from src.forecast_core import truth_sums, paths_to_sums
from src.splits import make_split


# ------------------------------------------------------------------ splits
def test_split_is_chronological_and_exhaustive():
    dates = pd.date_range("2011-01-29", periods=1941)
    sp = make_split(dates)
    assert (sp.train_end, sp.val_end, sp.n_days) == (1358, 1649, 1941)
    a, b = sp.bounds("train"), sp.bounds("validation")
    c = sp.bounds("test")
    assert a[1] == b[0] and b[1] == c[0] and c[1] == sp.n_days                      # contiguous, no overlap
    assert sp.date_range("train")[1] < sp.date_range("validation")[0] < sp.date_range("validation")[1] < sp.date_range("test")[0]
    assert abs((b[1] - b[0]) / 1941 - C.VAL_FRAC) < 0.01 and abs(a[1] / 1941 - C.TRAIN_FRAC) < 0.01


@pytest.mark.parametrize("H", C.HORIZONS)
def test_origins_keep_windows_inside_period(H):
    sp = make_split(pd.date_range("2011-01-29", periods=1941))
    for period in ("validation", "test"):
        a, b = sp.bounds(period)
        o = sp.eval_origins(period, H)
        assert (o + 1 >= a).all() and (o + H <= b - 1).all()
        assert len(o) == (b - a) - H + 1
    tr = sp.train_origins(H, sp.train_end)
    assert (tr + H).max() == sp.train_end - 1                                        # last usable target ends right before validation


def test_all_origins_cover_phase_plus_boundary_day():
    sp = make_split(pd.date_range("2011-01-29", periods=1941))
    assert sp.all_origins("test")[0] == sp.val_end - 1 and sp.all_origins("test")[-1] == sp.n_days - 1


def test_truth_sums_known_values_and_nan_tail():
    Y = np.arange(1, 11, dtype=float)[None, :]                                        # 1..10
    T = truth_sums(Y, 3)[0]
    assert T[0] == 2 + 3 + 4 and T[6] == 8 + 9 + 10
    assert np.isnan(T[7:]).all()


def test_paths_to_sums_cumulates_first_h_steps():
    paths = np.ones((1, 2, C.MAX_HORIZON))
    s = paths_to_sums(paths)
    assert s.shape == (1, 2, len(C.HORIZONS)) and s[0, 0].tolist() == list(C.HORIZONS)


# ------------------------------------------------------------------ metrics
def test_basic_metrics_known_values():
    y = np.array([10.0, 20.0, 30.0])
    f = np.array([12.0, 18.0, 33.0])
    assert M.mae(y, f) == pytest.approx((2 + 2 + 3) / 3)
    assert M.rmse(y, f) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert M.wape(y, f) == pytest.approx(100 * 7 / 60)
    assert M.bias(y, f) == pytest.approx((2 - 2 + 3) / 3)
    expected_smape = 100 * np.mean([2 * 2 / 22, 2 * 2 / 38, 2 * 3 / 63])
    assert M.smape(y, f) == pytest.approx(expected_smape)


def test_smape_zero_zero_convention_and_bounds():
    assert M.smape([0, 0], [0, 0]) == 0.0                                            # 0/0 := 0
    assert M.smape([0.0], [5.0]) == pytest.approx(200.0)                              # maximum value
    assert 0 <= M.smape([1, 2, 3], [3, 2, 1]) <= 200


def test_wape_undefined_when_all_actuals_zero():
    assert np.isnan(M.wape([0, 0], [1, 1]))


def test_all_metrics_ignores_nan_pairs():
    r = M.all_metrics([1.0, np.nan, 3.0], [1.0, 5.0, 4.0])
    assert r["n"] == 2 and r["MAE"] == pytest.approx(0.5)


def test_dm_equal_accuracy_not_significant_and_better_model_detected():
    rng = np.random.default_rng(0)
    n = 400
    e1 = rng.normal(0, 1, n)
    e2 = rng.normal(0, 1, n)
    same = M.dm_test(np.abs(e1), np.abs(e2), h=1)
    assert same["p_value"] > 0.05
    better = M.dm_test(np.abs(e1) * 0.5, np.abs(e2), h=1)                              # model A has half the error
    assert better["p_value"] < 0.001 and better["dm_stat"] < 0 and better["mean_d"] < 0


def test_dm_handles_overlapping_errors_with_larger_h():
    rng = np.random.default_rng(1)
    n = 300
    base = rng.normal(0, 1, n + 6)
    e = np.convolve(base, np.ones(7) / 7, mode="valid")[:n]                            # MA(6) errors as produced by h = 7 windows
    out = M.dm_test(np.abs(e), np.abs(e) * 1.0, h=7)
    assert np.isnan(out["dm_stat"])                                                    # identical losses -> degenerate, handled
    out2 = M.dm_test(np.abs(e) * 0.8, np.abs(e), h=7)
    assert out2["mean_d"] < 0 and np.isfinite(out2["p_value"])


@pytest.mark.parametrize("h", [1, 7, 14, 28])
def test_dm_statistic_matches_statsmodels_hac_with_uniform_kernel(h):
    """Independent cross-check of the Diebold-Mariano implementation: the statistic equals the HAC t-value of the mean loss differential
    (statsmodels, rectangular kernel, lag h-1, no small-sample correction); the Harvey-Leybourne-Newbold factor and the t(n-1) p-value follow."""
    import statsmodels.api as sm
    from scipy import stats
    rng = np.random.default_rng(0)
    e = rng.normal(size=400 + h)
    d = np.convolve(e, np.ones(h) / np.sqrt(h), mode="valid")[:400] + 0.05          # MA(h-1) differential with a small positive mean
    n = len(d)
    res = sm.OLS(d, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": h - 1, "kernel": "uniform", "use_correction": False})
    assert M.dm_test(d, np.zeros(n), h, hln=False)["dm_stat"] == pytest.approx(float(res.tvalues[0]), rel=1e-9)
    out = M.dm_test(d, np.zeros(n), h, hln=True)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    assert out["dm_stat"] == pytest.approx(float(res.tvalues[0]) * hln, rel=1e-9)
    assert out["p_value"] == pytest.approx(2 * stats.t.sf(abs(out["dm_stat"]), df=n - 1), rel=1e-9)
