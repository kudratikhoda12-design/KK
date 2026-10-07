"""Baselines, exponential smoothing, SARIMA mechanics, features, uncertainty and the leakage-audit machinery."""
import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src import baselines as bl
from src import config as C
from src import features as F
from src import leakage_audit as la
from src import stat_models as sm
from src import uncertainty as un
from src.forecast_core import ForecastBook, truth_sums
from tests.conftest import needs_forecasts, needs_prepared


# ------------------------------------------------------------------ baselines
def test_naive_seasonal_naive_moving_average_values():
    y = np.arange(1.0, 31.0)[None, :]                                                  # 1..30 (strictly increasing)
    t = 20
    assert bl.naive_paths(y)[0, t, 0] == y[0, t]
    sn = bl.seasonal_naive_paths(y)[0, t]
    for k in range(1, 15):
        # forecast for t+k equals the observation 7*ceil(k/7) days before t+k
        assert sn[k - 1] == y[0, t + k - 7 * int(np.ceil(k / 7))]
    ma = bl.moving_average_paths(y, 7)[0, t, 0]
    assert ma == pytest.approx(y[0, t - 6:t + 1].mean())


def test_seasonal_naive_equals_ma7_for_sums_over_full_weeks():
    """Mathematical identity behind identical SeasonalNaive / MA7 scores at H = 7, 14, 28."""
    rng = np.random.default_rng(3)
    Y = rng.poisson(3, (2, 120)).astype(float)
    sn, ma = bl.seasonal_naive_paths(Y), bl.moving_average_paths(Y, 7)
    for H in (7, 14, 28):
        a = sn[:, 10:, :H].sum(axis=2)
        b = ma[:, 10:, :H].sum(axis=2)
        assert np.allclose(a, b)
    assert not np.allclose(sn[:, 10:, :3].sum(axis=2), ma[:, 10:, :3].sum(axis=2))   # but NOT for H = 3


def test_croston_sba_constant_demand_and_zero_series():
    rate_const = bl.croston_sba_rate(np.full(200, 4.0), alpha=0.1)
    assert rate_const[-1] == pytest.approx((1 - 0.05) * 4.0)                           # z = 4, p = 1, SBA factor 0.95
    assert (bl.croston_sba_rate(np.zeros(50)) == 0).all()
    y = np.zeros(200); y[::5] = 2.0                                                    # demand 2 every 5th day
    assert bl.croston_sba_rate(y, alpha=0.1)[-1] == pytest.approx(0.95 * 2.0 / 5.0, rel=0.02)


# ------------------------------------------------------------------ ETS
def test_ets_regression_ses_and_holt_fit_do_not_fail_on_unused_nan_params():
    """Regression test: statsmodels reports unused parameters as NaN; that must not be mistaken for a failed fit."""
    rng = np.random.default_rng(5)
    y = rng.poisson(5, 300).astype(float)
    for kind in ("SES", "Holt", "HW_N", "HW_D"):
        fit = sm._fit_ets(y, kind)
        assert fit is not None, kind
        assert np.isfinite(fit["aicc"])


@pytest.mark.parametrize("kind", ["HW_N", "HW_D"])
def test_ets_state_path_forecast_matches_statsmodels(kind):
    rng = np.random.default_rng(7)
    season = np.tile([5, 6, 7, 6, 8, 12, 10], 60)[:400]
    y = rng.poisson(season).astype(float)
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fit = sm._fit_ets(y[:350], kind)
    assert fit is not None
    paths = sm.ets_rolling_paths(y, fit)                                                 # filter run over the whole series
    ref = ExponentialSmoothing(y[:350], trend="add" if kind == "HW_D" else None, damped_trend=(kind == "HW_D"), seasonal="add",
                               seasonal_periods=7, initialization_method="estimated").fit().forecast(28)
    assert np.allclose(paths[349], ref, atol=1e-6)                                      # forecast at origin 349 == statsmodels forecast


def test_ets_forecasts_are_causal():
    rng = np.random.default_rng(8)
    y = rng.poisson(4, 300).astype(float)
    fit = sm._fit_ets(y[:200], "HW_N")
    y2 = y.copy(); y2[250:] = 99.0
    p1, p2 = sm.ets_rolling_paths(y, fit), sm.ets_rolling_paths(y2, fit)
    assert np.allclose(p1[:250], p2[:250], equal_nan=True)


def test_sarima_extend_equals_refilter():
    rng = np.random.default_rng(9)
    y = rng.poisson(np.tile([5, 6, 7, 6, 8, 12, 10], 40)[:260]).astype(float)
    res, ok = sm._fit_sarima(y[:200], sm.SARIMA_CANDIDATES["S1:(1,0,1)(1,0,1)7+c"])
    assert ok
    paths = sm.roll_sarima(res, y, 199, 230)
    ref = res.apply(y[:216], refit=False).forecast(C.MAX_HORIZON)                       # origin 215 = index 16 in the rolled block
    assert np.allclose(paths[16], ref, atol=1e-8)


def test_audit_causality_checks_on_synthetic_data():
    assert all(ok for _, ok, _ in la.check_sarima_causality())


# ------------------------------------------------------------------ features
def test_feature_definitions_on_synthetic_panel(synth):
    H = 7
    df = F.build_features(synth.Y, synth.price, synth.dates, synth.snap_full, synth.event_full, synth.christmas_full, H)
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = int(rng.integers(synth.n_series)); t = int(rng.integers(F.FIRST_ORIGIN, synth.N))
        r = df[(df.series_idx == s) & (df.origin == t)].iloc[0]
        y = synth.Y[s]
        assert r["lag_1"] == y[t] and r["lag_7"] == y[t - 6] and r["lag_28"] == y[t - 27]            # lag measured from the first forecast day
        assert r["rolling_mean_7"] == pytest.approx(y[t - 6:t + 1].mean())
        assert r["rolling_std_28"] == pytest.approx(y[t - 27:t + 1].std(ddof=1))
        assert r["snap_days_window"] == synth.snap_full[s, t + 1:t + H + 1].sum()
        if t + H <= synth.N - 1:
            assert r["target"] == y[t + 1:t + H + 1].sum()
        else:
            assert np.isnan(r["target"])


def test_trailing_stat_uses_only_past_values():
    y = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    m = F.trailing_stat(y, 3, "mean")
    assert np.isnan(m[:2]).all() and m[2] == 2.0 and m[9] == 9.0                         # mean of y[t-2..t]


def test_features_carry_no_nan_and_no_future_information(synth):
    df = F.build_features(synth.Y, synth.price, synth.dates, synth.snap_full, synth.event_full, synth.christmas_full, 7)
    assert df[F.FEATURES].notna().all().all()
    res = la.check_feature_perturbation(synth, H=7, n_cuts=2)
    assert all(ok for _, ok, _ in res), [r for r in res if not r[1]]


def test_perturbation_test_has_power_it_detects_a_deliberate_leak(synth, monkeypatch):
    """Negative control: a centred rolling window peeks into the future; the audit must flag it."""
    from numpy.lib.stride_tricks import sliding_window_view

    def leaky(x, w, kind):                                                              # centred window t-w//2 .. t+w//2
        x = np.asarray(x, float)
        out = np.full(len(x), np.nan)
        h = w // 2
        win = sliding_window_view(x, w)
        out[h:len(x) - (w - 1 - h)] = win.mean(axis=1) if kind == "mean" else win.std(axis=1, ddof=1)
        return np.where(np.isnan(out), np.nanmean(x), out)

    monkeypatch.setattr(F, "trailing_stat", leaky)
    res = la.check_feature_perturbation(synth, H=7, n_cuts=2)
    assert any(not ok for _, ok, _ in res), "a leaky feature was NOT detected - the audit has no power"


def test_rows_for_fit_never_reach_into_next_period(synth):
    ds = F.dataset_for_horizon(synth, 14)
    sub = F.rows_for_fit(ds, 14, synth.split.train_end)
    assert (sub["origin"] + 14).max() <= synth.split.train_end - 1


def test_known_ahead_features_independent_of_sales(synth):
    base = F.build_features(synth.Y, synth.price, synth.dates, synth.snap_full, synth.event_full, synth.christmas_full, 14)
    Y2 = synth.Y * 5 + 3
    other = F.build_features(Y2, synth.price * 2, synth.dates, synth.snap_full, synth.event_full, synth.christmas_full, 14)
    for f in F.FEATURE_GROUPS["known_ahead"]:
        assert np.array_equal(base[f].to_numpy(), other[f].to_numpy())


# ------------------------------------------------------------------ forecast book + uncertainty
def _book_with_baselines(ctx):
    book = ForecastBook(ctx)
    bl.add_baselines(book)
    return book


def test_forecast_book_alignment_and_eval_arrays(synth):
    book = _book_with_baselines(synth)
    for H in C.HORIZONS:
        t, y, f = book.eval_arrays("MA7", "val", H)
        assert y.shape == f.shape == (synth.n_series, len(synth.split.eval_origins("validation", H)))
        assert np.allclose(y, truth_sums(synth.Y, H)[:, t])
        assert np.isfinite(f).all()


def test_gaussian_interval_coverage_close_to_nominal():
    rng = np.random.default_rng(11)
    resid = rng.normal(0, 3.0, (4, 4000))
    pred = np.full((4, 4000), 20.0)
    actual = pred + rng.normal(0, 3.0, (4, 4000))
    lo, hi = un._interval(pred, resid, "gaussian", 0.90)
    cov = np.mean((actual >= lo) & (actual <= hi))
    assert abs(cov - 0.90) < 0.02
    lo, hi = un._interval(pred, resid, "empirical", 0.80)
    assert abs(np.mean((actual >= lo) & (actual <= hi)) - 0.80) < 0.02


def test_interval_lower_bound_clipped_at_zero():
    lo, hi = un._interval(np.array([[1.0]]), np.random.default_rng(0).normal(0, 10, (1, 500)), "gaussian", 0.9)
    assert lo[0, 0] == 0.0 and hi[0, 0] > 1.0


def test_sigma_uses_validation_phase_only_synthetic(synth):
    book = _book_with_baselines(synth)
    s1 = un.sigma_table(book, ["MA7"], horizons=(7,))
    Y2 = synth.Y.copy(); Y2[:, synth.split.val_end:] = 123.0
    book2 = ForecastBook(replace(synth, Y=Y2), F=copy.deepcopy(book.F))
    book2.F["test"]["MA7"] = book2.F["test"]["MA7"] * 9 + 5
    s2 = un.sigma_table(book2, ["MA7"], horizons=(7,))
    assert np.allclose(s1["sigma_rmse"], s2["sigma_rmse"])


# ------------------------------------------------------------------ real-data audit
@needs_prepared
@needs_forecasts
def test_full_leakage_audit_passes_on_real_data():
    from src.forecast_core import load_context
    ctx = load_context()
    book = ForecastBook(ctx).load()
    df = la.run_audit(ctx, book=book, sigma_df=pd.DataFrame(), tuning_log=None, write=False)
    failed = df[~df["passed"]]
    assert failed.empty, failed.to_string()


# ------------------------------------------------------------------ diagnostics used in the narrative
def test_error_concentration_known_case():
    """95 perfect windows + 5 windows with error 10: the worst 5% produce all of the squared error; RMSE/MAE = sqrt(5)/0.5."""
    from src import residuals as rs

    class _Stub:
        def eval_arrays(self, model, phase, H):
            e = np.zeros((1, 100)); e[0, :5] = 10.0
            y = np.full((1, 100), 50.0)
            return np.arange(100), y, y + e

    out = rs.error_concentration(_Stub(), ["M"], [7])
    assert set(out.phase) == {"val", "test"} and len(out) == 2
    r = out.iloc[0]
    assert r.worst5pct_share_of_squared_error == pytest.approx(1.0)
    assert r.RMSE_over_MAE == pytest.approx(np.sqrt(5.0) / 0.5)


def test_error_concentration_on_baselines(synth):
    from src import residuals as rs
    book = _book_with_baselines(synth)
    out = rs.error_concentration(book, ["MA7", "Naive"], [7, 14])
    assert len(out) == 2 * 2 * 2
    assert (out.RMSE_over_MAE >= 1.0 - 1e-12).all()
    assert out.worst5pct_share_of_squared_error.between(0.05 - 1e-9, 1.0).all()


def test_seasonal_difference_of_white_noise_has_acf_minus_half_at_lag_7():
    """Used in the interview guide: independent noise differenced at lag 7 has autocorrelation -0.5 at lag 7 (over-differencing)."""
    from statsmodels.tsa.stattools import acf
    y = np.random.default_rng(3).normal(size=60000)
    d7 = y[7:] - y[:-7]
    assert acf(d7, nlags=7, fft=True)[7] == pytest.approx(-0.5, abs=0.02)


def test_zero_forecast_shares_known_case():
    from src import residuals as rs

    class _Stub:
        def eval_arrays(self, model, phase, H):
            return np.arange(4), np.array([[0.0, 0.0, 5.0, 5.0]]), np.array([[0.0, 3.0, 0.0, 4.0]])

    r = rs.zero_forecast_shares(_Stub(), ["M"], [7]).iloc[0]
    assert (r.share_zero_forecast, r.share_actual_zero, r.share_both_zero) == (0.5, 0.5, 0.25)
