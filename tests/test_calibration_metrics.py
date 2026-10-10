import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk import calibration as cal
from credit_risk import metrics as m


def _miscalibrated(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    true_p = rng.beta(1, 12, n)
    y = (rng.random(n) < true_p).astype(int)
    over = np.clip(true_p * 3.0, 0, 0.99)          # model that is 3x too pessimistic, but ranks perfectly
    return over, y, true_p


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_reduces_error_and_keeps_ranking(method):
    s, y, _ = _miscalibrated()
    before = m.expected_calibration_error(y, s)
    p = cal.cross_fit(s, y, method)
    assert m.expected_calibration_error(y, p) < before / 2
    assert abs(roc_auc_score(y, p) - roc_auc_score(y, s)) < 0.01   # ties broken, ranking preserved


def test_selection_prefers_a_calibrator_over_none_when_miscalibrated():
    s, y, _ = _miscalibrated()
    best, table, _ = cal.select_method(s, y)
    assert best in ("sigmoid", "isotonic") and table.loc[best, "brier"] < table.loc["none", "brier"]


def test_none_is_identity():
    s = np.array([0.1, 0.5])
    np.testing.assert_array_equal(cal.ScoreCalibrator("none").fit(s, [0, 1]).predict(s), s)


def test_shift_base_rate():
    p = np.array([0.02, 0.1, 0.4])
    np.testing.assert_allclose(cal.shift_base_rate(p, 0.07, 0.07), p, atol=1e-9)
    assert np.all(cal.shift_base_rate(p, 0.07, 0.02) < p) and np.all(cal.shift_base_rate(p, 0.07, 0.15) > p)


def test_ece_zero_for_perfectly_calibrated_and_slope_near_one():
    _, y, true_p = _miscalibrated(40000, seed=2)
    assert m.expected_calibration_error(y, true_p) < 0.01
    a, b = m.calibration_slope_intercept(y, true_p)
    assert abs(a) < 0.15 and abs(b - 1) < 0.1


def test_summary_and_bootstrap_ci():
    s, y, _ = _miscalibrated()
    out = m.summarize(y, s)
    assert out["gini"] == pytest.approx(2 * out["auc_roc"] - 1)
    assert 0 <= out["ks"] <= 1
    lo, hi = m.stratified_bootstrap_ci(y, s, n_boot=200)
    assert lo < out["auc_roc"] < hi
