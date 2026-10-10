import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from credit_risk.screening import SQCRatioScreener, benjamini_hochberg, one_sided_two_proportion_p


def status(screener):
    return dict(zip(screener.report_["ratio"], screener.report_["status"]))


def test_keeps_signal_drops_noise_and_degenerate(synthetic):
    X, y = synthetic
    s = SQCRatioScreener().fit(X, y)
    st = status(s)
    assert st["signal_low"] == "kept" or st["signal_low_copy"] == "kept"
    assert st["signal_high"] == "kept"
    assert st["noise"] == "dropped: no SQC signal"
    assert st["constant"] == "dropped: degenerate"


def test_adverse_tail_is_detected(synthetic):
    X, y = synthetic
    r = SQCRatioScreener().fit(X, y).report_.set_index("ratio")
    assert r.loc["signal_low", "adverse_tail"] == "low"
    assert r.loc["signal_high", "adverse_tail"] == "high"


def test_skewed_nonnegative_ratio_is_not_discarded(synthetic):
    """Regression: symmetric mean - 3 sigma limits fall below 0 for skewed ratios and never signal."""
    X, y = synthetic
    r = SQCRatioScreener().fit(X, y).report_.set_index("ratio")
    assert r.loc["skewed_low", "status"] == "kept"
    assert r.loc["skewed_low", "detection_rate"] > 0.5
    assert r.loc["skewed_low", "lcl"] > 0  # probability limit stays inside the feasible range


def test_redundant_column_removed(synthetic):
    X, y = synthetic
    st = status(SQCRatioScreener().fit(X, y))
    assert sorted([st["signal_low"], st["signal_low_copy"]]) == ["dropped: redundant", "kept"]


def test_limits_use_healthy_firms_only(synthetic):
    """Changing the defaulters' values must not move the control limits (no label leakage into the chart)."""
    X, y = synthetic
    a = SQCRatioScreener().fit(X, y)
    X2 = X.copy()
    X2.loc[y == 1] = X2.loc[y == 1] * 10 + 5
    b = SQCRatioScreener().fit(X2, y)
    for attr in ("centre_", "sigma_low_", "sigma_high_", "lcl_", "ucl_"):
        np.testing.assert_allclose(getattr(a, attr), getattr(b, attr))


def test_screen_false_keeps_everything(synthetic):
    X, y = synthetic
    s = SQCRatioScreener(screen=False).fit(X, y)
    assert len(s.kept_) == X.shape[1]


def test_missing_ratio_dropped_and_nan_passthrough(synthetic):
    X, y = synthetic
    X = X.copy()
    X.loc[::2, "noise"] = np.nan
    X.loc[::3, "signal_high"] = np.nan
    s = SQCRatioScreener(max_missing=0.4).fit(X, y)
    assert status(s)["noise"] == "dropped: missing"
    out = s.transform(X)
    assert np.isnan(out).any()           # NaNs are left for the imputer
    assert out.shape == (len(X), len(s.kept_))


@pytest.mark.parametrize("scale", ["none", "clip", "arcsinh"])
def test_transform_shapes_and_names(synthetic, scale):
    X, y = synthetic
    s = SQCRatioScreener(scale=scale, add_alarm_count=True).fit(X, y)
    out = s.transform(X)
    names = s.get_feature_names_out()
    assert out.shape[1] == len(names) == len(s.kept_) + 1
    assert names[-1] == "alarm_share"
    assert 0 <= out[:, -1].min() and out[:, -1].max() <= 1


def test_clip_bounds_zscore(synthetic):
    X, y = synthetic
    s = SQCRatioScreener(scale="clip", clip_k=4.0, screen=False).fit(X, y)
    out = s.transform(X * 1000)  # absurd outliers
    assert np.nanmax(np.abs(out)) <= 4.0 + 1e-9


def test_needs_both_classes(synthetic):
    X, _ = synthetic
    with pytest.raises(ValueError):
        SQCRatioScreener().fit(X, np.zeros(len(X), int))


def test_works_inside_sklearn_pipeline(synthetic):
    X, y = synthetic
    pipe = make_pipeline(SQCRatioScreener(scale="arcsinh"), LogisticRegression(max_iter=500))
    clone(pipe).fit(X, y)
    assert pipe.fit(X, y).predict_proba(X).shape == (len(X), 2)


def test_benjamini_hochberg_known_values():
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.005]))
    np.testing.assert_allclose(q, [0.02, 0.04, 0.04, 0.02])
    assert np.all(q >= np.array([0.01, 0.04, 0.03, 0.005]))


def test_two_proportion_test_direction():
    p_more = one_sided_two_proportion_p(np.array([40.0]), np.array([100.0]), np.array([5.0]), np.array([100.0]))
    p_less = one_sided_two_proportion_p(np.array([5.0]), np.array([100.0]), np.array([40.0]), np.array([100.0]))
    assert p_more[0] < 1e-6 and p_less[0] > 0.99
