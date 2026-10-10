import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crm import metrics  # noqa: E402


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(1)
    n = 20000
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-2.5 + 0.8 * x)))
    y = rng.binomial(1, p)
    return y, p, x


def test_delong_auc_matches_sklearn(sim):
    y, p, _ = sim
    a, _ = metrics.delong_cov(y, [p])
    assert a[0] == pytest.approx(roc_auc_score(y, p), abs=1e-10)


def test_delong_variance_matches_bootstrap(sim):
    y, p, _ = sim
    _, c = metrics.delong_cov(y, [p])
    rng = np.random.default_rng(0)
    boots = [roc_auc_score(y[i], p[i]) for i in
             (rng.integers(0, len(y), len(y)) for _ in range(300))]
    assert np.sqrt(c[0, 0]) == pytest.approx(np.std(boots), rel=0.15)


def test_delong_test_size_under_null(sim):
    """Two noisy copies of the same score: H0 true, p-values ~ U(0,1)."""
    y, _, x = sim
    rng = np.random.default_rng(3)
    pv = []
    for _ in range(60):
        s1 = x + rng.normal(scale=1.0, size=len(x))
        s2 = x + rng.normal(scale=1.0, size=len(x))
        pv.append(metrics.delong_test(y, s1, s2)["p_value"])
    assert 0.0 <= np.mean(np.array(pv) < 0.05) <= 0.15


def test_ks_matches_scipy(sim):
    y, p, _ = sim
    ks_ref = stats.ks_2samp(p[y == 1], p[y == 0]).statistic
    assert metrics.ks_stat(y, p) == pytest.approx(ks_ref, abs=1e-6)


def test_calibration_perfect_model(sim):
    y, p, _ = sim
    c = metrics.calibration_slope_intercept(y, p)
    assert c["citl_lo"] < 0 < c["citl_hi"]
    assert c["slope_lo"] < 1 < c["slope_hi"]
    assert metrics.spiegelhalter(y, p)["spiegelhalter_p"] > 0.01
    assert metrics.hosmer_lemeshow(y, p)["hl_p"] > 0.01


def test_calibration_detects_miscalibration(sim):
    y, p, _ = sim
    c = metrics.calibration_slope_intercept(y, p * 0.6)
    assert c["citl"] > 0.3 and c["citl_p"] < 1e-6
    assert metrics.hosmer_lemeshow(y, p * 0.6)["hl_p"] < 1e-6


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(size=50000)
    assert metrics.psi(a, rng.normal(size=50000)) < 0.01
    assert metrics.psi(a, rng.normal(0.5, 1, size=50000)) > 0.2


def test_jeffreys_flags_underestimation():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": 1, "y": rng.binomial(1, 0.10, 5000), "p": 0.05})
    out = metrics.binomial_backtest(df, "g")
    assert out["jeffreys_p"].iloc[0] < 0.01
    assert out["traffic_light"].iloc[0].startswith("red")


def test_holm():
    p = pd.Series([0.01, 0.04, 0.03], index=list("abc"))
    adj = metrics.holm(p)
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)
