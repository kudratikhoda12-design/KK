"""D42 estimator validation: LS-1..LS-5 (BJ, IPCW, transform), independent of predictive comparisons."""
import numpy as np
import pytest
from lifelines import KaplanMeierFitter

from kanfae.survival.censored_ls import (LinearLS, bj_impute_logscale, censoring_G, fit_censored_ls, ipcw_weights,
                                         km_survival, step_eval, transform_targets, _int_inv_G)

pytestmark = pytest.mark.stage("A2")


def linear_aft(n, seed, beta=(1.0, 0.8, -0.5), sigma=0.5, cens_scale=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    T = np.exp(beta[0] + X @ np.array(beta[1:]) + sigma * rng.normal(size=n))
    if cens_scale is None:
        return X, T, np.ones(n, bool)
    C = rng.exponential(cens_scale, n)                    # independent of (T, X): CA-2 holds
    return X, np.minimum(T, C), T <= C


@pytest.mark.criterion("LS-1")
@pytest.mark.kind("DET")
def test_LS1_no_censoring_reduces_to_ols():
    X, y, d = linear_aft(500, 1)
    G = censoring_G(y, d)
    assert np.allclose(ipcw_weights(y, d, G), 1.0)
    Ts, _ = transform_targets(y, d, G)
    np.testing.assert_allclose(Ts, y, rtol=1e-12)
    ols = LinearLS(X); ols.fit(np.log(y), np.ones(500))
    for method in ("ipcw", "transform", "bj"):
        m = LinearLS(X)
        fit_censored_ls(m.fit, m.predict, y, d, method, G=G)
        np.testing.assert_allclose(m.beta, ols.beta, atol=1e-8)


@pytest.mark.criterion("LS-2")
@pytest.mark.kind("DET")
def test_LS2_km_matches_lifelines_and_bj_conditional_mean():
    rng = np.random.default_rng(2)
    t, e = rng.exponential(1, 60), rng.random(60) < 0.7
    knots, S = km_survival(t, e)
    km = KaplanMeierFitter().fit(t, e)
    np.testing.assert_allclose(S, km.survival_function_at_times(knots).to_numpy(), atol=1e-12)
    # BJ conditional mean vs brute force from the KM distribution (largest residual treated as event)
    eta = np.zeros(60)
    logy = np.log(t)
    out = bj_impute_logscale(logy, e, eta)
    d2 = e.copy(); d2[np.argmax(logy)] = True
    k2, S2 = km_survival(logy, d2)
    p = -np.diff(np.r_[1.0, S2])
    for i in np.nonzero(~d2)[0]:
        sel = k2 > logy[i]
        brute = (k2[sel] * p[sel]).sum() / p[sel].sum()
        assert abs(out[i] - brute) < 1e-10
    assert np.array_equal(out[d2], logy[d2])


@pytest.mark.criterion("LS-3")
@pytest.mark.kind("DET")
def test_LS3_transform_integral_and_nonnegativity():
    X, y, d = linear_aft(800, 3, cens_scale=6.0)
    G = censoring_G(y, d)
    tt = np.linspace(0, y.max(), 50)
    fine = np.linspace(0, y.max(), 400001)
    g = step_eval(G[0], G[1], fine)
    ref = np.array([np.trapezoid(1 / g[fine <= x], fine[fine <= x]) if x > 0 else 0.0 for x in tt])
    np.testing.assert_allclose(_int_inv_G(G[0], G[1], tt), ref, rtol=2e-3, atol=2e-3)
    Ts, alpha = transform_targets(y, d, G)
    assert np.all(Ts >= -1e-12)


@pytest.mark.criterion("LS-4")
@pytest.mark.kind("STAT")
def test_LS4_slope_recovery_bj_ipcw():
    X, y, d = linear_aft(5000, 60, cens_scale=8.0)
    print("censored fraction", round(1 - d.mean(), 3))
    G = censoring_G(y, d)
    res = {}
    for method in ("bj", "ipcw", "transform"):
        m = LinearLS(X)
        info = fit_censored_ls(m.fit, m.predict, y, d, method, G=G)
        res[method] = m.beta
        print(method, np.round(m.beta, 3), info)
    for method in ("bj", "ipcw"):
        assert np.all(np.abs(res[method][1:] - np.array([0.8, -0.5])) < 0.1), method


@pytest.mark.criterion("LS-5")
@pytest.mark.kind("STAT")
def test_LS5_transform_unbiased_on_time_scale():
    rng = np.random.default_rng(61)
    T = rng.exponential(2.0, 200000)
    C = rng.exponential(4.0, 200000)
    y, d = np.minimum(T, C), T <= C
    Ts, alpha = transform_targets(y, d, censoring_G(y, d))
    print("mean T* =", Ts.mean(), "mean T =", T.mean(), "alpha =", alpha)
    assert abs(Ts.mean() - T.mean()) / T.mean() < 0.05
