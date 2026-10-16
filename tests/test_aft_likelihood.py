"""Stage A: AFT likelihood criteria A1..A8 (docs/experiment_protocol.md section 7.2).

Independent references: scipy.stats closed forms (A1, A3), scipy numerical integration and
autograd differentiation (A2), lifelines maximum-likelihood fits (A7).
"""
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import integrate, stats

from kanfae.survival.aft import (FAMILIES, AFTFamily, aft_log_likelihood, aft_median, aft_survival,
                                 get_family)
from kanfae.survival.linear_aft import fit_linear_aft

pytestmark = pytest.mark.stage("A")
torch.set_default_dtype(torch.float64)

ETAS = [-3.0, -1.5, 0.0, 1.5, 3.0]
SIGMAS = [0.2, 0.5, 1.0, 3.0]
EPS_Q = np.array([1e-6, 1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1 - 1e-4, 1 - 1e-6])


def scipy_dist(family, eta, sigma):
    if family == "weibull":
        return stats.weibull_min(c=1.0 / sigma, scale=np.exp(eta))
    if family == "lognormal":
        return stats.lognorm(s=sigma, scale=np.exp(eta))
    if family == "loglogistic":
        return stats.fisk(c=1.0 / sigma, scale=np.exp(eta))
    raise ValueError(family)


def times_at_eps_quantiles(family, eta, sigma):
    fam = get_family(family)
    return np.exp(eta + sigma * np.array([fam.quantile(q) for q in EPS_Q]))


@pytest.mark.criterion("A1")
@pytest.mark.kind("DET")
@pytest.mark.parametrize("family", FAMILIES)
def test_A1_terms_match_scipy(family):
    for eta in ETAS:
        for sigma in SIGMAS:
            y = times_at_eps_quantiles(family, eta, sigma)
            ref = scipy_dist(family, eta, sigma)
            yt = torch.tensor(y)
            ev = aft_log_likelihood(yt, torch.ones_like(yt), torch.full_like(yt, eta), sigma, family).numpy()
            ce = aft_log_likelihood(yt, torch.zeros_like(yt), torch.full_like(yt, eta), sigma, family).numpy()
            np.testing.assert_allclose(ev, ref.logpdf(y), rtol=1e-10, atol=1e-10,
                                       err_msg=f"logpdf {family} eta={eta} sigma={sigma}")
            np.testing.assert_allclose(ce, ref.logsf(y), rtol=1e-10, atol=1e-10,
                                       err_msg=f"logsf {family} eta={eta} sigma={sigma}")


@pytest.mark.criterion("A2")
@pytest.mark.kind("DET")
@pytest.mark.parametrize("family", FAMILIES)
def test_A2_numerical_integration_and_hazard(family):
    for eta in [-1.0, 0.0, 2.0]:
        for sigma in [0.2, 1.0, 3.0]:
            def dens_logt(u):  # density of log T at u = f_T(e^u) e^u, from our event term
                y = torch.tensor([np.exp(u)])
                return float(torch.exp(aft_log_likelihood(y, torch.ones(1), torch.tensor([eta]), sigma, family))) * np.exp(u)

            # Repair 2 (2026-10-16): adaptive quadrature over a wide u-range missed the narrow peak (sigma=0.2)
            # in attempts 1-2. Substitute u = eta + sigma*w and integrate dens_logt(u)*sigma over w in [-60, 60]
            # with breakpoints; this is still numerical integration of OUR density (mass outside < 1e-26).
            def dens_w(wv):
                return dens_logt(eta + sigma * wv) * sigma
            brk = list(np.arange(-10.0, 10.5, 1.0))
            total, _ = integrate.quad(dens_w, -60.0, 60.0, points=brk, epsabs=1e-14, epsrel=1e-12, limit=2000)
            assert abs(total - 1.0) < 1e-8, (family, eta, sigma, total)
            for q in [0.05, 0.5, 0.95]:
                y = float(np.exp(eta + sigma * get_family(family).quantile(q)))
                wy = (np.log(y) - eta) / sigma
                cdf, _ = integrate.quad(dens_w, -60.0, wy, points=[b for b in brk if b < wy],
                                        epsabs=1e-14, epsrel=1e-12, limit=2000)
                S = float(aft_survival(torch.tensor([y]), torch.tensor([eta]), sigma, family)[0, 0])
                assert abs(cdf - (1.0 - S)) < 1e-8, (family, eta, sigma, q, cdf, 1 - S)
                # hazard: d/dt[-log S(t)] by autograd vs exp(log f - log S)
                t = torch.tensor([y], requires_grad=True)
                logS = aft_log_likelihood(t, torch.zeros(1), torch.tensor([eta]), sigma, family)
                (g,) = torch.autograd.grad(-logS.sum(), t)
                logf = aft_log_likelihood(torch.tensor([y]), torch.ones(1), torch.tensor([eta]), sigma, family)
                h_ref = torch.exp(logf - logS.detach())
                assert abs(float(g) - float(h_ref)) <= 1e-8 * abs(float(h_ref)), (family, eta, sigma, q)


@pytest.mark.criterion("A3")
@pytest.mark.kind("DET")
@pytest.mark.parametrize("family", FAMILIES)
def test_A3_mixed_sample_and_censored_terms(family):
    rng = np.random.default_rng(3)
    n = 400
    eta = rng.normal(0.5, 1.0, n)
    sigma = 0.7
    y = np.exp(eta + sigma * get_family(family).sample(n, rng))
    delta = rng.random(n) < 0.65
    ll = aft_log_likelihood(torch.tensor(y), torch.tensor(delta), torch.tensor(eta), sigma, family).numpy()
    ref = np.where(delta, scipy_dist(family, eta, sigma).logpdf(y), scipy_dist(family, eta, sigma).logsf(y))
    assert abs(ll.sum() - ref.sum()) < 1e-9
    np.testing.assert_allclose(ll[~delta], scipy_dist(family, eta[~delta], sigma).logsf(y[~delta]),
                               rtol=1e-10, atol=1e-10)

    base = get_family(family)

    class Perturbed(AFTFamily):  # same survival function, density shifted by +1000
        name = "perturbed"

        def log_f(self, w):
            return base.log_f(w) + 1000.0

        def log_S(self, w):
            return base.log_S(w)

    ll_p = aft_log_likelihood(torch.tensor(y), torch.tensor(delta), torch.tensor(eta), sigma, Perturbed()).numpy()
    assert np.array_equal(ll_p[~delta], ll[~delta])          # censored terms do not use the density
    np.testing.assert_allclose(ll_p[delta] - ll[delta], 1000.0, rtol=0, atol=1e-9)


@pytest.mark.criterion("A4")
@pytest.mark.kind("DET")
@pytest.mark.parametrize("family", FAMILIES)
def test_A4_gradcheck(family):
    rng = np.random.default_rng(4)
    y = torch.tensor(np.exp(rng.normal(0, 1, 12)))
    delta = torch.tensor(rng.random(12) < 0.5)
    eta = torch.tensor(rng.normal(0, 1, 12), requires_grad=True)
    sigma = torch.tensor(0.8, requires_grad=True)
    assert torch.autograd.gradcheck(lambda e, s: aft_log_likelihood(y, delta, e, s, family), (eta, sigma),
                                    eps=1e-6, atol=1e-7, rtol=1e-5)


@pytest.mark.criterion("A5")
@pytest.mark.kind("DET")
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_A5_stability_extreme_w(family, dtype):
    w = torch.linspace(-50, 50, 201, dtype=dtype)
    for d in [0.0, 1.0]:
        eta = torch.zeros(201, dtype=dtype, requires_grad=True)
        sigma = torch.tensor(1.0, dtype=dtype, requires_grad=True)
        y = torch.exp(w)  # eta = 0, sigma = 1  =>  w = log y
        ll = aft_log_likelihood(y, torch.full((201,), d), eta, sigma, family)
        assert torch.isfinite(ll).all()
        ge, gs = torch.autograd.grad(ll.sum(), (eta, sigma))
        assert torch.isfinite(ge).all() and torch.isfinite(gs).all()


@pytest.mark.criterion("A6")
@pytest.mark.kind("DET")
@pytest.mark.parametrize("family", FAMILIES)
def test_A6_survival_function_valid(family):
    fam = get_family(family)
    for eta in ETAS:
        for sigma in SIGMAS:
            lo, hi = np.exp(eta + sigma * fam.quantile(1e-7)), np.exp(eta + sigma * fam.quantile(1 - 1e-7))
            t = torch.tensor(np.geomspace(lo, hi, 1000))
            S = aft_survival(t, torch.tensor([eta]), sigma, family)[0]
            assert (S >= 0).all() and (S <= 1).all()
            assert (torch.diff(S) <= 0).all()
            assert S[0] > 1 - 1e-6 and S[-1] < 1e-6
            med = float(aft_median(torch.tensor(eta), sigma, family))
            ref = np.exp(eta + sigma * fam.quantile(0.5))
            assert abs(med - ref) <= 1e-10 * ref
            assert abs(float(aft_survival(torch.tensor([med]), torch.tensor([eta]), sigma, family)[0, 0]) - 0.5) < 1e-10


def simulate_linear(family, n, seed, beta=(1.0, 0.5, -0.7), sigma=0.6, cens_rate=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    eta = beta[0] + X @ np.array(beta[1:])
    T = np.exp(eta + sigma * get_family(family).sample(n, rng))
    # independent exponential censoring with rate giving roughly 30% censoring
    rate = cens_rate if cens_rate is not None else 0.35 / np.median(T)
    C = rng.exponential(1 / rate, n)
    return X, np.minimum(T, C), (T <= C)


LIFELINES = {"weibull": ("WeibullAFTFitter", "lambda_", "rho_", -1.0),
             "lognormal": ("LogNormalAFTFitter", "mu_", "sigma_", 1.0),
             "loglogistic": ("LogLogisticAFTFitter", "alpha_", "beta_", -1.0)}


@pytest.mark.criterion("A7")
@pytest.mark.kind("DET")
@pytest.mark.parametrize("family", FAMILIES)
def test_A7_matches_lifelines(family):
    import lifelines
    X, y, d = simulate_linear(family, 3000, seed=7)
    print(f"{family}: censored fraction {1 - d.mean():.3f}")
    ours = fit_linear_aft(X, y, d, family, compute_se=False)
    assert ours.converged
    cls, loc_name, shape_name, sign = LIFELINES[family]
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "T": y, "E": d.astype(int)})
    # Repair 1 (2026-10-16): attempt 1 showed lifelines' default optimiser (SLSQP, ftol 1e-6) stopping ~8e-4 short of the maximum
    # for the log-logistic model (our log-likelihood was higher). Tighten the reference optimiser's gradient
    # tolerance; the 1e-4 criterion is unchanged.
    # Repair 2 (2026-10-16): lifelines uses SLSQP (Weibull, log-logistic; default ftol up to 1e-6, which ignores
    # gtol) or BFGS (log-normal; gtol=1e-10 raised a ConvergenceError in attempt 2). Tighten ftol for SLSQP only.
    fitter = getattr(lifelines, cls)()
    opts = {"ftol": 1e-12, "maxiter": 1000} if fitter._scipy_fit_method == "SLSQP" else {}
    ll_fit = fitter.fit(df, "T", "E", fit_options=opts)
    # Diagnostic: our likelihood evaluated at lifelines' own estimate must reproduce lifelines' value.
    from kanfae.survival.aft import aft_log_likelihood as _ll
    p0 = ll_fit.params_
    eta0 = p0[(loc_name, "Intercept")] + X @ np.array([p0[(loc_name, "x1")], p0[(loc_name, "x2")]])
    sig0 = np.exp(sign * p0[(shape_name, "Intercept")])
    ours_at_ref = float(_ll(torch.tensor(y), torch.tensor(d), torch.tensor(eta0), sig0, family).sum())
    print(f"{family}: our loglik at lifelines params {ours_at_ref:.8f}")
    assert abs(ours_at_ref - ll_fit.log_likelihood_) < 1e-8
    print(f"{family}: ours {ours.loglik:.8f}  lifelines {ll_fit.log_likelihood_:.8f}")
    assert abs(ours.loglik - ll_fit.log_likelihood_) < 1e-4
    p = ll_fit.params_
    mapped = np.array([p[(loc_name, "Intercept")], p[(loc_name, "x1")], p[(loc_name, "x2")],
                       sign * p[(shape_name, "Intercept")]])
    np.testing.assert_allclose(np.r_[ours.beta, ours.log_sigma], mapped, atol=1e-3)


@pytest.mark.criterion("A8")
@pytest.mark.kind("STAT")
@pytest.mark.parametrize("family", FAMILIES)
def test_A8_recovers_truth_within_3se(family):
    X, y, d = simulate_linear(family, 5000, seed=2026)
    fit = fit_linear_aft(X, y, d, family)
    est = np.r_[fit.beta, fit.log_sigma]
    truth = np.array([1.0, 0.5, -0.7, np.log(0.6)])
    z = (est - truth) / fit.se
    print(f"{family}: z-scores {np.round(z, 3)}")
    assert np.all(np.abs(z) < 3.0)
