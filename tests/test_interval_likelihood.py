"""Ledger L14: interval (integer-cycle) AFT likelihood, INT-1..3.

INT-1 reference (repair 1, 2026-10-17): event terms are compared with a 200-digit mpmath evaluation of
log[S(y-1) - S(y)]; the original float64 scipy reference log(sf(y-1) - sf(y)) loses up to 7 digits to cancellation
when both survival values are near 1 (verified against mpmath: scipy rel. error 1.6e-7, implementation <= 1e-13).
Censored terms are still compared with scipy logsf. Tolerances unchanged."""
import mpmath as mp
import numpy as np
import pytest
import torch
from scipy import stats

from kanfae.survival.aft import aft_interval_log_likelihood, aft_log_likelihood

pytestmark = [pytest.mark.stage("A3"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)
MP_S = {"weibull": lambda w: mp.e ** (-mp.e ** w), "lognormal": lambda w: mp.ncdf(-w),
        "loglogistic": lambda w: 1 / (1 + mp.e ** w)}
REF = {"weibull": lambda e, s: stats.weibull_min(c=1 / s, scale=np.exp(e)),
       "lognormal": lambda e, s: stats.lognorm(s=s, scale=np.exp(e)),
       "loglogistic": lambda e, s: stats.fisk(c=1 / s, scale=np.exp(e))}


def _mp_interval(y, eta, sigma, family):
    with mp.workdps(200):
        S = lambda t: mp.mpf(1) if t <= 0 else MP_S[family]((mp.log(t) - eta) / sigma)  # noqa: E731
        return mp.log(S(y - 1) - S(y))


@pytest.mark.criterion("INT-1")
@pytest.mark.parametrize("family", list(REF))
def test_INT1_matches_scipy_interval_probabilities(family):
    y = torch.tensor([1.0, 2.0, 5.0, 37.0, 120.0, 300.0])
    for eta in (2.0, 4.0, 5.5):
        for sigma in (0.3, 1.0):
            ll = aft_interval_log_likelihood(y, torch.ones(6), torch.full((6,), eta), sigma, family).numpy()
            dist = REF[family](eta, sigma)
            ref = np.array([float(_mp_interval(v, eta, sigma, family)) for v in y.tolist()])
            np.testing.assert_allclose(ll, ref, rtol=1e-9, atol=1e-12)
            cen = aft_interval_log_likelihood(y, torch.zeros(6), torch.full((6,), eta), sigma, family).numpy()
            np.testing.assert_allclose(cen, dist.logsf(y.numpy()), rtol=1e-10, atol=1e-12)


@pytest.mark.criterion("INT-2")
def test_INT2_approaches_density_for_narrow_intervals():
    y = torch.tensor([50.0, 80.0, 130.0])
    eta, sigma = torch.full((3,), 4.5), 0.5
    w = 1e-4
    ll = aft_interval_log_likelihood(y, torch.ones(3), eta, sigma, "weibull", width=w)
    dens = aft_log_likelihood(y, torch.ones(3), eta, sigma, "weibull")
    torch.testing.assert_close(ll - np.log(w), dens, rtol=0, atol=1e-4)


@pytest.mark.criterion("INT-3")
def test_INT3_gradients_and_stability():
    y = torch.tensor([1.0, 3.0, 200.0, 5000.0])
    eta = torch.tensor([0.5, 2.0, 5.0, 3.0], requires_grad=True)
    s = torch.tensor(0.7, requires_grad=True)
    assert torch.autograd.gradcheck(lambda e, ss: aft_interval_log_likelihood(y, torch.tensor([1., 1., 0., 1.]), e, ss),
                                    (eta, s), eps=1e-6, atol=1e-6)
    ll = aft_interval_log_likelihood(y, torch.ones(4), eta, s)
    assert torch.isfinite(ll).all()
