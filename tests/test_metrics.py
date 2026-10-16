"""Stage E (run before any metric is reported): evaluation criteria Q1..Q5 (protocol section 7.7)."""
import numpy as np
import pytest
from lifelines.utils import concordance_index
from sksurv.metrics import brier_score, integrated_brier_score
from sksurv.util import Surv

from kanfae.data.synthetic import SynthConfig, generate
from kanfae.evaluation.metrics import (brier_ipcw, d_calibration, harrell_c, ibs_marginal, marginal_censoring_G,
                                       survival_curve_error)

pytestmark = pytest.mark.stage("E")


def _fixture(seed=0, n=300):
    obs, ora = generate(SynthConfig(n=n, censoring=0.4, seed=seed, sampling="sparse"))
    half = n // 2
    tr, te = np.arange(half), np.arange(half, n)
    times = np.quantile(obs.y_obs[te], np.linspace(0.1, 0.8, 25))
    # Repair 1 (2026-10-16): the censoring KM is only defined below the largest training time (sksurv raised
    # ValueError for seed 0 in attempt 1); cap the fixture grid accordingly.
    times = times[times < obs.y_obs[tr].max()]
    # Repair 2 (2026-10-16): sksurv also evaluates the training censoring KM at each test Y_i, so the reference
    # comparison uses test individuals with Y_i below the largest training time (3 of 150 removed for seed 0).
    te = te[obs.y_obs[te] < obs.y_obs[tr].max()]
    S = ora.subset(te).survival(times) ** 0.8          # an imperfect but valid prediction
    return obs, tr, te, times, S


@pytest.mark.criterion("Q1")
@pytest.mark.kind("DET")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_Q1_brier_ibs_and_harrell_match_references(seed):
    obs, tr, te, times, S = _fixture(seed)
    s_tr = Surv.from_arrays(obs.delta[tr], obs.y_obs[tr])
    s_te = Surv.from_arrays(obs.delta[te], obs.y_obs[te])
    _, bs_ref = brier_score(s_tr, s_te, S, times)
    ibs_ref = integrated_brier_score(s_tr, s_te, S, times)
    ibs, bs = ibs_marginal(S, times, obs.y_obs[te], obs.delta[te], obs.y_obs[tr], obs.delta[tr])
    np.testing.assert_allclose(bs, bs_ref, atol=1e-8, rtol=0)
    assert abs(ibs - ibs_ref) < 1e-8
    risk = np.random.default_rng(seed).normal(size=len(te)) - 0.5 * np.log(obs.y_obs[te])
    c_ref = concordance_index(obs.y_obs[te], -risk, obs.delta[te])
    assert abs(harrell_c(obs.y_obs[te], obs.delta[te], risk) - c_ref) < 1e-12


@pytest.mark.criterion("Q2")
@pytest.mark.kind("DET")
def test_Q2_oracle_concordance_is_one():
    rng = np.random.default_rng(0)
    y = rng.exponential(1, 500)
    assert harrell_c(y, np.ones(500, bool), -y) == 1.0


@pytest.mark.criterion("Q2")
@pytest.mark.kind("STAT")
def test_Q2_random_score_concordance_near_half():
    rng = np.random.default_rng(31)
    y = rng.exponential(1, 5000)
    c = harrell_c(y, rng.random(5000) < 0.7, rng.normal(size=5000))
    print("random C =", c)
    assert abs(c - 0.5) < 0.03


@pytest.mark.criterion("Q3")
@pytest.mark.kind("DET")
def test_Q3_conditional_ipcw_with_marginal_G_equals_marginal():
    obs, tr, te, times, S = _fixture(5)
    G = marginal_censoring_G(obs.y_obs[tr], obs.delta[tr])
    yte, dte = obs.y_obs[te], obs.delta[te]
    bs_m = brier_ipcw(S, times, yte, dte, G(yte), G(times))
    G_t_per_ind = np.tile(G(times)[None, :], (len(te), 1))
    bs_c = brier_ipcw(S, times, yte, dte, G(yte), G_t_per_ind)
    np.testing.assert_allclose(bs_c, bs_m, atol=1e-12, rtol=0)


@pytest.mark.criterion("Q4")
@pytest.mark.kind("STAT")
def test_Q4_d_calibration_detects_miscalibration():
    import torch
    from kanfae.survival.aft import get_family
    obs, ora = generate(SynthConfig(n=5000, censoring=0.3, seed=32, sampling="sparse"))
    fam = get_family(ora.family)

    def S_at(y, eta, sigma):          # elementwise S(y_i | eta_i, sigma)
        return torch.exp(fam.log_S(torch.tensor((np.log(y) - eta) / sigma))).numpy()

    S_true_y = S_at(obs.y_obs, ora.eta_true, ora.sigma_true)
    _, p_true, _ = d_calibration(S_true_y, obs.delta)
    S_bad = S_at(obs.y_obs, ora.eta_true, 2 * ora.sigma_true)
    _, p_bad, _ = d_calibration(S_bad, obs.delta)
    print(f"D-cal p (true) = {p_true:.4f}; p (sigma doubled) = {p_bad:.3g}")
    assert p_true > 0.01 and p_bad < 0.01


@pytest.mark.criterion("Q5")
@pytest.mark.kind("DET")
def test_Q5_survival_curve_error():
    grid = np.linspace(0, 2, 2001)
    S1 = np.exp(-grid)[None, :]
    assert survival_curve_error(S1, S1, grid) == 0.0
    S2 = np.exp(-2 * grid)[None, :]
    # (1/2) int_0^2 (e^-t - e^-2t) dt = (1/2) [ (1 - e^-2) - (1 - e^-4)/2 ]
    exact = 0.5 * ((1 - np.exp(-2)) - (1 - np.exp(-4)) / 2)
    assert abs(survival_curve_error(S2, S1, grid) - exact) < 1e-6
