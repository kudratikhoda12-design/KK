"""Stage B: synthetic generator criteria G1..G12 (docs/experiment_protocol.md section 7.3)."""
import dataclasses
import itertools

import numpy as np
import pytest
import torch
from scipy import stats

from kanfae.data.schema import ORACLE_FIELDS, Observed, Oracle, make_observed_outcomes, to_training_arrays
from kanfae.data.synthetic import SynthConfig, generate
from kanfae.survival.aft import aft_nll, get_family

pytestmark = pytest.mark.stage("B")
COMBOS = list(itertools.product(["V1", "V2"], ["dense", "sparse"], [0.0, 0.3, 0.6]))


@pytest.mark.criterion("G1")
@pytest.mark.kind("DET")
@pytest.mark.parametrize("risk,sampling,cens", COMBOS)
def test_G1_observed_time_is_min(risk, sampling, cens):
    obs, ora = generate(SynthConfig(n=2000, risk=risk, sampling=sampling, censoring=cens, seed=1))
    assert np.array_equal(obs.y_obs, np.minimum(ora.T_true, ora.C))
    assert np.all(obs.y_obs > 0) and np.all(np.isfinite(obs.y_obs))


@pytest.mark.criterion("G2")
@pytest.mark.kind("DET")
@pytest.mark.parametrize("risk,sampling,cens", COMBOS)
def test_G2_event_indicator(risk, sampling, cens):
    obs, ora = generate(SynthConfig(n=2000, risk=risk, sampling=sampling, censoring=cens, seed=2))
    assert np.array_equal(obs.delta, ora.T_true <= ora.C)
    if cens == 0:
        assert obs.delta.all()


@pytest.mark.criterion("G3")
@pytest.mark.kind("STAT")
@pytest.mark.parametrize("target", [0.0, 0.3, 0.6])
def test_G3_censoring_fraction(target):
    obs, _ = generate(SynthConfig(n=10000, censoring=target, seed=11))
    frac = 1 - obs.delta.mean()
    print(f"target {target}: realised {frac:.4f}")
    if target == 0:
        assert frac == 0
    else:
        assert abs(frac - target) <= 0.03


@pytest.mark.criterion("G4")
@pytest.mark.kind("DET")
def test_G4_reproducibility():
    a_obs, a_ora = generate(SynthConfig(n=500, seed=5))
    b_obs, b_ora = generate(SynthConfig(n=500, seed=5))
    c_obs, _ = generate(SynthConfig(n=500, seed=6))
    for f in ("times", "values", "mask", "n_obs", "y_obs", "delta"):
        assert np.array_equal(getattr(a_obs, f), getattr(b_obs, f)), f
    for f in ("T_true", "C", "theta", "eta_true"):
        assert np.array_equal(getattr(a_ora, f), getattr(b_ora, f)), f
    assert not np.array_equal(a_obs.y_obs, c_obs.y_obs)
    assert not np.array_equal(a_obs.values, c_obs.values)


@pytest.mark.criterion("G5")
@pytest.mark.kind("DET")
def test_G5_oracle_isolation():
    obs, ora = generate(SynthConfig(n=100, seed=7))
    names = {f.name for f in dataclasses.fields(Observed)}
    assert not (names & set(ORACLE_FIELDS)), names & set(ORACLE_FIELDS)
    assert "T_true" not in obs.meta and "theta" not in str(obs.meta.keys())
    arrays = to_training_arrays(obs)
    assert set(arrays) == names
    with pytest.raises(TypeError):
        to_training_arrays(ora)


@pytest.mark.criterion("G6")
@pytest.mark.kind("DET")
def test_G6_hidden_event_times_do_not_affect_observed_or_loss():
    obs, ora = generate(SynthConfig(n=3000, censoring=0.6, seed=8))
    cens = ~obs.delta
    T_mod = ora.T_true.copy()
    rng = np.random.default_rng(0)
    T_mod[cens] = ora.C[cens] + rng.exponential(5.0, cens.sum()) + 1e-6   # keep T* > C
    y2, d2 = make_observed_outcomes(T_mod, ora.C)
    assert np.array_equal(y2, obs.y_obs) and np.array_equal(d2, obs.delta)
    eta = torch.tensor(np.random.default_rng(1).normal(1, 1, 3000))
    l1 = aft_nll(torch.tensor(obs.y_obs), torch.tensor(obs.delta), eta, 0.7)
    l2 = aft_nll(torch.tensor(y2), torch.tensor(d2), eta, 0.7)
    assert torch.equal(l1, l2)


@pytest.mark.criterion("G7")
@pytest.mark.kind("DET")
@pytest.mark.parametrize("sampling,p_miss", [("dense", 0.0), ("sparse", 0.1), ("dense", 0.3)])
def test_G7_time_ordering_padding_shapes(sampling, p_miss):
    cfg = SynthConfig(n=1500, sampling=sampling, p_miss=p_miss, seed=9)
    obs, _ = generate(cfg)
    lo, hi = cfg.L_range
    N, L, D = obs.values.shape
    assert (N, L, D) == (1500, hi, 3) and obs.times.shape == (N, L) and obs.mask.shape == (N, L, D)
    assert obs.n_obs.min() >= lo and obs.n_obs.max() <= hi
    real = np.arange(L)[None, :] < obs.n_obs[:, None]
    assert np.all((obs.times[real] >= 0) & (obs.times[real] <= 1))
    for i in range(N):
        assert np.all(np.diff(obs.times[i, :obs.n_obs[i]]) >= 0)
    assert not obs.mask[~real].any()
    assert np.all(obs.values[~real] == 0) and np.all(obs.times[~real] == 0)
    assert obs.mask.dtype == bool


@pytest.mark.criterion("G8")
@pytest.mark.kind("STAT")
@pytest.mark.parametrize("family", ["weibull", "lognormal", "loglogistic"])
def test_G8_event_noise_distribution(family):
    obs, ora = generate(SynthConfig(n=20000, family=family, censoring=0.0, seed=12, sampling="sparse"))
    eps = (np.log(ora.T_true) - ora.eta_true) / ora.sigma_true
    fam = get_family(family)
    cdf = (lambda w: 1 - np.exp(-np.exp(w))) if family == "weibull" else (
        stats.norm.cdf if family == "lognormal" else stats.logistic.cdf)
    p = stats.kstest(eps, cdf).pvalue
    print(f"{family}: KS p = {p:.4f}")
    assert p > 0.01


@pytest.mark.criterion("G9")
@pytest.mark.kind("STAT")
def test_G9_censoring_independent_of_risk():
    obs, ora = generate(SynthConfig(n=20000, censoring=0.3, seed=13, sampling="sparse"))
    r = np.corrcoef(np.log(ora.E), ora.eta_true)[0, 1]
    print(f"corr(log E, eta) = {r:.4f}")
    assert abs(r) < 0.03


@pytest.mark.criterion("G10")
@pytest.mark.kind("STAT")
def test_G10_oracle_survival_curve_consistent():
    obs, ora = generate(SynthConfig(n=100000, censoring=0.0, seed=14, sampling="sparse"))
    t = np.quantile(ora.T_true, np.linspace(0.05, 0.95, 20))
    S = ora.survival(t).mean(axis=0)
    emp = (ora.T_true[:, None] > t[None, :]).mean(axis=0)
    print("max |mean S_true - empirical| =", np.abs(S - emp).max())
    assert np.all(np.abs(S - emp) <= 0.01)


@pytest.mark.criterion("G11")
@pytest.mark.kind("DET")
def test_G11_common_random_numbers():
    outs = [generate(SynthConfig(n=800, censoring=c, seed=10)) for c in (0.0, 0.3, 0.6)]
    o0, r0 = outs[0]
    for o, r in outs[1:]:
        assert np.array_equal(r.T_true, r0.T_true) and np.array_equal(r.theta, r0.theta)
        for f in ("times", "values", "mask"):
            assert np.array_equal(getattr(o, f), getattr(o0, f)), f


@pytest.mark.criterion("G12")
@pytest.mark.kind("STAT")
def test_G12_missingness_mask():
    obs, _ = generate(SynthConfig(n=10000, p_miss=0.2, seed=15))
    real = np.arange(obs.mask.shape[1])[None, :] < obs.n_obs[:, None]
    real3 = np.repeat(real[:, :, None], 3, axis=2)
    frac = 1 - obs.mask[real3].mean()
    print(f"masked fraction of real entries = {frac:.4f}")
    assert abs(frac - 0.2) <= 0.01                      # STAT part
    assert np.all(obs.values[real3 & ~obs.mask] == 0)   # DET part
