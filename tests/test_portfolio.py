import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crm import portfolio as pf  # noqa: E402


def test_vasicek_density_integrates_to_one():
    f = lambda x: np.exp(pf.vasicek_logpdf(x, 0.05, 0.1))
    val, _ = integrate.quad(f, 1e-9, 1 - 1e-9, limit=200)
    assert val == pytest.approx(1.0, abs=1e-4)


def test_cond_pd_averages_to_unconditional():
    z = np.random.default_rng(0).standard_normal(400_000)
    assert pf.cond_pd(0.06, 0.08, z).mean() == pytest.approx(0.06, rel=0.01)


def test_rho_mle_ci_coverage():
    """95% Wald CI should cover the true rho in ~95% of replications."""
    rng = np.random.default_rng(1)
    cover = []
    for _ in range(200):
        dr = pf.cond_pd(0.05, 0.06, rng.standard_normal(150))
        fit = pf.fit_rho_mle(dr, np.full(150, 0.05))
        cover.append(fit["lo"] < 0.06 < fit["hi"])
    assert 0.90 <= np.mean(cover) <= 0.99


def test_mc_expected_loss_and_asrf_quantile():
    rng = np.random.default_rng(2)
    n = 20_000
    pd_ = rng.uniform(0.01, 0.15, n)
    lgd = np.full(n, 0.9)
    ead = rng.uniform(5e3, 3e4, n)
    L = pf.simulate_losses(pd_, lgd, ead, 0.05, n_scen=4000, seed=3)
    el = np.sum(pd_ * lgd * ead)
    assert abs(L.mean() - el) < 3 * L.std() / np.sqrt(len(L))
    # Large granular portfolio: MC 99% quantile ~ ASRF analytic quantile.
    assert np.quantile(L, 0.99) == pytest.approx(
        pf.asrf_quantile_loss(pd_, lgd, ead, 0.05, 0.99), rel=0.05)


def test_var_es_ordering():
    L = np.random.default_rng(0).lognormal(size=10_000)
    r = pf.var_es(L, 0.99)
    assert r["VaR_lo"] <= r["VaR"] <= r["VaR_hi"] and r["ES"] >= r["VaR"]
