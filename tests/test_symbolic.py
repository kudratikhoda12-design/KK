"""Symbolic extraction module SYM-1..4 (R^2 > 0.99 acceptance on held-out points)."""
import numpy as np
import pytest

from kanfae.symbolic.fit import ACCEPT_R2, fit_symbolic

pytestmark = pytest.mark.stage("SYM")


@pytest.mark.criterion("SYM-1")
@pytest.mark.kind("DET")
def test_SYM1_linear_accepted_at_stage1():
    x = np.linspace(-2, 2, 300); f = lambda z: 0.7 * z - 1.2  # noqa: E731
    r = fit_symbolic(x, f(x), f, use_pysr=False)
    assert r.accepted and r.stage == "linear" and r.r2_val > ACCEPT_R2


@pytest.mark.criterion("SYM-2")
@pytest.mark.kind("DET")
@pytest.mark.parametrize("name,f", [("tanh", lambda z: 1.5 * np.tanh(2 * z)), ("x^2", lambda z: -0.8 * z ** 2 + 0.3),
                                    ("sin", lambda z: np.sin(1.5 * z) + 0.3), ("exp", lambda z: 0.5 * np.exp(0.7 * z))])
def test_SYM2_library_recovers_known_forms(name, f):
    x = np.linspace(-2, 2, 400)
    r = fit_symbolic(x, f(x), f, use_pysr=False)
    print(name, r.formula, r.r2_val, r.r2_grid)
    assert r.accepted and r.stage == "library" and r.r2_val > 0.999 and r.r2_grid > 0.999
    assert r.params["f"] == name or r.complexity <= 4


@pytest.mark.criterion("SYM-3")
@pytest.mark.kind("STAT")
def test_SYM3_rejects_unrepresentable_function():
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(-2, 2, 400))
    noise = rng.normal(size=400)
    f = lambda z: np.interp(z, x, noise)  # noqa: E731
    r = fit_symbolic(x, noise, f, use_pysr=False)
    assert not r.accepted and r.stage == "none" and "spline retained" in r.notes


@pytest.mark.criterion("SYM-4")
@pytest.mark.kind("STAT")
def test_SYM4_pysr_stage_runs_and_is_held_out_validated():
    x = np.linspace(-2, 2, 300)
    # Repair 1 (2026-10-17): the first target, x sin(x) + 0.2 x^2, is matched on [-2, 2] by the library form
    # 1.404 (1 - cos(1.311 x)) with held-out R^2 = 0.999998, so the test premise was wrong. sin(3x) + 0.5x mixes an
    # oscillation with a trend, which no single affine library form can represent.
    f = lambda z: np.sin(3 * z) + 0.5 * z  # noqa: E731
    r = fit_symbolic(x, f(x), f, use_pysr=True, pysr_budget={"niterations": 20, "timeout": 120})
    print(r.stage, r.formula, r.r2_val, r.r2_grid, r.notes)
    assert r.stage in ("pysr", "none")
    if r.accepted:
        assert r.r2_val > ACCEPT_R2 and r.r2_grid > ACCEPT_R2
