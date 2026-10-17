"""Revision 3 (D38/D40): nonlinear functional layer criteria F6a..F6c."""
import numpy as np
import pytest
import torch

from kanfae.data.synthetic import SynthConfig, generate
from kanfae.functional.basis import BSplineBasis
from kanfae.functional.layers import FunctionalLinear, NonlinearFunctional
from kanfae.functional.smoothing import smooth

pytestmark = [pytest.mark.stage("C"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)


@pytest.mark.criterion("F6a")
def test_F6a_identity_map_equals_functional_linear():
    b, bb = BSplineBasis(12), BSplineBasis(8)
    lin = FunctionalLinear(3, 7, b, bb, seed=3)
    nfl = NonlinearFunctional(3, 7, b, bb, seed=3, identity_only=True)
    with torch.no_grad():
        nfl.weight.copy_(lin.weight.unsqueeze(2))
        nfl.bias.copy_(torch.randn(7))
        lin.bias.copy_(nfl.bias)
    c = torch.tensor(np.random.default_rng(0).normal(size=(5, 3, 12)))
    with torch.no_grad():
        diff = (nfl(c) - lin(c)).abs().max().item()
    print("max |NFL(identity) - FunctionalLinear| =", diff)
    assert diff < 1e-10


@pytest.mark.criterion("F6b")
def test_F6b_discretisation_consistency():
    b, bb = BSplineBasis(12), BSplineBasis(8)
    nfl = NonlinearFunctional(3, 16, b, bb, seed=0)

    def feats(t):
        t = t[None, :]
        y = np.stack([np.sin(2 * np.pi * t) + 0.5 * t, np.cos(3 * t), np.exp(-t)], axis=-1)
        c = smooth(t, y, np.ones(y.shape, bool), b, lam=1e-4)
        with torch.no_grad():
            return nfl(torch.tensor(c)).numpy()[0]

    ns = [25, 50, 100, 200, 400, 800]
    res = [feats((np.arange(n) + 0.5) / n) for n in ns]
    ch = [np.linalg.norm(res[i + 1] - res[i]) / np.linalg.norm(res[-1]) for i in range(len(ns) - 1)]
    print("NFL relative changes", np.array2string(np.array(ch), precision=3))
    assert all(ch[i + 1] <= ch[i] for i in range(len(ch) - 1)) and ch[-1] < 1e-2


@pytest.mark.criterion("F6c")
def test_F6c_shapes_and_gradcheck():
    b, bb = BSplineBasis(12), BSplineBasis(8)
    nfl = NonlinearFunctional(2, 4, b, bb, n_maps=3, width=5, seed=1)
    c = torch.randn(6, 2, 12, requires_grad=True)
    assert nfl(c).shape == (6, 4)
    assert nfl.feature_maps(c).shape[:2] == (6, 2) and nfl.feature_maps(c).shape[-1] == 3
    assert torch.autograd.gradcheck(nfl, (c,), eps=1e-6, atol=1e-6)
    assert torch.isfinite(nfl.roughness_penalty())
    obs, _ = generate(SynthConfig(n=50, seed=2))
    cc = smooth(obs.times, obs.values, obs.mask, b, lam=1e-4)
    out = NonlinearFunctional(3, 32, b, bb, seed=0)(torch.tensor(cc))
    assert out.shape == (50, 32) and torch.isfinite(out).all()


@pytest.mark.criterion("F7")
def test_F7_energy_front_exact():
    from scipy import integrate
    from kanfae.functional.layers import EnergyFunctional
    b, bb = BSplineBasis(12), BSplineBasis(8)
    e = EnergyFunctional(3, 5, b, bb, seed=0)
    c = torch.tensor(np.random.default_rng(1).normal(size=(2, 3, 12)))
    x = np.linspace(0, 1, 9 * 5 * 400 + 1)
    for i in range(2):
        X = np.einsum("dk,lk->ld", c[i].numpy(), b(x))
        ref = integrate.simpson(bb(x)[:, :, None] * X[:, None, :] ** 2, x=x, axis=0).T
        assert np.abs(e.energies(c[i:i + 1])[0].detach().numpy() - ref).max() < 1e-10
    with torch.no_grad():
        e.q.zero_()
    torch.testing.assert_close(e(c), e.lin(c), rtol=0, atol=0)


@pytest.mark.criterion("F8")
def test_F8_tensor_spline_front():
    from kanfae.functional.layers import TensorSplineFunctional
    b, bb = BSplineBasis(12), BSplineBasis(8)
    t = TensorSplineFunctional(2, 4, b, bb, seed=0)
    c = torch.randn(5, 2, 12, requires_grad=True)
    assert t(c).shape == (5, 4)
    assert torch.autograd.gradcheck(lambda cc: t(cc), (c,), eps=1e-6, atol=1e-6)
    with torch.no_grad():
        t.w.zero_()
    torch.testing.assert_close(t(c), t.lin(c), rtol=0, atol=0)
