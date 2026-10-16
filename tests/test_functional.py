"""Stage C: functional representation criteria F1..F5 (docs/experiment_protocol.md section 7.4)."""
import numpy as np
import pytest
import torch
from scipy import integrate
from scipy.interpolate import BSpline

from kanfae.data.synthetic import SynthConfig, generate
from kanfae.functional.basis import BSplineBasis
from kanfae.functional.fpca import FPCA
from kanfae.functional.layers import FunctionalLinear
from kanfae.functional.smoothing import evaluate, quadrature_weights, smooth

pytestmark = [pytest.mark.stage("C"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)


def scipy_basis(basis, x, deriv=0):
    out = np.zeros((len(x), basis.K))
    for j in range(basis.K):
        c = np.zeros(basis.K); c[j] = 1.0
        out[:, j] = BSpline(basis.knots, c, basis.degree, extrapolate=True)(x, nu=deriv)
    return out


@pytest.mark.criterion("F1")
@pytest.mark.parametrize("K", [4, 8, 12, 16])
def test_F1_basis_matches_scipy_and_partition_of_unity(K):
    b = BSplineBasis(K)
    x = np.r_[np.linspace(0, 1, 1001), np.random.default_rng(0).uniform(0, 1, 500)]
    np.testing.assert_allclose(b(x), scipy_basis(b, x), atol=1e-12, rtol=0)
    np.testing.assert_allclose(b(x).sum(axis=1), 1.0, atol=1e-12, rtol=0)
    xi = x[(x > 1e-9) & (x < 1 - 1e-9)]
    for d in (1, 2):
        ref = scipy_basis(b, xi, d)
        np.testing.assert_allclose(b(xi, deriv=d), ref, atol=1e-9 * max(1.0, np.abs(ref).max()), rtol=0)


@pytest.mark.criterion("F1b")
@pytest.mark.parametrize("K", [8, 12])
def test_F1b_roughness_and_gram_vs_simpson(K):
    b = BSplineBasis(K)
    n_int = (K - 3) * 2000                     # knots fall on panel boundaries
    x = np.linspace(0, 1, n_int + 1)
    B, D2 = scipy_basis(b, x), scipy_basis(b, x, 2)
    G_ref = integrate.simpson(B[:, :, None] * B[:, None, :], x=x, axis=0)
    R_ref = integrate.simpson(D2[:, :, None] * D2[:, None, :], x=x, axis=0)
    assert np.abs(b.gram() - G_ref).max() <= 1e-6 * np.abs(G_ref).max()
    assert np.abs(b.roughness() - R_ref).max() <= 1e-6 * np.abs(R_ref).max()


@pytest.mark.criterion("F2")
def test_F2_smoother_exact_recovery_and_sparse_stability():
    b = BSplineBasis(12)
    rng = np.random.default_rng(1)
    c_true = rng.normal(size=(5, 2, 12))
    t = np.sort(rng.uniform(0, 1, (5, 50)), axis=1)
    y = np.einsum("ndk,nlk->nld", c_true, b(t))
    mask = np.ones((5, 50, 2), bool)
    c = smooth(t, y, mask, b, lam=0.0, lam0=0.0)
    np.testing.assert_allclose(c, c_true, atol=1e-6)
    # sparse: fewer points than basis functions, single points, and empty channels
    obs, _ = generate(SynthConfig(n=400, sampling="sparse", p_miss=0.3, seed=3))
    m = obs.mask.copy(); m[0] = False; m[1, :, 0] = False; m[2] = False; m[2, 0, :] = True
    c = smooth(obs.times, np.where(m, obs.values, 0), m, b, lam=1e-4)
    assert np.all(np.isfinite(c))
    assert np.all(c[0] == 0) and np.all(c[1, 0] == 0)
    q = quadrature_weights(obs.times, m)
    has = m.any(axis=1)
    np.testing.assert_allclose(q.sum(axis=1)[has], 1.0, atol=1e-12)


def _test_function(t):
    return np.stack([np.sin(2 * np.pi * t) + 0.5 * t, np.cos(3 * t), np.exp(-t)], axis=-1)


@pytest.mark.criterion("F3")
def test_F3_discretisation_consistency():
    b, bb = BSplineBasis(12), BSplineBasis(8)
    obs, _ = generate(SynthConfig(n=600, seed=4))
    c_train = smooth(obs.times, obs.values, obs.mask, b, lam=1e-4)
    fp = FPCA(4).fit(c_train, b.gram())
    layer = FunctionalLinear(3, 16, b, bb, seed=0)

    def feats(t):
        t = t[None, :]
        y = _test_function(t)
        c = smooth(t, y, np.ones(y.shape, bool), b, lam=1e-4)
        with torch.no_grad():
            h = layer(torch.tensor(c)).numpy()[0]
        return h, fp.transform(c)[0]

    ns = [25, 50, 100, 200, 400, 800]
    res = [feats((np.arange(n) + 0.5) / n) for n in ns]
    for k, name in enumerate(["h", "fpca"]):
        ref = np.linalg.norm(res[-1][k])
        ch = [np.linalg.norm(res[i + 1][k] - res[i][k]) / ref for i in range(len(ns) - 1)]
        print(name, "relative changes", np.array2string(np.array(ch), precision=3))
        assert all(ch[i + 1] <= ch[i] for i in range(len(ch) - 1)), (name, ch)
        assert ch[-1] < 1e-2, (name, ch)
    jit = (np.arange(800) + 0.5 + np.random.default_rng(0).uniform(-0.4, 0.4, 800)) / 800
    hj, fj = feats(np.sort(jit))
    assert np.linalg.norm(hj - res[-1][0]) / np.linalg.norm(res[-1][0]) < 1e-2
    assert np.linalg.norm(fj - res[-1][1]) / np.linalg.norm(res[-1][1]) < 1e-2


@pytest.mark.criterion("F4")
def test_F4_fpca_properties():
    b = BSplineBasis(12)
    obs, _ = generate(SynthConfig(n=800, seed=5))
    c = smooth(obs.times, obs.values, obs.mask, b, lam=1e-4)
    fp = FPCA(36).fit(c, b.gram())
    s = fp.transform(c)
    assert np.abs(s.mean(axis=0)).max() < 1e-10
    cov = np.cov(s, rowvar=False, bias=True)
    off = cov - np.diag(np.diag(cov))
    assert np.abs(off).max() < 1e-8 * np.abs(np.diag(cov)).max()
    assert np.all(np.diff(fp.eigvals_) <= 1e-12)
    np.testing.assert_allclose(fp.inverse_transform(s), c, atol=1e-8)


@pytest.mark.criterion("F5")
def test_F5_functional_linear_matches_dense_quadrature():
    b, bb = BSplineBasis(12), BSplineBasis(8)
    rng = np.random.default_rng(6)
    c = rng.normal(size=(4, 3, 12))
    layer = FunctionalLinear(3, 5, b, bb, seed=1)
    with torch.no_grad():
        h = layer(torch.tensor(c)).numpy()
    x = np.linspace(0, 1, 9 * 5 * 400 + 1)   # union of breaks (1/9, 1/5) on panel boundaries
    X = np.einsum("ndk,lk->nld", c, scipy_basis(b, x))
    Bt = np.einsum("hdm,lm->hld", layer.weight.detach().numpy(), scipy_basis(bb, x))
    ref = integrate.simpson(np.einsum("nld,hld->nhl", X, Bt), x=x, axis=-1) + layer.bias.detach().numpy()
    np.testing.assert_allclose(h, ref, atol=1e-8, rtol=0)
