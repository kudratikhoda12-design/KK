"""Revision 4 (D57/D61): frequency-aware functional front "fa" (docs/design.md section 3.2c), criteria FQ-1..FQ-9.

Independent references use dense composite Simpson quadrature and ordinary least squares in numpy; the module
uses precomputed Gauss-Legendre quadratic forms."""
import numpy as np
import pytest
import torch
from scipy import integrate

from kanfae.data.preprocess import Preprocessor
from kanfae.data.synthetic import SynthConfig, generate
from kanfae.functional.basis import BSplineBasis
from kanfae.functional.layers import FrequencyAwareFunctional, TensorSplineFunctional
from kanfae.functional.smoothing import smooth

pytestmark = [pytest.mark.stage("C4"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)
TT = np.linspace(0, 1, 40001)


def ref_spectrum(x, freqs, t=TT):
    """Independent reference on a dense grid: R(f) via least squares with the Simpson L2 metric."""
    w = np.full(len(t), (t[1] - t[0]) / 3.0)
    w[1:-1:2] *= 4
    w[2:-1:2] *= 2
    def pe(B):
        G = B.T @ (B * w[:, None]); b = B.T @ (w * x)
        return b @ np.linalg.solve(G, b)
    base = np.column_stack([np.ones_like(t), t])
    e0, tot = pe(base), (w * x * x).sum()
    R = np.array([(pe(np.column_stack([base, np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)])) - e0)
                  / (tot - e0 + 1e-8) for f in freqs])
    return R, np.log(1e-8 + tot - e0)


def project(fun, basis):
    """L2 projection of a function onto the B-spline basis (dense Gauss-Legendre)."""
    tq, wq = basis._gauss_legendre(64)
    Phi = basis(tq)
    return np.linalg.solve((Phi * wq[:, None]).T @ Phi, (Phi * wq[:, None]).T @ fun(tq))


@pytest.mark.criterion("FQ-1")
def test_FQ1_quadratic_forms_match_dense_reference():
    b = BSplineBasis(12)
    m = FrequencyAwareFunctional(2, 8, b, BSplineBasis(8), seed=0, fourier=True)
    c = np.random.default_rng(1).normal(size=(3, 2, 12))
    with torch.no_grad():
        R, en, fp = m.spectrum(torch.tensor(c))
    for i in range(3):
        for d in range(2):
            x = b(TT) @ c[i, d]
            Rr, e0 = ref_spectrum(x, m.freqs_np)
            assert np.abs(R[i, d, 0].numpy() - Rr).max() < 1e-8
            assert abs(en[i, d, 0].item() - e0) < 1e-8
            dx = b(TT, deriv=1) @ c[i, d]
            assert abs(en[i, d, 1].item() - np.log(1e-8 + integrate.simpson(dx ** 2, x=TT))) < 1e-8
            s = np.array([integrate.simpson(x * np.sin(2 * np.pi * f * TT), x=TT) for f in m.freqs_np])
            co = np.array([integrate.simpson(x * np.cos(2 * np.pi * f * TT), x=TT) for f in m.freqs_np])
            assert np.abs(fp[i, d].numpy() - np.log(1e-8 + s ** 2 + co ** 2)).max() < 1e-7


@pytest.mark.criterion("FQ-2")
def test_FQ2_phase_invariance():
    freqs = np.geomspace(0.25, 4, 33)
    f0 = freqs[16]                                                   # exactly on the grid (= 1.0)
    # (a) the definition (independent reference): R(f0) = 1 for every phase, to rounding
    for ph in np.linspace(0, 2 * np.pi, 9):
        R, _ = ref_spectrum(0.3 + 0.7 * TT + 1.3 * np.sin(2 * np.pi * f0 * TT + ph), freqs)
        assert abs(R[16] - 1) < 1e-7          # repair 1: R = res / (res + eps) with eps = 1e-8 by definition
    # (b) the module on a fine basis (K = 60): R across phases varies by < 1e-4 at every grid frequency
    b = BSplineBasis(60)
    m = FrequencyAwareFunctional(1, 4, b, BSplineBasis(8), seed=0)
    C = np.stack([project(lambda t, ph=ph: 0.3 + 0.7 * t + 1.3 * np.sin(2 * np.pi * f0 * t + ph), b)
                  for ph in np.linspace(0, 2 * np.pi, 13)])
    with torch.no_grad():
        R = m.spectrum(torch.tensor(C[:, None, :]))[0][:, 0, 0].numpy()
    # repair 2: invariance holds AT the true frequency (B_f0 contains every phase); off the peak R depends on the
    # phase through leakage, which the original assertion wrongly required to vanish. Reported, not asserted.
    print("R(f0) over phases: min", R[:, 16].min(), "| off-peak phase variation (leakage):", np.ptp(R, axis=0).max())
    assert R[:, 16].min() > 1 - 1e-4 and np.ptp(R[:, 16]) < 1e-4 and (R.argmax(1) == 16).all()


@pytest.mark.criterion("FQ-3")
@pytest.mark.parametrize("j", [8, 16, 24])
def test_FQ3_peak_at_true_frequency(j):
    b = BSplineBasis(60)
    m = FrequencyAwareFunctional(1, 4, b, BSplineBasis(8), seed=0)
    f0 = m.freqs_np[j]
    C = np.stack([project(lambda t, ph=ph: np.sin(2 * np.pi * f0 * t + ph), b) for ph in np.linspace(0, 6, 7)])
    with torch.no_grad():
        R = m.spectrum(torch.tensor(C[:, None, :]))[0][:, 0, 0].numpy()
        fh = m.soft_frequency(m.spectrum(torch.tensor(C[:, None, :]))[0]).numpy()[:, 0]
    assert (R.argmax(1) == j).all()
    print(f"f0={f0:.3f}: soft-frequency estimates {np.round(fh, 3)}")
    assert np.abs(np.log(fh / f0)).max() < np.log(1.25)              # within 25% (softmax over a log grid)


@pytest.mark.criterion("FQ-4")
def test_FQ4_scale_and_trend_invariance():
    b = BSplineBasis(12)
    m = FrequencyAwareFunctional(1, 4, b, BSplineBasis(8), seed=0, windows=True)
    rng = np.random.default_rng(4)
    c = rng.normal(size=(4, 1, 12))
    ones = np.ones(12)
    greville = np.array([b.knots[k + 1:k + 4].mean() for k in range(12)])      # t = sum_k greville_k phi_k
    assert np.abs(b(TT) @ greville - TT).max() < 1e-12 and np.abs(b(TT) @ ones - 1).max() < 1e-12
    c2 = 3.7 * c + 2.0 * ones - 1.5 * greville
    with torch.no_grad():
        R1, e1, _ = m.spectrum(torch.tensor(c))
        R2, e2, _ = m.spectrum(torch.tensor(c2))
    # repair 2: the only permitted difference is the eps = 1e-8 in the denominator, bounded by 2 eps / residual
    # (the residual energy within a half window can be ~0.02, so a fixed 1e-7 was too tight).
    den = torch.einsum("bdk,wkl,bdl->bdw", torch.tensor(c), m.N - m.M0, torch.tensor(c)).unsqueeze(-1)
    assert ((R1 - R2).abs() <= 2 * m.EPS / den + 1e-9).all()
    # repair 1 of this assertion (first reached in attempt 3): log(eps + s^2 r) - log(eps + r) = 2 log s up to
    # eps / r (definition), not to 1e-9.
    assert ((e2[..., 0] - e1[..., 0] - 2 * np.log(3.7)).abs() <= 2 * m.EPS / den[..., 0, 0] + 1e-12).all()
    assert (R1 >= -1e-12).all() and (R1 <= 1 + 1e-12).all()


@pytest.mark.criterion("FQ-5")
def test_FQ5_gradcheck():
    m = FrequencyAwareFunctional(2, 3, BSplineBasis(12), BSplineBasis(8), seed=1, n_freq=9, fourier=True,
                                 windows=True)
    m.eval()                                                        # fixed (identity) standardisation
    c = torch.randn(4, 2, 12, requires_grad=True)
    assert torch.autograd.gradcheck(lambda cc: m(cc), (c,), eps=1e-6, atol=1e-6)
    R = m.spectrum(c.detach())[0]
    rho = m.rho.detach().clone().requires_grad_(True)      # repair 1: pass rho explicitly (keeps the graph)
    assert torch.autograd.gradcheck(lambda r: m.soft_frequency(R, rho=r), (rho,), eps=1e-6, atol=1e-6)


@pytest.mark.criterion("FQ-6")
def test_FQ6_reduces_to_A3_when_spectral_weights_are_zero():
    b, bb = BSplineBasis(12), BSplineBasis(8)
    fa, a3 = FrequencyAwareFunctional(3, 16, b, bb, seed=5), TensorSplineFunctional(3, 16, b, bb, seed=5)
    with torch.no_grad():
        fa.spec.weight.zero_(); fa.spec.bias.zero_()
    c = torch.randn(6, 3, 12)
    torch.testing.assert_close(fa(c), a3(c), rtol=0, atol=0)


@pytest.mark.criterion("FQ-7")
def test_FQ7_spectral_cache_is_identity_checked():
    m = FrequencyAwareFunctional(2, 4, BSplineBasis(12), BSplineBasis(8), seed=0, n_freq=9)
    ref = FrequencyAwareFunctional(2, 4, BSplineBasis(12), BSplineBasis(8), seed=0, n_freq=9)
    for _ in range(10):
        c1 = torch.randn(64, 2, 12)
        m.spectrum(c1)
        del c1                                                       # its address may be reused below
        c2 = torch.randn(64, 2, 12)
        torch.testing.assert_close(m.spectrum(c2)[0], ref.spectrum(c2.clone())[0], rtol=0, atol=0)
        c2.mul_(2.0)                                                 # in place: the version changes
        torch.testing.assert_close(m.spectrum(c2)[0], ref.spectrum(c2.clone())[0], rtol=0, atol=0)


@pytest.mark.criterion("FQ-8")
def test_FQ8_discretisation_consistency():
    b = BSplineBasis(12)
    m = FrequencyAwareFunctional(3, 8, b, BSplineBasis(8), seed=0)
    m.eval()

    def feats(n):
        t = ((np.arange(n) + 0.5) / n)[None, :]
        y = np.stack([np.sin(2 * np.pi * 1.3 * t + 0.4) + 0.5 * t, np.cos(3 * t), np.exp(-t)], axis=-1)
        c = smooth(t, y, np.ones(y.shape, bool), b, lam=1e-4)
        with torch.no_grad():
            return m.spectral_vector(torch.tensor(c)).numpy()[0]
    ns = [25, 50, 100, 200, 400, 800]
    res = [feats(n) for n in ns]
    ch = [np.linalg.norm(res[i + 1] - res[i]) / np.linalg.norm(res[-1]) for i in range(len(ns) - 1)]
    print("relative changes", np.array2string(np.array(ch), precision=4))
    assert ch[-1] < 1e-2 and ch[-1] <= ch[0]


@pytest.mark.criterion("FQ-9")
@pytest.mark.kind("STAT")
def test_FQ9_frequency_information_on_synthetic_curves():
    """Pre-specified (D61): on synthetic data seed 7 (used nowhere else), the untrained soft-frequency estimate of
    channel x2 from the model's smoothed curves correlates with omega at >= 0.9, and the phase explains < 5% of
    its error variance."""
    obs, ora = generate(SynthConfig(n=600, censoring=0.0, sigma_y=0.05, seed=7))
    pre = Preprocessor(n_basis=12).fit(obs)
    c = torch.tensor(pre.transform(obs).c)
    m = FrequencyAwareFunctional(3, 8, pre.basis, BSplineBasis(8), seed=0)
    with torch.no_grad():
        fh = m.soft_frequency(m.spectrum(c)[0])[:, 1].numpy()
    om, ph = ora.theta[:, 3], ora.theta[:, 4]
    r = np.corrcoef(fh, om)[0, 1]
    e = fh - np.polyval(np.polyfit(om, fh, 1), om)
    Z = np.column_stack([np.ones_like(ph), np.cos(ph), np.sin(ph), np.cos(2 * ph), np.sin(2 * ph)])
    share = 1 - ((e - Z @ np.linalg.lstsq(Z, e, rcond=None)[0]) ** 2).sum() / ((e - e.mean()) ** 2).sum()
    print(f"corr(f_hat, omega) = {r:.3f}; phase share of error = {share:.3f}")
    assert r >= 0.9 and share < 0.05
