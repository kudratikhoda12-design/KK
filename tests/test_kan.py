"""Stage C: in-house KAN layer criteria F-KAN-1..4 (docs/experiment_protocol.md section 7.4)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.interpolate import BSpline

from kanfae.models.kan import KANBlock, KANLayer

pytestmark = [pytest.mark.stage("C"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)


@pytest.mark.criterion("F-KAN-1")
@pytest.mark.parametrize("G,k", [(3, 3), (5, 3), (10, 3), (5, 2)])
def test_FKAN1_basis_matches_scipy(G, k):
    layer = KANLayer(2, 3, G, k)
    x = torch.linspace(-2, 2 - 1e-9, 2001)
    B = layer.basis(torch.stack([x, x], 1))[:, 0, :].numpy()
    t = layer.knots.numpy()
    for j in range(G + k):
        ref = BSpline.basis_element(t[j:j + k + 2], extrapolate=False)(x.numpy())
        np.testing.assert_allclose(B[:, j], np.nan_to_num(ref), atol=1e-10, rtol=0)


@pytest.mark.criterion("F-KAN-2")
def test_FKAN2_shapes_and_gradcheck():
    layer = KANLayer(4, 3, 5, 3, seed=0)
    x = torch.randn(7, 4, requires_grad=True)
    assert layer(x).shape == (7, 3)
    assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-6)
    blk = KANBlock([6, 5, 2], seed=0)
    assert blk(torch.randn(9, 6)).shape == (9, 2)


@pytest.mark.criterion("F-KAN-3")
def test_FKAN3_outside_grid_spline_is_zero():
    layer = KANLayer(3, 2, 5, 3, seed=1)
    lo, hi = float(layer.knots[0]), float(layer.knots[-1])
    x = torch.tensor([[lo - 0.1, hi + 0.5, -100.0], [hi, lo - 1e-6, 50.0]])
    assert torch.equal(layer.spline_term(x), torch.zeros(2, 2))
    torch.testing.assert_close(layer(x), F.silu(x) @ layer.w_base + layer.bias, rtol=0, atol=0)


@pytest.mark.criterion("F-KAN-4")
@pytest.mark.parametrize("n_in,n_out,G,k", [(16, 8, 5, 3), (4, 12, 3, 3), (8, 4, 10, 2)])
def test_FKAN4_parameter_count(n_in, n_out, G, k):
    layer = KANLayer(n_in, n_out, G, k)
    assert sum(p.numel() for p in layer.parameters()) == KANLayer.n_params(n_in, n_out, G, k) \
        == n_in * n_out * (G + k + 2) + n_out
