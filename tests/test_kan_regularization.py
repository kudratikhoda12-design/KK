"""KAN regularisation, pruning and symbolic substitution (CoxKAN Eqs. 5-8; KAN-AFT §3.2.2): KREG-1..4."""
import numpy as np
import pytest
import torch

from kanfae.models.kan import KANBlock, KANLayer

pytestmark = [pytest.mark.stage("C2"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)


@pytest.mark.criterion("KREG-1")
def test_KREG1_regularization_matches_definition_and_has_gradients():
    layer = KANLayer(3, 2, 5, 3, seed=0)
    x = torch.randn(64, 3)
    R, info = layer.regularization(x, lam_ent=2.0, lam_coef=1.0)
    phi = torch.stack([torch.stack([layer.w_base[i, o] * torch.nn.functional.silu(x[:, i])
                                    + layer.w_spline[i, o] * (layer.basis(x)[:, i, :] @ layer.coef[i, o])
                                    for o in range(2)], -1) for i in range(3)], 1)
    l1 = phi.abs().mean(0)
    p = l1 / l1.sum()
    ent = -(p * torch.log(p)).sum()
    coef = layer.coef.abs().mean(-1).sum()
    assert abs(float(R) - float(l1.sum() + 2.0 * ent + coef)) < 1e-10
    R.backward()
    assert all(torch.isfinite(q.grad).all() for q in (layer.coef, layer.w_base, layer.w_spline))


@pytest.mark.criterion("KREG-2")
def test_KREG2_edge_sum_equals_forward():
    layer = KANLayer(4, 3, 5, 3, seed=1)
    x = torch.randn(20, 4)
    torch.testing.assert_close(layer.edge_activations(x).sum(1) + layer.bias, layer(x), rtol=0, atol=1e-12)


@pytest.mark.criterion("KREG-3")
def test_KREG3_edge_and_node_pruning():
    blk = KANBlock([3, 4, 2], seed=2)
    with torch.no_grad():
        l0, l1 = blk.layers
        l0.coef[0, 1] = 0.0; l0.w_base[0, 1] = 0.0                    # dead input edge 0 -> hidden 1
        l1.coef[2, :] = 0.0; l1.w_base[2, :] = 0.0                    # hidden node 2 has no outgoing signal
    x = torch.randn(200, 3)
    blk.eval()
    blk.norm.train(); blk(x); blk.eval()                               # populate BatchNorm statistics
    before = blk(x)
    info = blk.prune(x, theta=1e-2)
    after = blk(x)
    assert l0.mask[0, 1] == 0 and bool((l0.mask[:, 2] == 0).all()) and bool((l1.mask[2, :] == 0).all())
    assert info["edges_pruned"] >= 1 and info["nodes_pruned"] >= 1
    torch.testing.assert_close(before, after, rtol=0, atol=1e-10)     # pruned parts carried no signal


@pytest.mark.criterion("KREG-4")
def test_KREG4_exact_symbolic_substitution_leaves_output_unchanged():
    layer = KANLayer(2, 2, 5, 3, seed=3)
    x = torch.randn(50, 2)
    ref = layer(x).detach()
    w_b, w_s, c = layer.w_base[1, 0].detach(), layer.w_spline[1, 0].detach(), layer.coef[1, 0].detach()
    layer.symbolic[(1, 0)] = lambda xi: w_b * torch.nn.functional.silu(xi) + w_s * (
        layer.basis(torch.stack([xi, xi], 1))[:, 1, :] @ c)
    torch.testing.assert_close(layer(x).detach(), ref, rtol=0, atol=1e-12)
