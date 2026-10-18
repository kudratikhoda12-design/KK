"""Step 3 (D104) tools: KAN-FAE sparsity regularisation and the two pruning variants, criteria ST-1..ST-3."""
import copy

import pytest
import torch

from kanfae.data.splits import split_groups
from kanfae.data.synthetic import SynthConfig, generate
from kanfae.models import fae_kan_tools as KT
from kanfae.models.kan import KANBlock
from kanfae.models.pipelines import NeuralAFT

pytestmark = [pytest.mark.stage("D4f"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)
CFG = dict(front="fa", fa_compact=True, fa_nfreq=17, r=16, fa_balance=True, clip=1.0, head_wd=1e-2, lr=1e-3,
           es_mode="two_checkpoint", min_epochs=5, max_epochs=10, patience=5, head="mlp")


@pytest.mark.criterion("ST-1")
def test_ST1_kan_regularisation_is_the_cached_kan_aft_penalty():
    obs, _ = generate(SynthConfig(n=120, censoring=0.3, seed=79))
    tr, va, _ = split_groups(obs.ids, 79, strata=obs.delta)
    m0 = NeuralAFT("fae_kan", **CFG).fit(obs.subset(tr), obs.subset(va), seed=0)
    m1 = copy.deepcopy(m0)
    m1.cfg["kan_lam"] = 1e-3
    T = m0._tensors(m0.prepare(obs.subset(tr)))
    m0.model.eval(); m1.model.eval()
    with torch.no_grad():
        l0, _ = m0._objective(T, "joint")
        l1, _ = m1._objective(T, "joint")                      # forward caches each KAN layer's inputs
        reg = m1.model.encoder.cached_regularization() + m1.model.decoder.cached_regularization()
        x = m1.model.flin(T["inp"])
        ref = m1.model.encoder.regularization(x) + m1.model.decoder.regularization(m1.model.encoder(x))
    assert float(reg) > 0 and torch.allclose(l1, l0 + 1e-3 * reg) and torch.allclose(reg, ref)


@pytest.mark.criterion("ST-2")
def test_ST2_centred_pruning_absorbs_constant_edges_exactly():
    torch.manual_seed(0)
    blk = KANBlock([4, 3], seed=1).eval()
    x = torch.randn(200, 4)
    with torch.no_grad():
        blk(x)
        lay = blk.layers[0]
        lay.coef[1, 2] = 0.0; lay.w_base[1, 2] = 0.0            # edge (1, 2): phi == 0 ...
        lay.symbolic[(1, 2)] = lambda z: torch.full_like(z, 0.7)  # ... replaced by the constant 0.7
        before = blk(x)
        info = KT.prune_centred(blk, x, theta=1e-2)
        after = blk(x)
    assert lay.mask[1, 2] == 0 and info["edges_pruned"] >= 1
    assert torch.allclose(before, after, atol=1e-12)


@pytest.mark.criterion("ST-3")
def test_ST3_uncentred_pruning_masks_small_edges():
    torch.manual_seed(0)
    blk = KANBlock([4, 3], seed=1).eval()
    x = torch.randn(200, 4)
    with torch.no_grad():
        lay = blk.layers[0]
        lay.coef[0, 0] = 0.0; lay.w_base[0, 0] = 1e-4
        n0 = KT.active_edges(blk)
        info = KT.prune_uncentred(blk, x, theta=1e-2)
    assert lay.mask[0, 0] == 0 and KT.active_edges(blk) == n0 - info["edges_pruned"]
    assert float(lay.edge_activations(blk.layer_inputs(x)[0])[:, 0, 0].abs().max()) == 0.0
