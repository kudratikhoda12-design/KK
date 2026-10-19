"""Round 9 (D108): KAN-fairness decoders, criteria FC-1..FC-5 (written and run before any fairness fit)."""
import numpy as np
import pytest
import torch

from kanfae.data.splits import split_groups
from kanfae.data.synthetic import SynthConfig, generate
from kanfae.experiments.runner import reconstruction_eval
from kanfae.models.ensemble import SeedEnsembleAFT
from kanfae.models.kan import KANBlock
from kanfae.models.pipelines import NeuralAFT

pytestmark = [pytest.mark.stage("D9"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)
CFG = dict(front="fa", fa_compact=True, fa_nfreq=17, r=16, H=32, hidden=64, fa_balance=True, clip=1.0, head_wd=1e-2,
           lr=1e-3, es_mode="two_checkpoint", min_epochs=0, max_epochs=2, patience=5, head="mlp")


def _fit(kind, **kw):
    obs, _ = generate(SynthConfig(n=80, censoring=0.3, seed=79))
    tr, va, te = split_groups(obs.ids, 79, strata=obs.delta)
    return NeuralAFT(kind, **{**CFG, **kw}).fit(obs.subset(tr), obs.subset(va), seed=0), obs, te


def _np(m):
    return sum(p.numel() for p in m.parameters())


@pytest.mark.criterion("FC-1")
def test_FC1_default_fae_kan_is_the_D31_design():
    m, _, _ = _fit("fae_kan")
    assert _np(m.model.encoder) == 5136 and _np(m.model.decoder) == 1599 and m.model.block_params() == 6735
    assert [(l.n_in, l.n_out) for l in m.model.decoder.layers] == [(16, 3), (3, 36)]       # width-3 bottleneck


@pytest.mark.criterion("FC-2")
def test_FC2_mode_B_shares_the_mlp_decoder_and_keeps_the_kan_encoder():
    base, _, _ = _fit("fae_kan")
    b, _, _ = _fit("fae_kan", dec_block="mlp")
    mlp, _, _ = _fit("fae_mlp")
    assert isinstance(b.model.encoder, KANBlock) and _np(b.model.encoder) == 5136
    assert [p.shape for p in b.model.decoder.parameters()] == [p.shape for p in mlp.model.decoder.parameters()]
    assert _np(b.model.decoder) == _np(mlp.model.decoder) == 3428 and b.model.block_params() == 8564
    fresh_b, fresh_k = b._build(3, 0, 0), base._build(3, 0, 0)          # untrained builds, same seed
    assert all(torch.equal(x, y) for x, y in zip(fresh_b.encoder.state_dict().values(), fresh_k.encoder.state_dict().values()))


@pytest.mark.criterion("FC-3")
def test_FC3_mode_A_single_layer_kan_decoder_has_no_bottleneck():
    a, _, _ = _fit("fae_kan", kan_dec="single")
    assert [(l.n_in, l.n_out) for l in a.model.decoder.layers] == [(16, 36)]
    assert _np(a.model.decoder) == 5796 and a.model.block_params() == 10932


@pytest.mark.criterion("FC-4")
def test_FC4_wide_mlp_is_parameter_matched_to_mode_A():
    w, _, _ = _fit("fae_mlp", hidden=107)
    assert w.model.block_params() == 10966 and abs(10966 - 10932) / 10932 <= 0.10


@pytest.mark.criterion("FC-5")
def test_FC5_reconstruction_eval_is_the_member_mean():
    m0, obs, te = _fit("fae_mlp")
    m1 = NeuralAFT("fae_mlp", **CFG).fit(*[obs.subset(i) for i in split_groups(obs.ids, 79, strata=obs.delta)[:2]], seed=1)
    ens = SeedEnsembleAFT.from_members([m0, m1])
    r = reconstruction_eval(ens, obs.subset(te))
    single = [reconstruction_eval(m, obs.subset(te))["reconstruction_test"] for m in (m0, m1)]
    assert np.isclose(r["reconstruction_test"], np.mean(single)) and r["reconstruction_test_members"] == single
