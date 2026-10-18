"""Revision 4f (D100): nonlinear head refit on the frozen 4e latent and multi-seed risk averaging, criteria NF-1..NF-4."""
import numpy as np
import pytest
import torch

from kanfae.data.splits import split_groups
from kanfae.data.synthetic import SynthConfig, generate
from kanfae.models.ensemble import SeedEnsembleAFT
from kanfae.models.pipelines import NeuralAFT

pytestmark = [pytest.mark.stage("D4f"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)
CFG = dict(front="fa", fa_compact=True, fa_nfreq=17, r=16, fa_balance=True, clip=1.0, head_wd=1e-2, lr=1e-3,
           es_mode="two_checkpoint", min_epochs=20, max_epochs=80, patience=10)


def _data():
    obs, _ = generate(SynthConfig(n=200, censoring=0.3, seed=79))
    tr, va, te = split_groups(obs.ids, 79, strata=obs.delta)
    return obs, tr, va, te


@pytest.mark.criterion("NF-1")
def test_NF1_refit_keeps_encoder_decoder_and_latent_bit_identical():
    """nonlinear_refit: the joint run is untouched, so every non-head weight and the latent equal the linear fit's
    exactly; only the head (now nonlinear) and the risk predictions change."""
    obs, tr, va, te = _data()
    lin = NeuralAFT("fae_mlp", **CFG).fit(obs.subset(tr), obs.subset(va), seed=0)
    ref = NeuralAFT("fae_mlp", head_refit="mlp", **CFG).fit(obs.subset(tr), obs.subset(va), seed=0)
    a, b = lin.model.state_dict(), ref.model.state_dict()
    body = [k for k in a if not k.startswith("head.")]
    assert body and all(torch.equal(a[k], b[k]) for k in body)
    assert np.array_equal(lin.latent(obs.subset(te)), ref.latent(obs.subset(te)))
    assert not isinstance(ref.model.head.net, torch.nn.Linear) and ref.risk_model is None
    assert ref.info["runs"][-1]["refit"] and ref.info["runs"][0]["stop_epoch"] == lin.info["runs"][0]["stop_epoch"]
    assert not np.allclose(lin.predict_risk(obs.subset(te)), ref.predict_risk(obs.subset(te)))


@pytest.mark.criterion("NF-2")
def test_NF2_single_seed_ensemble_equals_single_model():
    obs, tr, va, te = _data()
    single = NeuralAFT("fae_mlp", **CFG).fit(obs.subset(tr), obs.subset(va), seed=0)
    ens = SeedEnsembleAFT("fae_mlp", ensemble_seeds=1, **CFG).fit(obs.subset(tr), obs.subset(va))
    grid = np.linspace(0.0, 2.0, 7)
    assert np.array_equal(ens.predict_risk(obs.subset(te)), single.predict_risk(obs.subset(te)))
    assert np.array_equal(ens.predict_survival(obs.subset(te), grid), single.predict_survival(obs.subset(te), grid))


@pytest.mark.criterion("NF-3")
def test_NF3_ensemble_is_member_mean_and_deterministic():
    obs, tr, va, te = _data()
    e1 = SeedEnsembleAFT("fae_mlp", ensemble_seeds=2, **CFG).fit(obs.subset(tr), obs.subset(va))
    e2 = SeedEnsembleAFT("fae_mlp", ensemble_seeds=2, **CFG).fit(obs.subset(tr), obs.subset(va))
    grid = np.linspace(0.0, 2.0, 7)
    r1 = e1.predict_risk(obs.subset(te))
    assert np.array_equal(r1, e2.predict_risk(obs.subset(te)))
    assert np.allclose(r1, (e1.members[0].predict_risk(obs.subset(te)) + e1.members[1].predict_risk(obs.subset(te))) / 2)
    S = e1.predict_survival(obs.subset(te), grid)
    assert np.allclose(S, np.mean([m.predict_survival(obs.subset(te), grid) for m in e1.members], axis=0))
    assert not np.array_equal(e1.members[0].predict_risk(obs.subset(te)), e1.members[1].predict_risk(obs.subset(te)))


@pytest.mark.criterion("NF-4")
def test_NF4_head_weight_decay_covers_nonlinear_head():
    """head_wd applies to every weight matrix of a nonlinear head (and, unchanged, to the linear head's weight)."""
    obs, tr, va, _ = _data()
    seen = {}
    orig = torch.optim.Adam.__init__

    def spy(self, params, *a, **k):
        seen["groups"] = params
        return orig(self, params, *a, **k)

    m = NeuralAFT("fae_mlp", head="mlp", **dict(CFG, max_epochs=2, min_epochs=0))
    torch.optim.Adam.__init__ = spy
    try:
        m.fit(obs.subset(tr), obs.subset(va), seed=0)
    finally:
        torch.optim.Adam.__init__ = orig
    wd_ids = {id(p) for p in seen["groups"][1]["params"]}
    heads = {n: p for n, p in m.model.named_parameters() if n.startswith("head.net") and n.endswith("weight")}
    assert len(heads) == 2 and all(id(p) in wd_ids for p in heads.values())
    assert seen["groups"][1]["weight_decay"] == 1e-2
