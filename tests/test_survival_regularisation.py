"""Revision 4b (D71): survival-path regularisation options, criteria HR-1..HR-3."""
import numpy as np
import pytest
import torch

from kanfae.data.splits import split_groups
from kanfae.data.synthetic import SynthConfig, generate
from kanfae.models.fae import FunctionalAE
from kanfae.functional.basis import BSplineBasis
from kanfae.models.pipelines import NeuralAFT

pytestmark = [pytest.mark.stage("D4b"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)
CFG = dict(front="fa", fa_compact=True, fa_nfreq=17, r=12, max_epochs=25, patience=100)


def _data():
    obs, _ = generate(SynthConfig(n=200, censoring=0.3, seed=77))
    tr, va, te = split_groups(obs.ids, 77, strata=obs.delta)
    return obs, tr, va, te


@pytest.mark.criterion("HR-1")
def test_HR1_head_param_group_with_same_decay_is_identical():
    obs, tr, va, te = _data()
    a = NeuralAFT("fae_mlp", **CFG).fit(obs.subset(tr), obs.subset(va), seed=0)
    b = NeuralAFT("fae_mlp", head_wd=1e-4, **CFG).fit(obs.subset(tr), obs.subset(va), seed=0)   # = global wd
    np.testing.assert_array_equal(a.predict_risk(obs.subset(te)), b.predict_risk(obs.subset(te)))


@pytest.mark.criterion("HR-2")
def test_HR2_head_dropout_touches_only_the_survival_path():
    b = BSplineBasis(12)
    m0 = FunctionalAE(3, b, H=32, r=12, front="fa", fa_nfreq=17, fa_compact=True, seed=3)
    m1 = FunctionalAE(3, b, H=32, r=12, front="fa", fa_nfreq=17, fa_compact=True, seed=3, head_dropout=0.5)
    m1.load_state_dict(m0.state_dict())
    c, u = torch.randn(40, 3, 12), torch.zeros(40, 0)
    m0.train(); m1.train()
    with torch.no_grad():
        z0, ch0, e0, _ = m0(c, u)
        torch.manual_seed(0)
        z1, ch1, e1, _ = m1(c, u)
    m0.eval(); m1.eval()                                   # BatchNorm in train mode used batch stats in both
    torch.testing.assert_close(z1, z0, rtol=0, atol=0)
    torch.testing.assert_close(ch1, ch0, rtol=0, atol=0)   # decoder sees the undropped latent
    assert not torch.allclose(e1, e0)                      # the head sees the dropped latent
    with torch.no_grad():                                  # evaluation mode: dropout is the identity
        torch.testing.assert_close(m1(c, u)[2], m0(c, u)[2], rtol=0, atol=0)


@pytest.mark.criterion("HR-3")
def test_HR3_head_weight_decay_shrinks_only_head_weights():
    obs, tr, va, te = _data()
    cfg = dict(CFG, max_epochs=150)
    a = NeuralAFT("fae_mlp", **cfg).fit(obs.subset(tr), obs.subset(va), seed=0)
    b = NeuralAFT("fae_mlp", head_wd=1.0, **cfg).fit(obs.subset(tr), obs.subset(va), seed=0)
    na, nb = a.model.head.net.weight.norm().item(), b.model.head.net.weight.norm().item()
    print("head weight norm: default", round(na, 4), "head_wd=1", round(nb, 4))
    assert nb < 0.5 * na
