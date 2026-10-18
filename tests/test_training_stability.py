"""Revision 4c (D81): spectral balance, gradient clipping and two-state stopping, criteria RS-2..RS-5
(RS-1, exact reproduction of a pre-change fit, is recorded in reports/logs/rs1_regression_check.log)."""
import numpy as np
import pytest
import torch

from kanfae.data.splits import split_groups
from kanfae.data.synthetic import SynthConfig, generate
from kanfae.models.pipelines import NeuralAFT

pytestmark = [pytest.mark.stage("D4c"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)
CFG = dict(front="fa", fa_compact=True, fa_nfreq=17, r=12, max_epochs=60, patience=1000)


def _data():
    obs, _ = generate(SynthConfig(n=200, censoring=0.3, seed=78))
    tr, va, te = split_groups(obs.ids, 78, strata=obs.delta)
    return obs, tr, va, te


@pytest.mark.criterion("RS-2")
def test_RS2_balance_matches_a3_rms_on_training_data():
    obs, tr, va, _ = _data()
    m = NeuralAFT("fae_mlp", fa_balance=True, **dict(CFG, max_epochs=1)).fit(obs.subset(tr), obs.subset(va), seed=0)
    b = m.balance_info
    print(b)
    assert abs(b["init_rms_spec_after"] / b["init_rms_a3"] - 1) < 1e-10 and b["spec_gain"] != 1.0


@pytest.mark.criterion("RS-3")
def test_RS3_gradient_clipping_is_applied(monkeypatch):
    calls = []
    orig = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **k):
        params = list(params)
        calls.append((len(params), max_norm))
        out = orig(params, max_norm, *a, **k)
        assert torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None)) <= max_norm * (1 + 1e-6)
        return out
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    obs, tr, va, _ = _data()
    m = NeuralAFT("fae_mlp", clip=0.5, **dict(CFG, max_epochs=5)).fit(obs.subset(tr), obs.subset(va), seed=0)
    assert len(calls) == 5 and all(mx == 0.5 for _, mx in calls)
    assert calls[0][0] == len(list(m.model.parameters()))


@pytest.mark.criterion("RS-4")
def test_RS4_two_checkpoint_risk_comes_from_the_validation_nll_minimum():
    obs, tr, va, te = _data()
    m = NeuralAFT("fae_mlp", es_mode="two_checkpoint", **CFG).fit(obs.subset(tr), obs.subset(va), seed=0)
    run = m.info["runs"][0]
    assert run["risk_state_epoch"] == run["epoch_min_val_nll"]
    Pva = m.prepare(obs.subset(va)); Tva = m._tensors(Pva)
    m.risk_model.eval()
    with torch.no_grad():
        _, _, eta, sigma = m.risk_model(Tva["inp"], Tva["u"])
        from kanfae.survival.aft import aft_nll
        v = float(aft_nll(Tva["y"], Tva["d"], eta, sigma, m.cfg["family"]))
    assert abs(v - run["min_val_nll"]) < 1e-10
    single = NeuralAFT("fae_mlp", **CFG).fit(obs.subset(tr), obs.subset(va), seed=0)
    np.testing.assert_array_equal(m.latent(obs.subset(te)), single.latent(obs.subset(te)))   # latent: objective state


@pytest.mark.criterion("RS-5")
def test_RS5_two_state_refit_changes_only_the_head():
    obs, tr, va, te = _data()
    m = NeuralAFT("fae_mlp", es_mode="two_state_refit", **CFG).fit(obs.subset(tr), obs.subset(va), seed=0)
    single = NeuralAFT("fae_mlp", **CFG).fit(obs.subset(tr), obs.subset(va), seed=0)
    np.testing.assert_array_equal(m.latent(obs.subset(te)), single.latent(obs.subset(te)))
    sd_m, sd_s = m.model.state_dict(), single.model.state_dict()
    changed = [k for k in sd_m if not torch.equal(sd_m[k], sd_s[k])]
    print("changed tensors:", changed)
    assert changed and all(k.startswith("head.") for k in changed)
    assert m.info["runs"][-1]["refit"] and m.info["runs"][-1]["parts"] == "nll"
