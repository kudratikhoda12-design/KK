"""Revision 4d (D87): burn-in before validation-NLL-based selection, criteria BI-2..BI-4 (BI-1, exact reproduction
of a pre-change fit with min_epochs = 0, is recorded in reports/logs/bi1_regression_check.log)."""
import numpy as np
import pytest
import torch

from kanfae.data.splits import split_groups
from kanfae.data.synthetic import SynthConfig, generate
from kanfae.models.pipelines import NeuralAFT

pytestmark = [pytest.mark.stage("D4d"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)
CFG = dict(front="fa", fa_compact=True, fa_nfreq=17, r=12, fa_balance=True, clip=1.0, head_wd=1e-2, max_epochs=120,
           patience=10)


def _data():
    obs, _ = generate(SynthConfig(n=200, censoring=0.3, seed=79))
    tr, va, te = split_groups(obs.ids, 79, strata=obs.delta)
    return obs, tr, va, te


@pytest.mark.criterion("BI-2")
def test_BI2_two_checkpoint_risk_state_respects_burn_in():
    obs, tr, va, _ = _data()
    m = NeuralAFT("fae_mlp", es_mode="two_checkpoint", min_epochs=60, **CFG).fit(obs.subset(tr), obs.subset(va), seed=0)
    run = m.info["runs"][0]
    assert run["epochs"] >= 61                                   # no early stopping before the burn-in
    assert run["risk_state_epoch"] >= 60


@pytest.mark.criterion("BI-3")
def test_BI3_nll_selection_ignores_pre_burn_in_minimum(monkeypatch):
    """With burn-in, the selected NLL state is the minimum over epochs >= min_epochs, even if an earlier epoch was
    lower (the epoch-91 failure mode of seed 135)."""
    obs, tr, va, _ = _data()
    m = NeuralAFT("fae_mlp", es_target="nll", min_epochs=50, **dict(CFG, patience=1000, max_epochs=80)).fit(
        obs.subset(tr), obs.subset(va), seed=0)
    run = m.info["runs"][0]
    assert run["stop_epoch"] >= 50 and run["epochs"] == 80


@pytest.mark.criterion("BI-4")
def test_BI4_refit_head_respects_burn_in():
    obs, tr, va, _ = _data()
    m = NeuralAFT("fae_mlp", es_mode="two_state_refit", min_epochs=40, **CFG).fit(obs.subset(tr), obs.subset(va), seed=0)
    refit = m.info["runs"][-1]
    assert refit["refit"] and refit["epochs"] >= 41 and refit["stop_epoch"] >= 40
