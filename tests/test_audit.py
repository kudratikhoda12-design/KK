"""Source-code audit checks (2026-10-17): each equation in the training objective is recomputed independently."""
import numpy as np
import pytest
import torch

from kanfae.data.synthetic import SynthConfig, generate
from kanfae.functional.smoothing import individual_weighted_mse
from kanfae.models.pipelines import NeuralAFT, _ind_weighted_rec
from kanfae.survival.aft import aft_median, aft_nll

pytestmark = [pytest.mark.stage("AUDIT"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def small():
    obs, _ = generate(SynthConfig(n=120, censoring=0.3, seed=50, p_miss=0.2))
    return obs.subset(np.arange(80)), obs.subset(np.arange(80, 120))


@pytest.mark.criterion("AUD-1")
@pytest.mark.parametrize("kind,front", [("fae_mlp", "linear"), ("fae_mlp", "nonlinear"), ("fae_kan", "nonlinear"),
                                        ("flat_ae", "linear")])
def test_AUD1_joint_objective_decomposition(small, kind, front):
    tr, va = small
    m = NeuralAFT(kind, front=front, max_epochs=2, alpha=0.7, lam_beta=0.05).fit(tr, va, seed=0)
    T = m._tensors(m.prepare(tr))
    m.model.eval()
    with torch.no_grad():
        joint, _ = m._objective(T, "joint")
        z, out, eta, sigma = m.model(T["inp"], T["u"])
        nll = aft_nll(T["y"], T["d"], eta, sigma, "weibull")
        xhat = m.model.reconstruct_at(out, T["design"]).numpy()
        rec = individual_weighted_mse((xhat - T["values"].numpy()) ** 2, T["mask"].numpy())
        pen = 0.05 * float(m.model.roughness_penalty())
    assert abs(float(joint) - (float(nll) + 0.7 * rec + pen)) < 1e-10


@pytest.mark.criterion("AUD-2")
def test_AUD2_reconstruction_weighting_matches_numpy():
    rng = np.random.default_rng(0)
    err_x, val = rng.normal(size=(30, 12, 3)), rng.normal(size=(30, 12, 3))
    mask = rng.random((30, 12, 3)) < 0.6
    mask[0] = False
    mask[1, :, 0] = False
    t = _ind_weighted_rec(torch.tensor(err_x), torch.tensor(val), torch.tensor(mask))
    assert abs(float(t) - individual_weighted_mse((err_x - val) ** 2, mask)) < 1e-12


@pytest.mark.criterion("AUD-3")
def test_AUD3_S1_freezes_autoencoder_in_head_phase(small):
    tr, va = small
    snaps = {}

    class Spy(NeuralAFT):
        def _run(self, Ttr, Tva, parts, params):
            if parts == "nll":
                snaps["before"] = {k: v.clone() for k, v in self.model.state_dict().items() if not k.startswith("head.")}
            out = super()._run(Ttr, Tva, parts, params)
            if parts == "nll":
                snaps["after"] = {k: v.clone() for k, v in self.model.state_dict().items() if not k.startswith("head.")}
            return out

    Spy("fae_mlp", strategy="S1", max_epochs=5).fit(tr, va, seed=0)
    for k in snaps["before"]:
        if "running" in k or "num_batches" in k:
            continue
        assert torch.equal(snaps["before"][k], snaps["after"][k]), k


@pytest.mark.criterion("AUD-4")
def test_AUD4_survival_only_gives_no_decoder_gradient(small):
    tr, va = small
    m = NeuralAFT("fae_mlp", alpha=0.0, max_epochs=1).fit(tr, va, seed=0)
    T = m._tensors(m.prepare(tr))
    m.model.train()
    m.model.zero_grad()
    loss, _ = m._objective(T, "nll")
    loss.backward()
    for p in m.model.decoder.parameters():
        assert p.grad is None or float(p.grad.abs().max()) == 0.0


@pytest.mark.criterion("AUD-5")
def test_AUD5_time_scaling_consistent(small):
    tr, va = small
    m = NeuralAFT("fae_mlp", max_epochs=3).fit(tr, va, seed=0)
    P = m.prepare(va)
    _, eta, sigma, _ = m._forward(P)
    med = aft_median(torch.tensor(eta), sigma, "weibull").numpy() * m.tscale
    S = m.predict_survival(va, np.sort(med)[:5], P=P)
    order = np.argsort(med)[:5]
    np.testing.assert_allclose(S[order, np.arange(5)], 0.5, atol=1e-10)


@pytest.mark.criterion("AUD-6")
def test_AUD6_front_end_shared_between_arms(small):
    tr, va = small
    a = NeuralAFT("fae_mlp", max_epochs=1).fit(tr, va, seed=0)
    b = NeuralAFT("fae_kan", max_epochs=1).fit(tr, va, seed=0)
    na = sum(p.numel() for p in a.model.flin.parameters())
    nb = sum(p.numel() for p in b.model.flin.parameters())
    assert na == nb and type(a.model.flin) is type(b.model.flin)
