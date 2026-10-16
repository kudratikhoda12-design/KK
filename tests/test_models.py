"""Stage D: model criteria M1, M2, M3, M5 and leakage criteria K1, K2 (protocol sections 7.5)."""
import numpy as np
import pytest
import torch

from kanfae.data.preprocess import Preprocessor
from kanfae.data.splits import split_groups
from kanfae.data.synthetic import SynthConfig, generate
from kanfae.functional.smoothing import evaluate, select_lambda
from kanfae.models.fae import kan2_match, kan_match
from kanfae.models.pipelines import DEFAULTS, NeuralAFT, _ind_weighted_rec, make_model

pytestmark = [pytest.mark.stage("D"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)
ALL = ["fae_mlp", "fae_kan", "fae_kan2", "fae_linear", "flat_ae", "fpca_aft", "fpca_cox", "sum_aft", "sum_cox", "sum_rsf"]


@pytest.fixture(scope="module")
def small():
    obs, ora = generate(SynthConfig(n=240, sampling="sparse", p_miss=0.2, censoring=0.3, seed=40))
    m = obs.mask.copy()
    m[0] = False; m[0, 0, :] = True           # individual with a single observation time
    m[1, :, 2] = False                        # an empty channel
    obs = type(obs)(obs.ids, obs.times, np.where(m, obs.values, 0), m, obs.n_obs, obs.y_obs, obs.delta,
                    obs.u, obs.channel_names, obs.meta)
    tr, va, te = split_groups(obs.ids, 0, strata=obs.delta)
    return obs, ora, tr, va, te


@pytest.mark.criterion("M1")
@pytest.mark.parametrize("name", ALL)
def test_M1_shapes(small, name):
    obs, _, tr, va, te = small
    kw = {"n_components": 4} if name.startswith("fpca") else {}
    model = make_model(name, max_epochs=3, **kw).fit(obs.subset(tr), obs.subset(va), seed=0)
    times = np.linspace(0.1, 10, 7)
    S = model.predict_survival(obs.subset(te), times)
    r = model.predict_risk(obs.subset(te))
    assert S.shape == (len(te), 7) and r.shape == (len(te),)
    assert np.all(np.isfinite(S)) and np.all(np.isfinite(r))
    if isinstance(model, NeuralAFT):
        assert model.latent(obs.subset(te)).shape == (len(te), 4)
        xhat, P = model.reconstruct(obs.subset(te))
        assert xhat.shape == obs.subset(te).values.shape


@pytest.mark.criterion("M2")
@pytest.mark.parametrize("name", ["fae_mlp", "fae_kan", "fae_linear", "flat_ae"])
def test_M2_memorise_one_batch(name):
    obs, _ = generate(SynthConfig(n=16, seed=41))
    model = NeuralAFT(name)
    model._prep_fit(obs, seed=0)
    P = model.prepare(obs)
    # D32: targets are representable by each model, so the 1% criterion is attainable in principle:
    # smoothed spline curves (functional AEs) or the hat-basis interpolant of the input grid (flat AE).
    if name == "flat_ae":
        P.values = np.where(P.fit_mask, np.einsum("ndg,nlg->nld", P.grid_values, P.H), 0.0)
    else:
        P.values = np.where(P.fit_mask, evaluate(P.c, P.Phi), 0.0)
    T = model._tensors(P)
    model.model = model._build(3, 0, seed=0)
    opt = torch.optim.Adam(model.model.parameters(), lr=3e-3)
    model.model.train()

    def rec():
        _, out, _, _ = model.model(T["inp"], T["u"])
        return _ind_weighted_rec(model.model.reconstruct_at(out, T["design"]), T["values"], T["mask"])

    init = float(rec())
    for _ in range(3000):
        opt.zero_grad(); loss = rec(); loss.backward(); opt.step()
    model.model.eval()
    with torch.no_grad():
        final = float(rec())
    print(f"{name}: initial {init:.4g} final {final:.4g} ratio {final / init:.4g}")
    if name == "fae_linear":
        # a linear autoencoder with r = 4 cannot memorise 16 distinct 36-dim targets; record ratio only
        pytest.skip(f"linear AE rank-limited (r=4 < 16 targets); ratio {final / init:.4g} recorded, not a criterion")
    assert final < 0.01 * init


@pytest.mark.criterion("M3")
@pytest.mark.parametrize("D", [3, 6, 14])
def test_M3_parameter_matching(D):
    H, hid = DEFAULTS["H"], DEFAULTS["hidden"]          # matching uses the current model defaults
    he, hd, info = kan_match(D, 12, H=H, hidden=hid)
    w2, info2 = kan2_match(D, 12, H=H, hidden=hid)
    print(D, he, hd, info, "kan2", w2, info2)
    assert abs(info["rel_dev"]) <= 0.10 and abs(info2["rel_dev"]) <= 0.10
    obs, _ = generate(SynthConfig(n=60, seed=42))
    if D == 3:
        a = NeuralAFT("fae_mlp", max_epochs=1).fit(obs, obs)
        b = NeuralAFT("fae_kan", max_epochs=1).fit(obs, obs)
        c = NeuralAFT("fae_kan2", max_epochs=1).fit(obs, obs)
        dev = (b.info["block_params"] - a.info["block_params"]) / a.info["block_params"]
        dev2 = (c.info["block_params"] - a.info["block_params"]) / a.info["block_params"]
        print("instantiated blocks", a.info["block_params"], b.info["block_params"], c.info["block_params"])
        assert abs(dev) <= 0.10 and abs(dev2) <= 0.10
        assert b.info["block_params"] == info["kan_block_params"]


@pytest.mark.criterion("M5")
@pytest.mark.parametrize("name", ALL)
def test_M5_fit_rejects_oracle(name):
    obs, ora = generate(SynthConfig(n=60, seed=43))
    with pytest.raises(TypeError):
        make_model(name, max_epochs=1).fit(ora, obs)


@pytest.mark.criterion("K1")
def test_K1_split_disjoint_and_complete():
    obs, _ = generate(SynthConfig(n=1000, seed=44))
    for s in range(5):
        tr, va, te = split_groups(obs.ids, s, strata=obs.delta)
        a, b, c = set(obs.ids[tr]), set(obs.ids[va]), set(obs.ids[te])
        assert not (a & b) and not (a & c) and not (b & c)
        assert len(a | b | c) == 1000
    groups = np.repeat(np.arange(50), 7)          # multi-row groups (windows)
    tr, va, te = split_groups(groups, 0)
    for part in (tr, va, te):
        for g in np.unique(groups[part]):
            assert set(np.nonzero(groups == g)[0]) <= set(part)


@pytest.mark.criterion("K2")
def test_K2_preprocessing_uses_training_only():
    obs, _ = generate(SynthConfig(n=600, seed=45))
    tr, va, te = split_groups(obs.ids, 0, strata=obs.delta)
    pre = Preprocessor().fit(obs.subset(tr))
    otr = obs.subset(tr)
    for d in range(3):
        v = otr.values[:, :, d][otr.mask[:, :, d]]
        assert np.isclose(pre.mean_[d], v.mean()) and np.isclose(pre.sd_[d], v.std())
    vals = np.where(otr.mask, (otr.values - pre.mean_) / pre.sd_, 0)
    lam, _ = select_lambda(otr.times, vals, otr.mask, otr.n_obs, pre.basis, seed=0)
    assert lam == pre.lam_
    # perturbing validation/test data must not change fitted statistics
    obs2 = type(obs)(obs.ids, obs.times, obs.values.copy(), obs.mask, obs.n_obs, obs.y_obs, obs.delta)
    obs2.values[te] += 100.0
    pre2 = Preprocessor().fit(obs2.subset(tr))
    assert np.array_equal(pre.mean_, pre2.mean_) and pre.lam_ == pre2.lam_
    m = make_model("fpca_aft", n_components=4).fit(obs.subset(tr), obs.subset(va))
    m2 = make_model("fpca_aft", n_components=4).fit(obs2.subset(tr), obs2.subset(va))
    assert np.allclose(m.fpca.mean_, m2.fpca.mean_)
