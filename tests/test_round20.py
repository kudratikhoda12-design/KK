"""Round 20 (D150): deterministic checks of the NP-KAN-AFT (family "np_aft") and of CrossConfirmRefitAnchor
(no formal seed is used).

R20-1  NP-AFT building blocks: the vectorised KM equals km_survival; the KM conditional mean equals the brute-force
       Buckley-James imputation (censored_ls) and the integral form r + int_r S / S(r); the gradient of the (nearly
       unsmoothed) rank loss equals the Gehan estimating function sum_i d_i sum_j (x_i - x_j) 1{e_j > e_i} / n^2, and a
       linear model fitted by it recovers the AFT slope under censoring
R20-2  NPKANAFT (bj_km and rank_km) recovers the true survival curves without the error family: mean |S_hat - S_true|
       < 0.03 on bimodal-error and log-normal-error data, and on bimodal errors it beats the (misspecified) Weibull and
       log-normal KAN-AFT
R20-3  NP survival prediction: S(t | Z) = S_KM(log t - loc(Z)), non-increasing in t, 1 below the smallest training
       residual, S_KM(largest residual) above it; the functional (b2s) NP-KAN-AFT and its anchored two-way gate run and
       give finite survival in [0, 1]
R20-4  distill_and_replace on NP-KAN-AFT (both objectives): finite predictions, KM re-estimated on the distilled
       residuals; the non-degradation rollback guard returns the spline model unchanged when the validation loss
       would degrade
R20-5  CrossConfirmRefitAnchor: not confirmed -> exactly the full-validation 3-seed B2-S (max |diff| == 0.0; Cox,
       log-normal AFT, NP-AFT); the decision needs BOTH directions (absolute or relative margin); held-out losses use every ensemble member; when
       confirmed, the refit is the Round-19 "finetune" refit of the h1 direction (bit-identical), early-stopping on the
       whole validation set, training on the training split only
R20-6  default-off: FKANSurv (Cox, Weibull, log-normal), the Round-18 confirmation gate and the Round-19
       ConfirmRefitAnchor give bit-identical predictions to the Round-19 code (commit 495f57e, loaded from git)
"""
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import norm

from kanfae.models.fkan import FKANEnsemble, FKANSurv
from kanfae.models.fkan_surv import (NPKANAFT, bj_targets, distill_and_replace, gehan_loss, km_cond_mean, km_eval,
                                     km_fit, km_tail_fit)
from kanfae.models.fres import (ConfirmRefitAnchor, CrossConfirmRefitAnchor, ResFKAN, cross_confirm_decision,
                                ensemble_heldout_loss, heldout_loss)
from kanfae.survival.censored_ls import bj_impute_logscale, km_survival

pytestmark = [pytest.mark.stage("R20"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[1]
R19_COMMIT = "495f57e"
KW = dict(max_epochs=40, patience=40, lrs=(1e-2,), wds=(1e-4,), seeds=(0, 1))


# ---- data ---------------------------------------------------------------------------------------------------------------
def _static(n, err, seed):
    rng = np.random.default_rng(seed)
    z = rng.uniform(-2, 2, (n, 2))
    f = 0.8 * np.sin(z[:, 0]) + 0.5 * z[:, 1]
    eps = (np.where(rng.random(n) < 0.5, -0.6, 0.6) + 0.2 * rng.standard_normal(n) if err == "bimodal"
           else 0.5 * rng.standard_normal(n))
    T, C = np.exp(1 + f + eps), np.exp(rng.uniform(0.5, 3.5, n))
    return dict(xg=None, z=z, y=np.minimum(T, C), d=(T <= C).astype(float)), f


def _truth(f, t, err):
    w = np.log(t)[None, :] - 1 - f[:, None]
    if err == "bimodal":
        return 0.5 * norm.sf((w + 0.6) / 0.2) + 0.5 * norm.sf((w - 0.6) / 0.2)
    return norm.sf(w / 0.5)


def _func(n=200, seed=6, aft=False):
    """The Round-19 functional fixture: train, val h1, val h2, test (Cox-type hazard; AFT log-time for aft=True)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 50)
    xg = rng.normal(size=(n, 1, 2)) + 0.5 * rng.normal(size=(n, 1, 2)) * t[None, :, None]
    last = xg[:, -1, :]
    z = rng.normal(size=(n, 1))
    eta = np.tanh(last[:, 0]) + 0.5 * np.sin(xg[:, :, 1].mean(1)) + 0.3 * z[:, 0]
    T, C = rng.exponential(np.exp(-eta)), rng.exponential(2.0, n)
    p = dict(xg=xg, last=last, z=z, y=np.minimum(T, C), d=(T <= C).astype(float))
    sl = (slice(0, 100), slice(100, 140), slice(140, 180), slice(180, 200))
    return [{k: v[s] for k, v in p.items()} for s in sl]


def _plain(p):
    return {k: p[k] for k in ("xg", "z", "y", "d")}


def _fam(loss):
    return {"cox": ("cox", {}), "aft": ("aft", dict(family="lognormal")),
            "np": ("aft", dict(family="np_aft", np_objective="bj_km"))}[loss]


def _ens(loss, tr, va):
    l, fam = _fam(loss)
    return FKANEnsemble(lrs=KW["lrs"], wds=KW["wds"], seeds=KW["seeds"], kind="b2s", loss=l, max_epochs=40, patience=40,
                        **fam).fit(_plain(tr), _plain(va))


def _cross(loss, margin, tr, h1, h2, refs=None):
    if refs is None:
        vf = {k: np.concatenate([h1[k], h2[k]]) for k in ("xg", "last", "z", "y", "d")}
        refs = (_ens(loss, tr, h1), _ens(loss, tr, h2), _ens(loss, tr, vf))
    l, fam = _fam(loss)
    m = CrossConfirmRefitAnchor(loss=l, margin=margin, fine_tune_epochs=10, **KW, **fam).fit(tr, h1, h2, *refs)
    return m, refs


# ---- R20-1 --------------------------------------------------------------------------------------------------------------
@pytest.mark.criterion("R20-1")
def test_R20_1_km_bj_and_gehan_building_blocks():
    rng = np.random.default_rng(1)
    r = np.round(rng.normal(size=300), 2)                                   # ties included
    d = (rng.random(300) < 0.6).astype(float)
    k1, S1 = km_fit(r, d)
    k0, S0 = km_survival(r, d)
    assert np.array_equal(k1, k0) and np.max(np.abs(S1 - S0)) < 1e-14
    # Buckley-James conditional mean: brute force and integral form
    logy, eta = rng.normal(size=300), 0.3 * rng.normal(size=300)
    ys = bj_targets(logy, d, eta, km_tail_fit(logy - eta, d))
    assert np.max(np.abs(ys - bj_impute_logscale(logy, d, eta))) < 1e-12
    kn, S = km_tail_fit(logy - eta, d)
    for ri in (logy - eta)[d == 0][:20]:
        u = np.linspace(ri, kn[-1], 200001)
        integ = np.trapezoid(km_eval(kn, S, u), u)
        Sr = float(km_eval(kn, S, ri))
        if Sr > 1e-9:
            assert abs(float(km_cond_mean(kn, S, np.array([ri]))[0]) - (ri + integ / Sr)) < 1e-3
    # rank loss: gradient = the Gehan estimating function (pairs: event i against every j with e_j > e_i)
    n = 200
    x = rng.normal(size=n)
    logT = 1.0 * x + 0.5 * rng.normal(size=n)
    logC = rng.uniform(-1, 2.5, n)
    y, dd = np.exp(np.minimum(logT, logC)), (logT <= logC).astype(float)
    b = torch.tensor(0.4, requires_grad=True)
    loss = gehan_loss(b * torch.tensor(x), torch.tensor(y), torch.tensor(dd), tau=1e-7)
    loss.backward()
    e = np.log(y) - 0.4 * x
    U = sum(dd[i] * np.sum((x[i] - x) * (e > e[i])) for i in range(n)) / n ** 2
    assert abs(float(b.grad) - U) < 1e-8
    # a linear model fitted by the smoothed rank loss recovers the slope (the intercept is not identified)
    n = 2000
    x = rng.normal(size=n)
    logT = 1.0 + 1.0 * x + 0.5 * rng.normal(size=n)
    logC = 1.0 + 0.8 * x + rng.uniform(-0.5, 1.5, n)                        # covariate-dependent censoring
    y, dd = torch.tensor(np.exp(np.minimum(logT, logC))), torch.tensor((logT <= logC).astype(float))
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lo = gehan_loss(b * torch.tensor(x), y, dd)
        lo.backward()
        return lo
    opt.step(closure)
    assert 0.4 < float(dd.mean()) < 0.9 and abs(float(b) - 1.0) < 0.1


# ---- R20-2 --------------------------------------------------------------------------------------------------------------
@pytest.mark.criterion("R20-2")
def test_R20_2_np_kan_aft_recovers_survival_without_error_family():
    t = np.exp(np.linspace(0, 4, 40))
    for err, seed in (("bimodal", 11), ("lognormal", 12)):
        (tr, _), (va, _), (te, fte) = _static(600, err, seed), _static(200, err, seed + 100), _static(500, err, seed + 200)
        St = _truth(fte, t, err)
        e = {}
        for obj in ("bj_km", "rank_km"):
            m = NPKANAFT(obj, kind="static", lrs=(3e-3,), wds=(1e-4,), seeds=(0,)).fit(tr, va)
            S = m.predict_survival(None, te["z"], t)
            assert np.all(np.isfinite(S)) and np.all(np.diff(S, axis=1) <= 1e-12)
            e[obj] = float(np.mean(np.abs(S - St)))
            assert e[obj] < 0.03, (err, obj, e[obj])
        if err == "bimodal":
            for fam in ("weibull", "lognormal"):
                p = FKANSurv(kind="static", loss="aft", family=fam).fit(tr, va)
                ep = float(np.mean(np.abs(p.predict_survival(None, te["z"], t) - St)))
                assert max(e.values()) < ep, (fam, e, ep)


# ---- R20-3 --------------------------------------------------------------------------------------------------------------
@pytest.mark.criterion("R20-3")
def test_R20_3_np_survival_prediction_and_functional_model():
    (tr, _), (va, _), (te, _) = _static(300, "bimodal", 21), _static(100, "bimodal", 22), _static(50, "bimodal", 23)
    for obj in ("bj_km", "rank_km"):
        m = FKANSurv(kind="static", loss="aft", family="np_aft", np_objective=obj, max_epochs=200).fit(tr, va)
        loc = float(m.mu.detach()) + m.eta(None, te["z"])
        r = m.np_residuals
        assert np.allclose(r, np.log(tr["y"]) - float(m.mu.detach()) - m.eta(None, tr["z"]))
        rs = np.unique(r)                                      # midpoints between jumps (not at a jump, where rounding
        mid = 0.5 * (rs[1:] + rs[:-1])                         # of log(exp(.)) would decide the side)
        tt = np.exp(np.r_[r.min() - 1.0, mid, r.max() + 1.0][None, :] + loc[:5, None])
        for i in range(5):
            S = m.predict_survival(None, te["z"][i:i + 1], tt[i])[0]
            assert S[0] == 1.0 and S[-1] == km_fit(r, tr["d"])[1][-1]
            assert np.all(np.diff(S) <= 1e-12) and np.all((S >= 0) & (S <= 1))
            assert np.allclose(S, km_eval(*km_fit(r, tr["d"]), np.log(tt[i]) - loc[i]))
        if obj == "rank_km":                                   # the intercept is held fixed (not identified)
            assert float(m.mu.detach()) == float(np.log(np.mean(tr["y"])))
    # functional NP-KAN-AFT and the anchored two-way gate with the NP loss
    trf, h1, h2, tef = _func()
    e = NPKANAFT("bj_km", max_epochs=40, patience=40, lrs=(1e-2,), wds=(1e-4,), seeds=(0, 1)).fit(_plain(trf), _plain(h1))
    S = e.predict_survival(tef["xg"], tef["z"], np.array([0.1, 0.5, 2.0]))
    assert np.all(np.isfinite(S)) and np.all((S >= 0) & (S <= 1))
    g = ResFKAN(residual=False, gate_last_value=True, loss="aft", family="np_aft", np_objective="rank_km",
                fine_tune_epochs=10, **KW).fit(trf, h1, confirm=h2)
    S = g.anchored_alternative().predict_survival(tef["xg"], tef["last"], tef["z"], np.array([0.1, 0.5, 2.0]))
    assert np.all(np.isfinite(S)) and np.all((S >= 0) & (S <= 1)) and np.isfinite(g.confirm_loss_anchored)


# ---- R20-4 --------------------------------------------------------------------------------------------------------------
@pytest.mark.criterion("R20-4")
def test_R20_4_distillation_on_np_kan_aft_and_rollback_guard():
    trf, h1, _, tef = _func()
    tt = np.array([0.1, 0.5, 2.0])
    for obj in ("bj_km", "rank_km"):
        m = FKANSurv(kind="b2s", loss="aft", family="np_aft", np_objective=obj, max_epochs=60, patience=60,
                     lr=1e-2).fit(_plain(trf), _plain(h1))
        S0 = m.predict_survival(tef["xg"], tef["z"], tt)
        kept, rec = distill_and_replace(m, max_iter=50, val_tol=np.inf)
        assert rec["n_substituted"] > 0 and np.isfinite(rec["val_loss_before"])
        if rec["finite"]:
            assert kept is not m and not rec["rollback"]
            S = kept.predict_survival(tef["xg"], tef["z"], tt)
            assert np.all(np.isfinite(S)) and np.all((S >= 0) & (S <= 1))
            loc = float(kept.mu.detach()) + kept.eta(trf["xg"], trf["z"])
            assert np.allclose(kept.np_residuals, np.log(trf["y"]) - loc)              # KM on distilled residuals
        else:
            assert kept is m and rec["rollback"]
        back, rec2 = distill_and_replace(m, max_iter=50, val_tol=-np.inf)               # any degradation -> rollback
        assert back is m and rec2["rollback"]
        assert np.array_equal(back.predict_survival(tef["xg"], tef["z"], tt), S0)


# ---- R20-5 --------------------------------------------------------------------------------------------------------------
@pytest.mark.criterion("R20-5")
@pytest.mark.parametrize("loss", ["cox", "aft", "np"])
def test_R20_5_cross_confirm_reduction_and_refit(loss):
    tr, h1, h2, te = _func()
    tt = np.array([0.05, 0.3, 1.0, 3.0])
    m, refs = _cross(loss, 1e6, tr, h1, h2)
    rf = refs[2]
    assert m.reduced and m.members is None and not m.confirmed
    assert float(np.max(np.abs(m.predict_survival(te["xg"], te["last"], te["z"], tt) - rf.predict_survival(te["xg"], te["z"], tt)))) == 0.0
    assert float(np.max(np.abs(m.predict_risk(te["xg"], te["last"], te["z"]) - rf.predict_risk(te["xg"], te["z"])))) == 0.0
    # held-out losses: the mean over every ensemble member (not the seed-0 member), on the other half
    d0 = m.dirs[0]
    assert d0["loss_ref"] == ensemble_heldout_loss(refs[0].members, _plain(h2))
    assert d0["loss_ref"] == float(np.mean([heldout_loss(q, _plain(h2)) for q in refs[0].members]))
    assert d0["r18_loss_ref"] == heldout_loss(refs[0].members[0], _plain(h2))
    if loss != "cox":
        return
    # both directions are required
    a, b = dict(loss_anchored=0.0, loss_ref=1.0, active=True), dict(loss_anchored=2.0, loss_ref=1.0, active=True)
    assert cross_confirm_decision([a, a], 0.5) and not cross_confirm_decision([a, b], 0.5)
    assert not cross_confirm_decision([b, a], 0.5) and not cross_confirm_decision([a, dict(a, active=False)], 0.5)
    assert not cross_confirm_decision([a, a], 1.0)
    assert cross_confirm_decision([a, a], 0.99, relative=True) and not cross_confirm_decision([a, a], 1.0, relative=True)
    # confirmed: the Round-19 finetune refit of the h1 direction, early-stopping on the whole validation set
    mc, _ = _cross(loss, -1e6, tr, h1, h2, refs)
    assert mc.confirmed and len(mc.members) == len(KW["seeds"])
    r19 = ConfirmRefitAnchor(loss="cox", margin=-1e6, refit="finetune", fine_tune_epochs=10, **KW).fit(
        tr, h1, h2, refs[0], refs[2])
    for p, q in zip(mc.members, r19.members):
        assert all(torch.equal(x, y) for x, y in zip(p.net.state_dict().values(), q.net.state_dict().values()))
        assert torch.equal(p.data["yva"], torch.tensor(np.concatenate([h1["y"], h2["y"]])))
        assert len(p.data["ytr"]) == len(tr["y"])
    S = mc.predict_survival(te["xg"], te["last"], te["z"], tt)
    assert np.array_equal(S, r19.predict_survival(te["xg"], te["last"], te["z"], tt))
    assert not any(np.array_equal(te["y"], x) for x in (mc.val_full["y"], tr["y"]))


# ---- R20-6 --------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def r19(tmp_path_factory):
    """The kanfae package as of the Round-19 deliverable commit, importable as kanfae_r19."""
    d = tmp_path_factory.mktemp("r19")
    tar = subprocess.run(["git", "-C", str(ROOT), "archive", R19_COMMIT, "src/kanfae"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(d)], input=tar.stdout, check=True)
    shutil.move(str(d / "src" / "kanfae"), str(d / "kanfae_r19"))
    sys.path.insert(0, str(d))
    try:
        yield (importlib.import_module("kanfae_r19.models.fkan"), importlib.import_module("kanfae_r19.models.fres"))
    finally:
        sys.path.remove(str(d))


@pytest.mark.criterion("R20-6")
def test_R20_6_default_off_bit_identical_to_round19(r19):
    old_fkan, old_fres = r19
    tr, h1, h2, te = _func()
    tt = np.array([0.05, 0.3, 1.0, 3.0])
    for loss, fam in (("cox", {}), ("aft", dict(family="weibull")), ("aft", dict(family="lognormal"))):
        a = FKANSurv(kind="b2s", loss=loss, max_epochs=40, patience=40, **fam).fit(_plain(tr), _plain(h1))
        b = old_fkan.FKANSurv(kind="b2s", loss=loss, max_epochs=40, patience=40, **fam).fit(_plain(tr), _plain(h1))
        assert np.array_equal(a.predict_survival(te["xg"], te["z"], tt), b.predict_survival(te["xg"], te["z"], tt))
        assert a.val_loss == b.val_loss
    for loss, fam in (("cox", {}), ("aft", dict(family="lognormal"))):
        new_refs = [FKANEnsemble(**{**KW, "kind": "b2s", "loss": loss, **fam}).fit(_plain(tr), _plain(v)) for v in (h1, h2)]
        old_refs = [old_fkan.FKANEnsemble(**{**KW, "kind": "b2s", "loss": loss, **fam}).fit(_plain(tr), _plain(v))
                    for v in (h1, h2)]
        for margin in (-1e6, 0.0):
            a = ConfirmRefitAnchor(loss=loss, margin=margin, refit="finetune", fine_tune_epochs=10, **KW, **fam).fit(
                tr, h1, h2, new_refs[0], new_refs[1])
            b = old_fres.ConfirmRefitAnchor(loss=loss, margin=margin, refit="finetune", fine_tune_epochs=10, **KW,
                                            **fam).fit(tr, h1, h2, old_refs[0], old_refs[1])
            assert a.reduced == b.reduced
            assert a.gate.confirm_loss_anchored == b.gate.confirm_loss_anchored
            assert a.gate.confirm_loss_ref == b.gate.confirm_loss_ref
            assert np.array_equal(a.predict_survival(te["xg"], te["last"], te["z"], tt),
                                  b.predict_survival(te["xg"], te["last"], te["z"], tt))
            assert np.array_equal(a.predict_risk(te["xg"], te["last"], te["z"]), b.predict_risk(te["xg"], te["last"], te["z"]))
