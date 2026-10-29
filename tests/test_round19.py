"""Round 19 (D147): deterministic checks of the confirmation-split decision with a full-validation estimator
(no formal seed is used).

R19-1  val_confirm is untouched before the step-(2) decision: changing every confirmation outcome and covariate leaves
       the val_select comparator, the candidates, the step-(1) choice and its parameters bit-identical
R19-2  when anchoring is not confirmed, the estimator returns the full-validation B2-S survival matrix and risks exactly
       (max |diff| == 0.0), for Cox and log-normal AFT
R19-3  when anchoring is confirmed, the refit early-stops on the whole validation set (val_select + val_confirm), trains
       on the training split only, and uses the chosen masks on every member; the test split is not an input
R19-4  both refit variants ("retrain", "finetune") give finite survival in [0, 1]; "retrain" tunes a new ensemble on the
       whole validation set
"""
import numpy as np
import pytest
import torch

from kanfae.models.fkan import FKANEnsemble
from kanfae.models.fres import ConfirmRefitAnchor

pytestmark = [pytest.mark.stage("R19"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)
KW = dict(max_epochs=40, patience=40, lrs=(1e-2,), wds=(1e-4,), seeds=(0, 1))


def _data(n=200, seed=6):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 50)
    xg = rng.normal(size=(n, 1, 2)) + 0.5 * rng.normal(size=(n, 1, 2)) * t[None, :, None]
    last = xg[:, -1, :]
    z = rng.normal(size=(n, 1))
    eta = np.tanh(last[:, 0]) + 0.5 * np.sin(xg[:, :, 1].mean(1)) + 0.3 * z[:, 0]
    T, C = rng.exponential(np.exp(-eta)), rng.exponential(2.0, n)
    p = dict(xg=xg, last=last, z=z, y=np.minimum(T, C), d=(T <= C).astype(float))
    sl = (slice(0, 100), slice(100, 140), slice(140, 180), slice(180, 200))
    return [{k: v[s] for k, v in p.items()} for s in sl]                # train, val_select, val_confirm, test


def _plain(p):
    return {k: p[k] for k in ("xg", "z", "y", "d")}


def _refs(loss, tr, vs, vc):
    fam = dict(family="lognormal") if loss == "aft" else {}
    mk = lambda: FKANEnsemble(lrs=KW["lrs"], wds=KW["wds"], seeds=KW["seeds"], kind="b2s", loss=loss,  # noqa: E731
                              max_epochs=40, patience=40, **fam)
    vf = {k: np.concatenate([vs[k], vc[k]]) for k in ("xg", "z", "y", "d")}
    return mk().fit(_plain(tr), _plain(vs)), mk().fit(_plain(tr), vf), fam


def _fit(loss, margin, refit, tr, vs, vc, refs=None):
    rs, rf, fam = refs or _refs(loss, tr, vs, vc)
    m = ConfirmRefitAnchor(loss=loss, margin=margin, refit=refit, fine_tune_epochs=10, **KW, **fam).fit(tr, vs, vc, rs, rf)
    return m, rs, rf


@pytest.mark.criterion("R19-1")
def test_R19_1_confirmation_untouched_before_decision():
    tr, vs, vc, _ = _data()
    rng = np.random.default_rng(3)
    vc2 = dict(xg=vc["xg"][::-1] * 2.0 - 1.0, last=vc["last"][::-1] * 2.0, z=rng.normal(size=vc["z"].shape),
               y=rng.exponential(4.0, len(vc["y"])), d=1.0 - vc["d"])
    rs, _, fam = _refs("cox", tr, vs, vc)
    a = ConfirmRefitAnchor(loss="cox", margin=1e6, fine_tune_epochs=10, **KW).fit(tr, vs, vc, rs, rs)
    b = ConfirmRefitAnchor(loss="cox", margin=1e6, fine_tune_epochs=10, **KW).fit(tr, vs, vc2, rs, rs)
    ga, gb = a.gate, b.gate
    assert [g["val_loss"] for g in ga.gate_grid] == [g["val_loss"] for g in gb.gate_grid]
    assert ga.anchored_choice == gb.anchored_choice and ga.ref_loss == gb.ref_loss
    assert all(torch.equal(x, y) for x, y in zip(ga._cand_anch.net.state_dict().values(), gb._cand_anch.net.state_dict().values()))
    assert ga.confirm_loss_ref != gb.confirm_loss_ref


@pytest.mark.criterion("R19-2")
@pytest.mark.parametrize("loss", ["cox", "aft"])
def test_R19_2_not_confirmed_returns_full_validation_b2s(loss):
    tr, vs, vc, te = _data()
    m, rs, rf = _fit(loss, 1e6, "retrain", tr, vs, vc)
    tt = np.array([0.05, 0.3, 1.0, 3.0])
    assert m.reduced and m.members is None
    assert float(np.max(np.abs(m.predict_survival(te["xg"], te["last"], te["z"], tt) - rf.predict_survival(te["xg"], te["z"], tt)))) == 0.0
    assert float(np.max(np.abs(m.predict_risk(te["xg"], te["last"], te["z"]) - rf.predict_risk(te["xg"], te["z"])))) == 0.0
    assert float(np.max(np.abs(rf.predict_survival(te["xg"], te["z"], tt) - rs.predict_survival(te["xg"], te["z"], tt)))) > 0.0


@pytest.mark.criterion("R19-3")
def test_R19_3_refit_uses_whole_validation_set():
    tr, vs, vc, te = _data()
    m, _, _ = _fit("cox", -1e6, "retrain", tr, vs, vc)
    assert not m.reduced and len(m.members) == len(KW["seeds"])
    yfull = np.concatenate([vs["y"], vc["y"]])
    c = m.gate._cand_anch
    for mem in m.members:
        assert torch.equal(mem.data["yva"], torch.tensor(yfull)) and len(mem.data["ytr"]) == len(tr["y"])
        assert torch.equal(mem.net.g.mask, c.net.g.mask) and torch.equal(mem.net.stat.mask, c.net.stat.mask)
    assert len(m.val_full["y"]) == len(vs["y"]) + len(vc["y"])
    assert not any(np.array_equal(te["y"], x) for x in (m.val_full["y"], tr["y"]))


@pytest.mark.criterion("R19-4")
def test_R19_4_refit_variants():
    tr, vs, vc, te = _data()
    refs = _refs("aft", tr, vs, vc)
    tt = np.array([0.1, 0.5, 2.0])
    for how in ("retrain", "finetune"):
        m, _, _ = _fit("aft", -1e6, how, tr, vs, vc, refs=refs)
        S = m.predict_survival(te["xg"], te["last"], te["z"], tt)
        assert np.all(np.isfinite(S)) and np.all((S >= 0) & (S <= 1))
        assert all(torch.equal(mem.data["yva"], torch.tensor(np.concatenate([vs["y"], vc["y"]]))) for mem in m.members)
    other = m.refit_members("retrain")
    assert len(other) == len(KW["seeds"])
