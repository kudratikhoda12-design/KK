"""Round 18 (D144): deterministic checks of the confirmation-split gate (no formal seed is used).

R18-1  val_confirm is never used before the final decision: changing every confirmation outcome and covariate leaves
       the comparator, every candidate's validation loss, the step-(1) anchored choice and its parameters bit-identical
R18-2  when the confirmation check does not pass, the estimator returns the comparator's survival matrix and risks
       exactly (max |diff| == 0.0), for Cox and log-normal AFT
R18-3  the confirmation losses are the unpenalised held-out losses of the step-(1) candidate and of the comparator's
       seed-0 member on val_confirm, and the decision is loss_anchored < loss_comparator - last_tol
R18-4  confirm=None (default) is the Round-17 rule: same grid and the same decision as choose_twoway on val_loss
"""
import numpy as np
import pytest
import torch

from kanfae.models.fkan import FKANEnsemble
from kanfae.models.fres import ResFKAN, choose_twoway, heldout_loss, residual_pack
from kanfae.survival.aft import aft_nll
from kanfae.survival.cox import cox_nll

pytestmark = [pytest.mark.stage("R18"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)
KW = dict(max_epochs=40, patience=40, lrs=(1e-2,), wds=(1e-4,), seeds=(0, 1), fine_tune_epochs=10)


def _data(n=200, seed=4):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 50)
    xg = rng.normal(size=(n, 1, 2)) + 0.5 * rng.normal(size=(n, 1, 2)) * t[None, :, None]
    last = xg[:, -1, :]
    z = rng.normal(size=(n, 1))
    eta = np.tanh(last[:, 0]) + 0.5 * np.sin(xg[:, :, 1].mean(1)) + 0.3 * z[:, 0]
    T, C = rng.exponential(np.exp(-eta)), rng.exponential(2.0, n)
    p = dict(xg=xg, last=last, z=z, y=np.minimum(T, C), d=(T <= C).astype(float))
    return ({k: v[:100] for k, v in p.items()}, {k: v[100:150] for k, v in p.items()},
            {k: v[150:] for k, v in p.items()})


def _plain(p):
    return {k: p[k] for k in ("xg", "z", "y", "d")}


def _fit(loss, last_tol, confirm, ref=None, tr_va=None):
    tr, va, _ = tr_va or _data()
    fam = dict(family="lognormal") if loss == "aft" else {}
    if ref is None:
        ref = FKANEnsemble(lrs=KW["lrs"], wds=KW["wds"], seeds=KW["seeds"], kind="b2s", loss=loss, max_epochs=40,
                           patience=40, **fam).fit(_plain(tr), _plain(va))
    r = ResFKAN(residual=False, gate_last_value=True, last_tol=last_tol, loss=loss, **KW, **fam).fit(tr, va, ref=ref,
                                                                                                       confirm=confirm)
    return r, ref


@pytest.mark.criterion("R18-1")
def test_R18_1_confirmation_data_untouched_before_decision():
    tr, va, cf = _data()
    rng = np.random.default_rng(9)
    cf2 = dict(xg=cf["xg"][::-1] * 3.0 + 1.0, last=cf["last"][::-1] * 3.0, z=rng.normal(size=cf["z"].shape),
               y=rng.exponential(5.0, len(cf["y"])), d=1.0 - cf["d"])
    r1, ref1 = _fit("cox", 0.0, cf, tr_va=(tr, va, cf))
    r2, ref2 = _fit("cox", 0.0, cf2, tr_va=(tr, va, cf2))
    assert all(torch.equal(a, b) for m1, m2 in zip(ref1.members, ref2.members)
               for a, b in zip(m1.net.state_dict().values(), m2.net.state_dict().values()))
    assert [g["val_loss"] for g in r1.gate_grid] == [g["val_loss"] for g in r2.gate_grid]
    assert r1.anchored_choice == r2.anchored_choice
    assert all(torch.equal(a, b) for a, b in zip(r1._cand_anch.net.state_dict().values(), r2._cand_anch.net.state_dict().values()))
    assert r1.ref_loss == r2.ref_loss and r1.confirm_loss_ref != r2.confirm_loss_ref       # only the decision inputs differ


@pytest.mark.criterion("R18-2")
@pytest.mark.parametrize("loss", ["cox", "aft"])
def test_R18_2_exact_reduction_when_confirmation_fails(loss):
    tr, va, cf = _data()
    r, ref = _fit(loss, 1e6, cf, tr_va=(tr, va, cf))
    tt = np.array([0.05, 0.3, 1.0, 3.0])
    assert r.reduced and r.members is None
    S, Sr = r.predict_survival(cf["xg"], cf["last"], cf["z"], tt), ref.predict_survival(cf["xg"], cf["z"], tt)
    assert float(np.max(np.abs(S - Sr))) == 0.0
    assert float(np.max(np.abs(r.predict_risk(cf["xg"], cf["last"], cf["z"]) - ref.predict_risk(cf["xg"], cf["z"])))) == 0.0


@pytest.mark.criterion("R18-3")
@pytest.mark.parametrize("loss", ["cox", "aft"])
def test_R18_3_confirmation_losses_and_rule(loss):
    tr, va, cf = _data()
    r, ref = _fit(loss, 0.0, cf, tr_va=(tr, va, cf))
    m, m0 = r._cand_anch, ref.members[0]
    pk = residual_pack(cf, False)
    X, Z = m._tensors(pk["xg"], pk["z"])
    X0, Z0 = m0._tensors(cf["xg"], cf["z"])
    with torch.no_grad():
        if loss == "cox":
            la = float(cox_nll(m.net(X, Z), torch.tensor(cf["y"]), torch.tensor(cf["d"]), "efron"))
            lr = float(cox_nll(m0.net(X0, Z0), torch.tensor(cf["y"]), torch.tensor(cf["d"]), "efron"))
        else:
            sg = lambda q: torch.nn.functional.softplus(q.log_sigma) + 1e-3  # noqa: E731
            la = float(aft_nll(torch.tensor(cf["y"]), torch.tensor(cf["d"]), m.mu + m.net(X, Z), sg(m), "lognormal"))
            lr = float(aft_nll(torch.tensor(cf["y"]), torch.tensor(cf["d"]), m0.mu + m0.net(X0, Z0), sg(m0), "lognormal"))
    assert r.confirm_loss_anchored == la and r.confirm_loss_ref == lr
    assert heldout_loss(m0, _plain(cf)) == lr
    assert r.reduced == (not la < lr)
    r_yes, _ = _fit(loss, -1e6, cf, ref=ref, tr_va=(tr, va, cf))
    assert not r_yes.reduced and len(r_yes.members) == 2


@pytest.mark.criterion("R18-4")
def test_R18_4_default_is_round17_rule():
    tr, va, cf = _data()
    r, ref = _fit("cox", 0.01, None, tr_va=(tr, va, cf))
    assert not hasattr(r, "confirm_loss_anchored")
    k = choose_twoway(r.gate_grid, 0.01, 0.01, float(ref.members[0].val_loss))
    assert r.reduced == (k is None)
