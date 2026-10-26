"""Round 17 (D141): deterministic checks of the two-way gate and the log-normal KAN-AFT prediction (no formal seed).

R17-1  FKANSurv(loss="aft", family="lognormal").predict_survival uses the log-normal survival function (it used the
       Weibull one for every family before D141); the Weibull path is unchanged
R17-2  choose_twoway: parsimonious choice among joint states that keep a trajectory channel; a static state only if
       strictly better (non-cliff); the anchored state only if it beats the comparator by more than last_tol
R17-3  exact reduction: when the comparator wins, ResFKAN(gate_last_value=True) predicts exactly what the comparator
       ensemble predicts (Cox and AFT); anchored_alternative() returns the anchored ensemble with masks on all members
R17-4  gate_last_value=False (default) never touches the two-way path
"""
import numpy as np
import pytest
import torch

from kanfae.models.fkan import FKANEnsemble, FKANSurv
from kanfae.models.fres import ResFKAN, choose_twoway
from kanfae.survival.aft import aft_survival

pytestmark = [pytest.mark.stage("R17"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)
KW = dict(max_epochs=40, patience=40, lrs=(1e-2,), wds=(1e-4,), seeds=(0, 1), fine_tune_epochs=10)


def _data(n=150, seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 50)
    xg = rng.normal(size=(n, 1, 2)) + 0.5 * rng.normal(size=(n, 1, 2)) * t[None, :, None]
    last = xg[:, -1, :]
    z = rng.normal(size=(n, 1))
    eta = np.tanh(last[:, 0]) + 0.5 * np.sin(xg[:, :, 1].mean(1)) + 0.3 * z[:, 0]
    T, C = rng.exponential(np.exp(-eta)), rng.exponential(2.0, n)
    p = dict(xg=xg, last=last, z=z, y=np.minimum(T, C), d=(T <= C).astype(float))
    return {k: v[:100] for k, v in p.items()}, {k: v[100:] for k, v in p.items()}


def _plain(p):
    return {k: p[k] for k in ("xg", "z", "y", "d")}


@pytest.mark.criterion("R17-1")
def test_R17_1_lognormal_survival_prediction():
    tr, va = _data()
    tt = np.array([0.0, 0.2, 0.5, 1.0, 2.0])
    for fam in ("lognormal", "weibull"):
        m = FKANSurv(kind="b2s", loss="aft", family=fam, max_epochs=30, patience=30, lr=1e-2).fit(_plain(tr), _plain(va))
        e = m.eta(va["xg"], va["z"])
        ref = aft_survival(torch.tensor(tt[1:]), torch.tensor(float(m.mu.detach()) + e), m.sigma, fam).numpy()
        S = m.predict_survival(va["xg"], va["z"], tt)
        assert np.allclose(S[:, 1:], ref, atol=1e-12) and np.all(S[:, 0] == 1.0)
        if fam == "weibull":
            w = (np.log(tt[1:])[None, :] - float(m.mu) - e[:, None]) / m.sigma
            assert np.array_equal(S[:, 1:], np.exp(-np.exp(w)))                  # bit-identical to the old formula


@pytest.mark.criterion("R17-2")
def test_R17_2_two_way_rule():
    J = lambda th, v, on=(1, 1): dict(kind="joint", theta=th, val_loss=v, traj_on=list(on))  # noqa: E731
    S = lambda th, v: dict(kind="static", theta=th, val_loss=v, traj_on=[0, 0])  # noqa: E731
    g = [J(0.0, 4.300), J(0.1, 4.305), J(0.25, 4.320), S(0.0, 4.302), S(0.1, 4.301)]
    assert choose_twoway(g, 0.01) == 1                                   # parsimonious within tol; static not better
    g2 = g[:3] + [S(0.0, 4.299), S(0.1, 4.299)]
    assert choose_twoway(g2, 0.01) == 4                                  # static strictly better; tie -> larger theta
    assert choose_twoway(g, 0.01, last_tol=0.0, ref_loss=4.305) is None  # not strictly below the comparator
    assert choose_twoway(g, 0.01, last_tol=0.0, ref_loss=4.306) == 1
    assert choose_twoway(g, 0.01, last_tol=0.002, ref_loss=4.306) is None


@pytest.mark.criterion("R17-3")
@pytest.mark.parametrize("loss", ["cox", "aft"])
def test_R17_3_exact_reduction_and_alternative(loss):
    tr, va = _data()
    fam = dict(family="lognormal") if loss == "aft" else {}
    ens_kw = dict(lrs=KW["lrs"], wds=KW["wds"], seeds=KW["seeds"], kind="b2s", loss=loss, max_epochs=40, patience=40, **fam)
    ref = FKANEnsemble(**ens_kw).fit(_plain(tr), _plain(va))
    r = ResFKAN(residual=False, gate_last_value=True, last_tol=1e6, loss=loss, **KW, **fam).fit(tr, va, ref=ref)
    tt = np.array([0.1, 0.5, 1.0])
    assert r.reduced and r.residual_on == [0, 0] and r.last_on == [0, 0]
    assert np.array_equal(r.predict_survival(va["xg"], va["last"], va["z"], tt), ref.predict_survival(va["xg"], va["z"], tt))
    assert np.array_equal(r.predict_risk(va["xg"], va["last"], va["z"]), ref.predict_risk(va["xg"], va["z"]))
    assert {g["kind"] for g in r.gate_grid} == {"joint", "static"} and len(r.gate_grid) == 16
    assert 1 <= r.n_fine_tunes <= 16
    alt = r.anchored_alternative()
    assert not alt.reduced and len(alt.members) == 2
    gm, sm = alt.members[0].net.g.mask, alt.members[0].net.stat.mask
    assert all(torch.equal(m.net.g.mask, gm) and torch.equal(m.net.stat.mask, sm) for m in alt.members)
    S = alt.predict_survival(va["xg"], va["last"], va["z"], tt)
    assert np.all(np.isfinite(S)) and np.all((S >= 0) & (S <= 1))
    r0 = ResFKAN(residual=False, gate_last_value=True, last_tol=-1e6, loss=loss, **KW, **fam).fit(tr, va, ref=ref)
    assert not r0.reduced and r0.anchored_choice["index"] == next(i for i, g in enumerate(r0.gate_grid) if g.get("chosen"))


@pytest.mark.criterion("R17-4")
def test_R17_4_default_path_unchanged():
    tr, va = _data()
    r = ResFKAN(residual=False, **KW).fit(tr, va)
    assert not r.gate_last_value and not r.reduced and r.ref is None
    assert all("kind" not in g for g in r.gate_grid) and len(r.gate_grid) == 9
    assert all(m.net.stat.mask[:2, 0].min() > 0 for m in r.members)          # last-value edges never gated
