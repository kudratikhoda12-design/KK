"""Round 15 (D133): deterministic checks of the Phase-2 code (no formal seed is used).

R15-1  domain guard: equal to the unguarded formula on the training range (<= 5e-6 x floor for log/sqrt/1/x), finite
       for any affine parameters, absent for formulas without a domain restriction
R15-2  backward compatibility: refit_symbolic without an anchor and substitute_symbolic without guard are the
       Round-11 procedures (identical parameters after a refit)
R15-3  functional distillation: anchor_terms reproduce the B2-S surfaces; a very large mu keeps the refitted surfaces at
       the anchor, mu = 0 equals the plain refit
R15-4  b2s_res_cox: residual inputs; the "all off" gate gives a model whose residual layer contributes nothing
R15-5  afcm_ens: the bagged model's linear predictor equals the mean of its resample models' linear predictors
R15-6  fsim S-Last: eta is the stated function of the curves at t = 1; the other scenarios are unchanged
"""
import copy

import numpy as np
import pytest
import torch

from kanfae.data import fsim
from kanfae.models import afcm as A
from kanfae.models.fkan import FKANSurv, SymbolicEdge, anchor_terms, domain_guard, substitute_symbolic, symbolic_edges
from kanfae.models.fres import ALL_OFF, ResFKAN, _gate, residual_pack

pytestmark = [pytest.mark.stage("R15"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)


def _toy(n=160, seed=5, D=2, G=50):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, G)
    xg = rng.normal(size=(n, 1, D)) + 0.5 * rng.normal(size=(n, 1, D)) * t[None, :, None]
    z = rng.normal(size=(n, 1))
    eta = np.sin(xg[:, :, 0].mean(1)) + 0.5 * z[:, 0]
    T = rng.exponential(np.exp(-eta))
    C = rng.exponential(2.0, n)
    return dict(xg=xg, z=z, y=np.minimum(T, C), d=(T <= C).astype(float))


def _split(p, a, b):
    return {k: (v[a:b] if v is not None else None) for k, v in p.items()}


@pytest.mark.criterion("R15-1")
def test_R15_1_domain_guard():
    x = np.linspace(0.5, 4.0, 300)
    for f, a, b in [("log", 0.9, -0.2), ("sqrt", 1.1, 0.3), ("1/x", -0.7, -0.1)]:
        g = domain_guard(f, a, b, x)
        e0, e1 = SymbolicEdge(f, a, b, 1.2, 0.1), SymbolicEdge(f, a, b, 1.2, 0.1, guard=g)
        with torch.no_grad():
            u = a * x + b
            dev = float((e0(torch.tensor(x)) - e1(torch.tensor(x))).abs().max())
            assert dev <= 1.2 * 5e-6 * g[1] * max(1.0, float(np.max(np.abs(1.0 / np.minimum(np.abs(u), 1e9)))) ** 2) + 1e-12
            e1.p[0] = -5.0
            e1.p[1] = -3.0
            assert torch.isfinite(e1(torch.tensor(np.linspace(-100, 100, 201)))).all()
    assert domain_guard("sin", 1.0, 0.0, x) is None and domain_guard("tanh", 1.0, 0.0, x) is None
    assert domain_guard("log", 1.0, -1.0, x) is None                 # training arguments not all positive: unguarded
    ge = domain_guard("exp", 1.0, 0.0, x)
    e = SymbolicEdge("exp", 1.0, 0.0, 1.0, 0.0, guard=ge)
    with torch.no_grad():
        assert torch.isfinite(e(torch.tensor([1e6]))).all()


def _fitted(seed=0):
    p = _toy()
    tr, va = _split(p, 0, 110), _split(p, 110, 160)
    m = FKANSurv(kind="b2s", loss="cox", max_epochs=60, patience=60, seed=seed, lr=1e-2).fit(tr, va)
    return m


@pytest.mark.criterion("R15-2")
def test_R15_2_backward_compatible_refit_and_substitution():
    m = _fitted()
    recs = symbolic_edges(m)
    a, b = copy.deepcopy(m), copy.deepcopy(m)
    substitute_symbolic(a, recs)
    substitute_symbolic(b, recs, guard=False)
    for la, lb in ((a.net.g, b.net.g), (a.net.h, b.net.h)):
        for k in la.symbolic:
            assert la.symbolic[k].guard is None and torch.equal(la.symbolic[k].p, lb.symbolic[k].p)
    a.refit_symbolic(max_iter=20)
    b.refit_symbolic(max_iter=20, anchor=None, mu=0.0)
    for qa, qb in zip(a._params(), b._params()):
        assert torch.equal(qa, qb)


@pytest.mark.criterion("R15-3")
def test_R15_3_distillation_anchor():
    m = _fitted()
    X, Z = m.data["Xtr"], m.data["Ztr"]
    terms = anchor_terms(m.net, X, Z)
    G = X.shape[1]
    xs = m.data["Xtr"][:, :, 0].reshape(-1).numpy() * m.xs[0] + m.xm[0]
    ts = np.tile(np.linspace(0, 1, G), X.shape[0])
    assert np.allclose(terms[0].detach().numpy(), m.effect(0, xs, ts), atol=1e-10)
    recs = symbolic_edges(m)
    sub = copy.deepcopy(m)
    substitute_symbolic(sub, recs, guard=True)
    big = copy.deepcopy(sub).refit_symbolic(max_iter=30, anchor=m, mu=1e6)
    free = copy.deepcopy(sub).refit_symbolic(max_iter=30)
    zero = copy.deepcopy(sub).refit_symbolic(max_iter=30, anchor=m, mu=0.0)
    with torch.no_grad():
        ref = anchor_terms(m.net, X, Z)
        d_big = sum(float(((c - r) ** 2).mean()) for c, r in zip(anchor_terms(big.net, X, Z), ref))
        d_free = sum(float(((c - r) ** 2).mean()) for c, r in zip(anchor_terms(free.net, X, Z), ref))
        d_sub = sum(float(((c - r) ** 2).mean()) for c, r in zip(anchor_terms(sub.net, X, Z), ref))
    assert d_big <= d_sub + 1e-9 and d_big <= d_free + 1e-9
    for qa, qb in zip(free._params(), zero._params()):
        assert torch.equal(qa, qb)


@pytest.mark.criterion("R15-4")
def test_R15_4_residual_model_and_gate():
    p = _toy(D=2)
    last = p["xg"][:, -1, :] + 0.01
    pk = dict(p, last=last)
    r = residual_pack(pk)
    assert np.allclose(r["xg"], p["xg"] - last[:, None, :]) and np.allclose(r["z"][:, :2], last)
    tr, va = residual_pack(dict(_split(p, 0, 110), last=last[:110])), residual_pack(dict(_split(p, 110, 160), last=last[110:]))
    m = FKANSurv(kind="b2s", loss="cox", max_epochs=40, patience=40, seed=0, lr=1e-2).fit(tr, va)
    off = _gate(m, ALL_OFF, tr, 5)
    assert float(off.net.g.mask.sum()) == 0.0
    with torch.no_grad():
        assert float(off.net.channel_terms(off.data["Xtr"]).abs().max()) == 0.0
    rf = ResFKAN(lrs=(1e-2,), wds=(1e-4,), seeds=(0, 1), fine_tune_epochs=5, max_epochs=30, patience=30)
    rf.fit(dict(_split(p, 0, 110), last=last[:110]), dict(_split(p, 110, 160), last=last[110:]))
    assert sum(g.get("chosen", False) for g in rf.gate_grid) == 1 and len(rf.members) == 2
    e = rf.eta(p["xg"][:5], last[:5], p["z"][:5])
    assert e.shape == (5,) and np.all(np.isfinite(e))


@pytest.mark.criterion("R15-5")
def test_R15_5_bagged_afcm_is_the_mean_of_its_resample_models():
    d = fsim.generate("S-Add", n=240, seed=7, dgp="v2")
    xg = d.x_true_grid
    pk = lambda a, b: dict(xg=xg[a:b], z=d.z[a:b], y=d.y[a:b], d=d.delta[a:b].astype(float))  # noqa: E731
    kw = dict(lams_x=(1e-2,), lams_t=(1.0,), pen_norm=True, t_pen="d1d2")
    m, recs = A.fit_afcm_bagged(pk(0, 160), pk(160, 240), B=2, seed=3, **kw)
    assert len(recs) == 2
    proto = A.AFCM(pen_norm=True, t_pen="d1d2")
    proto._set_ranges(xg[:160])
    proto.zm, proto.zs = d.z[:160].mean(0), np.where(d.z[:160].std(0) > 0, d.z[:160].std(0), 1.0)
    W, Zt = proto.design(xg[:160]), proto._z(d.z[:160], 160)
    rng = np.random.default_rng(3)
    etas = []
    for _ in range(2):
        idx = rng.integers(0, 160, 160)
        mb = A._fit_one(proto, dict(pen_norm=True, t_pen="d1d2"), 1e-2, 1.0, W[idx], Zt[idx],
                        dict(y=d.y[:160][idx], d=d.delta[:160][idx].astype(float)))
        etas.append(mb.eta(xg[160:], d.z[160:]))
    assert np.allclose(m.eta(xg[160:], d.z[160:]), np.mean(etas, 0), atol=1e-10)


@pytest.mark.criterion("R15-6")
def test_R15_6_s_last_scenario():
    d = fsim.generate("S-Last", n=300, seed=8, sparse=True)
    x1 = d.x_true_grid[:, -1, :]
    assert np.allclose(d.eta, np.sin(np.pi * x1[:, 0] / 2) + 1.5 * np.tanh(x1[:, 1]) + 0.3 * d.z[:, 0] - 0.3 * d.z[:, 1])
    assert all(t.max() < 1.0 for t in d.times)
    a = fsim.generate("S-Add", n=50, seed=3000)
    assert np.allclose(a.y[:3], [0.71610892, 0.36745362, 0.40277489])


@pytest.mark.criterion("R15-7")
def test_R15_7_laml_information_and_selection():
    from kanfae.survival.cox import cox_nll
    rng = np.random.default_rng(0)
    n, q = 60, 4
    X = rng.normal(size=(n, q))
    y = np.round(rng.exponential(size=n), 1) + 0.1
    d = rng.random(n) < 0.7
    b = rng.normal(size=q) * 0.3
    H = A.cox_information(X, X @ b, y, d)
    f = lambda bb: cox_nll(torch.tensor(X) @ bb, torch.tensor(y), torch.tensor(d.astype(float)), "breslow", reduction="sum")  # noqa: E731
    assert np.allclose(H, torch.autograd.functional.hessian(f, torch.tensor(b)).numpy(), atol=1e-10)
    dat = fsim.generate("S-Add", n=240, seed=7, dgp="v2")
    pk = dict(xg=dat.x_true_grid[:160], z=dat.z[:160], y=dat.y[:160], d=dat.delta[:160].astype(float))
    m, recs = A.fit_afcm_laml(pk, lams_x=(1e-3, 1e-1), lams_t=(1e-2, 1.0), pen_norm=True, t_pen="d1d2")
    assert sum(r.get("chosen", False) for r in recs) == 1 and all(np.isfinite(r["laml"]) for r in recs)
    ch = [r for r in recs if r.get("chosen")][0]
    assert ch["laml"] == max(r["laml"] for r in recs) and 0 < ch["edf"] < 2 * 100 + 2


@pytest.mark.criterion("R15-8")
def test_R15_8_cross_validated_gate():
    p = _toy(D=2)
    last = p["xg"][:, -1, :] + 0.01
    rf = ResFKAN(lrs=(1e-2,), wds=(1e-4,), seeds=(0, 1), fine_tune_epochs=5, max_epochs=30, patience=30,
                 residual=False, gate="cv", cv_folds=3)
    rf.fit(dict(_split(p, 0, 110), last=last[:110]), dict(_split(p, 110, 160), last=last[110:]))
    assert all("cv_loss" in g and np.isfinite(g["cv_loss"]) for g in rf.gate_grid)
    assert sum(g.get("chosen", False) for g in rf.gate_grid) == 1 and len(rf.members) == 2
