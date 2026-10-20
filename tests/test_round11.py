"""Round 11 acceptance tests CK-1..CK-12 (D112; docs/design_note_round10.md section 6).

Thresholds are the registered ones; a STAT failure at its registered seed is recorded as FAIL and never converted.
Operational details fixed in this file before any CK run (recorded in D116):
  * STAT splits: random permutation with the test seed, 60/20/20 train/validation/test (CK-3: 80/20 train/validation).
  * Dense curves X_hat: per-subject penalised cubic B-spline smoothing (12 basis functions) evaluated on the 50-point grid,
    lambda by held-out-point CV on the training subjects (kanfae.functional.smoothing.select_lambda).
  * RISE (CK-5, CK-6): `fsim.rise`, evaluated at the TRUE test-split curves on the grid.
  * CK-3: data z ~ N(0, I_3), beta = (0.8, -0.5, 0.3), Weibull PH (shape 1.5), censoring U(0, 3); static 1-layer CoxKAN
    whose edges are linear in original units (edge="linear"; lambda 0, lr 3e-3, wd 1e-4, seed 0, the standard training
    loop with early stopping on the validation split); slopes vs lifelines on the training split. (Revised before any
    seed-11 run: requiring stage-1 symbolic acceptance of spline edges tested the symbolic stage, not the Cox machinery;
    D116.)
  * CK-5: FKANEnsemble(kind "b2s", Cox), the D112 protocol (lr x wd tuning, 3 seeds). Both phi_1 and phi_2 RISE < 0.10.
  * CK-6: fit_afcm (5 x 5 lambda grid) on S-Add; the known F is S-Add's phi_1 and phi_2; both RISE < 0.10.
  * CK-7: all three channels of S-Add sparse curves (all 1,000 subjects): |cos(first eigenfunction, sqrt2 sin 2 pi t)|
    > 0.95 on every channel, and mean ISE over subjects and channels (on the grid, against the true curves) lower than
    per-subject B-spline smoothing (12 basis functions, lambda by held-out-point CV).
  * CK-8: B2-S Cox (lr 3e-3, wd 1e-4, seed 0) through select_lambda_theta (full D112 grid); selected channels == [1, 1, 0].
  * CK-10: a KAN edge (grid 5, range [-2, 2]) fitted by least squares to sin, tanh and x^2 on [-2, 2]; fit_symbolic
    without PySR must accept (stage linear/library, validation R^2 > 0.99) a formula of the right family; exact
    identities count as the same family: sin ~ cos (phase shift), tanh ~ sigmoid (tanh u = 2 sigmoid(2u) - 1).
  * CK-11: S-Misspec-lognormal, B2-S KAN-AFT (Weibull loss; lr 3e-3, wd 1e-4, seed 0); D-calibration (10 bins) on the
    test split with and without residual calibration of the same fitted network.
STAT results are also written to reports/logs/round11_ck.jsonl for the report.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from lifelines import CoxPHFitter

from kanfae.data import fsim
from kanfae.data.landmark import build_stacked
from kanfae.functional.basis import BSplineBasis
from kanfae.functional.pace import PACE
from kanfae.functional.smoothing import curves_on_grid, pad_ragged, select_lambda
from kanfae.models.afcm import fit_afcm
from kanfae.models.fkan import FKANEnsemble, FKANNet, FKANSurv, select_lambda_theta, substitute_symbolic, symbolic_edges
from kanfae.models.kan import KANLayer
from kanfae.models.lme import blup
from kanfae.survival.cox import breslow_baseline, cox_nll
from kanfae.evaluation.metrics import d_calibration
from kanfae.symbolic.fit import fit_symbolic

pytestmark = [pytest.mark.stage("R11")]
torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "logs" / "round11_ck.jsonl"


def _record(ck, **kw):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(ck=ck, run_id=os.environ.get("KANFAE_RUN_ID", time.strftime("%Y%m%dT%H%M%S")), **kw)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def _split(n, seed, fr=(0.6, 0.2)):
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(fr[0] * n), int((fr[0] + fr[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def _smoothed(dat, tr_idx):
    basis = BSplineBasis(12, 3, (0.0, 1.0))
    T, V, M, n_obs = pad_ragged([dat.times[i] for i in tr_idx], [dat.values[i] for i in tr_idx])
    lam, _ = select_lambda(T, V, M, n_obs, basis, seed=0)
    return curves_on_grid(dat.times, dat.values, fsim.GRID, basis, lam), lam


def _pack(dat, xg, idx):
    return dict(xg=xg[idx], z=dat.z[idx], y=dat.y[idx], d=dat.delta[idx].astype(float))


def _rises(model, X):
    return [fsim.rise(lambda x, t, d=d: model.effect(d, x, t), lambda x, t, d=d: fsim.true_effect(d, x, t), X[:, :, d])
            for d in (0, 1)]


# ---------------------------------------------------------------------------------------------------------------- DET
def _tied_data(seed=5, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    beta = np.array([0.7, -0.4])
    T = np.ceil(rng.exponential(1 / np.exp(X @ beta)) * 10) / 10            # heavy ties
    C = np.ceil(rng.uniform(0, 2, n) * 10) / 10
    return X, beta, np.minimum(T, C), (T <= C)


@pytest.mark.criterion("CK-1")
@pytest.mark.kind("DET")
def test_CK1_efron_breslow_match_lifelines_and_direct_formula():
    X, beta, y, d = _tied_data()
    eta = X @ beta
    # direct Breslow
    br = -sum(eta[i] - np.log(np.exp(eta[y >= y[i]]).sum()) for i in np.nonzero(d)[0])
    ours_b = float(cox_nll(torch.tensor(eta), torch.tensor(y), torch.tensor(d), "breslow", reduction="sum"))
    assert abs(ours_b - br) < 1e-8
    # lifelines (Efron) at its own MLE: our Efron NLPL at the same beta equals -log_likelihood_
    df = pd.DataFrame(X, columns=["a", "b"]).assign(T=y, E=d.astype(int))
    cph = CoxPHFitter().fit(df, "T", "E")
    ours_e = float(cox_nll(torch.tensor(X @ cph.params_.values), torch.tensor(y), torch.tensor(d), "efron", reduction="sum"))
    assert abs(ours_e - (-cph.log_likelihood_)) < 1e-8
    # and a direct Efron formula
    ef = 0.0
    for u in np.unique(y[d]):
        D_ = (y == u) & d
        R, E, m = np.exp(eta[y >= u]).sum(), np.exp(eta[D_]).sum(), D_.sum()
        ef -= eta[D_].sum() - sum(np.log(R - l / m * E) for l in range(m))
    ours_e2 = float(cox_nll(torch.tensor(eta), torch.tensor(y), torch.tensor(d), "efron", reduction="sum"))
    assert abs(ours_e2 - ef) < 1e-8


@pytest.mark.criterion("CK-2")
@pytest.mark.kind("DET")
def test_CK2_breslow_baseline_matches_lifelines():
    X, _, y, d = _tied_data()
    df = pd.DataFrame(X, columns=["a", "b"]).assign(T=y, E=d.astype(int))
    cph = CoxPHFitter().fit(df, "T", "E")
    eta = (X - X.mean(0)) @ cph.params_.values                              # lifelines centres the covariates
    ut, H = breslow_baseline(eta, y, d)[None]
    ref = cph.baseline_cumulative_hazard_
    assert np.max(np.abs(H - ref.loc[ut].values[:, 0])) < 1e-6


@pytest.mark.criterion("CK-4")
@pytest.mark.kind("DET")
def test_CK4_functional_quadrature_identity_edge():
    net = FKANNet("b2h", D=3, p=0, seed=0)
    for d in range(3):
        net.phi.symbolic[(d, 0)] = lambda x: x
    t = fsim.GRID
    curves = np.stack([t ** 2, np.exp(t), np.sin(3 * t)], -1)[None]            # (1, G, 3)
    exact = np.array([1 / 3, np.e - 1, (1 - np.cos(3.0)) / 3])
    with torch.no_grad():
        got = net.channel_terms(torch.tensor(curves))[0].numpy()
    assert np.max(np.abs(got - exact)) < 1e-3


@pytest.mark.criterion("CK-9")
@pytest.mark.kind("DET")
def test_CK9_stacked_landmark_builder():
    rng = np.random.default_rng(3)
    n = 60
    y = rng.uniform(0.5, 10, n)
    d = rng.random(n) < 0.6
    times = [np.sort(rng.uniform(0, yi, rng.integers(1, 9))) for yi in y]
    values = [rng.normal(size=(len(t), 2)) for t in times]
    z = rng.normal(size=(n, 1))
    L, W, w = (2.0, 4.0, 6.0), 3.0, 2.0
    st = build_stacked(times, values, z, y, d, L, W, w)
    for k in range(len(st.y)):
        tL, i = st.landmark[k], st.sid[k]
        assert np.all(st.times_abs[k] <= tL) and np.all(st.times_abs[k] >= tL - W)
        assert np.all((st.times[k] >= 0) & (st.times[k] <= 1))
        assert y[i] > tL                                                         # at risk at the landmark
        assert st.y[k] <= w + 1e-12
        if y[i] > tL + w:
            assert not st.d[k] and abs(st.y[k] - w) < 1e-12                      # administratively censored
        else:
            assert abs(st.y[k] - (y[i] - tL)) < 1e-12 and st.d[k] == d[i]
    for tL in L:                                                                 # every at-risk subject with history
        expect = {i for i in range(n) if y[i] > tL and np.any((times[i] >= tL - W) & (times[i] <= tL))}
        assert set(st.sid[st.landmark == tL]) == expect


@pytest.mark.criterion("CK-10")
@pytest.mark.kind("DET")
@pytest.mark.parametrize("target,family", [("sin", {"sin", "cos"}), ("tanh", {"tanh", "sigmoid"}), ("x^2", {"x^2"})])
def test_CK10_symbolic_stages_recover_edges(target, family):
    f = {"sin": np.sin, "tanh": np.tanh, "x^2": np.square}[target]
    torch.manual_seed(0)
    layer = KANLayer(1, 1, 5, grid_range=(-2.0, 2.0), seed=0)
    x = torch.linspace(-2, 2, 401)[:, None]
    y = torch.tensor(f(x.numpy()[:, 0]))
    opt = torch.optim.LBFGS(layer.parameters(), lr=1.0, max_iter=2000, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((layer.edge_activations(x)[:, 0, 0] - y) ** 2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    def func(g):
        with torch.no_grad():
            return layer.edge_activations(torch.tensor(np.asarray(g, float))[:, None])[:, 0, 0].numpy()

    ya = func(x.numpy()[:, 0])
    res = fit_symbolic(x.numpy()[:, 0], ya, func, seed=0, use_pysr=False)
    assert res.accepted and res.stage in ("linear", "library") and res.r2_val > 0.99
    assert res.params.get("f") in family


@pytest.mark.criterion("CK-12")
@pytest.mark.kind("DET")
def test_CK12_lme_blup_closed_form():
    rng = np.random.default_rng(4)
    beta, Sigma, s2 = np.array([1.0, -0.5]), np.array([[0.8, 0.2], [0.2, 0.3]]), 0.25
    times = [np.sort(rng.uniform(0, 1, rng.integers(1, 9))) for _ in range(40)]
    values = []
    for t in times:
        u = rng.multivariate_normal(np.zeros(2), Sigma)
        values.append(beta[0] + u[0] + (beta[1] + u[1]) * t + rng.normal(0, np.sqrt(s2), len(t)))
    got = blup(times, values, beta, Sigma, s2)
    for i, (t, y) in enumerate(zip(times, values)):                         # Henderson mixed-model equations
        Z = np.column_stack([np.ones(len(t)), t])
        ref = np.linalg.solve(Z.T @ Z / s2 + np.linalg.inv(Sigma), Z.T @ (y - Z @ beta) / s2)
        assert np.max(np.abs(got[i] - ref)) < 1e-8


# --------------------------------------------------------------------------------------------------------------- STAT
@pytest.mark.criterion("CK-3")
@pytest.mark.kind("STAT")
def test_CK3_linear_coxkan_recovers_lifelines_coefficients():
    rng = np.random.default_rng(11)
    n, beta = 2000, np.array([0.8, -0.5, 0.3])
    Z = rng.normal(size=(n, 3))
    T = (-np.log(rng.random(n)) / np.exp(Z @ beta)) ** (1 / 1.5)
    C = rng.uniform(0, 3, n)
    y, d = np.minimum(T, C), (T <= C).astype(float)
    tr, va = np.split(rng.permutation(n), [1600])
    pk = lambda i: dict(xg=None, z=Z[i], y=y[i], d=d[i])  # noqa: E731
    m = FKANSurv(kind="static", loss="cox", lam=0.0, lr=3e-3, wd=1e-4, seed=0, edge="linear").fit(pk(tr), pk(va))
    slopes = np.array([float(e.p[2] * e.p[0]) for _, e in sorted(m.net.stat.symbolic.items())])
    df = pd.DataFrame(Z[tr], columns=["a", "b", "c"]).assign(T=y[tr], E=d[tr].astype(int))
    ref = CoxPHFitter().fit(df, "T", "E").params_.values
    err = float(np.max(np.abs(slopes - ref)))
    _record("CK-3", slopes=slopes.tolist(), lifelines=ref.tolist(), max_abs_err=err, epochs=len(m.history),
            threshold=0.05)
    assert err < 0.05, f"slopes {slopes}, lifelines {ref}"


@pytest.mark.criterion("CK-10")
@pytest.mark.kind("DET")
def test_CK10b_symbolic_extraction_and_substitution_on_a_coxkan():
    """Plumbing for the model-level symbolic path (stages 1-2 on every edge, substitution, refit): runs and keeps the
    partial likelihood finite. Fixed data (seed 7), short training; not a recovery claim."""
    rng = np.random.default_rng(7)
    Z = rng.normal(size=(300, 2))
    T = rng.exponential(1 / np.exp(np.sin(Z[:, 0]) + 0.5 * Z[:, 1]))
    C = rng.exponential(2.0, 300)
    pk = lambda i: dict(xg=None, z=Z[i], y=np.minimum(T, C)[i], d=(T <= C)[i].astype(float))  # noqa: E731
    m = FKANSurv(kind="static", loss="cox", lr=1e-2, max_epochs=200, seed=0).fit(pk(np.arange(200)), pk(np.arange(200, 300)))
    recs = symbolic_edges(m)
    assert len(recs) == 2 and all(r["result"].stage in ("linear", "library", "none") for r in recs)
    substitute_symbolic(m, recs)
    m.refit_symbolic(max_iter=50)
    assert np.isfinite(m.val_loss)


@pytest.mark.criterion("CK-5")
@pytest.mark.kind("STAT")
def test_CK5_b2s_cox_recovers_s_add_shapes():
    dat = fsim.generate("S-Add", n=2000, seed=12)
    tr, va, te = _split(2000, 12)
    xg, lam = _smoothed(dat, tr)
    e = FKANEnsemble(kind="b2s", loss="cox").fit(_pack(dat, xg, tr), _pack(dat, xg, va))
    r = _rises(e, dat.x_true_grid[te])
    member = [_rises(m, dat.x_true_grid[te]) for m in e.members]
    _record("CK-5", rise_phi1=r[0], rise_phi2=r[1], member_rise=member, tuning=e.tuning, lr=e.lr, wd=e.wd,
            smoothing_lambda=lam, threshold=0.10)
    assert r[0] < 0.10 and r[1] < 0.10, f"RISE phi1 {r[0]:.3f}, phi2 {r[1]:.3f}"


@pytest.mark.criterion("CK-6")
@pytest.mark.kind("STAT")
def test_CK6_afcm_recovers_known_surface():
    dat = fsim.generate("S-Add", n=2000, seed=13)
    tr, va, te = _split(2000, 13)
    xg, lam = _smoothed(dat, tr)
    m, recs = fit_afcm(_pack(dat, xg, tr), _pack(dat, xg, va))
    r = _rises(m, dat.x_true_grid[te])
    _record("CK-6", rise_phi1=r[0], rise_phi2=r[1], lam_x=m.lam_x, lam_t=m.lam_t, grid=recs, smoothing_lambda=lam,
            threshold=0.10)
    assert r[0] < 0.10 and r[1] < 0.10, f"RISE phi1 {r[0]:.3f}, phi2 {r[1]:.3f}"


@pytest.mark.criterion("CK-7")
@pytest.mark.kind("STAT")
def test_CK7_pace_sparse_eigenfunction_and_reconstruction():
    dat = fsim.generate("S-Add", n=1000, seed=14, sparse=True)
    G = fsim.GRID
    w = np.full(len(G), G[1] - G[0]); w[[0, -1]] *= 0.5
    psi1 = np.sqrt(2) * np.sin(2 * np.pi * G)
    basis = BSplineBasis(12, 3, (0.0, 1.0))
    T, V, M, n_obs = pad_ragged(dat.times, dat.values)
    lam, _ = select_lambda(T, V, M, n_obs, basis, seed=0)
    sm = curves_on_grid(dat.times, dat.values, G, basis, lam)
    cos, ise_p, ise_s = [], [], []
    for d in range(3):
        vals = [v[:, d] for v in dat.values]
        p = PACE(seed=0).fit(dat.times, vals)
        cos.append(float(abs((p.phi[:, 0] * psi1 * w).sum()) / np.sqrt((p.phi[:, 0] ** 2 * w).sum() * (psi1 ** 2 * w).sum())))
        rec = p.reconstruct(dat.times, vals)
        ise_p.append(float(np.mean(((rec - dat.x_true_grid[:, :, d]) ** 2 * w).sum(1))))
        ise_s.append(float(np.mean(((sm[:, :, d] - dat.x_true_grid[:, :, d]) ** 2 * w).sum(1))))
    _record("CK-7", abs_cos=cos, ise_pace=ise_p, ise_smooth=ise_s, mean_ise_pace=float(np.mean(ise_p)),
            mean_ise_smooth=float(np.mean(ise_s)), smoothing_lambda=lam, threshold_cos=0.95)
    assert min(cos) > 0.95 and np.mean(ise_p) < np.mean(ise_s), f"|cos| {cos}, ISE {np.mean(ise_p):.3f} vs {np.mean(ise_s):.3f}"


@pytest.mark.criterion("CK-8")
@pytest.mark.kind("STAT")
def test_CK8_pruning_selects_active_channels():
    dat = fsim.generate("S-Add", n=2000, seed=15)
    tr, va, te = _split(2000, 15)
    xg, lam = _smoothed(dat, tr)
    best, recs = select_lambda_theta(lambda l: FKANSurv(kind="b2s", loss="cox", lam=l, lr=3e-3, wd=1e-4, seed=0),
                                     _pack(dat, xg, tr), _pack(dat, xg, va))
    sel = best.selected_channels()
    _record("CK-8", selected=sel, grid=recs, smoothing_lambda=lam)
    assert sel == [1, 1, 0], f"selected channels {sel}"


@pytest.mark.criterion("CK-11")
@pytest.mark.kind("STAT")
def test_CK11_residual_calibrated_kan_aft_d_calibration():
    dat = fsim.generate("S-Misspec-lognormal", n=2000, seed=16)
    tr, va, te = _split(2000, 16)
    xg, lam = _smoothed(dat, tr)
    m = FKANSurv(kind="b2s", loss="aft", lr=3e-3, wd=1e-4, seed=0).fit(_pack(dat, xg, tr), _pack(dat, xg, va))
    yt, dt = dat.y[te], dat.delta[te]

    def s_at_y(model):
        return np.array([model.predict_survival(xg[te][i:i + 1], dat.z[te][i:i + 1], [yt[i]])[0, 0] for i in range(len(te))])

    p_w = d_calibration(s_at_y(m), dt)[1]
    m.cfg["residual_calibration"] = True
    m._finalise(m.train_)
    p_rc = d_calibration(s_at_y(m), dt)[1]
    _record("CK-11", p_weibull=p_w, p_residual_calibrated=p_rc, sigma=m.sigma, smoothing_lambda=lam)
    assert p_rc > 0.05 and p_w < 0.05, f"D-cal p: residual-calibrated {p_rc:.3g}, Weibull {p_w:.3g}"
