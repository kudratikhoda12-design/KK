"""Repaired acceptance tests CK-3b, CK-5b, CK-6b, CK-8b, CK-11b (owner decisions D120, closing L38) and two
deterministic checks of the D120 identifiability repair. The Round-11 records CK-1..CK-12 stay unchanged
(tests/test_round11.py; the five failures are strict expected failures).

Registered before any run on seeds 17-21 (D120):
  * Data: DGP "v2" (fsim default): random level and linear trend added to the curves (D120 decision 1).
  * Splits, curve smoothing and the ensemble protocol as in Round 11 (tests/test_round11.py helpers).
  * Shape metric (CK-5b, CK-6b): `fsim.rise_identifiable` (per-t centring, then projection orthogonal to the
    unidentified directions of the empirical trajectory operator on the true test curves; tensor 10 x 10 cubic B-spline
    candidate space; tolerance 3e-8) is primary; the raw Round-11 RISE is reported next to it.
  * CK-3b (seed 19): z ~ N(0, I_3), beta (0.8, -0.5, 0.3), Weibull PH shape 1.5, censoring U(0, 3), n 2,000 all used for
    training; linear-edge CoxKAN, weight decay 0, lambda 0, no early stopping: 500 Adam epochs (lr 3e-3) on the training
    partial likelihood, then L-BFGS to convergence (max 1,000 iterations); slopes vs lifelines on the same data, < 0.05.
  * CK-5b (seed 17): B2-S Cox, FKANEnsemble (D112 protocol); identifiable RISE of phi_1 and phi_2 each < 0.10.
  * CK-6b (seed 18): AFCM with the D120 lambda_t grid {1e-4, 1e-2, 1, 1e2, 1e4} x lambda_x {1e-4..1}; identifiable RISE
    of phi_1 and phi_2 each < 0.10.
  * CK-8b (seed 20): B2-S Cox (lr 3e-3, wd 1e-4, seed 0) through select_lambda_theta with the D120 defaults (integrated
    channel score, lambda {0, 1e-4, 1e-3, 1e-2, 3e-2} x theta_rel {0, .01, .02, .05, .10, .15, .20, .25}, the most
    parsimonious pair within 0.01 of the minimum validation loss); selected channels == [1, 1, 0].
  * CK-11b (seed 21): "S-Misspec-mixture" (log T = 1 - eta/1.5 + eps, eps the Round-2 MIS skewed two-component normal
    mixture, SD 0.5); B2-S KAN-AFT, Weibull loss (lr 3e-3, wd 1e-4, seed 0); test-split D-calibration (10 bins):
    residual-calibrated p > 0.05 and Weibull p < 0.05.
STAT results are appended to reports/logs/round12_ck.jsonl.
D123 (owner, L39): CK-6b failed (D121) and is kept as a strict expected failure with the Round-12a AFCM pinned; the
live AFCM check is CK-6c in tests/test_round12b_ck.py.
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
from kanfae.evaluation.metrics import d_calibration
from kanfae.models.afcm import LAM_GRID, LAM_T_GRID, fit_afcm
from kanfae.models.fkan import FKANEnsemble, FKANSurv, select_lambda_theta

import test_round11 as R11

pytestmark = [pytest.mark.stage("R12")]
torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "logs" / "round12_ck.jsonl"
SEEDS = {"CK-3b": 19, "CK-5b": 17, "CK-6b": 18, "CK-8b": 20, "CK-11b": 21}


def _record(ck, out=OUT, **kw):
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(ck=ck, run_id=os.environ.get("KANFAE_RUN_ID", time.strftime("%Y%m%dT%H%M%S")), **kw)
    with open(out, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def shapes(model, X):
    """Identifiable (primary) and raw RISE for phi_1, phi_2 at the true test curves X (n, G, D)."""
    out = {}
    for d in (0, 1):
        fh = lambda x, t, d=d: model.effect(d, x, t)  # noqa: E731
        ft = lambda x, t, d=d: fsim.true_effect(d, x, t)  # noqa: E731
        out[f"phi{d + 1}"] = dict(rise_id=fsim.rise_identifiable(fh, ft, X[:, :, d]), rise_raw=fsim.rise(fh, ft, X[:, :, d]))
    return out


# --------------------------------------------------------------------------------------------- the five procedures
def ck3b(seed):
    rng = np.random.default_rng(seed)
    n, beta = 2000, np.array([0.8, -0.5, 0.3])
    Z = rng.normal(size=(n, 3))
    T = (-np.log(rng.random(n)) / np.exp(Z @ beta)) ** (1 / 1.5)
    C = rng.uniform(0, 3, n)
    y, d = np.minimum(T, C), (T <= C).astype(float)
    m = FKANSurv(kind="static", loss="cox", lam=0.0, lr=3e-3, wd=0.0, seed=0, edge="linear", early_stopping=False,
                 max_epochs=500).fit(dict(xg=None, z=Z, y=y, d=d), None)
    m.refit_symbolic(max_iter=1000)
    slopes = np.array([float(e.p[2] * e.p[0]) for _, e in sorted(m.net.stat.symbolic.items())])
    ref = CoxPHFitter().fit(pd.DataFrame(Z, columns=["a", "b", "c"]).assign(T=y, E=d.astype(int)), "T", "E").params_.values
    return dict(slopes=slopes.tolist(), lifelines=ref.tolist(), max_abs_err=float(np.max(np.abs(slopes - ref))),
                grad_norm=m.refit_grad_norm, train_nlpl=m.refit_train_loss, censoring=float(1 - d.mean()))


def ck5b(seed):
    dat = fsim.generate("S-Add", n=2000, seed=seed)
    tr, va, te = R11._split(2000, seed)
    xg, lam = R11._smoothed(dat, tr)
    e = FKANEnsemble(kind="b2s", loss="cox").fit(R11._pack(dat, xg, tr), R11._pack(dat, xg, va))
    return dict(shapes=shapes(e, dat.x_true_grid[te]), member_shapes=[shapes(m, dat.x_true_grid[te]) for m in e.members],
                lr=e.lr, wd=e.wd, tuning=e.tuning, smoothing_lambda=lam)


def ck6b(seed):
    dat = fsim.generate("S-Add", n=2000, seed=seed)
    tr, va, te = R11._split(2000, seed)
    xg, lam = R11._smoothed(dat, tr)
    m, recs = fit_afcm(R11._pack(dat, xg, tr), R11._pack(dat, xg, va), lams_x=LAM_GRID, lams_t=LAM_T_GRID)  # 12a, pinned
    return dict(shapes=shapes(m, dat.x_true_grid[te]), lam_x=m.lam_x, lam_t=m.lam_t, grid=recs, smoothing_lambda=lam)


def ck8b(seed):
    dat = fsim.generate("S-Add", n=2000, seed=seed)
    tr, va, te = R11._split(2000, seed)
    xg, lam = R11._smoothed(dat, tr)
    best, recs = select_lambda_theta(lambda l: FKANSurv(kind="b2s", loss="cox", lam=l, lr=3e-3, wd=1e-4, seed=0),
                                     R11._pack(dat, xg, tr), R11._pack(dat, xg, va))
    return dict(selected=best.selected_channels(), grid=recs, channel_scores_selected=best.channel_scores().tolist(),
                smoothing_lambda=lam)


def ck11b(seed):
    dat = fsim.generate("S-Misspec-mixture", n=2000, seed=seed)
    tr, va, te = R11._split(2000, seed)
    xg, lam = R11._smoothed(dat, tr)
    m = FKANSurv(kind="b2s", loss="aft", lr=3e-3, wd=1e-4, seed=0).fit(R11._pack(dat, xg, tr), R11._pack(dat, xg, va))
    yt, dt = dat.y[te], dat.delta[te]

    def s_at_y():
        return np.array([m.predict_survival(xg[te][i:i + 1], dat.z[te][i:i + 1], [yt[i]])[0, 0] for i in range(len(te))])

    p_w = d_calibration(s_at_y(), dt)[1]
    m.cfg["residual_calibration"] = True
    m._finalise(m.train_)
    p_rc = d_calibration(s_at_y(), dt)[1]
    return dict(p_weibull=p_w, p_residual_calibrated=p_rc, sigma=m.sigma, smoothing_lambda=lam)


# ------------------------------------------------------------------------------------------------------------- DET
@pytest.mark.criterion("ID-1")
@pytest.mark.kind("DET")
def test_ID1_identifiable_rise_ignores_unidentified_directions():
    """Adding an unidentified direction (Round-11 DGP v1 curves) changes the raw RISE but not the identifiable RISE;
    adding an identified one (the linear function of x, under v2) changes both."""
    d1 = fsim.generate("S-Add", n=300, seed=4, dgp="v1")
    X = d1.x_true_grid[:, :, 1]
    F, _ = fsim.unidentified_directions(X)
    assert F.shape[1] >= 1
    f0 = F[:, 0].reshape(X.shape)
    truth = lambda x, t: fsim.true_effect(1, x, t)  # noqa: E731
    hat = lambda x, t: truth(x, t) + 0.5 * f0  # noqa: E731     (evaluated only at X)
    assert fsim.rise(hat, truth, X) > 0.01
    assert fsim.rise_identifiable(hat, truth, X) < 1e-10
    d2 = fsim.generate("S-Add", n=300, seed=4)
    X2 = d2.x_true_grid[:, :, 1]
    hat2 = lambda x, t: truth(x, t) + 0.3 * x  # noqa: E731
    assert fsim.rise_identifiable(hat2, truth, X2) > 0.01


@pytest.mark.criterion("ID-2")
@pytest.mark.kind("DET")
def test_ID2_v2_curves_identify_level_and_trend():
    """D120 verification (development seed 902, n 2,000): SD_i(int X_id dt) > 0.5 and SD_i(int t X_id dt) > 0.1 for
    every channel under DGP v2, while v1 has SD(int X_id dt) ~ 0 (D118)."""
    G = fsim.GRID
    w = np.full(len(G), G[1] - G[0]); w[[0, -1]] *= 0.5
    for dgp in ("v1", "v2"):
        X = fsim.generate("S-Add", n=2000, seed=902, dgp=dgp).x_true_grid
        sd0 = (X * w[None, :, None]).sum(1).std(0)
        sd1 = (X * (w * G)[None, :, None]).sum(1).std(0)
        if dgp == "v1":
            assert np.all(sd0 < 1e-12)
        else:
            assert np.all(sd0 > 0.5) and np.all(sd1 > 0.1)


# ------------------------------------------------------------------------------------------------------------ STAT
@pytest.mark.criterion("CK-3b")
@pytest.mark.kind("STAT")
def test_CK3b_converged_linear_coxkan_matches_lifelines():
    r = ck3b(SEEDS["CK-3b"])
    _record("CK-3b", seed=SEEDS["CK-3b"], threshold=0.05, **r)
    assert r["max_abs_err"] < 0.05, f"slopes {r['slopes']} vs lifelines {r['lifelines']}"


@pytest.mark.criterion("CK-5b")
@pytest.mark.kind("STAT")
def test_CK5b_b2s_cox_identifiable_rise():
    r = ck5b(SEEDS["CK-5b"])
    _record("CK-5b", seed=SEEDS["CK-5b"], threshold=0.10, **r)
    s = r["shapes"]
    assert s["phi1"]["rise_id"] < 0.10 and s["phi2"]["rise_id"] < 0.10, f"identifiable RISE {s}"


@pytest.mark.criterion("CK-6b")
@pytest.mark.kind("STAT")
@pytest.mark.xfail(strict=True, reason="historical record, D121: AFCM phi_1 identifiable RISE 0.556 (seed 18, Round-12a "
                                       "AFCM); live check CK-6c (D123)")
def test_CK6b_afcm_identifiable_rise():
    r = ck6b(SEEDS["CK-6b"])
    _record("CK-6b", seed=SEEDS["CK-6b"], threshold=0.10, **r)
    s = r["shapes"]
    assert s["phi1"]["rise_id"] < 0.10 and s["phi2"]["rise_id"] < 0.10, f"identifiable RISE {s}"


@pytest.mark.criterion("CK-8b")
@pytest.mark.kind("STAT")
def test_CK8b_integrated_channel_pruning():
    r = ck8b(SEEDS["CK-8b"])
    _record("CK-8b", seed=SEEDS["CK-8b"], **r)
    assert r["selected"] == [1, 1, 0], f"selected channels {r['selected']}"


@pytest.mark.criterion("CK-11b")
@pytest.mark.kind("STAT")
def test_CK11b_residual_calibration_on_mixture_errors():
    r = ck11b(SEEDS["CK-11b"])
    _record("CK-11b", seed=SEEDS["CK-11b"], **r)
    assert r["p_residual_calibrated"] > 0.05 and r["p_weibull"] < 0.05, f"D-cal p {r}"
