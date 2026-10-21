"""Repaired acceptance tests CK-6c (AFCM) and CK-7b (PACE under DGP v2), owner decisions D123 (closing L39).
Registered before any run on seeds 22 and 23 (verified unused in all code, tests and records).

  * CK-6c (seed 22): S-Add, DGP v2, n 2,000; the CK-6b procedure (splits 60/20/20, smoothed curves, RISE at the true
    test curves) with the repaired AFCM `AFCM_D123` (lambda_x {1e-7, ..., 1e1} x lambda_t {1e-4, 1e-2, 1, 1e2, 1e4},
    penalties normalised by basis size, first- plus second-order differences in t, common lambdas, raw x input).
    Criterion (owner): identifiable RISE phi_2 < 0.10 AND phi_1 < 0.15; whether each is < 0.10, and the raw RISE, are
    reported beside it.
  * CK-7b (seed 23): S-Add sparse (S-Sparse: 3-8 visits), DGP v2, n 1,000, all subjects. For every channel:
    |cos(first PACE eigenfunction, first eigenfunction of the DGP v2 population covariance)| > 0.90 (population
    covariance sum_k lam_k b_k(s) b_k(t) of fsim.basis_v2 / fsim.LAM_V2, eigen-decomposed on the 50-point grid with
    trapezoid weights; population eigenvalues 0.866, 0.500, ...) AND PACE's mean ISE against the true curves is lower
    than per-subject penalised B-spline smoothing (12 basis functions, lambda by held-out-point CV), on all 3 channels.
STAT results are appended to reports/logs/round12_ck.jsonl.
"""
import numpy as np
import pytest
import torch

from kanfae.data import fsim
from kanfae.functional.basis import BSplineBasis
from kanfae.functional.pace import PACE
from kanfae.functional.smoothing import curves_on_grid, pad_ragged, select_lambda
from kanfae.models.afcm import AFCM_D123, fit_afcm

import test_round11 as R11
from test_round12_ck import _record, shapes

pytestmark = [pytest.mark.stage("R12")]
torch.set_default_dtype(torch.float64)
SEEDS = {"CK-6c": 22, "CK-7b": 23}


def population_eigenfunction(t=fsim.GRID):
    w = np.full(len(t), t[1] - t[0]); w[[0, -1]] *= 0.5
    Wh = np.sqrt(w)
    B = fsim.basis_v2(t)
    C = (B * fsim.LAM_V2) @ B.T
    lam, vec = np.linalg.eigh(Wh[:, None] * C * Wh[None, :])
    return vec[:, -1] / Wh, lam[::-1], w


def ck6c(seed):
    dat = fsim.generate("S-Add", n=2000, seed=seed)
    tr, va, te = R11._split(2000, seed)
    xg, lam = R11._smoothed(dat, tr)
    m, recs = fit_afcm(R11._pack(dat, xg, tr), R11._pack(dat, xg, va), **AFCM_D123)
    return dict(shapes=shapes(m, dat.x_true_grid[te]), lam_x=m.lam_x, lam_t=m.lam_t, grid=recs, smoothing_lambda=lam)


def ck7b(seed):
    dat = fsim.generate("S-Add", n=1000, seed=seed, sparse=True)
    G = fsim.GRID
    phi1, pop_lam, w = population_eigenfunction(G)
    basis = BSplineBasis(12, 3, (0.0, 1.0))
    T, V, M, n_obs = pad_ragged(dat.times, dat.values)
    lam, _ = select_lambda(T, V, M, n_obs, basis, seed=0)
    sm = curves_on_grid(dat.times, dat.values, G, basis, lam)
    cos, ise_p, ise_s, K = [], [], [], []
    for d in range(3):
        vals = [v[:, d] for v in dat.values]
        p = PACE(seed=0).fit(dat.times, vals)
        cos.append(float(abs((p.phi[:, 0] * phi1 * w).sum()) / np.sqrt((p.phi[:, 0] ** 2 * w).sum() * (phi1 ** 2 * w).sum())))
        rec = p.reconstruct(dat.times, vals)
        ise_p.append(float(np.mean(((rec - dat.x_true_grid[:, :, d]) ** 2 * w).sum(1))))
        ise_s.append(float(np.mean(((sm[:, :, d] - dat.x_true_grid[:, :, d]) ** 2 * w).sum(1))))
        K.append(int(p.K))
    return dict(abs_cos=cos, ise_pace=ise_p, ise_smooth=ise_s, K=K, population_eigenvalues=pop_lam[:6].tolist(),
                smoothing_lambda=lam, mean_visits=float(np.mean(n_obs)))


@pytest.mark.criterion("CK-6c")
@pytest.mark.kind("STAT")
def test_CK6c_repaired_afcm_identifiable_rise():
    r = ck6c(SEEDS["CK-6c"])
    s = r["shapes"]
    _record("CK-6c", seed=SEEDS["CK-6c"], threshold_phi1=0.15, threshold_phi2=0.10,
            phi1_below_010=bool(s["phi1"]["rise_id"] < 0.10), phi2_below_010=bool(s["phi2"]["rise_id"] < 0.10), **r)
    assert s["phi1"]["rise_id"] < 0.15 and s["phi2"]["rise_id"] < 0.10, f"identifiable RISE {s}"


@pytest.mark.criterion("CK-7b")
@pytest.mark.kind("STAT")
def test_CK7b_pace_under_dgp_v2():
    r = ck7b(SEEDS["CK-7b"])
    _record("CK-7b", seed=SEEDS["CK-7b"], threshold_cos=0.90, **r)
    assert min(r["abs_cos"]) > 0.90 and all(p < s for p, s in zip(r["ise_pace"], r["ise_smooth"])), f"{r}"
