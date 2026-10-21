"""Exploratory diagnostic after CK-6b (D121; NOT a registered test, changes no outcome): the approximation floor of
AFCM's x basis for the homogeneous truth phi_1(x) = sin(pi x / 2) on the CK-6b data (seed 18, DGP v2).

AFCM's x basis is 10 cubic B-splines on the training range of the smoothed curves X_hat_1 (kanfae.models.afcm). The
best L2 fit of phi_1 in span{B_j(x)} (least squares at the true test-curve values, i.e. density-weighted, plus a free
constant) gives the smallest RISE any coefficient vector could reach with the time-constant part of the surface;
compared with the same floor for wider/narrower ranges and more basis functions. Output: reports/round12/explore_afcm_floor.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import test_round11 as R11  # noqa: E402
from kanfae.data import fsim  # noqa: E402
from kanfae.functional.basis import BSplineBasis  # noqa: E402


def floor(xv, lo, hi, k):
    B = BSplineBasis(k, 3, (lo, hi))(np.clip(xv, lo, hi))
    y = fsim.phi1(xv)
    A = np.column_stack([B, np.ones(len(xv))])
    fit = A @ np.linalg.lstsq(A, y, rcond=None)[0]
    yc, fc = y - y.mean(), fit - fit.mean()
    return float(np.mean((fc - yc) ** 2) / np.mean(yc ** 2))


def main():
    dat = fsim.generate("S-Add", n=2000, seed=18)
    tr, va, te = R11._split(2000, 18)
    xg, _ = R11._smoothed(dat, tr)
    lo, hi = float(xg[tr][:, :, 0].min()), float(xg[tr][:, :, 0].max())
    xv = dat.x_true_grid[te][:, :, 0].ravel()
    q = np.percentile(xv, [1, 99])
    out = dict(seed=18, train_range=[lo, hi], test_p1_p99=q.tolist(),
               floor_afcm_basis_10=floor(xv, lo, hi, 10),
               floor_basis_15=floor(xv, lo, hi, 15), floor_basis_20=floor(xv, lo, hi, 20),
               floor_10_on_p1_p99=floor(xv, float(q[0]), float(q[1]), 10),
               note="exploratory; the half-period of sin(pi x/2) is 2 x-units")
    (ROOT / "reports/round12").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/round12/explore_afcm_floor.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
