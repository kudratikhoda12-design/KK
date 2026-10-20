"""Round 11 exploratory analysis (D116; NOT a registered test, does not change any CK outcome).

Refits the CK-5 (B2-S Cox ensemble, seed 12) and CK-6 (AFCM, seed 13) models with exactly the test code paths and
reports (a) the registered RISE (must reproduce the CK records: determinism check) and (b) RISE after removing, at
each t, the best affine function of x from the error ("RISE-aff"). In S-Add, int X_d(t) dt is the same for every
subject, so components of phi linear in x are (partly) invisible to the likelihood (scripts/diag_round11_identifiability.py);
RISE-aff removes all per-t linear-in-x components, i.e. more than the unidentified part, so it is optimistic.
Output: reports/round11/explore_rise.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import test_round11 as T  # noqa: E402
from kanfae.data import fsim  # noqa: E402
from kanfae.models.afcm import fit_afcm  # noqa: E402
from kanfae.models.fkan import FKANEnsemble  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def rise_aff(phi_hat, phi_true, X, t=fsim.GRID):
    tt = np.broadcast_to(t[None, :], X.shape)
    a, b = np.asarray(phi_hat(X, tt), float), np.asarray(phi_true(X, tt), float)
    den_b = b - b.mean(0, keepdims=True)
    e = a - b
    res = np.empty_like(e)
    for g in range(X.shape[1]):
        A = np.column_stack([np.ones(len(X)), X[:, g]])
        res[:, g] = e[:, g] - A @ np.linalg.lstsq(A, e[:, g], rcond=None)[0]
    return float(np.mean(res ** 2) / np.mean(den_b ** 2))


def both(model, X):
    out = {}
    for d in (0, 1):
        fh = lambda x, t, d=d: model.effect(d, x, t)  # noqa: E731
        ft = lambda x, t, d=d: fsim.true_effect(d, x, t)  # noqa: E731
        out[f"phi{d + 1}"] = dict(rise=fsim.rise(fh, ft, X[:, :, d]), rise_aff=rise_aff(fh, ft, X[:, :, d]))
    return out


def main():
    res = {}
    dat = fsim.generate("S-Add", n=2000, seed=12)
    tr, va, te = T._split(2000, 12)
    xg, _ = T._smoothed(dat, tr)
    e = FKANEnsemble(kind="b2s", loss="cox").fit(T._pack(dat, xg, tr), T._pack(dat, xg, va))
    res["CK5_b2s_cox_seed12"] = both(e, dat.x_true_grid[te])
    dat = fsim.generate("S-Add", n=2000, seed=13)
    tr, va, te = T._split(2000, 13)
    xg, _ = T._smoothed(dat, tr)
    m, _ = fit_afcm(T._pack(dat, xg, tr), T._pack(dat, xg, va))
    res["CK6_afcm_seed13"] = both(m, dat.x_true_grid[te])
    (ROOT / "reports/round11/explore_rise.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
