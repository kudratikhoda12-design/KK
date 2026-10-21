"""Round 11 diagnostic (D116; exploratory, development seed 902 only): how well is a bivariate surface F(x, t) identified
from int F(X(t), t) dt when the curves are S-Add's 4-term Fourier curves?

The design matrix of the AFCM tensor basis (10 x 10 cubic B-splines, as in kanfae.models.afcm) for channel 1 is
W_i,(j,k) = int B_j(X_i(t)) B_k(t) dt. A coefficient vector in (or near) the null space of W changes F without changing
any subject's eta, so it is invisible to the likelihood and only the penalty decides it. Reported: singular values of
the column-centred W (centring removes the direction the baseline absorbs), numerical rank at relative tolerances,
and the same for the homogeneous (x-only) basis of B2-H. Also the across-subject SD of int X_d(t) dt: the Fourier
terms integrate to zero over [0, 1], so int X_d dt is the same for every subject and any component of phi linear in x
(homogeneous case) is invisible to the likelihood.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kanfae.data import fsim  # noqa: E402
from kanfae.models.afcm import AFCM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def summary(W):
    Wc = W - W.mean(0)
    s = np.linalg.svd(Wc, compute_uv=False)
    rel = s / s[0]
    return dict(n_cols=W.shape[1], sv_rel=[float(v) for v in rel],
                rank_1e2=int((rel > 1e-2).sum()), rank_1e3=int((rel > 1e-3).sum()), rank_1e4=int((rel > 1e-4).sum()))


def main():
    dat = fsim.generate("S-Add", n=2000, seed=902, dgp="v1")          # D120: Round-11 DGP pinned
    xg = dat.x_true_grid
    m = AFCM()
    m._set_ranges(xg)
    W = m.design(xg)                                                       # (n, D, 100)
    out = {"seed": 902, "n": 2000, "note": "development seed; exploratory diagnostic, not a registered test"}
    for d in range(2):
        tensor = summary(W[:, d])
        Bx = m._bx(d, xg[:, :, d])                                          # (n, G, 10)
        w = np.full(len(fsim.GRID), fsim.GRID[1] - fsim.GRID[0]); w[[0, -1]] *= 0.5
        homog = summary(np.einsum("ngj,g->nj", Bx, w))
        ix = (xg[:, :, d] * w).sum(1)                                       # int X_id(t) dt per subject
        ip = (fsim.true_effect(d, xg[:, :, d], fsim.GRID[None, :]) * w).sum(1)
        out[f"channel{d + 1}"] = dict(tensor_10x10=tensor, homogeneous_10=homog, sd_int_X=float(ix.std()),
                                      sd_int_true_phi=float(ip.std()))
    (ROOT / "reports/round11").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/round11/diag_identifiability.json").write_text(json.dumps(out, indent=1))
    for d in (1, 2):
        c = out[f"channel{d}"]
        print(f"channel {d}: tensor rank (1e-2/1e-3/1e-4) {c['tensor_10x10']['rank_1e2']}/{c['tensor_10x10']['rank_1e3']}/"
              f"{c['tensor_10x10']['rank_1e4']} of 100; homogeneous {c['homogeneous_10']['rank_1e2']}/"
              f"{c['homogeneous_10']['rank_1e3']}/{c['homogeneous_10']['rank_1e4']} of 10; SD of int X dt "
              f"{c['sd_int_X']:.2e} vs SD of int phi_true {c['sd_int_true_phi']:.3f}")


if __name__ == "__main__":
    main()
