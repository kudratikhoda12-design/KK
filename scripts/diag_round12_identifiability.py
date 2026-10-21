"""D120 verification before Round 12 (development seed 902; not a registered test): identifiability under the repaired
DGP v2 compared with v1. Reports SD_i(int X_id dt), SD_i(int t X_id dt), the rank of the AFCM 10 x 10 tensor design and
of the homogeneous 10-function design (column-centred; relative tolerances 1e-2/1e-3/1e-4), and the number of
unidentified directions used by fsim.rise_identifiable (tolerance fsim.ID_TOL) with the smallest ratios, on a test-sized
sample (n 400) and on n 2,000. Output: reports/round12/diag_identifiability_v2.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from diag_round11_identifiability import summary  # noqa: E402
from kanfae.data import fsim  # noqa: E402
from kanfae.models.afcm import AFCM  # noqa: E402


def main():
    G = fsim.GRID
    w = np.full(len(G), G[1] - G[0]); w[[0, -1]] *= 0.5
    out = {"seed": 902, "id_tol": fsim.ID_TOL, "note": "development seed; exploratory verification (D120)"}
    for dgp in ("v1", "v2"):
        dat = fsim.generate("S-Add", n=2000, seed=902, dgp=dgp)
        xg = dat.x_true_grid
        m = AFCM()
        m._set_ranges(xg)
        W = m.design(xg)
        res = {}
        for d in range(3):
            Bx = m._bx(d, xg[:, :, d])
            res[f"channel{d + 1}"] = dict(
                sd_int_X=float((xg[:, :, d] * w).sum(1).std()), sd_int_tX=float((xg[:, :, d] * w * G).sum(1).std()),
                tensor_10x10=summary(W[:, d]), homogeneous_10=summary(np.einsum("ngj,g->nj", Bx, w)))
            for n_eval in (400, 2000):
                F, r = fsim.unidentified_directions(xg[:n_eval, :, d])
                res[f"channel{d + 1}"][f"n_unidentified_n{n_eval}"] = int(F.shape[1])
                res[f"channel{d + 1}"][f"smallest_ratios_n{n_eval}"] = [float(v) for v in r[:16]]
        out[dgp] = res
    (ROOT / "reports/round12").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/round12/diag_identifiability_v2.json").write_text(json.dumps(out, indent=1))
    for dgp in ("v1", "v2"):
        for ch in ("channel1", "channel2", "channel3"):
            c = out[dgp][ch]
            print(f"{dgp} {ch}: SD int X {c['sd_int_X']:.3g}, SD int tX {c['sd_int_tX']:.3g}; tensor rank "
                  f"{c['tensor_10x10']['rank_1e2']}/{c['tensor_10x10']['rank_1e3']}/{c['tensor_10x10']['rank_1e4']}; "
                  f"homogeneous {c['homogeneous_10']['rank_1e2']}/{c['homogeneous_10']['rank_1e3']}/"
                  f"{c['homogeneous_10']['rank_1e4']}; unidentified dirs n400 {c['n_unidentified_n400']}, "
                  f"n2000 {c['n_unidentified_n2000']}")


if __name__ == "__main__":
    main()
