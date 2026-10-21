"""D123 step 1 (owner, L39): diagnose and repair AFCM's phi_1 estimation error on DEVELOPMENT seed 914 only
(S-Add, DGP v2, n 2,000, the CK-6b procedure otherwise). Not a registered test; seeds 22 and 3000-3419 untouched.

Variants (cumulative unless stated):
  V0  Round-12a AFCM (CK-6b settings): lam_x {1e-4..1}, lam_t {1e-4, 1e-2, 1, 1e2, 1e4}, D2 penalties
  V1  (a) lam_x widened to {1e-7, ..., 1e1} (9 values)
  V2  (a) + penalty normalisation by basis size (x penalty / n_t, t penalty / n_x)
  V3  (b) V2 + first-order difference penalty in t (D1'D1 + D2'D2)
  V4  (c) V3 + per-channel (lam_x_d, lam_t_d) (one coordinate sweep)
  V5  (c) V3 + Cui et al. quantile input x~ = F_t(X(t))
  V6  (c) V3 + per-channel lambdas + quantile input
Reported: identifiable (primary) and raw RISE of phi_1 and phi_2, selected lambdas, validation NLPL, time.
Output: reports/round12/diag_afcm_fix_seed914_<first variant of the lane>.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import test_round11 as R11  # noqa: E402
from test_round12_ck import shapes  # noqa: E402
from kanfae.data import fsim  # noqa: E402
from kanfae.models.afcm import LAM_GRID, LAM_T_GRID, fit_afcm  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
SEED = 914
LX_WIDE = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1)
VARIANTS = {
    "V0_r12a": dict(lams_x=LAM_GRID, lams_t=LAM_T_GRID),
    "V1_wide_lx": dict(lams_x=LX_WIDE, lams_t=LAM_T_GRID),
    "V2_wide_norm": dict(lams_x=LX_WIDE, lams_t=LAM_T_GRID, pen_norm=True),
    "V3_norm_d1d2": dict(lams_x=LX_WIDE, lams_t=LAM_T_GRID, pen_norm=True, t_pen="d1d2"),
    "V4_d1d2_perchannel": dict(lams_x=LX_WIDE, lams_t=LAM_T_GRID, pen_norm=True, t_pen="d1d2", per_channel=True),
    "V5_d1d2_quantile": dict(lams_x=LX_WIDE, lams_t=LAM_T_GRID, pen_norm=True, t_pen="d1d2", xin="quantile"),
    "V6_d1d2_perchannel_quantile": dict(lams_x=LX_WIDE, lams_t=LAM_T_GRID, pen_norm=True, t_pen="d1d2",
                                        per_channel=True, xin="quantile"),
}


def main():
    only = sys.argv[1:] or list(VARIANTS)
    dat = fsim.generate("S-Add", n=2000, seed=SEED)
    tr, va, te = R11._split(2000, SEED)
    xg, lam = R11._smoothed(dat, tr)
    out_p = ROOT / f"reports/round12/diag_afcm_fix_seed914_{only[0]}.json"          # one file per lane
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    for name in only:
        t0 = time.time()
        m, recs = fit_afcm(R11._pack(dat, xg, tr), R11._pack(dat, xg, va), **VARIANTS[name])
        best = min(r["val_nll"] for r in recs)
        out[name] = dict(spec={k: v for k, v in VARIANTS[name].items()}, shapes=shapes(m, dat.x_true_grid[te]),
                         lam_x=np.asarray(m.lam_x).tolist(), lam_t=np.asarray(m.lam_t).tolist(), val_nll=best,
                         n_fits=len(recs), seconds=round(time.time() - t0, 1), smoothing_lambda=lam)
        s = out[name]["shapes"]
        print(f"{name}: phi1 id {s['phi1']['rise_id']:.3f} raw {s['phi1']['rise_raw']:.3f} | phi2 id "
              f"{s['phi2']['rise_id']:.3f} raw {s['phi2']['rise_raw']:.3f} | lam_x {out[name]['lam_x']} lam_t "
              f"{out[name]['lam_t']} | val {best:.4f} | {out[name]['seconds']}s", flush=True)
        (ROOT / "reports/round12").mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
