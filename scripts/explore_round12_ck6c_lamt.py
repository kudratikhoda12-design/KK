"""Exploratory check for D126 (NOT a registered test; CK-6c stays FAIL): on the CK-6c data (seed 22, S-Add, DGP v2),
refit the repaired AFCM (AFCM_D123) with lambda_t restricted to {1, 1e2, 1e4} (lambda_x grid unchanged) and report
identifiable/raw RISE, to check the owner's reading that CK-6c's phi_1 error comes from the validation choice
lambda_t = 1e-2 rather than from the basis or solver. Output: reports/round12/explore_ck6c_lamt.json."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import test_round11 as R11  # noqa: E402
from test_round12_ck import shapes  # noqa: E402
from kanfae.data import fsim  # noqa: E402
from kanfae.models.afcm import AFCM_D123, fit_afcm  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def main():
    dat = fsim.generate("S-Add", n=2000, seed=22)
    tr, va, te = R11._split(2000, 22)
    xg, _ = R11._smoothed(dat, tr)
    out = {}
    for name, lt in (("registered_grid", AFCM_D123["lams_t"]), ("lam_t_ge_1", (1.0, 1e2, 1e4))):
        m, recs = fit_afcm(R11._pack(dat, xg, tr), R11._pack(dat, xg, va), **{**AFCM_D123, "lams_t": lt})
        out[name] = dict(shapes=shapes(m, dat.x_true_grid[te]), lam_x=m.lam_x, lam_t=m.lam_t,
                         val_nll=min(r["val_nll"] for r in recs))
        print(name, json.dumps(out[name]), flush=True)
    (ROOT / "reports/round12/explore_ck6c_lamt.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
