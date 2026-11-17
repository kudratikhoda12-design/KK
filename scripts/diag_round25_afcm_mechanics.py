"""Round 25 readiness audit, Task 1D: ONE development diagnostic of the Python AFCM's mechanics (seed 22 only).

Seed 22 is a development seed already used in Rounds 15 and 22; it is reused here as a historical diagnostic
reproduction, not as a fresh or confirmatory seed. Nothing here changes the registered CK-6c / CK-6d thresholds, records
or verdicts, and no smoothing parameter is searched: the model is fitted ONCE at the per-channel smoothing parameters
already saved in reports/round22/dev/results_afcmdev_22.jsonl (the converged Fellner-Schall values of that record).

What is compared (identical saved inputs for R and Python):
  1. the cold-start fit reproduces the saved seed-22 shape metrics (identifiable RISE of phi_1, phi_2);
  2. scripts/diag_round25_afcm_check.R (packages `survival` and `splines` of the project-local R; mgcv is NOT installed
     and must not be installed) recomputes, from the exported design matrix, times, events and coefficients,
       - the Efron log partial likelihood (survival::coxph at the Python coefficients, no iteration),
       - the score vector, and with the exported penalty the gradient of the penalised objective (stationarity),
       - the cubic B-spline basis on the exported knots and points (splines::splineDesign).
This checks that the Python code solves the penalised Cox problem it states. It does NOT check agreement with mgcv's
AFCM (basis "cr"/"cc", marginal centring constraint, Newton LAML): that needs mgcv.
Outputs: reports/round25/diag/ (inputs for R, R result, summary JSON).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
OUT = ROOT / "reports/round25/diag"
SEED = 22
BUDGET_S = 300


def main():
    t0 = time.time()
    import run_round22 as R22
    from kanfae.models.afcm import AFCM, AFCM_D123
    from kanfae.survival.cox import cox_nll

    saved = json.loads((ROOT / "reports/round22/dev/results_afcmdev_22.jsonl").read_text().splitlines()[0])["afcm_reml"]
    lam_x, lam_t = saved["fit"]["lam_x"], saved["fit"]["lam_t"]
    torch.set_default_dtype(torch.float64)
    dat, ptr, X, lam_smooth = R22.ck_data(SEED)
    t_data = time.time() - t0
    kw = {k: v for k, v in AFCM_D123.items() if k not in ("lams_x", "lams_t")}
    m = AFCM(lam_x=lam_x, lam_t=lam_t, **kw)
    t1 = time.time()
    m.fit(ptr["xg"], ptr["z"], ptr["y"], ptr["d"])                     # the single fit of this diagnostic
    t_fit = time.time() - t1
    shp = R22.shapes_ck(m, X)

    W = m.design(ptr["xg"])
    n, D, P = W.shape
    Z = m._z(ptr["z"], n)
    Xd = np.hstack([W.reshape(n, D * P), Z])
    beta = np.concatenate([m.theta.reshape(-1), m.gamma])
    y, d = np.asarray(ptr["y"], float), np.asarray(ptr["d"], float)
    E = float(d.sum())
    lp_py = -float(cox_nll(torch.tensor(Xd @ beta), torch.tensor(y), torch.tensor(d), "efron", reduction="sum"))
    Kx, Kt = (k.numpy() for k in m._penalty_mats())
    q = Xd.shape[1]
    S = np.zeros((q, q))                                                # objective = -lp / E + beta' S beta
    for k in range(D):
        sl = slice(k * P, (k + 1) * P)
        S[sl, sl] = lam_x[k] * Kx + lam_t[k] * Kt + m.ridge * np.eye(P)
    bt = torch.tensor(beta, requires_grad=True)
    obj = cox_nll(torch.tensor(Xd) @ bt, torch.tensor(y), torch.tensor(d), "efron") + bt @ torch.tensor(S) @ bt
    obj.backward()
    grad_py = bt.grad.numpy()

    xs = np.linspace(m.lo[0], m.hi[0], 203)[1:-1]                       # interior points of channel 1's x range
    OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT / "design.csv", Xd, delimiter=",")
    np.savetxt(OUT / "y_d.csv", np.c_[y, d], delimiter=",")
    np.savetxt(OUT / "beta.csv", beta, delimiter=",")
    np.savetxt(OUT / "penalty.csv", S, delimiter=",")
    np.savetxt(OUT / "basis_knots.csv", m.bx[0].knots, delimiter=",")
    np.savetxt(OUT / "basis_points.csv", xs, delimiter=",")
    np.savetxt(OUT / "basis_python.csv", m.bx[0](xs), delimiter=",")
    (OUT / "r_inputs.json").write_text(json.dumps(dict(n_events=E, lp_python=lp_py)))
    left = BUDGET_S - (time.time() - t0)
    rs = ROOT / ".rlocal/env/bin/Rscript"
    t2 = time.time()
    p = subprocess.run([str(rs), str(ROOT / "scripts/diag_round25_afcm_check.R"), str(OUT)], capture_output=True, text=True,
                       timeout=max(left, 10))
    t_r = time.time() - t2
    r = json.loads((OUT / "r_result.json").read_text()) if (OUT / "r_result.json").exists() else dict(
        status="failed", error=(p.stderr or p.stdout)[-2000:])
    res = dict(seed=SEED, role="historical diagnostic reproduction of a development seed (not fresh, not confirmatory)",
               n_train=int(n), n_events=E, n_coef=int(q), lam_x=lam_x, lam_t=lam_t,
               shapes_now=shp, shapes_saved=saved["shapes"],
               rise_id_abs_diff={k: abs(shp[k]["rise_id"] - saved["shapes"][k]["rise_id"]) for k in ("phi1", "phi2")},
               lp_python=lp_py, objective_python=float(obj), grad_python_max_abs=float(np.max(np.abs(grad_py))),
               r=r, seconds=dict(data=round(t_data, 1), fit=round(t_fit, 1), r_check=round(t_r, 1),
                                 total=round(time.time() - t0, 1)), budget_seconds=BUDGET_S,
               within_budget=bool(time.time() - t0 <= BUDGET_S))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("lam_x", "lam_t")}, indent=1, default=float))


if __name__ == "__main__":
    main()
