"""D75: data-information ceiling for R2's concordance criterion (seed-21 TRAIN/VALIDATION only).
Writes reports/diag_data_ceiling.json."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kanfae.data.preprocess import Preprocessor  # noqa: E402
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.data.synthetic import SynthConfig, generate  # noqa: E402
from kanfae.evaluation.metrics import harrell_c  # noqa: E402
from kanfae.functional.basis import BSplineBasis  # noqa: E402
from kanfae.functional.layers import FrequencyAwareFunctional  # noqa: E402

OMEGA_GRID = np.linspace(0.3, 2.5, 221)


def true_eta(a, b, A, om):
    return 2.0 + 1.5 * np.tanh(2 * b) - 0.8 * A ** 2 + 0.5 * a * om


def parametric(t, x1, x2):
    """Least-squares a, b from x1 on (1, t); A, omega from a phase-free sinusoid fit to x2 (grid, then refinement)."""
    X = np.column_stack([np.ones_like(t), t])
    a, b = np.linalg.lstsq(X, x1, rcond=None)[0]
    best = None
    for om in OMEGA_GRID:
        S = np.column_stack([np.sin(2 * np.pi * om * t), np.cos(2 * np.pi * om * t)])
        coef, res, *_ = np.linalg.lstsq(S, x2, rcond=None)
        rss = float(((x2 - S @ coef) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, om, coef)
    _, om0, (s0, c0) = best
    f = lambda p: p[0] * np.sin(2 * np.pi * p[2] * t) + p[1] * np.cos(2 * np.pi * p[2] * t) - x2  # noqa: E731
    p = least_squares(f, [s0, c0, om0], bounds=([-10, -10, 0.2], [10, 10, 3.0])).x
    return a, b, float(np.hypot(p[0], p[1])), float(p[2])


def main():
    torch.set_default_dtype(torch.float64)
    obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=21))
    tr, va, te = split_groups(obs.ids, 21, strata=obs.delta)          # te unused
    yv, dv = obs.y_obs[va], obs.delta[va]
    out = {"oracle_val_C": harrell_c(yv, dv, -ora.eta_true[va]),
           "true_theta_plugin_val_C": harrell_c(yv, dv, -true_eta(*ora.theta[va, :4].T))}
    # (a) per-individual parametric estimates from the raw observations
    est = []
    for i in range(len(obs)):
        m = obs.mask[i, :, 0]
        est.append(parametric(obs.times[i, m], obs.values[i, m, 0], obs.values[i, m, 1]))
    est = np.array(est)
    out["parametric_plugin_val_C"] = harrell_c(yv, dv, -true_eta(*est[va].T))
    out["parametric_param_R2_val"] = [float(1 - ((ora.theta[va, j] - est[va, j]) ** 2).sum() /
                                            ((ora.theta[va, j] - ora.theta[va, j].mean()) ** 2).sum()) for j in range(4)]
    # (b) flexible regression of each parameter from smoothed coefficients + compact spectral features (+ (a))
    pre = Preprocessor(n_basis=12).fit(obs.subset(tr))
    c = pre.transform(obs).c
    fa = FrequencyAwareFunctional(3, 4, pre.basis, BSplineBasis(8), seed=0, n_freq=17, compact=True)
    with torch.no_grad():
        spec = fa.spectral_vector(torch.tensor(c)).numpy()
    F = np.hstack([c.reshape(len(c), -1), spec, est])
    th_hat = np.column_stack([HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
                              .fit(F[tr], ora.theta[tr, j]).predict(F) for j in range(4)])
    out["flexible_param_R2_val"] = [float(1 - ((ora.theta[va, j] - th_hat[va, j]) ** 2).sum() /
                                          ((ora.theta[va, j] - ora.theta[va, j].mean()) ** 2).sum()) for j in range(4)]
    out["flexible_plugin_val_C"] = harrell_c(yv, dv, -true_eta(*th_hat[va].T))
    # (c) direct flexible regression of the oracle eta on the same inputs (trained on train)
    g = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, random_state=0).fit(F[tr], ora.eta_true[tr])
    out["flexible_direct_eta_val_C"] = harrell_c(yv, dv, -g.predict(F[va]))
    out["R2_threshold_val"] = out["oracle_val_C"] - 0.03
    (ROOT / "reports" / "diag_data_ceiling.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
