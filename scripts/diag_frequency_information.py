"""D60 diagnostic (round 3, before any design choice is fixed): is the oscillation frequency omega recoverable,
phase-invariantly, from the representation the models see? Uses ONLY the training split of the search data
(seed 21). Writes reports/diag_frequency_information.json.

Features compared (channel x2 = A sin(2 pi omega t + phi); smoothed with the model's B-spline smoother):
  (i)   fixed Fourier power |int x e^{-2 pi i f t}|^2 at f on a grid (no Gram correction): phase-dependent
  (ii)  derivative-energy ratio int x'^2 / int (x - mean)^2 (A4-type): phase-dependent for few cycles
  (iii) quadrature-pair projection spectrum R(f) = ||P_{1,t,s_f,c_f} x||^2 - ||P_{1,t} x||^2, normalised by
        ||x - P_{1,t} x||^2 (exactly phase-invariant at the true frequency)
Each is summarised by an argmax/centroid frequency estimate and by a linear regression of omega on the
features (fit on 70% of the training split, evaluated on the other 30%).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kanfae.data.preprocess import Preprocessor  # noqa: E402
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.data.synthetic import GRID, SynthConfig, generate, _curves_on_grid  # noqa: E402

FREQS = np.geomspace(0.25, 4.0, 33)


def quad(n=2001):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1), 0.5 * w


def spectra(X, t, w):
    """X: (N, Q) curves at quadrature nodes t with weights w. Returns fourier power, deriv ratio, R(f)."""
    W = w[None, :]
    base = np.stack([np.ones_like(t), t], 1)                       # (Q, 2)
    def proj_energy(Bm):                                           # ||P_B x||^2 with the L2 metric
        G = Bm.T @ (Bm * w[:, None]); b = (X * W) @ Bm             # (N, k)
        return np.einsum("nk,kl,nl->n", b, np.linalg.inv(G), b)
    e_base = proj_energy(base)
    resid = (X ** 2 * W).sum(1) - e_base
    R, P = [], []
    for f in FREQS:
        s, c = np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)
        R.append((proj_energy(np.column_stack([base, s, c])) - e_base) / np.maximum(resid, 1e-12))
        P.append(((X * W) @ s) ** 2 + ((X * W) @ c) ** 2)
    return np.array(P).T, np.array(R).T


def r2(y, yh):
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def evaluate(name, F, om, est=None):
    n = len(om); k = int(0.7 * n)
    A = np.column_stack([np.ones(n), F])
    coef = np.linalg.lstsq(A[:k], om[:k], rcond=None)[0]
    out = {"linear_probe_R2_heldout": r2(om[k:], A[k:] @ coef)}
    if est is not None:
        out["estimate_corr"] = float(np.corrcoef(est, om)[0, 1])
        out["estimate_R2_after_affine"] = r2(om[k:], np.polyval(np.polyfit(est[:k], om[:k], 1), est[k:]))
    print(f"{name:55s}", {k_: round(v, 3) for k_, v in out.items()}, flush=True)
    return out


def main():
    obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=21))
    tr, va, te = split_groups(obs.ids, 21, strata=obs.delta)       # only tr is used below
    om, ph, A = ora.theta[tr, 3], ora.theta[tr, 4], ora.theta[tr, 2]
    t, w = quad()
    res = {"n_train": int(len(tr)), "freq_grid": FREQS.tolist()}
    # (a) noiseless dense truth (upper bound)
    curves = _curves_on_grid(ora.theta[tr], np.zeros((len(tr), len(GRID))))[:, :, 1]
    Xt = np.array([np.interp(t, GRID, c) for c in curves])
    # (b) the model's view: smoothed B-spline curve of the (noisy, irregular) observations
    pre = Preprocessor(n_basis=12).fit(obs.subset(tr))
    Pc = pre.transform(obs.subset(tr)).c[:, 1, :]                   # channel x2 coefficients
    Xs = Pc @ pre.basis(t).T
    for label, X in (("truth", Xt), ("smoothed_K12", Xs)):
        P, R = spectra(X, t, w)
        dX = np.gradient(X, t, axis=1)
        Xc = X - (X * w).sum(1, keepdims=True)
        ratio = np.sqrt((dX ** 2 * w).sum(1) / np.maximum((Xc ** 2 * w).sum(1), 1e-12)) / (2 * np.pi)
        res[label] = {
            "fourier_power": evaluate(f"{label}: fixed Fourier power (argmax)", np.log(P + 1e-9), om, FREQS[P.argmax(1)]),
            "deriv_ratio": evaluate(f"{label}: derivative-energy ratio", ratio[:, None], om, ratio),
            "quadrature_projection": evaluate(f"{label}: quadrature-pair projection R(f) (argmax)", R, om,
                                              FREQS[R.argmax(1)]),
        }
        # phase dependence: correlation of the estimate's error with cos(2 phi) / sin(2 phi) at fixed omega
        for key, est in (("fourier_power", FREQS[P.argmax(1)]), ("deriv_ratio", ratio), ("quadrature_projection", FREQS[R.argmax(1)])):
            e = est - np.polyval(np.polyfit(om, est, 1), om)
            Z = np.column_stack([np.ones_like(ph), np.cos(2 * ph), np.sin(2 * ph), np.cos(ph), np.sin(ph)])
            res[label][key]["error_variance_explained_by_phase"] = r2(e, Z @ np.linalg.lstsq(Z, e, rcond=None)[0])
            print(f"   {key}: share of estimate error explained by phase = {res[label][key]['error_variance_explained_by_phase']:.3f}")
    (ROOT / "reports" / "diag_frequency_information.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
