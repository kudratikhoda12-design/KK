"""KAN-AFT symbolic-recovery validation (SYNTHETIC, known truth) and censoring-loss comparison.

eta(z) = 1 + 1.5 tanh(2 z1) - 0.8 z2^2 + 0.5 sin(1.5 z3) + 0 * z4,  z ~ U(-2, 2)^4,  log T = eta + 0.5 eps (Weibull),
independent exponential censoring calibrated to ~30%. Data seeds 201-203; 60/20/20 split.
Per seed and loss (parametric | bj | ipcw | transform): lam chosen on validation C from {0, 1e-3, 1e-2}; then
(unpruned) -> prune(theta = 1e-2) -> symbolic substitution; validation and test metrics after each step.
Symbolic approximation quality (R^2 vs learned spline) and truth recovery (R^2 vs true component) are reported
separately from survival-prediction quality.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kanfae.evaluation import metrics as M  # noqa: E402
from kanfae.models.kan_aft import KANAFT  # noqa: E402

OUT = ROOT / "reports" / "kanaft_symbolic.jsonl"
TRUE = [lambda z: 1.5 * np.tanh(2 * z), lambda z: -0.8 * z ** 2, lambda z: 0.5 * np.sin(1.5 * z), lambda z: 0 * z]
TRUE_NAMES = ["1.5 tanh(2 z1)", "-0.8 z2^2", "0.5 sin(1.5 z3)", "0 (null covariate)"]
SIGMA = 0.5


def simulate(n, seed, cens=0.3):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-2, 2, (n, 4))
    eta = 1.0 + sum(f(Z[:, j]) for j, f in enumerate(TRUE))
    T = np.exp(eta + SIGMA * (-rng.gumbel(size=n)))
    # calibrate exponential censoring on an independent pilot
    pr = np.random.default_rng(seed + 10000)
    Zp = pr.uniform(-2, 2, (20000, 4))
    Tp = np.exp(1.0 + sum(f(Zp[:, j]) for j, f in enumerate(TRUE)) + SIGMA * (-pr.gumbel(size=20000)))
    E1 = pr.exponential(1, 20000)
    lo, hi = -20, 20
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if np.mean(E1 / np.exp(mid) < Tp) < cens else (lo, mid)
    C = rng.exponential(1, n) / np.exp((lo + hi) / 2)
    return Z, np.minimum(T, C), T <= C, eta


def true_S(eta, t):
    return np.exp(-np.exp((np.log(np.asarray(t))[None, :] - eta[:, None]) / SIGMA))


def evaluate(m, Ztr, ytr, dtr, Z, y, d, eta_true, grid):
    times = M.ibs_time_grid(y, ytr)
    S = m.predict_survival(Z, times)
    ibs, _ = M.ibs_marginal(S, times, y, d, ytr, dtr)
    eh = m.eta(Z)
    return dict(harrell_c=M.harrell_c(y, d, -eh), uno_c=M.uno_c(ytr, dtr, y, d, -eh, np.quantile(y, 0.9)), ibs=ibs,
                curve_error=M.survival_curve_error(m.predict_survival(Z, grid), true_S(eta_true, grid), grid),
                eta_centered_rmse=float(np.sqrt(np.mean(((eh - eh.mean()) - (eta_true - eta_true.mean())) ** 2))))


def main():
    torch.set_default_dtype(torch.float64)
    with open(OUT, "w") as fh:
        pass
    for seed in (201, 202, 203):
        Z, y, d, eta = simulate(2000, seed)
        idx = np.random.default_rng(seed).permutation(2000)
        tr, va, te = idx[:1200], idx[1200:1600], idx[1600:]
        grid = np.linspace(0, np.quantile(np.exp(eta + SIGMA * 0.5772 * -1), 0.9) * 3, 101)
        for loss in ("parametric", "bj", "ipcw", "transform"):
            t0 = time.time()
            tuning = []
            for lam in (0.0, 1e-3, 1e-2):
                m = KANAFT(loss=loss, lam=lam, seed=0).fit(Z[tr], y[tr], d[tr], Z[va], y[va], d[va])
                vc = M.harrell_c(y[va], d[va], -m.eta(Z[va]))
                tuning.append((vc, lam, m))
            vc, lam, m = max(tuning, key=lambda t: t[0])
            rec = dict(seed=seed, loss=loss, lam_selected=lam, tuning=[(float(a), b) for a, b, _ in tuning],
                       censored_frac=float(1 - d.mean()), fit_info=m.info)
            rec["unpruned_val"] = evaluate(m, Z[tr], y[tr], d[tr], Z[va], y[va], d[va], eta[va], grid)
            rec["unpruned_test"] = evaluate(m, Z[tr], y[tr], d[tr], Z[te], y[te], d[te], eta[te], grid)
            rec["prune"] = m.prune(1e-2)
            rec["pruned_mask"] = m.net.layers[0].mask[:, 0].tolist()
            rec["pruned_val"] = evaluate(m, Z[tr], y[tr], d[tr], Z[va], y[va], d[va], eta[va], grid)
            rec["pruned_test"] = evaluate(m, Z[tr], y[tr], d[tr], Z[te], y[te], d[te], eta[te], grid)
            syms = m.symbolic(use_pysr=True, pysr_budget={"niterations": 30, "timeout": 120})
            # truth recovery on the covariate scale: learned component vs true component (up to a constant)
            for s in syms:
                j = s["edge"][0]
                zg = np.linspace(-2, 2, 401)
                xg = (zg - s["input_mean"]) / s["input_sd"]
                Zg = torch.zeros(401, 4); Zg[:, j] = torch.tensor(xg)
                with torch.no_grad():
                    learned = m.net.layers[0].edge_activations(Zg)[:, j, 0].numpy()   # after substitution if accepted
                tr_ = TRUE[j](zg)
                s["true_component"] = TRUE_NAMES[j]
                s["truth_recovery_r2"] = (M_r2(tr_ - tr_.mean(), learned - learned.mean()) if np.ptp(tr_) > 0 else None)
                s["edge"] = list(s["edge"])
            rec["symbolic"] = syms
            rec["symbolic_val"] = evaluate(m, Z[tr], y[tr], d[tr], Z[va], y[va], d[va], eta[va], grid)
            rec["symbolic_test"] = evaluate(m, Z[tr], y[tr], d[tr], Z[te], y[te], d[te], eta[te], grid)
            rec["runtime_s"] = round(time.time() - t0, 1)
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")
            acc = sum(s["accepted"] for s in syms)
            print(f"seed {seed} {loss}: lam={lam} test C {rec['unpruned_test']['harrell_c']:.4f} -> pruned "
                  f"{rec['pruned_test']['harrell_c']:.4f} -> symbolic {rec['symbolic_test']['harrell_c']:.4f}; "
                  f"mask {rec['pruned_mask']}; symbolic accepted {acc}/{len(syms)}; "
                  + "; ".join(f"z{s['edge'][0] + 1}:{s['stage']}:{s['formula'][:40]} (R2val {s['r2_val']:.4f}, truth "
                              f"{s['truth_recovery_r2'] if s['truth_recovery_r2'] is None else round(s['truth_recovery_r2'], 4)})"
                              for s in syms) + f" [{rec['runtime_s']}s]", flush=True)


def M_r2(y, yhat):
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / ss)


if __name__ == "__main__":
    main()
