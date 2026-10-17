"""Ledger L16 / D50: slope bias of AFT estimators under covariate-dependent censoring (CA1) and error
misspecification (MIS), against independent censoring (CA2). Writes reports/censoring_variants.jsonl."""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kanfae.survival.censored_ls import LinearLS, censoring_G, fit_censored_ls  # noqa: E402
from kanfae.survival.linear_aft import fit_linear_aft  # noqa: E402

OUT = ROOT / "reports" / "censoring_variants.jsonl"
BETA = np.array([1.0, 0.8, -0.5])
SIGMA = 0.5


def errors(rng, n, scenario):
    if scenario != "MIS":
        return SIGMA * rng.normal(size=n)
    comp = rng.random(n) < 0.7                       # skewed two-component mixture, standardised to sd SIGMA
    e = np.where(comp, rng.normal(-0.6, 0.5, n), rng.normal(1.4, 1.0, n))
    mu = 0.7 * -0.6 + 0.3 * 1.4
    sd = np.sqrt(0.7 * (0.25 + 0.36) + 0.3 * (1.0 + 1.96) - mu ** 2)
    return SIGMA * (e - mu) / sd


def simulate(seed, scenario, n=2000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    T = np.exp(BETA[0] + X @ BETA[1:] + errors(rng, n, scenario))
    cmean = 8.0 * np.exp(-1.5 * X[:, 0]) if scenario == "CA1" else np.full(n, 8.0)
    C = rng.exponential(cmean)
    return X, np.minimum(T, C), T <= C, cmean


def main():
    OUT.write_text("")
    for scenario in ("CA2", "CA1", "MIS"):
        for seed in range(300, 320):
            X, y, d, cmean = simulate(seed, scenario)
            G = censoring_G(y, d)
            est = {}
            for fam in ("lognormal", "weibull"):
                est[f"param_{fam}"] = fit_linear_aft(X, y, d, family=fam, compute_se=False).beta[1:]
            for method in ("bj", "ipcw", "transform"):
                m = LinearLS(X)
                fit_censored_ls(m.fit, m.predict, y, d, method, G=G)
                est[method] = m.beta[1:]
            m = LinearLS(X)                                  # oracle reference: true conditional G(y- | x)
            m.fit(np.log(y), d / np.exp(-y / cmean))
            est["ipcw_oracle_condG"] = m.beta[1:]
            rec = dict(scenario=scenario, seed=seed, censored_frac=float(1 - d.mean()),
                       slopes={k: [float(v) for v in b] for k, b in est.items()})
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        recs = [json.loads(l) for l in OUT.read_text().splitlines() if json.loads(l)["scenario"] == scenario]
        print(f"== {scenario}: censored fraction {np.mean([r['censored_frac'] for r in recs]):.3f}")
        for k in recs[0]["slopes"]:
            b = np.array([r["slopes"][k] for r in recs])
            bias, rmse = b.mean(0) - BETA[1:], np.sqrt(((b - BETA[1:]) ** 2).mean(0))
            print(f"   {k:20s} bias {np.round(bias, 4)}  rmse {np.round(rmse, 4)}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time() - t0:.0f}s")
