"""D66: distribution of the R1 oracle-feature check (identical procedure to tests/test_learnability*.py::test_R1)
over data seeds 21, 41 and 501-510. Writes reports/diag_r1_seeds.jsonl."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.data.synthetic import SynthConfig, generate  # noqa: E402
from kanfae.evaluation.metrics import harrell_c  # noqa: E402
from kanfae.survival.aft import aft_nll  # noqa: E402


def r1(seed):
    obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.1, seed=seed))
    tr, va, te = split_groups(obs.ids, seed, strata=obs.delta)
    X = ora.theta[:, :4]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Xt = torch.tensor((X - mu) / sd)
    ys = np.median(obs.y_obs[tr])
    y, d = torch.tensor(obs.y_obs / ys), torch.tensor(obs.delta)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))
    rho = nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam(list(net.parameters()) + [rho], lr=3e-3)
    best, state, bad = np.inf, None, 0
    for ep in range(3000):
        opt.zero_grad()
        loss = aft_nll(y[tr], d[tr], net(Xt[tr]).squeeze(-1), nn.functional.softplus(rho) + 1e-3)
        loss.backward(); opt.step()
        with torch.no_grad():
            v = float(aft_nll(y[va], d[va], net(Xt[va]).squeeze(-1), nn.functional.softplus(rho) + 1e-3))
        if v < best - 1e-6:
            best, state, bad = v, (copy.deepcopy(net.state_dict()), float(rho)), 0
        else:
            bad += 1
            if bad > 100:
                break
    net.load_state_dict(state[0])
    sigma = float(nn.functional.softplus(torch.tensor(state[1])) + 1e-3)
    with torch.no_grad():
        eta = net(Xt[te]).squeeze(-1).numpy()
        eta_tr = net(Xt[tr]).squeeze(-1).numpy()
    c = harrell_c(obs.y_obs[te], obs.delta[te], -eta)
    c_or = harrell_c(obs.y_obs[te], obs.delta[te], -ora.eta_true[te])
    # how well does the head reproduce the true eta on the training data (up to a constant)?
    e, t = eta_tr - eta_tr.mean(), ora.eta_true[tr] - ora.eta_true[tr].mean()
    fit_r2 = 1 - ((t - e - np.log(ys) * 0) ** 2).sum() / (t ** 2).sum()
    # profile MLE of sigma with the learned eta held fixed (training data): separates "sigma not converged at the
    # early-stopping epoch" from "the likelihood / eta fit implies sigma != 0.5"
    from scipy.optimize import minimize_scalar
    with torch.no_grad():
        eta_t = net(Xt[tr]).squeeze(-1)
    prof = minimize_scalar(lambda s_: float(aft_nll(y[tr], d[tr], eta_t, torch.tensor(s_))), bounds=(0.05, 2.0),
                           method="bounded", options={"xatol": 1e-8}).x
    return dict(seed=seed, C=c, oracle_C=c_or, sigma=sigma, sigma_profile_train=float(prof), epochs=ep + 1,
                eta_fit_R2_train=float(fit_r2),
                pass_C=bool(c >= c_or - 0.02), pass_sigma=bool(abs(sigma - 0.5) <= 0.05))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="21,41," + ",".join(str(s) for s in range(501, 511)))
    ap.add_argument("--out", default="reports/diag_r1_seeds.jsonl")       # D68 R1-A fresh seeds: reports/r1a_fresh_seeds.jsonl
    args = ap.parse_args()
    SEEDS = [int(x) for x in args.seeds.split(",")]
    torch.set_default_dtype(torch.float64)
    out = ROOT / args.out
    # resumable (the first profile run hit the background time limit): completed seeds with the profile field are kept
    keep = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    keep = [r for r in keep if "sigma_profile_train" in r]
    out.write_text("".join(json.dumps(r) + "\n" for r in keep))
    for s in SEEDS:
        if s in {r["seed"] for r in keep}:
            continue
        r = r1(s)
        with open(out, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(f"seed {s}: C {r['C']:.4f} (oracle {r['oracle_C']:.4f}) sigma {r['sigma']:.4f} epochs {r['epochs']} "
              f"profile sigma {r['sigma_profile_train']:.4f} eta-fit R2 {r['eta_fit_R2_train']:.3f} pass C {r['pass_C']} pass sigma {r['pass_sigma']}", flush=True)
