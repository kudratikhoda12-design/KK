"""D74: information ceiling of the frozen latent (seed-21 TRAIN/VALIDATION only). Writes reports/diag_4b_ceiling.json."""
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
from kanfae.models.pipelines import NeuralAFT  # noqa: E402
from kanfae.survival.aft import aft_nll  # noqa: E402
from kanfae.survival.linear_aft import fit_linear_aft, predict_eta  # noqa: E402

A3 = dict(front="tspline", r=8, alpha=1.0, optimizer="adam", lr=3e-3, es_target="objective", max_epochs=3000,
          patience=100, H=32, hidden=64)


def true_eta(th):
    a, b, A, om = th.T[:4]
    return 2.0 + 1.5 * np.tanh(2 * b) - 0.8 * A ** 2 + 0.5 * a * om


def mlp_head(Ztr, ytr, dtr, Zev, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Ztr)); k = int(0.8 * len(Ztr)); fi, si = idx[:k], idx[k:]
    mu, sd = Ztr[fi].mean(0), Ztr[fi].std(0) + 1e-9
    X = torch.tensor((Ztr - mu) / sd); Xe = torch.tensor((Zev - mu) / sd)
    ys = np.median(ytr[fi]); y, d = torch.tensor(ytr / ys), torch.tensor(dtr)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))
    rho = nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam(list(net.parameters()) + [rho], lr=3e-3, weight_decay=1e-4)
    best, state, bad = np.inf, None, 0
    for ep in range(5000):
        opt.zero_grad()
        loss = aft_nll(y[fi], d[fi], net(X[fi]).squeeze(-1), nn.functional.softplus(rho) + 1e-3)
        loss.backward(); opt.step()
        with torch.no_grad():
            v = float(aft_nll(y[si], d[si], net(X[si]).squeeze(-1), nn.functional.softplus(rho) + 1e-3))
        if v < best - 1e-6:
            best, state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad > 200:
                break
    net.load_state_dict(state)
    with torch.no_grad():
        return net(Xe).squeeze(-1).numpy()


def main():
    torch.set_default_dtype(torch.float64)
    obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=21))
    tr, va, te = split_groups(obs.ids, 21, strata=obs.delta)          # te unused
    sel = json.loads((ROOT / "reports" / "learnability_selected_fa4b.json").read_text())["config"]
    yv, dv = obs.y_obs[va], obs.delta[va]
    out = {"oracle_val_C": harrell_c(yv, dv, -ora.eta_true[va])}
    for name, cfg in (("revision 4b selection", sel), ("A3 reference", A3)):
        m = NeuralAFT("fae_mlp", **cfg).fit(obs.subset(tr), obs.subset(va), seed=0)
        Zt, Zv = m.latent(obs.subset(tr)), m.latent(obs.subset(va))
        res = {"model_val_C": harrell_c(yv, dv, m.predict_risk(obs.subset(va)))}
        f = fit_linear_aft(Zt, obs.y_obs[tr], obs.delta[tr], "weibull", l2=1e-3, compute_se=False)
        res["linear_head_refit_val_C"] = harrell_c(yv, dv, -predict_eta(f, Zv))
        res["mlp_head_val_C"] = harrell_c(yv, dv, -mlp_head(Zt, obs.y_obs[tr], obs.delta[tr], Zv))
        A = np.column_stack([np.ones(len(Zt)), Zt]); B = np.column_stack([np.ones(len(Zv)), Zv])
        th_hat = np.column_stack([B @ np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ ora.theta[tr, j])
                                  for j in range(4)])
        res["plugin_true_function_val_C"] = harrell_c(yv, dv, -true_eta(th_hat))
        out[name] = res
        print(name, {k: round(v, 4) for k, v in res.items()}, "| oracle", round(out["oracle_val_C"], 4),
              "| R2 threshold", round(out["oracle_val_C"] - 0.03, 4), flush=True)
    (ROOT / "reports" / "diag_4b_ceiling.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
