"""Revised learnability gate R1..R3 for revision 4 (D61): the original gate's code and thresholds, with FRESH data
seed 41 and the configuration selected by the pre-registered search (reports/learnability_selected_fa.json)."""
import json
from pathlib import Path
import copy

import numpy as np
import pytest
import torch
from torch import nn

from kanfae.data.splits import split_groups
from kanfae.data.synthetic import SynthConfig, generate
from kanfae.evaluation.metrics import harrell_c, survival_curve_error
from kanfae.models.pipelines import NeuralAFT
from kanfae.survival.aft import aft_nll

pytestmark = [pytest.mark.stage("D4"), pytest.mark.kind("STAT"), pytest.mark.slow]
torch.set_default_dtype(torch.float64)
GATE = {}
SEED = 41
SEL = Path(__file__).resolve().parents[1] / "reports" / "learnability_selected_fa.json"
if not SEL.exists():
    pytest.skip("revised gate: pre-registered selection not produced yet (D61)", allow_module_level=True)
CFG = json.loads(SEL.read_text())["config"]


def _data(sigma_y, censoring):
    obs, ora = generate(SynthConfig(n=2000, censoring=censoring, sigma_y=sigma_y, seed=SEED))
    tr, va, te = split_groups(obs.ids, SEED, strata=obs.delta)
    return obs, ora, tr, va, te


@pytest.mark.criterion("R1")
def test_R1_oracle_features_head():
    obs, ora, tr, va, te = _data(0.1, 0.0)
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
    c = harrell_c(obs.y_obs[te], obs.delta[te], -eta)
    c_or = harrell_c(obs.y_obs[te], obs.delta[te], -ora.eta_true[te])
    print(f"R1: C = {c:.4f}, oracle C = {c_or:.4f}, sigma = {sigma:.4f} (truth 0.5)")
    assert c >= c_or - 0.02 and abs(sigma - 0.5) <= 0.05


@pytest.mark.criterion("R2")
def test_R2_fae_mlp_end_to_end():
    obs, ora, tr, va, te = _data(0.05, 0.0)
    m = NeuralAFT("fae_mlp", **CFG).fit(obs.subset(tr), obs.subset(va), seed=0)
    risk = m.predict_risk(obs.subset(te))
    c = harrell_c(obs.y_obs[te], obs.delta[te], risk)
    c_or = harrell_c(obs.y_obs[te], obs.delta[te], -ora.eta_true[te])
    Ztr, Zte = m.latent(obs.subset(tr)), m.latent(obs.subset(te))
    A = np.column_stack([np.ones(len(Ztr)), Ztr])
    r2 = []
    for j in range(4):
        tgt = ora.theta[tr, j]
        coef = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ tgt)
        pred = np.column_stack([np.ones(len(Zte)), Zte]) @ coef
        yt = ora.theta[te, j]
        r2.append(1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2))
    GATE["R2_C"] = c
    print(f"R2: C = {c:.4f}, oracle C = {c_or:.4f}; probe R2 (a,b,A,omega) = {np.round(r2, 3)}, mean {np.mean(r2):.3f}; "
          f"epochs {[r['epochs'] for r in m.info['runs']]}")
    assert c >= c_or - 0.03
    assert np.mean(r2) >= 0.8


@pytest.mark.criterion("R3")
def test_R3_censoring_awareness():
    obs, ora, tr, va, te = _data(0.1, 0.3)
    grid = np.linspace(0, ora.t_max, 101)
    S_true = ora.subset(te).survival(grid)
    errs = {}
    cs = []
    for naive in (False, True):
        for seed in (0, 1, 2):
            m = NeuralAFT("fae_mlp", naive=naive, **CFG).fit(obs.subset(tr), obs.subset(va), seed=seed)
            errs[(naive, seed)] = survival_curve_error(m.predict_survival(obs.subset(te), grid), S_true, grid)
            if not naive and seed == 0:
                cs.append(harrell_c(obs.y_obs[te], obs.delta[te], m.predict_risk(obs.subset(te))))
    print("R3: survival-curve error aware", [round(errs[(False, s)], 4) for s in (0, 1, 2)],
          "naive", [round(errs[(True, s)], 4) for s in (0, 1, 2)], f"; C (aware, seed 0) = {cs[0]:.4f}")
    r2_c = GATE.get("R2_C")
    assert r2_c is not None, "R2 must run first in the same session"
    print(f"R3: |C - C_R2| = {abs(cs[0] - r2_c):.4f}")
    assert abs(cs[0] - r2_c) <= 0.02
    assert all(errs[(True, s)] > errs[(False, s)] for s in (0, 1, 2))
