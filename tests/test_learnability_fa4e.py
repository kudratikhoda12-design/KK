"""Revision-4e learnability gate (D91/D93): the original gate's code with FRESH data seed 92 (never generated: the 4d gate
did not run, D89) and the configuration selected by the pre-registered Step-1 search (reports/learnability_selected_fa4e.json).
R2's probe part is re-referenced by the owner (D91): mean probe >= 0.75 and omega >= 0.75; the 0.78/0.80 (D82) and 0.80
(original) definitions are printed side by side.

R1 is fitted once and evaluated two ways (owner decision D68, prospective amendment):
  test_R1_oracle_features_head  the ORIGINAL registered check (early-stopped sigma), reported as is
  test_R1A_profile_sigma        R1-A: same C criterion; sigma = profile MLE with the trained eta held fixed, on the
                                TRAINING data only; same tolerance |sigma - 0.5| <= 0.05
The 4e verdict uses R1-A (D80/D82); the original R1 is reported side by side, never relabelled."""
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

pytestmark = [pytest.mark.stage("D4e"), pytest.mark.kind("STAT"), pytest.mark.slow]
torch.set_default_dtype(torch.float64)
GATE = {}
SEED = 92
SEL = Path(__file__).resolve().parents[1] / "reports" / "learnability_selected_fa4e.json"
if not SEL.exists():
    pytest.skip("4e gate: pre-registered selection not produced yet (D92)", allow_module_level=True)
if json.loads(SEL.read_text()).get("STEP1") != "PASSED":
    pytest.skip("4e gate NOT RUN: Step 1 did not pass (D92)",
                allow_module_level=True)
CFG = json.loads(SEL.read_text())["config"]


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    """D79: deterministic single-threaded fits, scoped to this module (the previous count is restored)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def _data(sigma_y, censoring):
    obs, ora = generate(SynthConfig(n=2000, censoring=censoring, sigma_y=sigma_y, seed=SEED))
    tr, va, te = split_groups(obs.ids, SEED, strata=obs.delta)
    return obs, ora, tr, va, te


R1FIT = {}


def _r1_fit():
    if R1FIT:
        return R1FIT
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
    from scipy.optimize import minimize_scalar
    with torch.no_grad():
        eta_tr = net(Xt[tr]).squeeze(-1)
    prof = minimize_scalar(lambda s_: float(aft_nll(y[tr], d[tr], eta_tr, torch.tensor(s_))), bounds=(0.05, 2.0),
                           method="bounded", options={"xatol": 1e-8}).x          # training data only (D68)
    R1FIT.update(c=c, c_or=c_or, sigma=sigma, sigma_profile=float(prof))
    return R1FIT




@pytest.mark.criterion("R1")
@pytest.mark.xfail(strict=True, reason="historical: revision-4e gate (seed 92), original R1 sigma failed (D95); superseded by revision 4f (D100/D101); kept as a record (D99)")
def test_R1_oracle_features_head():
    r = _r1_fit()
    print(f"R1: C = {r['c']:.4f}, oracle C = {r['c_or']:.4f}, sigma = {r['sigma']:.4f} (truth 0.5)")
    assert r["c"] >= r["c_or"] - 0.02 and abs(r["sigma"] - 0.5) <= 0.05


@pytest.mark.criterion("R1-A")
def test_R1A_profile_sigma():
    r = _r1_fit()
    print(f"R1-A: C = {r['c']:.4f}, oracle C = {r['c_or']:.4f}, profile sigma (train, eta fixed) = "
          f"{r['sigma_profile']:.4f}; early-stopped sigma = {r['sigma']:.4f} (truth 0.5)")
    assert r["c"] >= r["c_or"] - 0.02 and abs(r["sigma_profile"] - 0.5) <= 0.05


@pytest.mark.criterion("R2")
@pytest.mark.xfail(strict=True, reason="historical: revision 4e failed R2's concordance part on seed 92 (D95); superseded by revision 4f (D100/D101); kept as a record (D99)")
def test_R2_fae_mlp_end_to_end():
    """R2 (D93): C >= min(C_oracle - 0.03, C_ceiling - 0.020) on the test split (0.015 reported), where C_ceiling is the
    per-individual parametric plug-in (kanfae.evaluation.ceiling, D75 definition) on the SAME test split; and mean
    linear-probe R^2 >= 0.75 with the omega probe >= 0.75 (D91; 0.78/0.80 and 0.80 reported side by side)."""
    from kanfae.evaluation.ceiling import ceiling_C
    obs, ora, tr, va, te = _data(0.05, 0.0)
    m = NeuralAFT("fae_mlp", **CFG).fit(obs.subset(tr), obs.subset(va), seed=0)
    risk = m.predict_risk(obs.subset(te))
    c = harrell_c(obs.y_obs[te], obs.delta[te], risk)
    c_or = harrell_c(obs.y_obs[te], obs.delta[te], -ora.eta_true[te])
    c_ceil = ceiling_C(obs, te)
    Ztr, Zte = m.latent(obs.subset(tr)), m.latent(obs.subset(te))
    A = np.column_stack([np.ones(len(Ztr)), Ztr])
    r2 = []
    for j in range(4):
        tgt = ora.theta[tr, j]
        coef = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ tgt)
        pred = np.column_stack([np.ones(len(Zte)), Zte]) @ coef
        yt = ora.theta[te, j]
        r2.append(1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2))
    thr = min(c_or - 0.03, c_ceil - 0.020)
    thr15 = min(c_or - 0.03, c_ceil - 0.015)
    GATE["R2_C"] = c
    print(f"R2: C = {c:.4f}, oracle C = {c_or:.4f}; probe R2 (a,b,A,omega) = {np.round(r2, 3)}, mean {np.mean(r2):.3f}; "
          f"epochs {[r['epochs'] for r in m.info['runs']]}")
    print(f"R2-amended: C_ceiling = {c_ceil:.4f}; threshold = min({c_or - 0.03:.4f}, {c_ceil - 0.020:.4f}) = {thr:.4f}; "
          f"C = {c:.4f}; probe mean {np.mean(r2):.3f} (>= 0.75), omega {r2[3]:.3f} (>= 0.75)")
    print(f"R2 concordance with the 0.015 margin (reported only): threshold {thr15:.4f}; pass {c >= thr15}")
    print(f"R2 probe part under the three definitions: 4e (mean >= 0.75, omega >= 0.75) "
          f"{bool(np.mean(r2) >= 0.75 and r2[3] >= 0.75)}; D82 (mean >= 0.78, omega >= 0.80) "
          f"{bool(np.mean(r2) >= 0.78 and r2[3] >= 0.80)}; original (mean >= 0.80) {bool(np.mean(r2) >= 0.80)}")
    assert c >= thr
    assert np.mean(r2) >= 0.75 and r2[3] >= 0.75


@pytest.mark.criterion("R3")
@pytest.mark.xfail(strict=True, reason="historical: revision 4e failed the two-sided R3 on seed 92 (D95); superseded by revision 4f (D100/D101); kept as a record (D99)")
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
