"""Revision-4f learnability gate (D99/D101): the original gate's code with FRESH data seed 93 (never generated before) and
the configuration selected by the pre-registered Step-1 search (reports/learnability_selected_fa4f.json), including its
head mode and ensemble size k (risk = mean over training seeds 0..k-1).
R2 (D101): C >= min(C_oracle - 0.03, C_ceiling - 0.025) (0.020 and 0.015 reported); probe part restored to D82:
mean >= 0.78 and omega >= 0.80, required for the seed-0 member and (k > 1) for the probe averaged over the members.
R3-A (D101): censoring-aware error < naive on training seeds 0, 1, 2, and one-sided C_censored >= C_R2 - 0.02 (the
two-sided |C_censored - C_R2| <= 0.02 is printed side by side).

R1 is fitted once and evaluated two ways (owner decision D68, prospective amendment):
  test_R1_oracle_features_head  the ORIGINAL registered check (early-stopped sigma), reported as is
  test_R1A_profile_sigma        R1-A: same C criterion; sigma = profile MLE with the trained eta held fixed, on the
                                TRAINING data only; same tolerance |sigma - 0.5| <= 0.05
The 4f verdict uses R1-A (D80/D82); the original R1 is reported side by side, never relabelled."""
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
from kanfae.models.ensemble import SeedEnsembleAFT
from kanfae.models.pipelines import NeuralAFT
from kanfae.survival.aft import aft_nll

pytestmark = [pytest.mark.stage("D4f"), pytest.mark.kind("STAT"), pytest.mark.slow]
torch.set_default_dtype(torch.float64)
GATE = {}
SEED = 93
SEL = Path(__file__).resolve().parents[1] / "reports" / "learnability_selected_fa4f.json"
if not SEL.exists():
    pytest.skip("4f gate: pre-registered selection not produced yet (D100)", allow_module_level=True)
if json.loads(SEL.read_text()).get("STEP1") != "PASSED":
    pytest.skip("4f gate NOT RUN: Step 1 did not pass (D100)",
                allow_module_level=True)
SELJ = json.loads(SEL.read_text())
CFG, K = SELJ["config"], int(SELJ["ensemble_seeds"])


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
def test_R2_fae_mlp_end_to_end():
    """R2 (D101): C >= min(C_oracle - 0.03, C_ceiling - 0.025) on the test split (0.020 and 0.015 reported), C_ceiling
    the D75 per-individual parametric plug-in on the SAME test split; probe mean >= 0.78 and omega >= 0.80 for the
    seed-0 member and (k > 1) for the member-averaged probe."""
    from kanfae.evaluation.ceiling import ceiling_C
    obs, ora, tr, va, te = _data(0.05, 0.0)
    m = SeedEnsembleAFT("fae_mlp", ensemble_seeds=K, **CFG).fit(obs.subset(tr), obs.subset(va))
    risk = m.predict_risk(obs.subset(te))
    c = harrell_c(obs.y_obs[te], obs.delta[te], risk)
    c_or = harrell_c(obs.y_obs[te], obs.delta[te], -ora.eta_true[te])
    c_ceil = ceiling_C(obs, te)
    probes = []
    for mem in m.members:
        Ztr, Zte = mem.latent(obs.subset(tr)), mem.latent(obs.subset(te))
        A = np.column_stack([np.ones(len(Ztr)), Ztr])
        r2 = []
        for j in range(4):
            tgt = ora.theta[tr, j]
            coef = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ tgt)
            pred = np.column_stack([np.ones(len(Zte)), Zte]) @ coef
            yt = ora.theta[te, j]
            r2.append(1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2))
        probes.append(r2)
    r2, r2m = np.array(probes[0]), np.mean(probes, axis=0)
    thr = min(c_or - 0.03, c_ceil - 0.025)
    GATE["R2_C"] = c
    print(f"R2: C = {c:.4f}, oracle C = {c_or:.4f}; probe R2 (a,b,A,omega) = {np.round(r2, 3)}, mean {np.mean(r2):.3f}; "
          f"epochs {[[r['epochs'] for r in mem.info['runs']] for mem in m.members]}")
    print(f"R2-amended: C_ceiling = {c_ceil:.4f}; threshold = min({c_or - 0.03:.4f}, {c_ceil - 0.025:.4f}) = {thr:.4f}; "
          f"C = {c:.4f}; probe mean {np.mean(r2):.3f} (>= 0.78), omega {r2[3]:.3f} (>= 0.80)")
    print(f"R2 ensemble: k = {K}, head_mode = {SELJ['head_mode']}; member C "
          f"{[round(harrell_c(obs.y_obs[te], obs.delta[te], mem.predict_risk(obs.subset(te))), 4) for mem in m.members]}; "
          f"member-averaged probe {np.round(r2m, 3)}, mean {np.mean(r2m):.3f}")
    for mg in (0.020, 0.015):
        t_ = min(c_or - 0.03, c_ceil - mg)
        print(f"R2 concordance with the {mg:.3f} margin (reported only): threshold {t_:.4f}; pass {c >= t_}")
    print(f"R2 probe part under the three definitions: D82 (mean >= 0.78, omega >= 0.80; gating) "
          f"{bool(np.mean(r2) >= 0.78 and r2[3] >= 0.80 and np.mean(r2m) >= 0.78 and r2m[3] >= 0.80)}; "
          f"original (mean >= 0.80) {bool(np.mean(r2) >= 0.80)}; D91 (0.75/0.75) {bool(np.mean(r2) >= 0.75 and r2[3] >= 0.75)}")
    assert c >= thr
    assert np.mean(r2) >= 0.78 and r2[3] >= 0.80
    assert np.mean(r2m) >= 0.78 and r2m[3] >= 0.80


@pytest.mark.criterion("R3-A")
def test_R3_censoring_awareness():
    """R3-A (D101): aware error < naive error on training seeds 0, 1, 2; C_censored >= C_R2 - 0.02 (one-sided), where
    C_censored is the selected rule's prediction (k > 1: mean risk over the aware seeds 0..k-1)."""
    obs, ora, tr, va, te = _data(0.1, 0.3)
    grid = np.linspace(0, ora.t_max, 101)
    S_true = ora.subset(te).survival(grid)
    errs, fits = {}, {}
    for naive in (False, True):
        for seed in (0, 1, 2):
            m = NeuralAFT("fae_mlp", naive=naive, **CFG).fit(obs.subset(tr), obs.subset(va), seed=seed)
            fits[(naive, seed)] = m
            errs[(naive, seed)] = survival_curve_error(m.predict_survival(obs.subset(te), grid), S_true, grid)
    ens = {nv: SeedEnsembleAFT.from_members([fits[(nv, s)] for s in range(K)]) for nv in (False, True)}
    cs = harrell_c(obs.y_obs[te], obs.delta[te], ens[False].predict_risk(obs.subset(te)))
    ens_err = {nv: survival_curve_error(ens[nv].predict_survival(obs.subset(te), grid), S_true, grid) for nv in (False, True)}
    print("R3: survival-curve error aware", [round(errs[(False, s)], 4) for s in (0, 1, 2)],
          "naive", [round(errs[(True, s)], 4) for s in (0, 1, 2)], f"; C (aware, k = {K}) = {cs:.4f}; ensemble error aware "
          f"{ens_err[False]:.4f} naive {ens_err[True]:.4f}")
    r2_c = GATE.get("R2_C")
    assert r2_c is not None, "R2 must run first in the same session"
    print(f"R3: |C - C_R2| = {abs(cs - r2_c):.4f}; one-sided C - C_R2 = {cs - r2_c:+.4f} (>= -0.02 gating); two-sided "
          f"(reported only) pass {abs(cs - r2_c) <= 0.02}")
    assert cs >= r2_c - 0.02
    assert all(errs[(True, s)] > errs[(False, s)] for s in (0, 1, 2))
