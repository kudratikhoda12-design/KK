"""Round 22 (D156): per-channel REML AFCM, the three new fsim error laws, the external-cohort loaders and the
Round 15-21 regression.

R22-1  fit_afcm(..., selection="reml_per_channel") on a small S-Add sample (development data, seed 11, n 400; not a
       registered seed): deterministic (two runs bit-identical), converged, finite LAML and per-channel EDF, lambdas
       within REML_BOUNDS, LAML not below the common-lambda LAML pilot; the bagged variant runs and is finite; the
       default fit_afcm and fit_afcm_laml reproduce their pre-Round-22 fits bit for bit (stored hashes)
R22-2  fsim "S-Misspec-bimodal", "-studentt3", "-logistic05": error law of mean 0 and SD 0.5 (exact moments of the
       standardised law, and a large-sample check), censoring 0.30 +- 0.02 on the calibration design, oracle survival
       equal to the empirical survival of the event times (max |diff| < 0.01, n 40,000); every earlier scenario
       generates bit-identical data (hashes recorded before the change)
R22-3  loaders: renal (407 patients, 126 graft failures; landmark 1 year: 407 / 126), heart_valve (256 patients, 54
       deaths; landmark 1 year: 175 / 39), C-MAPSS FD003 Option B (100 training engines). Leakage: K1 subject / engine
       IDs disjoint across train, validation and test for outer splits 0-4, and the validation halves disjoint; K2 the
       B-spline smoother's lambda equals the one recomputed from the training subjects only; K3 all windows of an
       engine in one part; K4 no input observation after the landmark
R22-4  the Round 15-21 deterministic tests (tests/test_round15.py .. test_round20.py, test_round21_real_npaft.py; 39
       tests) pass unchanged (subprocess; its records go to a temporary file)
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
pytestmark = [pytest.mark.stage("R22"), pytest.mark.kind("DET")]

from kanfae.data import fsim  # noqa: E402
from kanfae.models import afcm as A  # noqa: E402

# recorded on 2026-11-06 before any Round-22 change (D156): default fits and earlier fsim scenarios
AFCM_HASH = {"fit_afcm": "8d4d6751914af2ad", "fit_afcm_laml": "9a61184dcfebea43"}
FSIM_HASH = {("S-Add", False): "3d03839eb5300f75", ("S-Add", True): "712710d824e45b04",
             ("S-Int", False): "97686f75f01941ee", ("S-Int", True): "92201e45fe44c031",
             ("S-Misspec-lognormal", False): "5fdbf1e1aa3df560", ("S-Misspec-lognormal", True): "d124cc3ecb88bb85",
             ("S-Misspec-loglogistic", False): "53bd4c6b862b8614", ("S-Misspec-loglogistic", True): "fd7a8de48f62c13f",
             ("S-Misspec-mixture", False): "2171d2f1fe0d416f", ("S-Misspec-mixture", True): "fbcd05e4db3f202f",
             ("S-Last", False): "af025e77a3f9a46f", ("S-Last", True): "6728286f21893d49"}
NEW = ("S-Misspec-bimodal", "S-Misspec-studentt3", "S-Misspec-logistic05")


def _theta_hash(m):
    return hashlib.sha256(np.ascontiguousarray(np.r_[m.theta.ravel(), m.gamma]).tobytes()).hexdigest()[:16]


def _data_hash(d):
    h = hashlib.sha256()
    for a in (d.y, d.delta, d.eta, d.T, d.C, d.z, d.x_true_grid, np.concatenate([v.ravel() for v in d.values]),
              np.concatenate(d.times)):
        h.update(np.ascontiguousarray(np.asarray(a, float)).tobytes())
    return h.hexdigest()[:16]


@pytest.fixture(scope="module")
def small():
    import test_round11 as R11
    torch.set_default_dtype(torch.float64)
    dat = fsim.generate("S-Add", n=400, seed=11)
    tr, va, te = R11._split(400, 11)
    xg, _ = R11._smoothed(dat, tr)
    kw = dict(A.AFCM_D123)
    kw.update(lams_x=(1e-5, 1e-3), lams_t=(1e-2, 1.0))
    return R11._pack(dat, xg, tr), R11._pack(dat, xg, va), kw


@pytest.mark.criterion("R22-1")
def test_r22_1_reml_per_channel_afcm(small):
    ptr, pva, kw = small
    m0, _ = A.fit_afcm(ptr, pva, **kw)
    m1, _ = A.fit_afcm_laml(ptr, **kw)
    assert _theta_hash(m0) == AFCM_HASH["fit_afcm"] and _theta_hash(m1) == AFCM_HASH["fit_afcm_laml"]
    a, ra = A.fit_afcm(ptr, None, selection="reml_per_channel", **kw)
    b, rb = A.fit_afcm(ptr, pva, selection="reml_per_channel", **kw)            # validation data are not used
    assert _theta_hash(a) == _theta_hash(b) and ra["lam_x"] == rb["lam_x"] and ra["lam_t"] == rb["lam_t"]
    assert ra["converged"] and np.isfinite(ra["laml"]) and np.isfinite(ra["edf"])
    assert len(ra["edf_channel"]) == 3 and all(np.isfinite(e) and e > 0 for e in ra["edf_channel"])
    lo, hi = A.REML_BOUNDS
    assert all(lo <= v <= hi for v in ra["lam_x"] + ra["lam_t"])
    pilot = max(g["laml"] for g in ra["pilot"]["grid"] if np.isfinite(g["laml"]))
    assert ra["laml"] >= pilot - 1e-6, (ra["laml"], pilot)
    tt = np.quantile(ptr["y"], [0.25, 0.5, 0.75])
    assert np.all(np.isfinite(a.theta)) and np.all(np.isfinite(a.predict_survival(pva["xg"], pva["z"], tt)))
    e, re_ = A.fit_afcm_reml_bagged(ptr, B=2, seed=0, **dict(kw))
    assert len(re_) == 2 and np.all(np.isfinite(e.theta)) and np.all(np.isfinite(e.predict_survival(pva["xg"], pva["z"], tt)))
    with pytest.raises(ValueError):
        A.fit_afcm(ptr, pva, selection="unknown", **kw)


@pytest.mark.criterion("R22-2")
def test_r22_2_new_error_laws():
    for (sc, sp), h in FSIM_HASH.items():
        assert _data_hash(fsim.generate(sc, n=200, seed=7, sparse=sp)) == h, sc
    from scipy.stats import t as student_t
    sd = {"S-Misspec-bimodal": np.sqrt(0.5 * (0.09 + 1.21) * 2), "S-Misspec-studentt3": np.sqrt(student_t(3).var()),
          "S-Misspec-logistic05": np.pi / np.sqrt(3.0)}
    for sc in NEW:
        assert abs(fsim.R22_LAWS[sc] - sd[sc]) < 1e-12                       # exact SD of the unstandardised law
        d = fsim.generate(sc, n=40000, seed=123)
        eps = np.log(d.T) - (1.0 - d.eta / fsim.SHAPE)
        q10, q90 = np.quantile(eps, [0.1, 0.9])                               # symmetric, mean 0 (the sample
        assert abs(eps.mean()) < 0.01 and abs(q10 + q90) < 0.02                # median of the bimodal law is unstable)
        tol = 0.04 if sc.endswith("studentt3") else 0.01                      # t3: heavy tails, slow sample SD
        assert abs(eps.std() - 0.5) < tol, (sc, eps.std())
        assert abs(1 - d.delta.mean() - 0.30) < 0.02
        g = np.quantile(d.T, [0.1, 0.3, 0.5, 0.7, 0.9])
        emp = np.array([(d.T > x).mean() for x in g])
        assert np.max(np.abs(d.survival(g).mean(0) - emp)) < 0.01, sc
    with pytest.raises(ValueError):
        fsim.generate("S-Misspec-unknown", n=10, seed=0)


@pytest.mark.criterion("R22-3")
def test_r22_3_loaders_and_leakage():
    from kanfae.data.cmapss import informative_sensors, option_b, read_fd
    from kanfae.data.realdata import LOADERS, single_landmark
    from kanfae.data.splits import split_groups
    from kanfae.functional.basis import BSplineBasis
    from kanfae.functional.smoothing import pad_ragged, select_lambda
    import run_round18 as R18
    expect = {"renal": (407, 126, 407, 126), "heart_valve": (256, 54, 175, 39)}
    for ds, (n, ev, nl, evl) in expect.items():
        ld = LOADERS[ds]()
        assert len(ld.ids) == n and int(ld.d.sum()) == ev and len(np.unique(ld.ids)) == n
        assert len(ld.channels) == 2 and ld.z.shape == (n, 3) and np.all(np.isfinite(ld.z))
        lm = single_landmark(ld, 1.0)
        assert len(lm.ids) == nl and int(lm.delta.sum()) == evl and np.all(lm.y > 0)
        for i, pid in enumerate(lm.ids):                                          # K4
            j = int(np.where(ld.ids == pid)[0][0])
            assert np.all(np.asarray(lm.times[i]) <= 1.0) and np.all(np.asarray(lm.times[i]) >= 0.0)
            raw = np.asarray(ld.times[j])
            assert len(lm.times[i]) == int(np.sum((raw <= 1.0) & np.isfinite(ld.values[j]).any(1)))
        for s in range(5):                                                         # K1
            a, b, c = split_groups(lm.ids, s, strata=lm.delta)
            A_, B_, C_ = (set(lm.ids[p].tolist()) for p in (a, b, c))
            assert not (A_ & B_) and not (A_ & C_) and not (B_ & C_) and len(a) + len(b) + len(c) == nl
            h1, h2 = R18.split_val(b, lm.delta.astype(int), s)
            assert not (set(h1.tolist()) & set(h2.tolist())) and len(h1) + len(h2) == len(b)
    import run_round13 as R13                                                       # K2 (split 0, both cohorts)
    R13.SINGLE.update(renal=1.0, heart_valve=1.0)
    for ds in ("renal", "heart_valve"):
        ctx = R13.RealCtx(ds, 0)
        T, V, Mk, n_obs = pad_ragged([ctx.lm.times[i] for i in ctx.tr], [ctx.lm.values[i] for i in ctx.tr])
        lam, _ = select_lambda(T, V, Mk, n_obs, BSplineBasis(12, 3, (0.0, 1.0)), seed=0)
        assert lam == ctx.front("bspline")["lam"]
    tr, _, _ = read_fd(ROOT / "data/raw/cmapss", "FD003")                           # K3, K4 on FD003
    units = np.array(sorted(tr.unit.unique()))
    assert len(units) == 100
    sens = informative_sensors(tr, units)
    ob = option_b(tr, units, sens)
    assert np.all(ob.meta["max_input_cycle"] <= ob.meta["landmark"]) and np.all(ob.delta)
    for s in range(5):
        a, b, c = split_groups(units, s)
        ga, gb, gc = (set(units[p].tolist()) for p in (a, b, c))
        assert not (ga & gb) and not (ga & gc) and not (gb & gc) and len(a) + len(b) + len(c) == 100
        win = {k: ob.meta["group"][np.isin(ob.meta["group"], list(g))] for k, g in (("a", ga), ("b", gb), ("c", gc))}
        assert sum(len(v) for v in win.values()) == len(ob)


@pytest.mark.criterion("R22-4")
def test_r22_4_rounds_15_to_21_regression(tmp_path):
    env = dict(os.environ, KANFAE_TEST_RECORD=str(tmp_path / "records.jsonl"), KANFAE_RUN_ID="r22_4_nested")
    files = [f"tests/test_round{r}.py" for r in range(15, 21)] + ["tests/test_round21_real_npaft.py"]
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=5400)
    tail = p.stdout.strip().splitlines()[-1]
    assert p.returncode == 0 and "39 passed" in tail and "failed" not in tail, p.stdout[-3000:]
