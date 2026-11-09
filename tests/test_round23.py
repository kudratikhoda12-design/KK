"""Round 23 (D160): C-MAPSS operating-condition normalisation, the landmark/NP-KAN-AFT blend, history thinning and
the Round-23 harness.

R23-1  cmapss.condition_normalise on FD002 / FD004 (inputs only, no model fit): the k-means fitted on the training
       engines' operational settings finds six conditions with at least 4,000 training rows each; within every
       condition the normalised training-row sensors have mean 0 (|mean| < 1e-8) and SD 1 or are identically 0;
       two calls give identical output; the FD001 rows read by read_fd are unchanged by the new function's presence
R23-2  choose_w / blend: w = 1 gives the landmark curve exactly, w = 0 the NP-KAN-AFT curve exactly; the chosen w has
       the smallest validation score on the grid {0, .25, .5, .75, 1}; ties go to the larger w; the blend risk is
       minus the mean blended survival over the grid
R23-3  thin: at most three visits per subject, the first and last observed visits kept, times sorted, values unchanged
R23-4  harness smoke run (shrunk budgets; real_pbc_E2_0, a partition already used in Rounds 13-22): status ok, finite
       curves for every model, w on the grid, and the blend's metrics equal the landmark's when w = 1 or the
       NP-KAN-AFT's when w = 0
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
pytestmark = [pytest.mark.stage("R23"), pytest.mark.kind("DET")]

from kanfae.data.cmapss import SENSORS, condition_normalise, read_fd  # noqa: E402
from kanfae.data.splits import split_groups  # noqa: E402


@pytest.mark.criterion("R23-1")
@pytest.mark.parametrize("fd", ["FD002", "FD004"])
def test_r23_1_condition_normalisation(fd):
    df = read_fd(ROOT / "data/raw/cmapss", fd)[0]
    units = np.sort(df.unit.unique())
    utr = units[split_groups(units, 0)[0]]
    dn, info = condition_normalise(df, utr)
    dn2, info2 = condition_normalise(df, utr)
    assert dn[SENSORS].equals(dn2[SENSORS]) and info == info2
    assert len(info["n_rows_train"]) == 6 and min(info["n_rows_train"]) >= 4000
    tr = dn[dn.unit.isin(utr)]
    for c in range(6):
        X = tr[tr.condition == c][SENSORS].to_numpy(float)
        assert np.all(np.abs(X.mean(0)) < 1e-8)
        sd = X.std(0)
        assert np.all((np.abs(sd - 1) < 1e-8) | (np.abs(X).max(0) == 0))
    assert len(dn) == len(df) and dn.unit.equals(df.unit) and dn.cycle.equals(df.cycle)


@pytest.mark.criterion("R23-2")
def test_r23_2_blend_and_weight_rule():
    import run_round23 as H
    tt = np.linspace(0.1, 2.0, 7)
    Sa = lambda t: np.exp(-np.outer([0.5, 1.0, 2.0], t))  # noqa: E731
    Sb = lambda t: np.exp(-np.outer([1.0, 0.7, 0.3], t))  # noqa: E731
    assert np.array_equal(H.blend(Sa, Sb, 1.0)(tt), Sa(tt)) and np.array_equal(H.blend(Sa, Sb, 0.0)(tt), Sb(tt))
    w, tab = H.choose_w(lambda w: (w - 0.6) ** 2)
    assert w == 0.5 and [r["w"] for r in tab] == list(H.W_GRID)
    assert tab[2]["val_ibs"] == min(r["val_ibs"] for r in tab)
    w, _ = H.choose_w(lambda w: 1.0)                       # all tied -> the simpler (landmark) model
    assert w == 1.0
    w, _ = H.choose_w(lambda w: abs(w - 0.375))           # 0.25 and 0.5 tied -> 0.5
    assert w == 0.5
    S = H.blend(Sa, Sb, 0.25)
    assert np.allclose(H.blend_risk(S, tt), -S(tt).mean(1))


@pytest.mark.criterion("R23-3")
def test_r23_3_thinning():
    import run_round23 as H
    rng = np.random.default_rng(0)
    times, values = [], []
    for n in (1, 2, 3, 4, 10, 97):
        t = np.sort(rng.uniform(0, 1, n))
        v = rng.normal(size=(n, 2))
        if n > 3:
            v[1, :] = np.nan                                  # a visit with nothing observed is never kept
        times.append(t[::-1]); values.append(v[::-1])       # given in reverse order: thin must sort
    tt, vv = H.thin(times, values, 3)
    for t0, v0, t, v in zip(times, values, tt, vv):
        o = np.argsort(t0)
        ts, vs = t0[o], v0[o]
        obs = np.isfinite(vs).any(1)
        assert len(t) == min(3, obs.sum()) and np.all(np.diff(t) > 0)
        assert t[0] == ts[obs][0] and t[-1] == ts[obs][-1]
        for ti, vi in zip(t, v):
            assert np.array_equal(vi, vs[ts == ti][0], equal_nan=True)


@pytest.mark.criterion("R23-4")
def test_r23_4_harness_smoke(tmp_path):
    os.environ["R23_OUT"] = str(tmp_path / "r23.jsonl")
    import importlib
    import run_round23 as H
    H = importlib.reload(H)
    H.ENS_KW.update(lrs=(3e-3,), wds=(1e-4,), seeds=(0,))
    try:
        r = H.run_job("real_pbc_E2_0")
    finally:
        H.ENS_KW.clear()
    assert r["status"] == "ok", r.get("traceback")
    assert all(m["finite"] for m in r["models"].values())
    w = r["stack"]["w"]
    assert w in H.W_GRID and r["models"]["stack"]["w"] == w
    keys = ("ibs", "ibs_cond")
    if w == 1.0:
        assert all(abs(r["models"]["stack"][k] - r["models"]["landmark_cox"][k]) < 1e-12 for k in keys)
    if w == 0.0:
        assert all(abs(r["models"]["stack"][k] - r["models"]["b2s_npaft"][k]) < 1e-12 for k in keys)
    rec = [json.loads(x) for x in Path(os.environ["R23_OUT"]).read_text().splitlines()]
    assert rec[-1]["job_name"] == "real_pbc_E2_0"


@pytest.mark.criterion("R23-5")
def test_r23_5_per_seed_ensemble_equals_fkanensemble():
    """The C-MAPSS ensemble assembled from per-seed fits equals FKANEnsemble at the same fixed setting (small budget,
    pbc_E2 split 0 inputs, a partition already used)."""
    os.environ.setdefault("R23_OUT", "/dev/null")
    import run_round23 as H
    from kanfae.models.fkan import FKANEnsemble
    ctx = H.R13.RealCtx("pbc_E2", 0)
    xg, z, y, d = ctx.front("bspline")["xg"], ctx.lm.z, ctx.y, ctx.d.astype(float)
    P = {n: dict(xg=xg[i], z=z[i], y=y[i], d=d[i]) for n, i in (("tr", ctx.tr), ("va", ctx.va), ("te", ctx.te))}
    H.ENS_KW.update(max_epochs=30)
    try:
        e1, _ = H.cm_ensemble(P, seeds=(0, 1))
        e2 = FKANEnsemble(kind="b2s", **H.NP, **H.CM_ENS, max_epochs=30, seeds=(0, 1)).fit(P["tr"], P["va"])
    finally:
        H.ENS_KW.clear()
    tt = np.quantile(P["te"]["y"], [0.25, 0.5, 0.75])
    S1 = np.mean([m.predict_survival(P["te"]["xg"], P["te"]["z"], tt) for m in e1.members], 0)
    S2 = e2.predict_survival(P["te"]["xg"], P["te"]["z"], tt)
    assert len(e1.members) == 2 and np.array_equal(S1, S2) and (e1.lr, e1.wd) == (e2.lr, e2.wd)
