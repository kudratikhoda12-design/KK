"""Round 24 (D163): the Bayesian joint-model comparator (JMbayes2, project-local R) on the Round-23 clinical partitions.

R24-1  jm_input (inputs only, no fit; partition 0, already used in Rounds 13-22): the joint model sees training
       subjects' full histories only up to their own event or censoring time, validation and test subjects' histories
       only inside the landmark window [t_L - W, t_L]; the three subject sets are disjoint and equal the RealCtx split;
       training survival rows enter at t_L (delayed entry) and end at t_L + residual time; validation and test
       survival rows are event-free at t_L; pbc times are in years
R24-2  pairing with the Round-23 records: on real_heart_valve_5 the cohort size, test-set size, test events and the
       landmark Cox test metrics recomputed now equal the Round-23 record exactly, so the Round-23 NP-KAN-AFT, landmark
       Cox and blend results can be paired with the joint model partition by partition
R24-3  surv_fn interpolates linearly on the grid and holds the last value after it; choose_form picks the lower
       validation score, breaks ties towards "lin" and skips forms without a score
R24-4  end-to-end smoke fit (short MCMC; real_heart_valve_0): both forms fit; validation and test curves are finite, in
       [0, 1], equal 1 at the landmark and non-increasing; a second run with the same seed gives identical curves
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
pytestmark = [pytest.mark.stage("R24"), pytest.mark.kind("DET")]

import run_round24 as R  # noqa: E402

SHORT = dict(n_chains=1, n_iter=300, n_burnin=100)
needs_r = pytest.mark.skipif(not R.RSCRIPT.exists(), reason="project-local R (.rlocal/) not installed")


@pytest.mark.criterion("R24-1")
@pytest.mark.parametrize("ds", ["pbc_E2", "aids"])
def test_r24_1_inputs(ds):
    ctx, ld = R.R13.RealCtx(ds, 0), R.R13._load(ds)
    inp, grid = R.jm_input(ctx, ld, 1)
    sc = R.TIME_SCALE.get(ds, 1.0)
    assert sc == (365.25 if ds.startswith("pbc") else 1.0)
    tL, W = ctx.lm.t_L / sc, ctx.lm.window / sc
    sets = {}
    for part, idx in (("train", ctx.tr), ("val", ctx.va), ("test", ctx.te)):
        ids = {r["id"] for r in inp[f"{part}_surv"]}
        assert ids == {int(i) for i in idx}
        assert {r["id"] for r in inp[f"{part}_long"]} <= ids
        sets[part] = ids
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    end = {r["id"]: r["time"] for r in inp["train_surv"]}
    for r in inp["train_surv"]:
        assert r["entry"] == pytest.approx(tL) and r["time"] > tL
        assert r["time"] == pytest.approx(tL + ctx.y[r["id"]] / sc) and r["event"] == int(ctx.d[r["id"]])
    assert all(r["t"] <= end[r["id"]] + 1e-12 for r in inp["train_long"])
    assert any(r["t"] > tL for r in inp["train_long"])                       # training histories go past t_L
    for part in ("val", "test"):
        assert all(tL - W - 1e-12 <= r["t"] <= tL + 1e-12 for r in inp[f"{part}_long"])
        assert all(r["time"] == pytest.approx(tL) and r["event"] == 0 for r in inp[f"{part}_surv"])
    assert grid[0] == 0.0 and grid[-1] == pytest.approx(float(np.max(ctx.y_tr)))
    assert inp["forms"] == ["lin", "ns2"] and inp["n_chains"] == 3 and inp["n_iter"] == 6500 and inp["n_burnin"] == 1500


@pytest.mark.criterion("R24-2")
def test_r24_2_pairing_with_round23_records():
    from kanfae.models.fbaselines import LinearCox
    rec = None
    for f in sorted((ROOT / "reports/round23").glob("results_lane*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("job_name") == "real_heart_valve_5" and r.get("status") == "ok":
                rec = r
    assert rec is not None
    ctx = R.R13.RealCtx("heart_valve", 5)
    assert (int(ctx.n), len(ctx.te), int(ctx.d_te.sum())) == (rec["n"], rec["n_test"], rec["events_test"])
    X = np.hstack([ctx.lastv(), ctx.lm.z])
    lc = LinearCox().fit(X[ctx.tr], ctx.y_tr, ctx.d_tr)
    m = R.R13.real_metrics(ctx, lambda tt: lc.predict_survival(X[ctx.te], tt), lc.predict_risk(X[ctx.te]))
    for k in ("ibs", "ibs_cond", "harrell_c", "uno_c"):
        assert m[k] == rec["models"]["landmark_cox"][k], k


@pytest.mark.criterion("R24-3")
def test_r24_3_interpolation_and_form_choice():
    grid = np.array([0.0, 1.0, 2.0])
    f = R.surv_fn(grid, [[1.0, 0.8, 0.4], [1.0, 0.9, 0.9]])
    out = f(np.array([0.5, 1.5, 2.0, 5.0]))
    assert np.allclose(out, [[0.9, 0.6, 0.4, 0.4], [0.95, 0.9, 0.9, 0.9]])
    assert R.choose_form({"jm_lin": {"val_ibs_cond": 0.2}, "jm_ns2": {"val_ibs_cond": 0.1}}) == "ns2"
    assert R.choose_form({"jm_lin": {"val_ibs_cond": 0.1}, "jm_ns2": {"val_ibs_cond": 0.1}}) == "lin"
    assert R.choose_form({"jm_lin": {"status": "failed"}, "jm_ns2": {"val_ibs_cond": 0.3}}) == "ns2"
    assert R.choose_form({"jm_lin": {"val_ibs_cond": None}, "jm_ns2": {"status": "failed"}}) is None


@needs_r
@pytest.mark.criterion("R24-4")
def test_r24_4_smoke_fit_and_determinism():
    ctx, ld = R.R13.RealCtx("heart_valve", 0), R.R13._load("heart_valve")
    inp, grid = R.jm_input(ctx, ld, 7, SHORT)
    outs = [R.run_r(inp, timeout=1800) for _ in range(2)]
    assert all(o["status"] == "ok" for o in outs)
    for form in R.FORMS:
        o = outs[0]["forms"][form]
        assert o["status"] == "ok", o
        for part, idx in (("val", ctx.va), ("test", ctx.te)):
            S, _ = R._curves(o[part], idx, grid)
            assert np.all(np.isfinite(S)) and S.min() >= 0.0 and S.max() <= 1.0 + 1e-12
            assert np.allclose(S[:, 0], 1.0, atol=1e-9)
            assert np.all(np.diff(S, axis=1) <= 1e-9)
            S2, _ = R._curves(outs[1]["forms"][form][part], idx, grid)
            assert np.array_equal(S, S2)
