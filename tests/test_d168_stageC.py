"""D168 / D170: tests of the Stage-C harness and analysis (written before any Stage-C seed is used).

C-1  the Stage-C seeds are exactly 6100..6209 (110), exclude the Stage-B seed 6001, and appear in no earlier result record
C-2  analysis: an equivalent synthetic data set is "supported", a shifted one "not supported", more than 11 failures of
     either method "not evaluable", and the worst-value imputation is used only for failed replicates
C-3  the reference AFCM's quantile map: training values map to the plotting positions at every grid time, test values
     into [0.5/n, 1 - 0.5/n], monotone in x
C-4  the Stage-C runner refuses to start unless Stage B passed
C-5  (needs the project-local R with mgcv) a small reference fit converges, returns finite surfaces and predictions
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
pytestmark = [pytest.mark.stage("D170"), pytest.mark.kind("DET")]

import analyze_d168_stageC as A  # noqa: E402
import d168_stageC as C  # noqa: E402
from kanfae.models.afcm_mgcv import RSCRIPT, AFCMReference, QuantileMap  # noqa: E402


@pytest.mark.criterion("C-1")
def test_c1_seeds_fresh():
    assert C.SEEDS == list(range(6100, 6210)) and len(C.SEEDS) == 110 and 6001 not in C.SEEDS
    used = set()
    for f in (ROOT / "reports").rglob("*.jsonl"):
        if "d168/stageC" in str(f):
            continue
        for line in f.read_text(errors="ignore").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            for k in ("seed", "data_seed"):
                if isinstance(r, dict) and isinstance(r.get(k), int):
                    used.add(r[k])
    assert not (used & set(C.SEEDS))


def _recs(diffs, fail_b2s=(), fail_ref=()):
    recs = {}
    for i, dv in enumerate(diffs):
        s = 6100 + i
        base = 0.2
        recs[s] = dict(seed=s, status="ok",
                       b2s_cox=dict(status="failed" if s in fail_b2s else "ok",
                                    metrics=dict(shape=dict(mean_id=base + dv))),
                       reference=dict(status="failed" if s in fail_ref else "ok", metrics=dict(shape=dict(mean_id=base))))
    return recs


@pytest.mark.criterion("C-2")
def test_c2_analysis_rules():
    rng = np.random.default_rng(0)
    eq = A.analyse(_recs(rng.normal(0.0, 0.05, 110)))
    assert eq["evaluable"] and eq["H_R25a"]["verdict"].startswith("supported")
    sh = A.analyse(_recs(rng.normal(0.08, 0.05, 110)))
    assert sh["H_R25a"]["verdict"] == "not supported" and "higher" in sh["H_R25a"]["direction"]
    ne = A.analyse(_recs(rng.normal(0.0, 0.05, 110), fail_ref=set(range(6100, 6112))))
    assert not ne["evaluable"] and ne["H_R25a"]["verdict"] == "not evaluable"
    ok = A.analyse(_recs(rng.normal(0.0, 0.05, 110), fail_b2s=set(range(6100, 6111))))
    assert ok["evaluable"] and ok["n_complete_pairs"] == 99 and ok["sensitivity_worst_imputation"]["n"] == 110


@pytest.mark.criterion("C-3")
def test_c3_quantile_map():
    rng = np.random.default_rng(1)
    xg = rng.normal(size=(50, 5, 2))
    q = QuantileMap(xg)
    t = q.transform(xg)
    for g in range(5):
        for d in range(2):
            assert np.allclose(np.sort(t[:, g, d]), (np.arange(1, 51) - 0.5) / 50)
    x = np.linspace(-5, 5, 101)
    v = q(0, x, np.zeros_like(x, dtype=int))
    assert np.all(np.diff(v) >= 0) and v.min() >= 0.5 / 50 - 1e-12 and v.max() <= 1 - 0.5 / 50 + 1e-12


@pytest.mark.criterion("C-4")
def test_c4_runner_needs_stage_b_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "BASE", tmp_path / "stageC")
    with pytest.raises(SystemExit):
        C.guards()


@pytest.mark.criterion("C-5")
@pytest.mark.skipif(not RSCRIPT.exists(), reason="project-local R not installed")
def test_c5_small_reference_fit():
    import test_round11 as R11
    from kanfae.data import fsim
    dat = fsim.generate("S-Add", n=300, seed=11)
    tr, va, te = R11._split(300, 11)
    xg, _ = R11._smoothed(dat, tr)
    m = AFCMReference(k_x=6, k_t=6).fit(R11._pack(dat, xg, tr), R11._pack(dat, xg, te))
    assert m.ok and m.finite and m.info["outer_conv"] == "full convergence"
    S = m.predict_survival_test(np.array([0.1, 0.5, 1.0]))
    assert S.shape == (len(te), 3) and np.all((S >= 0) & (S <= 1)) and np.all(np.diff(S, axis=1) <= 1e-12)
