"""Round 21 (D153): the Round-20 NP-KAN-AFT, log-normal KAN-AFT, cross-confirmed anchor and fixed-mu symbolic
distillation on a real longitudinal split (pbc_E2, outer split 0 of the Round-13 harness) with reduced budgets, and the
Round 15-20 regression.

R21-1  b2s_npaft (rank_km) and b2s_lnaft on pbc_E2 split 0: finite conditional and marginal IPCW IBS, Harrell C and Uno
       C in (0.5, 1], D-calibration p in [0, 1], finite survival predictions
R21-2  CrossConfirmRefitAnchor on the real split, not confirming (margin forced high): its test-split survival matrix and
       risk equal the full-validation B2-S ensemble's bit for bit (max |diff| == 0.0), for NP-AFT and Cox
R21-3  distill_and_replace (stages 1-2, guarded substitution, fixed mu = 10, NaN-safe refit, validation rollback) on the
       seed-0 member of b2s_npaft and b2s_cox over the six PBC biomarkers: finite before/after objectives, a complete
       record with per-edge stages and formulas, and a finite returned model
R21-4  the Round 15-20 deterministic tests (tests/test_round15.py .. test_round20.py) pass unchanged (subprocess; its
       records go to a temporary file, not to the suite's record)
"""
import functools
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
pytestmark = [pytest.mark.stage("R21"), pytest.mark.kind("DET")]


@pytest.fixture(scope="module")
def H(tmp_path_factory):
    os.environ["R21_OUT"] = str(tmp_path_factory.mktemp("r21") / "results.jsonl")
    import run_round21_real as H
    import kanfae.models.fkan as K
    torch.set_default_dtype(torch.float64)
    H.ENS_KW.update(lrs=(3e-3,), wds=(1e-4,), seeds=(0, 1))
    o = K.FKANSurv.__init__
    K.FKANSurv.__init__ = functools.partialmethod(lambda self, *a, **k: o(self, *a, **{**k, "max_epochs": 60, "patience": 60}))
    H.CROSS_KW.update(fine_tune_epochs=5, lrs=(3e-3,), wds=(1e-4,))
    yield H
    K.FKANSurv.__init__ = o
    H.ENS_KW.clear()
    H.CROSS_KW.clear()


@pytest.fixture(scope="module")
def split0(H):
    ctx = H.R13.RealCtx("pbc_E2", 0)
    xg, last, z = ctx.front("bspline")["xg"], ctx.lastv(), ctx.lm.z
    h1, h2 = H.R18.split_val(ctx.va, ctx.d.astype(int), 0)
    pk = lambda idx: dict(xg=xg[idx], z=z[idx], y=ctx.y[idx], d=ctx.d[idx].astype(float), last=last[idx])  # noqa: E731
    P = {"tr": pk(ctx.tr), "va": pk(ctx.va), "h1": pk(h1), "h2": pk(h2), "te": pk(ctx.te)}
    return ctx, P


@pytest.mark.criterion("R21-1")
def test_r21_1_real_split_npaft_and_lnaft_smoke(H, split0):
    ctx, P = split0
    rec = dict(models={}, alternatives={}, gates={}, symbolic={})
    for f in ("npaft", "lnaft"):
        full, _ = H.fit_family(f, P, lambda S, r: H.R13.real_metrics(ctx, S, r), rec, cross=False)
        m = rec["models"][f"b2s_{f}"]
        for k in ("ibs", "ibs_cond", "harrell_c", "uno_c", "dcal_p", "dcal_chi2"):
            assert np.isfinite(m[k]), (f, k, m[k])
        assert 0.5 < m["harrell_c"] <= 1.0 and 0.5 < m["uno_c"] <= 1.0, (f, m)
        assert 0.0 <= m["dcal_p"] <= 1.0 and abs(m["dcal_p"] - m["dcal_p_check"]) == 0.0
        S_fn, _ = H.surv_risk(full.members, P["te"])
        tg = H.M.ibs_time_grid(P["te"]["y"], P["tr"]["y"])
        S = S_fn(tg)
        assert np.all(np.isfinite(S)) and np.all((S >= 0) & (S <= 1))


@pytest.mark.criterion("R21-2")
def test_r21_2_real_split_cross_confirm_exact_fallback(H, split0):
    ctx, P = split0
    saved = dict(H.MARGIN)
    try:
        for f in ("npaft", "cox"):
            H.MARGIN[f] = (1e6, False)
            rec = dict(models={}, alternatives={}, gates={}, symbolic={})
            H.fit_family(f, P, lambda S, r: H.R13.real_metrics(ctx, S, r), rec, cross=True)
            g = rec["gates"][f"b2s_anchor_{f}_cross"]
            assert g["confirmed"] is False
            assert g["fallback_max_abs_diff_S"] == 0.0 and g["fallback_max_abs_diff_risk"] == 0.0, g
            a, b = rec["models"][f"b2s_anchor_{f}_cross"], rec["models"][f"b2s_{f}"]
            assert all(a[k] == b[k] for k in ("ibs", "ibs_cond", "harrell_c", "uno_c", "dcal_p"))
            assert len(g["dirs"]) == 2 and all(np.isfinite(dd["loss_ref"]) for dd in g["dirs"])
    finally:
        H.MARGIN.clear()
        H.MARGIN.update(saved)


@pytest.mark.criterion("R21-3")
def test_r21_3_real_split_fixed_mu_symbolic_distillation(H, split0):
    ctx, P = split0
    rec = dict(models={}, alternatives={}, gates={}, symbolic={})
    for f in ("npaft", "cox"):
        full, _ = H.fit_family(f, P, lambda S, r: H.R13.real_metrics(ctx, S, r), rec, cross=False)
        H.distill(f, full, P, lambda S, r: H.R13.real_metrics(ctx, S, r), rec)
        s = rec["symbolic"][f"b2s_{f}"]
        assert np.isfinite(s["val_loss_before"]) and np.isfinite(s["val_loss_after"]), s
        assert s["n_edges"] == len(s["edges"]) > 0 and 0.0 <= s["replacement_rate"] <= 1.0
        assert all({"stage", "accepted", "formula"} <= set(e) for e in s["edges"])
        assert s["rollback"] in (True, False) and (s["rollback"] or s["val_loss_after"] <= s["val_loss_before"])
        assert all(np.isfinite(v) for v in s["after"].values())


@pytest.mark.criterion("R21-4")
def test_r21_4_rounds_15_to_20_regression(tmp_path):
    env = dict(os.environ, KANFAE_TEST_RECORD=str(tmp_path / "records.jsonl"), KANFAE_RUN_ID="r21_4_nested")
    files = [f"tests/test_round{r}.py" for r in range(15, 21)]
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=3000)
    tail = p.stdout.strip().splitlines()[-1]
    assert p.returncode == 0 and "35 passed" in tail and "failed" not in tail, p.stdout[-3000:]
