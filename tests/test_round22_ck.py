"""Round 22 (D156): CK-6d, the per-channel REML AFCM on the held-out seed 24 (verified unused before Round 22).

CK-6d runs only if the development gate (reports/round22/afcm_gate.json, written by scripts/analyze_round22.py gate
from seeds 22, 914, 966 under the rule fixed in D156) is GO; otherwise it is skipped with the reason and seed 24 stays
unused. Procedure: the CK-6c procedure (S-Add, DGP v2, n 2,000, R11 split and smoothing) with the GO variant
(afcm_reml = fit_afcm(selection="reml_per_channel"), or afcm_reml_ens = fit_afcm_reml_bagged(B=3, seed=0)), AFCM_D123
basis, penalties and pilot grid. Criterion (as CK-6c): identifiable RISE phi_1 < 0.15 AND phi_2 < 0.10. The first run
writes reports/round22/ck6d.json (the formal record; later runs of the deterministic test do not overwrite it); every
run appends to reports/logs/round22_ck.jsonl.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from test_round12_ck import _record  # noqa: E402

pytestmark = [pytest.mark.stage("R22")]
SEED = 24
GATE = ROOT / "reports/round22/afcm_gate.json"
OUT = ROOT / "reports/round22/ck6d.json"


@pytest.mark.criterion("CK-6d")
@pytest.mark.kind("STAT")
def test_CK6d_per_channel_reml_afcm_identifiable_rise():
    if not GATE.exists():
        pytest.skip("CK-6d not run: the Round-22 development gate has not been evaluated (D156)")
    g = json.loads(GATE.read_text())
    if not g["go"]:
        pytest.skip("CK-6d not run: development gate NO-GO (D157); seed 24 stays unused")
    import run_round22 as H
    dat, ptr, X, lam = H.ck_data(SEED)
    m, fit = H.fit_variant(g["go_variant"], ptr)
    s = H.shapes_ck(m, X)
    passed = bool(s["phi1"]["rise_id"] < 0.15 and s["phi2"]["rise_id"] < 0.10)
    r = dict(seed=SEED, variant=g["go_variant"], shapes=s, passed=passed, smoothing_lambda=lam,
             lam_x=fit["lam_x"] if isinstance(fit, dict) else [x["lam_x"] for x in fit],
             lam_t=fit["lam_t"] if isinstance(fit, dict) else [x["lam_t"] for x in fit])
    _record("CK-6d", out=ROOT / "reports/logs/round22_ck.jsonl", threshold_phi1=0.15, threshold_phi2=0.10, **r)
    if not OUT.exists():
        OUT.write_text(json.dumps(r, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    assert passed, f"identifiable RISE {s}"
