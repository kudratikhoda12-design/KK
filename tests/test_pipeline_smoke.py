"""End-to-end smoke run on the real data (quick mode). Opt in with RUN_SLOW=1 - needs the download."""
import json
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 to run the end-to-end smoke test")


def test_quick_pipeline_end_to_end(tmp_path):
    from credit_risk.pipeline import run
    out = run(quick=True, out_dir=tmp_path, skip_robustness=True)
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["test_metrics"]["rf"]["auc_roc"] > 0.8
    assert saved["test_metrics"]["rf"]["auc_roc"] > saved["test_metrics"]["altman_z"]["auc_roc"]
    assert (tmp_path / "watchlist_full.csv").exists() and (tmp_path / "figures" / "roc_test.png").exists()
    assert out["threshold"]["t_star"] < 0.5
