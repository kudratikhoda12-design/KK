"""Owner decision D80 / D82: the data-attainable reference C_ceiling (D75 definition), criteria CEIL-1..CEIL-3."""
import sys
from pathlib import Path

import numpy as np
import pytest

from kanfae.data.synthetic import SynthConfig, generate
from kanfae.evaluation.ceiling import ceiling_C, estimate_one, estimate_parameters, plugin_eta

pytestmark = [pytest.mark.stage("G5"), pytest.mark.kind("DET")]


@pytest.mark.criterion("CEIL-1")
def test_CEIL1_recovers_parameters_from_noiseless_curves():
    t = np.sort(np.random.default_rng(0).uniform(0, 1, 40))
    for a, b, A, om, ph in [(0.3, -1.2, 1.5, 0.7, 2.0), (-1.0, 0.5, 0.6, 1.9, 5.0), (0.0, 2.0, 2.0, 1.2, 0.1)]:
        est = estimate_one(t, a + b * t, A * np.sin(2 * np.pi * om * t + ph))
        np.testing.assert_allclose(est, (a, b, A, om), atol=1e-6)


@pytest.mark.criterion("CEIL-2")
def test_CEIL2_per_individual_no_cross_individual_fitting():
    obs, _ = generate(SynthConfig(n=60, censoring=0.0, sigma_y=0.05, seed=91))
    full = estimate_parameters(obs)
    sub = estimate_parameters(obs, np.arange(10, 30))
    np.testing.assert_array_equal(sub, full[10:30])            # an individual's estimate ignores everyone else


@pytest.mark.criterion("CEIL-3")
def test_CEIL3_identical_to_the_D75_diagnostic():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from diag_data_ceiling import parametric, true_eta
    obs, _ = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=701))
    idx = np.arange(40)
    mine = estimate_parameters(obs, idx)
    ref = np.array([parametric(obs.times[i, obs.mask[i, :, 0]], obs.values[i, obs.mask[i, :, 0], 0],
                               obs.values[i, obs.mask[i, :, 0], 1]) for i in idx])
    np.testing.assert_array_equal(mine, ref)
    np.testing.assert_allclose(plugin_eta(mine), true_eta(*ref.T), rtol=0, atol=1e-12)
    import json
    rec = [json.loads(l) for l in (Path(__file__).resolve().parents[1] / "reports/diag_data_ceiling_seeds.jsonl")
           .read_text().splitlines() if json.loads(l)["seed"] == 701][0]
    assert abs(ceiling_C(obs, np.arange(2000)) - rec["parametric_plugin_C"]) < 1e-12
