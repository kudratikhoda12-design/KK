"""Round 13 (D129): deterministic checks of the real-data layer and the NaN-aware helpers added for it."""
import numpy as np
import pytest

from kanfae.data import fsim
from kanfae.data.realdata import load_aids, load_pbcseq, load_prothro, single_landmark
from kanfae.functional.smoothing import pad_ragged
from kanfae.models.fbaselines import impute_train_mean, last_values, summaries
from kanfae.models.lme import blup

pytestmark = [pytest.mark.stage("R13"), pytest.mark.kind("DET")]


@pytest.mark.criterion("R13-1")
def test_R13_1_pbc_cohort_matches_rounds_1_to_9():
    """The single-landmark PBC cohort equals kanfae.data.pbc.landmark_dataset (Rounds 1-9): same patients, outcome and
    endpoint coding, and the same (log-transformed) values at the same rescaled times."""
    from pathlib import Path
    from kanfae.data.pbc import landmark_dataset, read_pbcseq
    root = Path(__file__).resolve().parents[1]
    old = landmark_dataset(read_pbcseq(root / "data/raw/pbc/data/pbc.rda"), "E2")
    new = single_landmark(load_pbcseq("E2"), 730.0)
    assert np.array_equal(old.ids, new.ids) and np.allclose(old.y_obs, new.y) and np.array_equal(old.delta, new.delta)
    for i in range(len(new.ids)):
        n = old.n_obs[i]
        assert np.allclose(old.times[i, :n], new.times[i])
        ok = old.mask[i, :n]
        assert np.array_equal(ok, np.isfinite(new.values[i])) and np.allclose(old.values[i, :n][ok], new.values[i][ok])


@pytest.mark.criterion("R13-2")
def test_R13_2_jm_data_counts():
    a, p = load_aids(), load_prothro()
    assert len(a.ids) == 467 and sum(len(t) for t in a.times) == 1405 and a.unit == "months"
    assert len(p.ids) == 488 and p.unit == "years"
    assert max(np.nanmax(v) for v in a.values) < 25                       # CD4 already on the square-root scale (D128)


@pytest.mark.criterion("R13-3")
def test_R13_3_nan_aware_helpers_reduce_to_the_complete_data_versions():
    d = fsim.generate("S-Add", n=50, seed=943, sparse=True)
    T, V, M, n = pad_ragged(d.times, d.values)
    assert all(M[i, :n[i]].all() and not M[i, n[i]:].any() for i in range(50))   # complete data: every real entry kept
    S = summaries(d.times, d.values)
    assert np.isfinite(S).all() and np.allclose(last_values(d.times, d.values),
                                                np.vstack([v[np.argmax(t)] for t, v in zip(d.times, d.values)]))
    vals = [v.copy() for v in d.values]
    vals[0][:, 1] = np.nan                                                 # one channel never observed
    S2 = summaries(d.times, vals)
    assert np.isnan(S2[0, 4:8]).all() and np.allclose(S2[0, :4], S[0, :4]) and np.allclose(S2[1:], S[1:])
    F = impute_train_mean(S2, np.arange(1, 50))
    assert np.allclose(F[0, 4:8], S[1:, 4:8].mean(0))
    assert np.allclose(blup([np.array([])], [np.array([])], np.zeros(2), np.eye(2), 1.0), 0.0)
