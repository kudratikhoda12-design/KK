"""Stage G: PBC provenance, outcome coding and leakage criteria (K1, K4) (protocol 4.3, 7.5)."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from kanfae.data.pbc import describe, landmark_dataset, read_pbcseq
from kanfae.data.splits import split_groups

RDA = Path(__file__).resolve().parents[1] / "data" / "raw" / "pbc" / "data" / "pbc.rda"
pytestmark = [pytest.mark.stage("G"), pytest.mark.kind("DET"),
              pytest.mark.skipif(not RDA.exists(), reason="PBC data not downloaded")]


@pytest.fixture(scope="module")
def df():
    return read_pbcseq(RDA)


@pytest.mark.criterion("G-PROV")
def test_provenance_and_structure(df):
    assert hashlib.sha256(RDA.read_bytes()).hexdigest() == "86ed4d908bccc801a01053b4e619a23e92d1b1bf8000a8429c7ced7632de44e2"
    d = describe(df)
    assert d["n_patients"] == 312 and d["n_rows"] == 1945
    assert d["status_counts"] == {0: 143, 1: 29, 2: 140}
    assert d["futime_status_constant_within_patient"] and d["visits_after_followup"] == 0
    assert "futime" in d["columns"] and "time" not in d["columns"]
    assert "trig" not in d["columns"] and "copper" not in d["columns"]


@pytest.mark.criterion("G-OUTCOME")
def test_endpoint_coding(df):
    e1, e2 = landmark_dataset(df, "E1"), landmark_dataset(df, "E2")
    assert len(e1) == len(e2) == 278
    st = e1.meta["status"]
    assert np.array_equal(e1.delta, st == 2) and np.array_equal(e2.delta, (st == 1) | (st == 2))
    assert e1.delta.sum() == 107 and e2.delta.sum() == 107 + 28
    assert np.all(e1.y_obs > 0) and np.array_equal(e1.y_obs, e2.y_obs)
    with pytest.raises(ValueError):
        landmark_dataset(df, "E3")


@pytest.mark.criterion("K4")
def test_K4_inputs_before_landmark(df):
    obs = landmark_dataset(df, "E2")
    assert np.all(obs.meta["max_input_day"] <= 730)
    real = np.arange(obs.times.shape[1])[None, :] < obs.n_obs[:, None]
    assert np.all(obs.times[real] <= 1.0) and np.all(obs.times[real] >= 0)
    assert np.median(obs.n_obs) == 3


@pytest.mark.criterion("K1")
def test_K1_patient_level_split(df):
    obs = landmark_dataset(df, "E2")
    assert len(np.unique(obs.ids)) == len(obs)
    for s in range(5):
        a, b, c = split_groups(obs.ids, s, strata=obs.delta)
        assert not (set(obs.ids[a]) & set(obs.ids[b])) and not (set(obs.ids[a]) & set(obs.ids[c]))
        assert len(a) + len(b) + len(c) == len(obs)
