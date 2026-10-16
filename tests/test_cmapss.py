"""Stage F: C-MAPSS leakage criteria K3, K4 and loader sanity (protocol 7.5)."""
from pathlib import Path

import numpy as np
import pytest

from kanfae.data.cmapss import informative_sensors, official_test_life, option_a, option_b, read_fd
from kanfae.data.splits import split_groups

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "cmapss"
pytestmark = [pytest.mark.stage("F"), pytest.mark.kind("DET"),
              pytest.mark.skipif(not (RAW / "train_FD001.txt").exists(), reason="C-MAPSS not downloaded")]


@pytest.fixture(scope="module")
def fd001():
    return read_fd(RAW, "FD001")


@pytest.mark.criterion("K3")
def test_K3_windows_of_an_engine_in_one_split(fd001):
    tr, te, rul = fd001
    units = np.array(sorted(tr.unit.unique()))
    sens = informative_sensors(tr, units)
    obs = option_b(tr, units, sens)
    for seed in range(5):
        a, b, c = split_groups(obs.meta["group"], seed)
        ga, gb, gc = (set(obs.meta["group"][p]) for p in (a, b, c))
        assert not (ga & gb) and not (ga & gc) and not (gb & gc)
        assert len(a) + len(b) + len(c) == len(obs)


@pytest.mark.criterion("K4")
def test_K4_no_input_after_landmark(fd001):
    tr, te, rul = fd001
    units = np.array(sorted(tr.unit.unique()))
    sens = informative_sensors(tr, units)
    ob = option_b(tr, units, sens)
    assert np.all(ob.meta["max_input_cycle"] <= ob.meta["landmark"])
    assert np.all(ob.y_obs >= 1) and np.all(ob.delta)
    s0 = int(tr.groupby("unit").cycle.max().min()) - 1
    oa, excl = option_a(tr, units, s0, sens)
    assert np.all(oa.meta["max_input_cycle"] <= s0) and excl == []
    life = official_test_life(te, rul)
    ot = option_b(te, sorted(te.unit.unique()), sens, life=life, last_only=True)
    assert len(ot) == 100 and np.all(ot.meta["max_input_cycle"] == ot.meta["landmark"])
    assert np.array_equal(np.sort(ot.y_obs), np.sort(rul.astype(float)))
