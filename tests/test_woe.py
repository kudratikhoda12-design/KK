import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crm.woe import CategoricalBinner, NumericBinner  # noqa: E402


def _sim(n=100_000, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.normal(size=n))
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(-3 + x)))).astype(int))
    return x, y


def test_numeric_bins_monotone_and_min_share():
    x, y = _sim()
    b = NumericBinner().fit(x, y)
    t = b.table.drop(index=-1, errors="ignore").sort_index()
    assert np.all(np.diff(t["dr"].values) >= 0)       # weakly monotone risk
    assert (t["n"] / len(x)).min() >= 0.05 - 1e-9
    assert np.all(np.diff(t["woe"].values) >= 0)


def test_missing_bin_and_transform():
    x, y = _sim()
    x[:3000] = np.nan
    b = NumericBinner().fit(x, y)
    w = b.transform(x)
    assert np.isfinite(w).all()
    assert -1 in b.table.index


def test_categorical_rare_levels_pooled():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.choice(list("abc"), 50_000).tolist() + ["rare"] * 10)
    y = pd.Series(rng.integers(0, 2, len(x)))
    b = CategoricalBinner().fit(x, y)
    assert "OTHER" in b.table.index and "rare" not in b.table.index
