"""Test fixtures.

SYNTHETIC DATA LIVES ONLY IN THE TEST SUITE. It is used to check that the code
is correct (timing, leakage, accounting) and that the pipeline neither invents
signals (negative control) nor misses real ones (positive control). It is never
presented as, or mixed with, Binance data.
"""

import numpy as np
import pandas as pd
import pytest


def make_synthetic_grid(start="2021-01-01", days=60, beta=0.0, sigma_m=0.0008, seed=0):
    """Regular 1-minute OHLCV grid from a random walk.

    beta > 0 injects hourly momentum: each hour's drift = beta x previous hour's return.
    beta = 0 is a pure random walk: no feature can predict future returns.
    """
    rng = np.random.default_rng(seed)
    n_hours = days * 24
    r = np.empty((n_hours, 60))
    prev = 0.0
    for h in range(n_hours):
        r[h] = beta * prev / 60 + sigma_m * rng.standard_normal(60)
        prev = r[h].sum()
    r = r.ravel()
    idx = pd.date_range(start, periods=len(r), freq="1min", tz="UTC", name="open_time")
    close = 30000 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[30000.0], close[:-1]])
    wiggle = np.abs(rng.standard_normal(len(r))) * sigma_m / 2
    vol = rng.lognormal(3, 0.5, len(r))
    g = pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) * np.exp(wiggle),
        "low": np.minimum(open_, close) * np.exp(-wiggle), "close": close,
        "volume": vol, "quote_volume": vol * close, "n_trades": rng.integers(50, 500, len(r)).astype(float),
        "taker_buy_base": vol * rng.uniform(0.3, 0.7, len(r)),
    }, index=idx)
    g["taker_buy_quote"] = g["taker_buy_base"] * close
    g["present"] = True
    return g.asfreq("1min")


@pytest.fixture(scope="session")
def small_grid():
    return make_synthetic_grid(days=20, seed=1)
