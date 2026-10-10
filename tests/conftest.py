import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic():
    """600 healthy + 120 defaulted firms with known structure.

    signal_low   : defaulters shifted down (adverse = low tail)
    signal_high  : defaulters shifted up   (adverse = high tail)
    skewed_low   : non-negative, right-skewed; defaulters sit near zero (the case a symmetric
                   mean - 3 sigma lower limit can never flag)
    noise        : identical distribution in both groups
    """
    rng = np.random.default_rng(0)
    n0, n1 = 600, 120
    y = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    def col(f0, f1):
        return np.r_[f0(n0), f1(n1)]
    X = pd.DataFrame({
        "signal_low": col(lambda n: rng.normal(0, 1, n), lambda n: rng.normal(-2.5, 1, n)),
        "signal_high": col(lambda n: rng.normal(0, 1, n), lambda n: rng.normal(2.5, 1, n)),
        "skewed_low": col(lambda n: rng.lognormal(0.5, 0.6, n), lambda n: rng.lognormal(-2.0, 0.6, n)),
        "noise": rng.normal(0, 1, n0 + n1),
        "constant": np.full(n0 + n1, 3.0),
    })
    X["signal_low_copy"] = X["signal_low"] * 2 + 1  # perfectly redundant
    return X, y
