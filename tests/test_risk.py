import numpy as np
import pandas as pd

from src.risk import deflated_sharpe, kupiec_pof, max_drawdown, var_es


def test_max_drawdown_known_path():
    r = pd.Series([0.10, -0.50, 0.20])
    assert np.isclose(max_drawdown(r), -0.5)


def test_historical_var_es_on_known_sample():
    x = pd.Series(np.linspace(-0.05, 0.05, 1001))
    t = var_es(x, levels=(0.95,))
    h = t[t["method"] == "historical"].iloc[0]
    assert np.isclose(h["VaR"], 0.045, atol=1e-3)
    assert h["ES"] > h["VaR"]


def test_kupiec_accepts_correct_var_model():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.standard_normal(5000) * 0.02)
    res = kupiec_pof(x, level=0.99, window=500)
    assert res["p_value"] > 0.01


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(1)
    net = rng.normal(0.0005, 0.01, 5000)
    few = deflated_sharpe(net, rng.normal(0, 0.01, 5))["dsr"]
    many = deflated_sharpe(net, rng.normal(0, 0.01, 500))["dsr"]
    assert many < few
