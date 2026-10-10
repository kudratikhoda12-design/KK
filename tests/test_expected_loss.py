import numpy as np
import pandas as pd
import pytest

from credit_risk import expected_loss as el
from credit_risk.config import LossConfig


def _frame(tl_ta, ln_ta=4.0):
    return pd.DataFrame({"Attr2": tl_ta, "Attr29": np.full(len(tl_ta), ln_ta)})


def test_expected_loss_is_product():
    assert el.expected_loss(0.1, 0.4, 200.0) == pytest.approx(8.0)
    np.testing.assert_allclose(el.expected_loss([0.1, 0.2], [0.5, 0.5], [10, 10]), [0.5, 1.0])


def test_lgd_monotone_in_leverage_and_bounded():
    cfg = LossConfig()
    lgd = el.estimate_lgd(_frame([0.1, 0.5, 1.0, 1.5, 2.5]), cfg)
    assert np.all(np.diff(lgd) >= 0)
    assert lgd.min() >= cfg.lgd_floor and lgd.max() <= cfg.lgd_cap


def test_outliers_in_leverage_are_clipped():
    cfg = LossConfig()
    wild = el.estimate_ead(_frame([-430.0, 72.0, np.nan]), cfg)
    assert np.all(np.isfinite(wild)) and np.all(wild > 0)


def test_ead_scales_with_assets_and_bank_share():
    base = el.estimate_ead(_frame([0.5], 4.0), LossConfig())
    bigger = el.estimate_ead(_frame([0.5], 5.0), LossConfig())
    assert bigger[0] == pytest.approx(base[0] * np.e)
    assert el.estimate_ead(_frame([0.5]), LossConfig(bank_share_of_liabilities=0.2))[0] == pytest.approx(2 * base[0])


def test_score_doubles_odds_every_pdo():
    base = el.pd_to_score(1 / 51)          # odds good:bad = 50:1
    assert base == pytest.approx(600.0)
    assert el.pd_to_score(1 / 101) - base == pytest.approx(20.0)   # 100:1 odds -> +20 points
    assert el.pd_to_score(0.2) < el.pd_to_score(0.02)


def test_grades():
    np.testing.assert_array_equal(el.risk_grade([0.005, 0.02, 0.04, 0.08, 0.2, 0.6]), list("ABCDEF"))


def test_loss_capture_perfect_and_random():
    loss = np.zeros(100); loss[:10] = 1.0
    perfect = el.loss_capture(loss, loss + np.linspace(0, 0.01, 100)[::-1], [0.1, 0.5, 1.0])
    assert perfect[0.1] == pytest.approx(1.0) and perfect[1.0] == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    cap = np.mean([el.loss_capture(loss, rng.random(100), [0.3])[0.3] for _ in range(300)])
    assert cap == pytest.approx(0.3, abs=0.04)


def test_watchlist_is_sorted_by_el_and_shares_sum_to_one():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"Attr2": rng.uniform(0.1, 2, 50), "Attr29": rng.uniform(2, 6, 50)}, index=range(100, 150))
    pd_ = rng.uniform(0.001, 0.5, 50)
    y = (rng.random(50) < pd_).astype(int)
    wl = el.build_watchlist(X, pd_, 0.1, y)
    assert wl["expected_loss"].is_monotonic_decreasing
    assert wl["cum_el_share"].iloc[-1] == pytest.approx(1.0)
    assert set(wl["firm_id"]) == set(range(100, 150))
    row = wl.iloc[0]
    assert row["expected_loss"] == pytest.approx(row["pd"] * row["lgd"] * row["ead"])
    assert (wl["flagged"] == (wl["pd"] >= 0.1)).all()
