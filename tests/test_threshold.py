import numpy as np
import pytest

from credit_risk import threshold as thr


def _toy(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 14, n)                 # calibrated PDs by construction
    y = (rng.random(n) < p).astype(int)
    lgd = rng.uniform(0.2, 0.7, n)
    ead = rng.lognormal(0, 0.8, n)
    return p, y, lgd * ead, 0.03 * ead, lgd


def test_cost_curve_matches_brute_force():
    p, y, cfn, cfp, _ = _toy(300)
    curve = thr.cost_curve(p, y, cfn, cfp)
    for t, cost in curve.sample(15, random_state=0)[["threshold", "cost"]].to_numpy():
        if np.isfinite(t):
            assert cost == pytest.approx(thr.evaluate_rule(p, y, cfn, cfp, t)["cost"])
    assert curve["cost"].iloc[-1] == pytest.approx(cfn[y == 1].sum())   # flag nobody


def test_optimum_is_global_minimum_and_beats_naive_half():
    p, y, cfn, cfp, _ = _toy()
    t = thr.optimal_threshold(p, y, cfn, cfp)
    best = thr.evaluate_rule(p, y, cfn, cfp, t)["cost"]
    for other in (0.05, 0.1, 0.2, 0.5):
        assert best <= thr.evaluate_rule(p, y, cfn, cfp, other)["cost"] + 1e-9
    assert t < 0.5


def test_optimum_near_bayes_threshold_for_calibrated_pd():
    p, y, cfn, cfp, lgd = _toy(60000, seed=3)
    t_boot, (lo, hi) = thr.bootstrap_threshold(p, y, cfn, cfp, n_boot=40)
    bayes = float(np.mean(thr.bayes_threshold(lgd, 0.03)))
    assert abs(t_boot - bayes) < 0.06       # cost curve is flat near the optimum
    assert lo <= t_boot <= hi


def test_bayes_threshold_formula():
    assert thr.bayes_threshold(0.45, 0.03) == pytest.approx(0.03 / 0.48)
    assert thr.bayes_threshold(np.array([0.3, 0.6]), 0.03)[0] > thr.bayes_threshold(np.array([0.3, 0.6]), 0.03)[1]


def test_evaluate_flags_counts():
    y = np.array([1, 1, 0, 0, 0])
    flag = np.array([True, False, True, False, False])
    r = thr.evaluate_flags(flag, y, cost_fn=np.full(5, 10.0), cost_fp=np.full(5, 1.0))
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 2)
    assert r["cost"] == 11.0 and r["recall"] == 0.5 and r["precision"] == 0.5
