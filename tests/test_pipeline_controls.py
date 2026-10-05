"""End-to-end pipeline controls on SYNTHETIC data (never presented as results about BTC).

Negative control: a pure random walk contains no predictable signal, so the
full pipeline (features -> walk-forward LR -> signal -> costed backtest) must
show no out-of-sample skill and must lose money after costs.

Positive control: with injected hourly momentum (beta = 0.3) the same
pipeline must detect skill. Together they show the machinery is neither
biased towards finding signals nor blind to real ones.
"""

import numpy as np
import pandas as pd

from src import config
from src.backtest import positions_from_probs, run_backtest
from src.features import FEATURES, build_dataset
from src.models import auc_bootstrap_ci, classification_metrics, walk_forward_predict
from src.risk import performance
from src.validation import walk_forward_folds

from conftest import make_synthetic_grid


def _run(beta, seed):
    grid = make_synthetic_grid(start="2021-01-01", days=240, beta=beta, seed=seed)
    d = build_dataset(grid)
    folds = walk_forward_folds(d, pd.Timestamp("2021-03-01", tz="UTC"),
                               pd.Timestamp("2021-08-28", tz="UTC"), block_months=1)
    pr = walk_forward_predict(d, folds, "logreg", {"C": 0.1}, FEATURES)
    y = d.loc[pr.index, "y"]
    auc = classification_metrics(y, pr["p"])["auc"]
    lo, hi = auc_bootstrap_ci(y, pr["p"], n_boot=500)
    pos = positions_from_probs(pr["p"], pr["train_pred_sd"], 0.0, "long_short")
    bt = run_backtest(d, pos, config.BASE_COST_PER_SIDE)
    return auc, lo, hi, performance(bt, 8766)


def test_negative_control_random_walk_shows_no_edge():
    """Across independent random walks: no average skill, no gross edge, always a net loss.

    A single run can show a gross Sharpe of +-1.5 by chance (the standard error of
    a half-year annualised Sharpe is ~1.4), so bias is judged across seeds.
    """
    runs = [_run(beta=0.0, seed=s) for s in range(6)]
    aucs = np.array([r[0] for r in runs])
    gross = np.array([r[3]["gross_sharpe"] for r in runs])
    net = np.array([r[3]["sharpe"] for r in runs])
    assert abs(aucs.mean() - 0.5) < 0.01, aucs
    assert abs(gross.mean()) < 2 * gross.std(ddof=1) / np.sqrt(len(gross)) + 0.5, gross
    assert (net < 0).all(), net                  # trading noise always loses after costs


def test_positive_control_injected_momentum_is_detected():
    auc, lo, hi, perf = _run(beta=0.3, seed=7)
    assert lo > 0.5, (auc, lo, hi)
    assert auc > 0.55
