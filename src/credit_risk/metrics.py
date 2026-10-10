"""Discrimination, calibration and bootstrap utilities."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

EPS = 1e-6


def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Equal-frequency ECE: weighted mean |observed rate - mean predicted PD| per bin."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    ece = 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx):
            ece += len(idx) / len(y) * abs(y[idx].mean() - p[idx].mean())
    return float(ece)


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Logistic recalibration y ~ a + b*logit(p). Perfect calibration: a=0, b=1."""
    z = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def reliability_table(y, p, n_bins: int = 10):
    import pandas as pd
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    rows = [{"bin": b + 1, "n": len(idx), "mean_pd": p[idx].mean(), "observed_rate": y[idx].mean()}
            for b, idx in enumerate(np.array_split(order, n_bins)) if len(idx)]
    return pd.DataFrame(rows)


def summarize(y, p) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    auc = roc_auc_score(y, p)
    intercept, slope = calibration_slope_intercept(y, p)
    return {
        "auc_roc": float(auc), "gini": float(2 * auc - 1), "ks": ks_statistic(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS))),
        "ece": expected_calibration_error(y, p), "mean_pd": float(p.mean()), "default_rate": float(y.mean()),
        "calibration_intercept": intercept, "calibration_slope": slope,
    }


def stratified_bootstrap_ci(y, p, fn=roc_auc_score, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile CI of ``fn(y, p)`` resampling defaulters and non-defaulters separately."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    stats_ = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats_[b] = fn(y[idx], p[idx])
    lo, hi = np.quantile(stats_, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
