"""Cost-sensitive decision threshold.

A missed default costs ``LGD * EAD`` (cost_fn); a false alarm - restricting a good
borrower - costs the foregone margin ``margin * EAD`` (cost_fp). The threshold minimises
total cost, not error count, so it sits far below the naive 0.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def cost_curve(p, y, cost_fn, cost_fp) -> pd.DataFrame:
    """Total cost of the rule "flag if PD >= t" for every distinct t (plus flag-none)."""
    p, y = np.asarray(p, float), np.asarray(y).astype(int)
    cost_fn, cost_fp = np.asarray(cost_fn, float), np.asarray(cost_fp, float)
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    miss = np.concatenate([[0.0], np.cumsum(cost_fn[order] * ys)])          # defaulters below t, not flagged
    alarm = np.concatenate([[0.0], np.cumsum(cost_fp[order] * (1 - ys))])   # good firms below t, not flagged
    n = len(p)
    thresholds = np.unique(ps)
    k = np.searchsorted(ps, thresholds, side="left")                        # firms with p < t
    cost = miss[k] + (alarm[n] - alarm[k])
    thresholds = np.append(thresholds, np.inf)                              # flag nobody
    cost = np.append(cost, miss[n])
    flagged = np.append(n - k, 0) / n
    return pd.DataFrame({"threshold": thresholds, "cost": cost, "flagged_share": flagged})


def optimal_threshold(p, y, cost_fn, cost_fp) -> float:
    curve = cost_curve(p, y, cost_fn, cost_fp)
    finite = curve[np.isfinite(curve["threshold"])]
    return float(finite.loc[finite["cost"].idxmin(), "threshold"])


def bootstrap_threshold(p, y, cost_fn, cost_fp, n_boot: int = 300, seed: int = 0):
    """Median optimum over bootstrap resamples: the raw argmin of a step-wise cost curve is noisy."""
    p, y = np.asarray(p, float), np.asarray(y).astype(int)
    cost_fn, cost_fp = np.asarray(cost_fn, float), np.asarray(cost_fp, float)
    rng = np.random.default_rng(seed)
    opts = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(p), len(p))
        opts[b] = optimal_threshold(p[idx], y[idx], cost_fn[idx], cost_fp[idx])
    return float(np.median(opts)), (float(np.quantile(opts, 0.05)), float(np.quantile(opts, 0.95)))


def evaluate_flags(flag, y, cost_fn, cost_fp) -> dict:
    """Confusion counts and total cost for any boolean flag vector (scalar or per-firm rule)."""
    flag, y = np.asarray(flag, bool), np.asarray(y).astype(int)
    cost_fn, cost_fp = np.asarray(cost_fn, float), np.asarray(cost_fp, float)
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    cost = float(cost_fn[~flag & (y == 1)].sum() + cost_fp[flag & (y == 0)].sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "flagged_share": float(flag.mean()),
            "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1), "cost": cost}


def evaluate_rule(p, y, cost_fn, cost_fp, threshold: float) -> dict:
    return {"threshold": float(threshold),
            **evaluate_flags(np.asarray(p, float) >= threshold, y, cost_fn, cost_fp)}


def youden_threshold(p, y) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def bayes_threshold(lgd, margin: float):
    """Theoretical cut-off for *calibrated* PD: flag iff PD*LGD > (1-PD)*margin  <=>  PD > margin/(margin+LGD).
    EAD cancels because both costs scale with exposure."""
    lgd = np.asarray(lgd, float)
    return margin / (margin + lgd)
