"""Classification / regression metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def classification(y: np.ndarray, p: np.ndarray, fwd: np.ndarray | None = None) -> dict:
    y = y.astype(int)
    yhat = (p > 0.5).astype(int)
    out = {
        "n": int(len(y)), "base_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "log_loss_baseline": float(log_loss(y, np.full(len(y), np.clip(y.mean(), 1e-6, 1 - 1e-6)), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "ece": ece(y, p),
        "pred_up_share": float(yhat.mean()),
    }
    if fwd is not None:
        out["ic_spearman"] = float(spearmanr(p, fwd).statistic)
    return out


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    return [{"bin": b, "mean_pred": float(p[idx == b].mean()), "frac_up": float(y[idx == b].mean()),
             "n": int((idx == b).sum())} for b in range(bins) if (idx == b).any()]
