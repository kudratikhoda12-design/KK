"""PD calibration: map model scores to probabilities that match observed default rates.

Cross-fitted calibration: out-of-fold (OOF) scores are produced once by the base model;
a calibration map (Platt sigmoid or isotonic) is learnt on OOF scores of *other* firms to
evaluate each method honestly, and learnt on all OOF scores for the final map. The base
model is refitted on all training firms and the final map is applied to its scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config
from .metrics import EPS, expected_calibration_error

METHODS = ("none", "sigmoid", "isotonic")


def _logit(s):
    s = np.clip(np.asarray(s, float), EPS, 1 - EPS)
    return np.log(s / (1 - s))


class ScoreCalibrator:
    """Learn p = f(score). ``none`` passes scores through (they are already probabilities)."""

    def __init__(self, method: str = "sigmoid"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method

    def fit(self, scores, y):
        scores = np.asarray(scores, float)
        if self.method == "sigmoid":      # Platt scaling on the logit of the score
            self._model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(scores).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, y)
        return self

    def predict(self, scores):
        scores = np.asarray(scores, float)
        if self.method == "none":
            return scores
        if self.method == "sigmoid":
            return self._model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]
        # isotonic output is a step function; a 1e-6 slope on the raw score breaks ties so the
        # ranking (and hence AUC) of the underlying model is preserved.
        return np.clip(self._model.predict(scores) + 1e-6 * scores, 0.0, 1.0)


def cross_fit(scores, y, method: str, folds: int = config.CV_FOLDS, seed: int = config.RANDOM_STATE + 2):
    """OOF calibrated probabilities for ``scores`` (each firm calibrated by a map fitted without it)."""
    scores, y = np.asarray(scores, float), np.asarray(y).astype(int)
    out = np.empty_like(scores)
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(scores, y):
        out[va] = ScoreCalibrator(method).fit(scores[tr], y[tr]).predict(scores[va])
    return out


def select_method(oof_scores, y):
    """Choose the calibration method with the lowest cross-fitted Brier score.

    Returns (best_method, comparison_table, {method: calibrated OOF PDs}).
    """
    rows, calibrated = [], {}
    for m in METHODS:
        p = cross_fit(oof_scores, y, m)
        calibrated[m] = p
        rows.append({"method": m, "brier": brier_score_loss(y, p),
                     "log_loss": log_loss(y, np.clip(p, EPS, 1 - EPS)),
                     "ece": expected_calibration_error(y, p), "auc_roc": roc_auc_score(y, p)})
    table = pd.DataFrame(rows).set_index("method")
    return str(table["brier"].idxmin()), table, calibrated


def shift_base_rate(pd_: np.ndarray, sample_rate: float, target_rate: float) -> np.ndarray:
    """Prior-shift correction for deploying on a portfolio whose default rate differs from the
    training sample's (Saerens/Elkan odds adjustment). Identity when the two rates are equal."""
    pd_ = np.clip(np.asarray(pd_, float), 1e-9, 1 - 1e-9)
    odds = pd_ / (1 - pd_) * (target_rate / (1 - target_rate)) / (sample_rate / (1 - sample_rate))
    return odds / (1 + odds)
