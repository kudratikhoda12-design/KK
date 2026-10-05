"""Baselines, logistic regression, LightGBM and predictive-performance metrics.

The models output P(next-period return > 0). Predictive metrics alone do NOT
show that a strategy makes money - that is decided in backtest.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import stats as sstats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .stats import bootstrap_stat, hac_mean_test, percentile_ci
from .validation import Fold


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class BaseRate:
    """Predicts the training-window up-rate for every row (no information)."""

    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.p_)
        return np.column_stack([1 - p, p])


def make_lr(C: float) -> Pipeline:
    # Scaling is inside the pipeline, so it is fitted on the training window only.
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(C=C, max_iter=2000))])


def make_lgbm(params: dict) -> LGBMClassifier:
    return LGBMClassifier(**params, random_state=config.RANDOM_SEED, n_jobs=4,
                          deterministic=True, force_row_wise=True, verbose=-1)


def model_factory(name: str, params: dict):
    if name == "base_rate":
        return BaseRate()
    if name == "logreg":
        return make_lr(params["C"])
    if name == "lgbm":
        return make_lgbm(params)
    raise ValueError(name)


# --------------------------------------------------------------------------
# Walk-forward prediction
# --------------------------------------------------------------------------
def walk_forward_predict(data: pd.DataFrame, folds: list[Fold], name: str, params: dict,
                         features: list[str]) -> pd.DataFrame:
    """Fit on each fold's training rows, predict its evaluation rows.

    Also stores sd of the model's TRAINING predictions per fold; the signal
    threshold is scaled by it (past information only).
    """
    usable = data[features + ["y"]].notna().all(axis=1)
    out = []
    for f in folds:
        tr = f.train_mask & usable
        ev = f.eval_mask & usable
        Xtr, ytr = data.loc[tr, features].to_numpy(), data.loc[tr, "y"].to_numpy()
        m = model_factory(name, params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            m.fit(Xtr, ytr)
            p_tr = m.predict_proba(Xtr)[:, 1]
            p_ev = m.predict_proba(data.loc[ev, features].to_numpy())[:, 1]
        out.append(pd.DataFrame({"p": p_ev, "fold": f.name, "train_pred_sd": float(np.std(p_tr)),
                                 "train_base_rate": float(np.mean(ytr))},
                                index=data.index[ev]))
    return pd.concat(out) if out else pd.DataFrame(columns=["p", "fold", "train_pred_sd"])


def fit_final(data: pd.DataFrame, mask: pd.Series, name: str, params: dict, features: list[str]):
    usable = mask & data[features + ["y"]].notna().all(axis=1)
    m = model_factory(name, params)
    m.fit(data.loc[usable, features].to_numpy(), data.loc[usable, "y"].to_numpy())
    return m


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def classification_metrics(y: np.ndarray, p: np.ndarray, r: np.ndarray | None = None) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p > 0.5).astype(int)
    out = dict(n=len(y), base_rate=float(y.mean()), mean_p=float(p.mean()),
               auc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
               logloss=float(log_loss(y, p, labels=[0, 1])),
               brier=float(brier_score_loss(y, p)),
               accuracy=float(accuracy_score(y, pred)),
               precision=float(precision_score(y, pred, zero_division=0)),
               recall=float(recall_score(y, pred, zero_division=0)),
               f1=float(f1_score(y, pred, zero_division=0)),
               share_predicted_up=float(pred.mean()))
    if r is not None:
        out["ic_spearman"] = float(sstats.spearmanr(p, r).statistic)
    return out


def confusion(y, p) -> pd.DataFrame:
    cm = confusion_matrix(np.asarray(y, int), (np.asarray(p) > 0.5).astype(int), labels=[0, 1])
    return pd.DataFrame(cm, index=["actual_down", "actual_up"], columns=["pred_down", "pred_up"])


def auc_bootstrap_ci(y, p, mean_block: float = config.BOOTSTRAP_BLOCK_HOURS,
                     n_boot: int = config.BOOTSTRAP_N) -> tuple[float, float]:
    y, p = np.asarray(y, int), np.asarray(p, float)

    def _auc(a, b):
        return roc_auc_score(a, b) if 0 < a.mean() < 1 else np.nan

    return percentile_ci(bootstrap_stat((y, p), _auc, mean_block, n_boot))


def logloss_improvement_test(y, p_model, p_base) -> dict:
    """Per-observation log-loss difference (base - model); HAC test that its mean > 0."""
    y = np.asarray(y, float)
    pm = np.clip(np.asarray(p_model, float), 1e-6, 1 - 1e-6)
    pb = np.clip(np.asarray(p_base, float), 1e-6, 1 - 1e-6)
    ll_m = -(y * np.log(pm) + (1 - y) * np.log(1 - pm))
    ll_b = -(y * np.log(pb) + (1 - y) * np.log(1 - pb))
    t = hac_mean_test(ll_b - ll_m)
    t["p_value_one_sided"] = t["p_value"] / 2 if t["t"] > 0 else 1 - t["p_value"] / 2
    return t


def calibration_deciles(y, p, r=None, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed up-rate by decile, with Wilson 95% intervals.

    Same idea as the PD risk-decile calibration in the credit module, plus the
    mean forward return per decile (the economically relevant column).
    """
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    if r is not None:
        d["r"] = np.asarray(r, float)
    d["decile"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False) + 1
    g = d.groupby("decile")
    out = g.agg(n=("y", "size"), mean_p=("p", "mean"), observed_up=("y", "mean"))
    z = 1.96
    n, ph = out["n"], out["observed_up"]
    centre = (ph + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    out["ci_low"], out["ci_high"] = centre - half, centre + half
    if r is not None:
        out["mean_fwd_ret_bp"] = 1e4 * g["r"].mean()
    return out.reset_index()


def feature_importance(model, features: list[str]) -> pd.DataFrame:
    if isinstance(model, Pipeline):
        coef = model.named_steps["lr"].coef_[0]
        return pd.DataFrame({"feature": features, "std_coef": coef,
                             "abs_std_coef": np.abs(coef)}).sort_values("abs_std_coef", ascending=False)
    if isinstance(model, LGBMClassifier):
        gain = model.booster_.feature_importance("gain")
        return pd.DataFrame({"feature": features, "gain": gain,
                             "gain_share": gain / gain.sum()}).sort_values("gain", ascending=False)
    return pd.DataFrame()


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index with decile bins taken from the reference sample.

    Same definition as the credit module's vintage PSI:
    sum((cur% - ref%) * ln(cur% / ref%)). Rule of thumb: < 0.1 stable,
    0.1-0.25 moderate shift, > 0.25 large shift.
    """
    ref = np.asarray(reference, float); ref = ref[np.isfinite(ref)]
    cur = np.asarray(current, float); cur = cur[np.isfinite(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.nan
    r = np.histogram(np.clip(ref, edges[0], edges[-1]), edges)[0] / len(ref)
    c = np.histogram(np.clip(cur, edges[0], edges[-1]), edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_psi_by_block(data: pd.DataFrame, features: list[str], reference_mask: pd.Series,
                         blocks: pd.Series) -> pd.DataFrame:
    """PSI of each feature in each evaluation block vs the reference (development) period."""
    rows = []
    for b, idx in blocks.groupby(blocks).groups.items():
        rows.append({"block": b, **{f: psi(data.loc[reference_mask, f], data.loc[idx, f]) for f in features}})
    return pd.DataFrame(rows)
