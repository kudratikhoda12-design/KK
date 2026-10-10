"""SQC (statistical quality control) screening of financial ratios.

Idea: treat healthy firms as the "in-control process". For every ratio, fit
Shewhart-style individuals-chart limits on healthy firms only. A ratio is a good
early-warning signal when defaulters breach a limit more often than healthy firms
do, i.e. the chart's detection rate beats its false-alarm rate.

Design choices (each one fixes a failure seen on this data):

* *Probability limits, not mean +/- 3 sigma.* Ratios are heavy-tailed and many are
  non-negative and right-skewed, so a symmetric ``centre - 3 sigma`` lower limit falls
  below zero and can never signal (it silently discarded the liquidity ratios). The
  limits are therefore the ``tail_prob`` and ``1 - tail_prob`` empirical quantiles of
  the healthy firms, the distribution-free "probability limits" of non-normal charts.
  Every ratio then has the same, known false-alarm rate per tail, so ratios are
  directly comparable.
* *Each tail is charted separately* (an LCL chart and a UCL chart), because the adverse
  direction differs by ratio and healthy firms with extreme *favourable* values would
  otherwise mask the signal.
* *Robust scale.* The in-control z-score uses the healthy median and a semi-MAD on each
  side of it (``1.4826 * MAD`` of the lower / upper half), robust to outliers and skew.

Screening pipeline (fitted on training data only, inside CV folds):

1. data quality   - drop ratios that are too sparse or have zero spread
2. SQC signal     - one-sided two-proportion z-test (detection rate > false-alarm rate)
                    on each tail, Benjamini-Hochberg corrected over all 2 x p tests
3. redundancy     - of two ratios with |Spearman| > max_corr keep the stronger signal
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

MAD_TO_SIGMA = 1.4826


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


def one_sided_two_proportion_p(x1: np.ndarray, n1: np.ndarray, x0: np.ndarray, n0: np.ndarray) -> np.ndarray:
    """P-value for H1: p1 > p0 (pooled-variance z-test), vectorised over ratios."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p1, p0 = x1 / n1, x0 / n0
        pooled = (x1 + x0) / (n1 + n0)
        se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
        z = np.where(se > 0, (p1 - p0) / se, 0.0)
    return stats.norm.sf(z)


class SQCRatioScreener(BaseEstimator, TransformerMixin):
    """Fit control limits on healthy firms, screen ratios, and re-express them.

    Parameters
    ----------
    tail_prob : false-alarm probability per tail of the control chart (0.025 is the
        distribution-free analogue of the classic 2-sigma warning limit).
    max_missing : drop ratios whose training missing rate exceeds this.
    fdr : BH q-value below which a ratio counts as a significant SQC signal.
    max_corr : redundancy threshold (|Spearman|); ``None`` disables pruning.
    screen : if False, keep every ratio (ablation: "no screening").
    scale : "none" (raw ratios) | "clip" (in-control z-score clipped to +/- clip_k)
        | "arcsinh" (in-control z-score, variance-stabilised) - how kept ratios are output.
    clip_k : half-width (in robust sigmas) used when ``scale="clip"``.
    add_alarm_count : append the share of kept ratios breaching their adverse-tail
        control limit (a per-firm "number of chart alarms" feature).
    """

    def __init__(self, tail_prob=0.025, max_missing=0.40, fdr=0.05, max_corr=0.95,
                 screen=True, scale="none", clip_k=10.0, add_alarm_count=False):
        self.tail_prob = tail_prob
        self.max_missing = max_missing
        self.fdr = fdr
        self.max_corr = max_corr
        self.screen = screen
        self.scale = scale
        self.clip_k = clip_k
        self.add_alarm_count = add_alarm_count

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        X = validate_data(self, X, ensure_all_finite="allow-nan", dtype=np.float64)
        y = np.asarray(y).astype(int)
        if set(np.unique(y)) - {0, 1} or y.sum() == 0 or y.sum() == len(y):
            raise ValueError("SQC screening needs both healthy (0) and defaulted (1) firms.")
        p_cols = X.shape[1]
        names = (np.asarray(self.feature_names_in_) if hasattr(self, "feature_names_in_")
                 else np.array([f"x{i}" for i in range(p_cols)]))
        healthy, default = X[y == 0], X[y == 1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            centre = np.nanmedian(healthy, axis=0)
            sigma = MAD_TO_SIGMA * np.nanmedian(np.abs(healthy - centre), axis=0)
            sigma = np.where(sigma > 0, sigma, np.nanstd(healthy, axis=0))  # MAD = 0 fallback
            below = np.where(healthy < centre, centre - healthy, np.nan)
            above = np.where(healthy > centre, healthy - centre, np.nan)
            sigma_low = MAD_TO_SIGMA * np.nanmedian(below, axis=0)   # semi-MAD, lower half
            sigma_high = MAD_TO_SIGMA * np.nanmedian(above, axis=0)  # semi-MAD, upper half
            lcl = np.nanquantile(healthy, self.tail_prob, axis=0)
            ucl = np.nanquantile(healthy, 1 - self.tail_prob, axis=0)
            default_centre = np.nanmedian(default, axis=0)
        degenerate = ~np.isfinite(sigma) | (sigma <= 0) | ~np.isfinite(lcl) | (ucl <= lcl)
        safe_sigma = np.where(degenerate, 1.0, sigma)
        sigma_low = np.where(np.isfinite(sigma_low) & (sigma_low > 0), sigma_low, safe_sigma)
        sigma_high = np.where(np.isfinite(sigma_high) & (sigma_high > 0), sigma_high, safe_sigma)

        obs_h, obs_d = ~np.isnan(healthy), ~np.isnan(default)
        n_h, n_d = obs_h.sum(0).astype(float), obs_d.sum(0).astype(float)
        rates, pvals = {}, {}
        for tail, (h_out, d_out) in {
            "low": ((healthy < lcl) & obs_h, (default < lcl) & obs_d),
            "high": ((healthy > ucl) & obs_h, (default > ucl) & obs_d),
        }.items():
            xh, xd = h_out.sum(0).astype(float), d_out.sum(0).astype(float)
            rates[tail] = (np.divide(xh, n_h, out=np.zeros(p_cols), where=n_h > 0),
                           np.divide(xd, n_d, out=np.zeros(p_cols), where=n_d > 0))
            pvals[tail] = one_sided_two_proportion_p(xd, n_d, xh, n_h)
        unusable = degenerate | (n_d == 0) | (n_h == 0)
        p_all = np.where(np.concatenate([unusable, unusable]), 1.0, np.concatenate([pvals["low"], pvals["high"]]))
        q_all = benjamini_hochberg(p_all)
        q_low, q_high = q_all[:p_cols], q_all[p_cols:]
        side = np.where(q_low <= q_high, -1, 1)  # -1: low values are adverse, +1: high values are adverse
        qval = np.minimum(q_low, q_high)
        pval = np.where(side == -1, p_all[:p_cols], p_all[p_cols:])
        far = np.where(side == -1, rates["low"][0], rates["high"][0])
        dr = np.where(side == -1, rates["low"][1], rates["high"][1])
        missing = np.isnan(X).mean(0)
        shift = np.where(degenerate, 0.0, (default_centre - centre) /
                         np.where(default_centre < centre, sigma_low, sigma_high))

        status = np.array(["kept"] * p_cols, dtype=object)
        if self.screen:
            status[missing > self.max_missing] = "dropped: missing"
            status[degenerate] = "dropped: degenerate"
            signal = (qval < self.fdr) & (dr > far)
            status[(status == "kept") & ~signal] = "dropped: no SQC signal"
            if self.max_corr is not None:
                self._prune_redundant(X, status, dr - far)
        if not (status == "kept").any():
            raise ValueError("SQC screening removed every ratio; relax fdr / max_missing.")

        self.n_features_in_ = p_cols
        self.centre_, self.sigma_low_, self.sigma_high_ = centre, sigma_low, sigma_high
        self.lcl_, self.ucl_, self.side_ = lcl, ucl, side
        self.kept_ = np.flatnonzero(status == "kept")
        self.report_ = pd.DataFrame({
            "ratio": names, "missing_rate": missing, "centre": centre,
            "sigma_low": sigma_low, "sigma_high": sigma_high,
            "lcl": lcl, "ucl": ucl, "adverse_tail": np.where(side == -1, "low", "high"),
            "false_alarm_rate": far, "detection_rate": dr,
            "lift": np.divide(dr, far, out=np.full_like(dr, np.nan), where=far > 0),
            "shift_sigma": shift, "p_value": pval, "q_value": qval, "status": status,
        })
        return self

    def _prune_redundant(self, X, status, strength):
        idx = np.flatnonzero(status == "kept")
        corr = pd.DataFrame(X[:, idx]).corr(method="spearman").abs().to_numpy(copy=True)
        np.fill_diagonal(corr, 0.0)
        for i in np.argsort(-strength[idx]):  # strongest first, so it is the one retained
            if status[idx[i]] != "kept":
                continue
            for j in np.flatnonzero(corr[i] > self.max_corr):
                if status[idx[j]] == "kept":
                    status[idx[j]] = "dropped: redundant"

    # ------------------------------------------------------------ transform
    def transform(self, X):
        check_is_fitted(self, "kept_")
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan", dtype=np.float64)
        adverse = np.where(self.side_ == -1, X < self.lcl_, X > self.ucl_) & ~np.isnan(X)
        Xk = X[:, self.kept_]
        centre = self.centre_[self.kept_]
        sigma_low, sigma_high = self.sigma_low_[self.kept_], self.sigma_high_[self.kept_]
        if self.scale in ("clip", "arcsinh"):
            z = np.where(Xk < centre, (Xk - centre) / sigma_low, (Xk - centre) / sigma_high)  # in-control z-score
        if self.scale == "clip":
            Xk = np.clip(z, -self.clip_k, self.clip_k)
        elif self.scale == "arcsinh":
            Xk = np.arcsinh(z)
        elif self.scale != "none":
            raise ValueError(f"unknown scale: {self.scale!r}")
        if self.add_alarm_count:
            observed = (~np.isnan(X[:, self.kept_])).sum(1)
            share = adverse[:, self.kept_].sum(1) / np.maximum(observed, 1)
            Xk = np.column_stack([Xk, share])
        return Xk

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "kept_")
        names = list(self.report_["ratio"].to_numpy()[self.kept_])
        return np.array(names + (["alarm_share"] if self.add_alarm_count else []), dtype=object)
