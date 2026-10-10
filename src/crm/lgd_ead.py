"""LGD and EAD models: fractional-logit GLM vs benchmarks, with OOT tests."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

LGD_FORMULA = ("y ~ int_rate + term_m + np.log(funded_amnt) + fico + dti + revol_util_f"
               " + log_inc_f + default_mob + ead_ratio + C(home_ownership) + C(purpose_g)"
               " + C(verification_status)")
EAD_FORMULA = ("y ~ int_rate + term_m + np.log(funded_amnt) + fico + dti + revol_util_f"
               " + log_inc_f + C(home_ownership) + C(purpose_g)")
MAIN_PURPOSES = {"debt_consolidation", "credit_card", "home_improvement", "other",
                 "major_purchase", "small_business"}


def prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["revol_util_f"] = d["revol_util"].fillna(d["revol_util"].median()).clip(0, 150)
    d["log_inc_f"] = d["log_inc"].fillna(d["log_inc"].median())
    d["dti"] = d["dti"].fillna(d["dti"].median())
    d["purpose_g"] = d["purpose"].astype(str).where(d["purpose"].astype(str).isin(MAIN_PURPOSES), "rest")
    d["home_ownership"] = d["home_ownership"].astype(str)
    d["verification_status"] = d["verification_status"].astype(str)
    d["ead_ratio"] = (d["ead_default"] / d["funded_amnt"]).clip(0, 1)
    return d


def fit_fractional_logit(train: pd.DataFrame, formula: str):
    """Papke-Wooldridge quasi-ML fractional logit with robust (HC1) SEs."""
    return smf.glm(formula, data=train, family=sm.families.Binomial()).fit(cov_type="HC1")


def fit_lgb_fraction(X: pd.DataFrame, y: np.ndarray, n_iter=300):
    params = dict(objective="cross_entropy", learning_rate=0.03, num_leaves=31,
                  min_child_samples=500, feature_fraction=0.8, verbose=-1, n_jobs=4)
    return lgb.train(params, lgb.Dataset(X, y), n_iter)


def regression_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    sse = np.sum((y - p) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return {"n": len(y), "mean_obs": y.mean(), "mean_pred": p.mean(),
            "rmse": np.sqrt(np.mean((y - p) ** 2)), "mae": np.mean(np.abs(y - p)),
            "r2_oos": 1 - sse / sst, "spearman": stats.spearmanr(y, p).statistic,
            # H0: mean prediction error is zero (calibration-in-the-large).
            "bias_t_p": stats.ttest_1samp(y - p, 0).pvalue}


def paired_loss_test(y, p_model, p_bench) -> dict:
    """Diebold-Mariano style paired test of squared-error loss (iid errors)."""
    d = (np.asarray(y) - p_bench) ** 2 - (np.asarray(y) - p_model) ** 2
    t = stats.ttest_1samp(d, 0)
    return {"mse_reduction": float(d.mean()), "t": float(t.statistic),
            "p_value_one_sided": float(t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2)}


def ead_production_sample(loans: pd.DataFrame) -> pd.DataFrame:
    """Loans defaulting within 12 months, issue years <= 2016 (all labelled data
    before the 2017 cohort)."""
    ed = prep(loans[(loans["default_12m"] == 1) & (loans["obs_12m"] == 1) & (loans["issue_year"] <= 2016)])
    ed["y"] = ed["ead_ratio"]
    return ed


def fit_ead_production(loans: pd.DataFrame):
    """Deterministic refit (statsmodels formula results do not unpickle on py3.13)."""
    return fit_fractional_logit(ead_production_sample(loans), EAD_FORMULA)
