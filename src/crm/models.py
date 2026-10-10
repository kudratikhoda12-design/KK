"""PD model zoo: WOE logistic scorecard, LightGBM, XGBoost, Random Forest."""
from __future__ import annotations

import itertools
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .woe import WOETransformer

NUMERIC = [
    "loan_amnt", "term_m", "int_rate", "installment", "sub_grade_n", "emp_len",
    "emp_missing", "log_inc", "dti", "delinq_2yrs", "fico", "credit_hist_m",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "collections_12_mths_ex_med", "mths_since_last_major_derog",
    "acc_now_delinq", "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
    "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
    "mort_acc", "mths_since_recent_bc", "mths_since_recent_inq",
    "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl",
    "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
    "num_rev_tl_bal_gt_0", "num_sats", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
    "loan_to_inc", "inst_to_inc", "revol_to_inc", "joint", "whole_loan",
]
CATEGORICAL = ["home_ownership", "verification_status", "purpose", "addr_state"]
FEATURES = NUMERIC + CATEGORICAL


# --------------------------------------------------------------------------
# Logistic regression scorecard on WOE variables
# --------------------------------------------------------------------------
def vif_table(X: pd.DataFrame, sample=200_000, seed=0) -> pd.Series:
    Xs = X.sample(min(sample, len(X)), random_state=seed)
    Xc = sm.add_constant(Xs).to_numpy()
    return pd.Series([variance_inflation_factor(Xc, i + 1) for i in range(X.shape[1])],
                     index=X.columns)


class WOELogit:
    """IV filter -> correlation filter -> forward selection -> statsmodels Logit.

    Forward stepwise selection: at each step every remaining variable is tried
    and the best one is added if (a) it lifts the AUC on a time-holdout by
    >= min_gain, (b) its Wald p-value is < 0.001, (c) every coefficient stays
    positive (expected WOE sign), and (d) every VIF stays < 5.
    """

    def __init__(self, iv_min=0.02, corr_max=0.7, min_gain=5e-4, max_vars=25):
        self.iv_min, self.corr_max = iv_min, corr_max
        self.min_gain, self.max_vars = min_gain, max_vars

    def fit(self, X, y, X_hold, y_hold, log=print):
        self.woe = WOETransformer(NUMERIC, CATEGORICAL).fit(X, y)
        iv = self.woe.iv
        cand = list(iv[iv >= self.iv_min].index)
        W = self.woe.transform(X, cand)
        Wh = self.woe.transform(X_hold, cand)
        corr = W.sample(min(300_000, len(W)), random_state=0).corr().abs()
        kept = []
        for c in cand:  # IV-descending: drop a variable if a stronger one is too correlated
            if all(corr.loc[c, k] <= self.corr_max for k in kept):
                kept.append(c)
        log(f"IV>={self.iv_min}: {len(cand)} vars; after |r|<={self.corr_max}: {len(kept)}")
        sel, best = [], 0.5
        self.selection_log = []
        remaining = list(kept)
        step = 0
        while remaining and len(sel) < self.max_vars:
            step += 1
            trials = []
            for c in remaining:
                lr = LogisticRegression(C=1e6, max_iter=500).fit(W[sel + [c]], y)
                auc = roc_auc_score(y_hold, lr.predict_proba(Wh[sel + [c]])[:, 1])
                trials.append((auc, c, bool(np.all(lr.coef_ > 0))))
            trials.sort(reverse=True)
            added = None
            for auc, c, ok_sign in trials:   # best candidate first
                if auc - best < self.min_gain:
                    break
                vif_ok = len(sel) == 0 or vif_table(W[sel + [c]]).max() < 5
                pval = sm.Logit(y.to_numpy(), sm.add_constant(W[sel + [c]])).fit(disp=0).pvalues[c]
                ok = ok_sign and vif_ok and pval < 1e-3
                self.selection_log.append({"step": step, "var": c, "iv": iv[c], "holdout_auc": auc,
                                           "gain": auc - best, "positive_sign": ok_sign,
                                           "vif_ok": vif_ok, "wald_p": pval, "accepted": ok})
                if ok:
                    added = (c, auc)
                    break
            if added is None:
                break
            sel.append(added[0]); remaining.remove(added[0]); best = added[1]
            log(f"step {step}: + {added[0]} holdout AUC={best:.4f}")
        self.vars = sel
        self.model = sm.Logit(y.to_numpy(), sm.add_constant(W[sel])).fit(disp=0)
        self.vif = vif_table(W[sel])
        return self

    def coef_table(self) -> pd.DataFrame:
        m = self.model
        t = pd.DataFrame({"coef": m.params, "se": m.bse, "z": m.tvalues, "p_value": m.pvalues})
        t["iv"] = self.woe.iv.reindex(t.index)
        t["vif"] = self.vif.reindex(t.index)
        return t

    def predict(self, X) -> np.ndarray:
        W = sm.add_constant(self.woe.transform(X, self.vars), has_constant="add")
        return self.model.predict(W[["const"] + self.vars])


def scorecard_points(model: WOELogit, pdo=20, base_score=600, base_odds=50) -> pd.DataFrame:
    """Points table: score = offset + factor*ln(good/bad odds)."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    b = model.model.params
    k = len(model.vars)
    rows = []
    for v in model.vars:
        bnr = model.woe.binners[v]
        for bin_, r in bnr.table.iterrows():
            bin_ = bnr.labels.get(bin_, bin_)
            pts = -(b[v] * r["woe"] + b["const"] / k) * factor + offset / k
            rows.append({"variable": v, "bin": str(bin_), "n": r["n"], "dr": r["dr"],
                         "woe": r["woe"], "points": round(pts)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Gradient boosting & random forest
# --------------------------------------------------------------------------
def _sample_grid(grid: dict, n: int, seed: int) -> list[dict]:
    keys = list(grid)
    allc = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    rng = np.random.default_rng(seed)
    return [allc[i] for i in rng.choice(len(allc), size=min(n, len(allc)), replace=False)]


LGB_GRID = {
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [200, 500, 1000, 2000],
    "learning_rate": [0.03, 0.05],
    "feature_fraction": [0.5, 0.7, 0.9],
    "bagging_fraction": [0.7, 0.9],
    "lambda_l2": [0, 10, 50],
    "min_split_gain": [0.0, 0.1],
}


def tune_lgb(Xtr, ytr, Xva, yva, n_iter=20, seed=0, log=print, monotone=None):
    res = []
    for i, prm in enumerate(_sample_grid(LGB_GRID, n_iter, seed)):
        t0 = time.time()
        params = dict(objective="binary", metric="auc", verbose=-1, n_jobs=4,
                      bagging_freq=1, seed=seed, cat_smooth=50, **prm)
        if monotone is not None:
            params["monotone_constraints"] = monotone
        d_tr = lgb.Dataset(Xtr, ytr)
        d_va = lgb.Dataset(Xva, yva, reference=d_tr)
        b = lgb.train(params, d_tr, 3000, valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        res.append({**prm, "best_iter": b.best_iteration,
                    "auc": b.best_score["valid_0"]["auc"]})
        log(f"lgb {i:2d} auc={res[-1]['auc']:.4f} iters={b.best_iteration} {time.time()-t0:.0f}s {prm}")
    return pd.DataFrame(res).sort_values("auc", ascending=False)


def fit_lgb(X, y, prm: dict, n_iter: int, seed=0, monotone=None):
    params = dict(objective="binary", verbose=-1, n_jobs=4, bagging_freq=1,
                  seed=seed, cat_smooth=50, **prm)
    if monotone is not None:
        params["monotone_constraints"] = monotone
    return lgb.train(params, lgb.Dataset(X, y), n_iter)


XGB_GRID = {
    "max_depth": [3, 4, 5, 6, 8],
    "min_child_weight": [5, 20, 50, 100],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.5, 0.7, 0.9],
    "reg_lambda": [1, 10, 50],
}


def tune_xgb(Xtr, ytr, Xva, yva, n_iter=12, seed=0, log=print):
    res = []
    dtr = xgb.DMatrix(Xtr, ytr, enable_categorical=True)
    dva = xgb.DMatrix(Xva, yva, enable_categorical=True)
    for i, prm in enumerate(_sample_grid(XGB_GRID, n_iter, seed)):
        t0 = time.time()
        params = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                      nthread=4, seed=seed, max_cat_to_onehot=1, **prm)
        b = xgb.train(params, dtr, 3000, evals=[(dva, "va")], early_stopping_rounds=100,
                      verbose_eval=False)
        res.append({**prm, "best_iter": b.best_iteration + 1, "auc": b.best_score})
        log(f"xgb {i:2d} auc={b.best_score:.4f} iters={b.best_iteration} {time.time()-t0:.0f}s {prm}")
    return pd.DataFrame(res).sort_values("auc", ascending=False)


def fit_xgb(X, y, prm: dict, n_iter: int, seed=0):
    params = dict(objective="binary:logistic", tree_method="hist", nthread=4, seed=seed,
                  max_cat_to_onehot=1, **prm)
    return xgb.train(params, xgb.DMatrix(X, y, enable_categorical=True), n_iter)


def predict_xgb(model, X) -> np.ndarray:
    return model.predict(xgb.DMatrix(X, enable_categorical=True))


def rf_matrix(X: pd.DataFrame, medians: pd.Series | None = None):
    """RF input: numeric median-imputed + missing flags, categoricals as codes."""
    num = X[NUMERIC].astype(float)
    if medians is None:
        medians = num.median()
    flags = num.isna().add_suffix("_na").astype(np.int8)
    flags = flags.loc[:, [c + "_na" for c in NUMERIC if (c + "_na") in flags]]
    out = pd.concat([num.fillna(medians), flags,
                     X[CATEGORICAL].apply(lambda s: s.cat.codes)], axis=1)
    return out, medians


RF_GRID = {"min_samples_leaf": [50, 200, 500], "max_features": [0.2, 0.33]}


def tune_rf(Xtr, ytr, Xva, yva, seed=0, log=print):
    res = []
    for prm in _sample_grid(RF_GRID, 6, seed):
        t0 = time.time()
        m = RandomForestClassifier(n_estimators=200, max_samples=0.3, n_jobs=4,
                                   random_state=seed, **prm).fit(Xtr, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        res.append({**prm, "auc": auc})
        log(f"rf auc={auc:.4f} {time.time()-t0:.0f}s {prm}")
    return pd.DataFrame(res).sort_values("auc", ascending=False)


def fit_rf(X, y, prm: dict, seed=0):
    return RandomForestClassifier(n_estimators=500, max_samples=0.3, n_jobs=4,
                                  random_state=seed, **prm).fit(X, y)
