"""Model factories. Every learned preprocessing step lives inside the estimator pipeline, so it is
re-fitted on each training fold only (no global scaling or imputation)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20231

LGBM_DEFAULT = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=2000,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0)


def logistic(C: float = 0.1):
    """L2-regularised logistic regression. Median imputation + missing indicators + standardisation,
    all fitted inside the training fold."""
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=C, max_iter=500))


def random_forest(n_estimators: int = 200, min_samples_leaf: int = 500, max_depth: int = 10):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                                max_depth=max_depth, max_features="sqrt", n_jobs=-1,
                                                random_state=SEED))


def lightgbm(**params):
    import lightgbm as lgb
    p = {**LGBM_DEFAULT, **params}
    return lgb.LGBMClassifier(objective="binary", random_state=SEED, n_jobs=4, verbose=-1, **p)


def lightgbm_regressor(**params):
    import lightgbm as lgb
    p = {**LGBM_DEFAULT, **params}
    return lgb.LGBMRegressor(objective="regression", random_state=SEED, n_jobs=4, verbose=-1, **p)


def make(name: str, **params):
    return {"logit": logistic, "rf": random_forest, "lgbm": lightgbm}[name](**params)


def predict_up(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
