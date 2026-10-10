"""Model pipelines: SQC screening -> imputation -> (scaling) -> classifier.

Every preprocessing step sits inside the pipeline, so it is re-fitted on the
training part of each CV fold and never sees validation or test firms.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .screening import SQCRatioScreener


def _screener(sqc: config.SQCConfig, *, screen: bool, scale: str, alarm: bool) -> SQCRatioScreener:
    return SQCRatioScreener(tail_prob=sqc.tail_prob, max_missing=sqc.max_missing, fdr=sqc.fdr,
                            max_corr=sqc.max_corr, screen=screen, scale=scale, add_alarm_count=alarm)


def make_logistic(sqc: config.SQCConfig = config.SQCConfig(), *, C: float = 0.1, screen: bool = True,
                  scale: str = "arcsinh", alarm: bool = False) -> Pipeline:
    """Scorecard-style linear model on in-control z-scores of the screened ratios."""
    return Pipeline([
        ("sqc", _screener(sqc, screen=screen, scale=scale, alarm=alarm)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=5000, random_state=config.RANDOM_STATE)),
    ])


def make_forest(sqc: config.SQCConfig = config.SQCConfig(), *, n_estimators: int = 500,
                min_samples_leaf: int = 3, max_features="sqrt", screen: bool = True,
                alarm: bool = False, n_jobs: int = -1) -> Pipeline:
    """Random forest on the screened raw ratios (trees are scale-invariant)."""
    return Pipeline([
        ("sqc", _screener(sqc, screen=screen, scale="none", alarm=alarm)),
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                       max_features=max_features, n_jobs=n_jobs,
                                       random_state=config.RANDOM_STATE)),
    ])


def make_boosting(sqc: config.SQCConfig = config.SQCConfig(), *, learning_rate: float = 0.05, max_iter: int = 300,
                  max_leaf_nodes: int = 15, screen: bool = True) -> Pipeline:
    """Gradient-boosting *challenger* (reference model, outside the RF/LR scope).

    Histogram boosting handles missing values natively, so there is no imputer.
    """
    return Pipeline([
        ("sqc", _screener(sqc, screen=screen, scale="none", alarm=False)),
        ("clf", HistGradientBoostingClassifier(learning_rate=learning_rate, max_iter=max_iter,
                                               max_leaf_nodes=max_leaf_nodes,
                                               random_state=config.RANDOM_STATE)),
    ])


FACTORIES = {"lr": make_logistic, "rf": make_forest, "hgb": make_boosting}

# Small, pre-declared grids; searched by stratified CV on the training split only.
PARAM_GRIDS = {
    "lr": {"clf__C": [0.03, 0.1, 0.3, 1.0]},
    "rf": {"clf__min_samples_leaf": [1, 2, 3, 5], "clf__max_features": ["sqrt", 0.3, 0.5]},
    "hgb": {"clf__max_leaf_nodes": [8, 15, 31], "clf__max_iter": [200, 400]},
}
