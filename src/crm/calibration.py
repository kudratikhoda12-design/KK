"""PD recalibration methods (fit on the OOT test year, applied to validation)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from sklearn.isotonic import IsotonicRegression

EPS = 1e-9


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


class InterceptShift:
    """Calibration-in-the-large: logit(p') = logit(p) + a (slope fixed at 1)."""

    def fit(self, p, y=None, target_rate=None):
        lp = logit(p)
        if target_rate is None:
            target_rate = float(np.mean(y))
        self.a = optimize.brentq(lambda a: expit(lp + a).mean() - target_rate, -5, 5)
        return self

    def predict(self, p):
        return expit(logit(p) + self.a)


class Platt:
    """Logistic recalibration: logit(p') = a + b*logit(p)."""

    def fit(self, p, y):
        self.m = sm.GLM(y, sm.add_constant(logit(p)), family=sm.families.Binomial()).fit()
        self.a, self.b = self.m.params
        return self

    def predict(self, p):
        return expit(self.a + self.b * logit(p))


class Isotonic:
    def fit(self, p, y):
        self.m = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(p, y)
        return self

    def predict(self, p):
        return self.m.predict(p)


class Stack:
    """Logistic stacking of several models' logits (fit on the test year)."""

    def fit(self, P: np.ndarray, y):
        self.m = sm.GLM(y, sm.add_constant(logit(P)), family=sm.families.Binomial()).fit()
        return self

    def predict(self, P: np.ndarray):
        return self.m.predict(sm.add_constant(logit(P), has_constant="add"))


class PlattSegment:
    """Logistic recalibration with segment intercepts:
    logit(p') = a + b*logit(p) + gamma_segment (reference = first segment).
    Adopted only if a likelihood-ratio test against plain Platt rejects."""

    def fit(self, p, y, seg):
        self.levels = sorted(pd.unique(np.asarray(seg).astype(str)))
        X0 = sm.add_constant(logit(p))
        X1 = np.c_[X0, self._dummies(seg)]
        m0 = sm.GLM(y, X0, family=sm.families.Binomial()).fit()
        self.m = sm.GLM(y, X1, family=sm.families.Binomial()).fit()
        lr = 2 * (self.m.llf - m0.llf)
        dfree = X1.shape[1] - X0.shape[1]
        self.lr_test = {"chi2": float(lr), "df": int(dfree), "p": float(stats.chi2.sf(lr, dfree))}
        return self

    def _dummies(self, seg):
        s = np.asarray(seg).astype(str)
        return np.column_stack([(s == l).astype(float) for l in self.levels[1:]])

    def predict(self, p, seg):
        X = np.c_[np.ones(len(p)), logit(p), self._dummies(seg)]
        return expit(X @ self.m.params)
