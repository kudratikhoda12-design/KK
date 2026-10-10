"""PD recalibration methods (fit on the OOT test year, applied to validation)."""
from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy import optimize
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
