"""Censoring-aware accelerated failure time (AFT) likelihoods.

Model: log T = eta + sigma * eps, eps ~ F_eps (standardised family).  See docs/design.md section 2.

    l_i = delta_i [log f_eps(w_i) - log sigma - log y_i] + (1 - delta_i) log S_eps(w_i),
    w_i = (log y_i - eta_i) / sigma.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

FAMILIES = ("weibull", "lognormal", "loglogistic")


class AFTFamily:
    """Standardised error distribution of log T. Subclasses define log f, log S and the quantile."""

    name = "base"

    def log_f(self, w: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_S(self, w: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def quantile(self, p: float) -> float:
        raise NotImplementedError

    def sample(self, n: int, rng) -> "numpy.ndarray":  # noqa: F821
        raise NotImplementedError


class Weibull(AFTFamily):
    """eps ~ standard Gumbel (minimum): S(w) = exp(-e^w)."""

    name = "weibull"

    def log_f(self, w):
        return w - torch.exp(w)

    def log_S(self, w):
        return -torch.exp(w)

    def quantile(self, p):
        return math.log(-math.log1p(-p))

    def sample(self, n, rng):
        return rng.gumbel(size=n) * -1.0  # Gumbel(max) negated = Gumbel(min)


class LogNormal(AFTFamily):
    """eps ~ N(0, 1)."""

    name = "lognormal"

    def log_f(self, w):
        return -0.5 * w * w - 0.5 * math.log(2.0 * math.pi)

    def log_S(self, w):
        return torch.special.log_ndtr(-w)

    def quantile(self, p):
        return float(torch.special.ndtri(torch.tensor(p, dtype=torch.float64)))

    def sample(self, n, rng):
        return rng.standard_normal(n)


class LogLogistic(AFTFamily):
    """eps ~ standard logistic: S(w) = 1 / (1 + e^w)."""

    name = "loglogistic"

    def log_f(self, w):
        return w - 2.0 * F.softplus(w)

    def log_S(self, w):
        return -F.softplus(w)

    def quantile(self, p):
        return math.log(p / (1.0 - p))

    def sample(self, n, rng):
        return rng.logistic(size=n)


_REGISTRY = {"weibull": Weibull, "lognormal": LogNormal, "loglogistic": LogLogistic}


def get_family(family) -> AFTFamily:
    if isinstance(family, AFTFamily):
        return family
    try:
        return _REGISTRY[family]()
    except KeyError as exc:
        raise ValueError(f"unknown AFT family {family!r}; expected one of {FAMILIES}") from exc


def aft_log_likelihood(y, delta, eta, sigma, family="weibull"):
    """Per-individual log-likelihood terms (shape of y). Includes the -log y Jacobian term for events."""
    fam = get_family(family)
    y, eta = torch.as_tensor(y), torch.as_tensor(eta)
    sigma = torch.as_tensor(sigma, dtype=eta.dtype)
    log_y = torch.log(y)
    w = (log_y - eta) / sigma
    event_term = fam.log_f(w) - torch.log(sigma) - log_y
    cens_term = fam.log_S(w)
    return torch.where(torch.as_tensor(delta).bool(), event_term, cens_term)


def aft_nll(y, delta, eta, sigma, family="weibull"):
    """Mean negative log-likelihood (training loss)."""
    return -aft_log_likelihood(y, delta, eta, sigma, family).mean()


def aft_survival(t, eta, sigma, family="weibull"):
    """S(t | eta, sigma) on a time grid. t: (T,), eta: (B,) -> (B, T)."""
    fam = get_family(family)
    t = torch.as_tensor(t, dtype=torch.as_tensor(eta).dtype)
    eta = torch.as_tensor(eta)
    w = (torch.log(t)[None, :] - eta[:, None]) / sigma
    return torch.exp(fam.log_S(w))


def aft_median(eta, sigma, family="weibull"):
    fam = get_family(family)
    return torch.exp(torch.as_tensor(eta) + sigma * fam.quantile(0.5))
