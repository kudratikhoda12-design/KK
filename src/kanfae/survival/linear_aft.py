"""Linear AFT model fitted by full-batch L-BFGS on the validated likelihood (used for A7/A8 and baselines)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .aft import aft_log_likelihood, aft_survival, get_family


@dataclass
class LinearAFTFit:
    beta: np.ndarray          # (p + 1,) intercept first
    log_sigma: float
    loglik: float             # summed log-likelihood at the optimum (unpenalised)
    se: np.ndarray | None     # standard errors of (beta, log_sigma) from the inverse Hessian
    family: str
    converged: bool


def _design(X):
    X = np.asarray(X, dtype=np.float64)
    return np.column_stack([np.ones(len(X)), X])


def fit_linear_aft(X, y, delta, family="weibull", l2=0.0, max_iter=500, compute_se=True):
    """Maximise sum_i l_i - 0.5 * l2 * ||beta[1:]||^2 over (beta, log sigma)."""
    fam = get_family(family)
    Z = torch.tensor(_design(X))
    yt = torch.tensor(np.asarray(y, dtype=np.float64))
    dt = torch.tensor(np.asarray(delta).astype(bool))
    p = Z.shape[1]
    # Initialise intercept at mean log time, log sigma at log sd of log time.
    ly = torch.log(yt)
    theta = torch.zeros(p + 1, dtype=torch.float64)
    theta[0] = ly.mean()
    theta[-1] = torch.log(ly.std().clamp_min(1e-3))
    theta.requires_grad_(True)

    def objective(th):
        eta = Z @ th[:p]
        ll = aft_log_likelihood(yt, dt, eta, torch.exp(th[-1]), fam).sum()
        return -(ll - 0.5 * l2 * (th[1:p] ** 2).sum())

    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, tolerance_grad=1e-12,
                            tolerance_change=1e-15, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = objective(theta)
        loss.backward()
        return loss

    for _ in range(5):
        opt.step(closure)
    with torch.no_grad():
        g = torch.autograd.functional.jacobian(objective, theta.detach())
    converged = bool(g.abs().max() < 1e-5)
    th = theta.detach()
    with torch.no_grad():
        loglik = float(aft_log_likelihood(yt, dt, Z @ th[:p], torch.exp(th[-1]), fam).sum())
    se = None
    if compute_se:
        H = torch.autograd.functional.hessian(objective, th)
        try:
            se = np.sqrt(np.diag(np.linalg.inv(H.numpy())))
        except np.linalg.LinAlgError:
            se = None
    return LinearAFTFit(th[:p].numpy().copy(), float(th[-1]), loglik, se, fam.name, converged)


def predict_eta(fit: LinearAFTFit, X):
    return _design(X) @ fit.beta


def predict_survival(fit: LinearAFTFit, X, times):
    eta = torch.tensor(predict_eta(fit, X))
    return aft_survival(torch.tensor(np.asarray(times, float)), eta, float(np.exp(fit.log_sigma)), fit.family).numpy()
