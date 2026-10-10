"""One-factor Gaussian (Vasicek) portfolio credit-loss model.

Default of loan i:  -sqrt(rho)*Z + sqrt(1-rho)*e_i < Phi^-1(PD_i).
Conditional on the systematic factor Z:
    PD_i(Z) = Phi((Phi^-1(PD_i) + sqrt(rho) Z) / sqrt(1 - rho)).
Z > 0 is a bad (high-default) state, Z < 0 a good state.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

norm = stats.norm


def cond_pd(pd_, rho, z):
    return norm.cdf((norm.ppf(pd_) + np.sqrt(rho) * z) / np.sqrt(1 - rho))


def vasicek_logpdf(x, pd_, rho):
    """Log density of the large-portfolio default rate (Vasicek 2002)."""
    a = norm.ppf(x)
    b = norm.ppf(pd_)
    return (0.5 * np.log((1 - rho) / rho) + 0.5 * a ** 2
            - (np.sqrt(1 - rho) * a - b) ** 2 / (2 * rho))


def fit_rho_mle(dr: np.ndarray, pd_: np.ndarray, free_level: bool = True) -> dict:
    """MLE of asset correlation from cohort default rates and cohort PDs.

    pd_ carries each cohort's risk mix (e.g. the mean model PD). With
    free_level=True a common through-the-cycle shift c is estimated jointly,
    logit(PD_t) = logit(pd_t) + c, so that a level bias in the PDs is not
    mistaken for systematic risk (the implied Z_t are then centred).
    Returns rho with a Wald CI (profile curvature), c and the implied Z_t.
    """
    dr = np.clip(np.asarray(dr, float), 1e-6, 1 - 1e-6)
    lpd = np.log(np.asarray(pd_, float) / (1 - np.asarray(pd_, float)))

    def nll(r, c):
        p = 1 / (1 + np.exp(-(lpd + c)))
        return -np.sum(vasicek_logpdf(dr, p, r))

    def profile(r):
        if not free_level:
            return nll(r, 0.0), 0.0
        o = optimize.minimize_scalar(lambda c: nll(r, c), bounds=(-3, 3), method="bounded",
                                     options={"xatol": 1e-8})
        return o.fun, o.x

    opt = optimize.minimize_scalar(lambda r: profile(r)[0], bounds=(1e-4, 0.5), method="bounded",
                                   options={"xatol": 1e-8})
    r = opt.x
    c = profile(r)[1]
    h = max(1e-5, r * 1e-2)
    info = (profile(r + h)[0] - 2 * profile(r)[0] + profile(r - h)[0]) / h ** 2
    se = 1 / np.sqrt(info) if info > 0 else np.nan
    p_adj = 1 / (1 + np.exp(-(lpd + c)))
    z = (np.sqrt(1 - r) * norm.ppf(dr) - norm.ppf(p_adj)) / np.sqrt(r)
    return {"rho": r, "se": se, "lo": max(r - 1.96 * se, 0), "hi": r + 1.96 * se,
            "level_shift": c, "n_obs": len(dr), "z_t": z}


def fit_rho_pooled(dr_by_group: dict[str, np.ndarray]) -> dict:
    """Common rho across groups, each group with its own long-run PD (joint MLE)."""
    groups = list(dr_by_group)
    data = [np.clip(np.asarray(dr_by_group[g], float), 1e-6, 1 - 1e-6) for g in groups]

    def nll(theta):
        r = 1 / (1 + np.exp(-theta[0]))
        out = 0.0
        for k, x in enumerate(data):
            p = 1 / (1 + np.exp(-theta[k + 1]))
            out -= np.sum(vasicek_logpdf(x, p, r))
        return out

    th0 = np.r_[np.log(0.05 / 0.95), [np.log(x.mean() / (1 - x.mean())) for x in data]]
    opt = optimize.minimize(nll, th0, method="BFGS")
    r = 1 / (1 + np.exp(-opt.x[0]))
    se_theta = np.sqrt(np.diag(opt.hess_inv))[0]
    lo = 1 / (1 + np.exp(-(opt.x[0] - 1.96 * se_theta)))
    hi = 1 / (1 + np.exp(-(opt.x[0] + 1.96 * se_theta)))
    pds = {g: 1 / (1 + np.exp(-opt.x[k + 1])) for k, g in enumerate(groups)}
    return {"rho": r, "lo": lo, "hi": hi, "long_run_pd": pds}


def basel_retail_rho(pd_):
    """Basel II 'other retail' correlation, for benchmarking."""
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    return 0.03 * w + 0.16 * (1 - w)


def asrf_quantile_loss(pd_, lgd, ead, rho, q):
    """Analytic loss quantile in the asymptotic single-risk-factor model."""
    return float(np.sum(ead * lgd * cond_pd(pd_, rho, norm.ppf(q))))


def simulate_losses(pd_, lgd, ead, rho, n_scen=10_000, seed=0, chunk=250,
                    z=None, return_z=False):
    """Loan-level Monte Carlo: systematic Z ~ N(0,1), idiosyncratic Bernoulli.
    rho may be a scalar or a per-loan array (e.g. the Basel retail curve)."""
    rng = np.random.default_rng(seed)
    pd_ = np.asarray(pd_, float)
    exposure = (np.asarray(lgd, float) * np.asarray(ead, float)).astype(np.float32)
    thr = norm.ppf(pd_).astype(np.float32)
    if z is None:
        z = rng.standard_normal(n_scen)
    losses = np.empty(len(z))
    rho = np.broadcast_to(np.asarray(rho, np.float32), pd_.shape)   # scalar or per-loan
    sr, s1r = np.sqrt(rho)[None, :], np.sqrt(1 - rho)[None, :]
    for i in range(0, len(z), chunk):
        zc = z[i:i + chunk]
        p = special.ndtr((thr[None, :] + sr * zc[:, None].astype(np.float32)) / s1r).astype(np.float32)
        u = rng.random(p.shape, dtype=np.float32)
        losses[i:i + chunk] = ((u < p) * exposure[None, :]).sum(axis=1)
    return (losses, z) if return_z else losses


def var_es(losses, q) -> dict:
    """VaR / ES at level q with a distribution-free order-statistic CI for VaR
    and a normal-approximation CI for ES."""
    L = np.sort(np.asarray(losses))
    n = len(L)
    var = float(np.quantile(L, q))
    tail = L[L >= var]
    es = float(tail.mean())
    lo_i = int(stats.binom.ppf(0.025, n, q))
    hi_i = min(int(stats.binom.ppf(0.975, n, q)), n - 1)
    es_se = tail.std(ddof=1) / np.sqrt(len(tail)) if len(tail) > 1 else np.nan
    return {"q": q, "VaR": var, "VaR_lo": float(L[lo_i]), "VaR_hi": float(L[hi_i]),
            "ES": es, "ES_lo": es - 1.96 * es_se, "ES_hi": es + 1.96 * es_se}
