"""Semiparametric ("non-parametric error") functional KAN AFT, NP-KAN-AFT (Round 20, D150).

Model: log T_i = mu + KAN(Z_i) + eps_i, eps_i iid with an unspecified distribution (estimated by Kaplan-Meier), KAN(Z)
the functional B2-S location (or any FKANNet kind; with last-value anchor channels through ResFKAN). In the AFT
literature this is the semiparametric AFT model (location function + non-parametric error law); classical estimators
are Buckley-James (Buckley & James 1979) and the Gehan rank estimator (Jin, Lin, Wei & Ying 2003); KAN-AFT (Jose,
Francis & Kattumannil 2025/26) fits a static KAN location by Buckley-James. No novelty is claimed (D111).

It is FKANSurv(loss="aft", family="np_aft", np_objective=...) in fkan.py (the helpers below are called from there);
NPKANAFT is the D112 ensemble of it. Objectives (training loss and validation loss, the same function on each split):
  "bj_km"   Buckley-James / Kaplan-Meier residual EM: before every epoch the KM of the current training residuals
            r_j = log Y_j - loc_j (largest residual counted as an event, Efron's tail rule) is re-estimated; the loss on
            any split is mean_i (Y*_i - loc_i)^2 with Y*_i = log Y_i for events and, for censored i,
            Y*_i = loc_i + E_KM[eps | eps > r_i] = loc_i + r_i + int_{r_i}^inf S_KM(u) du / S_KM(r_i)
            (Y* held fixed: no gradient flows through the imputation). One Adam step per E-step.
  "rank_km" smoothed Gehan rank loss  (1/n^2) sum_{i: delta_i=1} sum_j tau * softplus((e_j - e_i) / tau),
            e = log Y - loc, tau = np_tau (0.05 on the log-time scale). Its unsmoothed form
            sum_{i: delta_i=1} sum_j max(0, e_j - e_i) has gradient equal to the Gehan estimating function
            sum_i delta_i sum_j (x_i - x_j) 1{e_j >= e_i} (Jin et al. 2003). The intercept mu is not identified by this
            loss and is held at its initial value; the KM of the residuals absorbs it.
Survival prediction (both objectives): S(t | Z) = S_KM(log t - loc(Z)), S_KM the plain (no tail rule) KM of the
training residuals at the fitted location, a right-continuous step function: 1 below the smallest residual, S_KM at the
largest residual above it (0 if that residual is an event, else the last KM level).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

NP_OBJECTIVES = ("bj_km", "rank_km")


def km_fit(r, d):
    """Kaplan-Meier estimate of P(eps > u) from (r, d): (unique sorted knots, S at each knot), right-continuous.
    Vectorised; equals survival.censored_ls.km_survival."""
    r, d = np.asarray(r, float), np.asarray(d, float)
    knots, inv = np.unique(r, return_inverse=True)
    n_ev = np.bincount(inv, weights=d, minlength=len(knots))
    n_tot = np.bincount(inv, minlength=len(knots))
    at_risk = len(r) - np.r_[0, np.cumsum(n_tot)[:-1]]
    return knots, np.cumprod(1.0 - n_ev / at_risk)


def km_eval(knots, S, x):
    """Right-continuous step function: 1 below the first knot, S at the last knot beyond it."""
    idx = np.searchsorted(knots, np.asarray(x, float), side="right") - 1
    return np.where(idx >= 0, S[np.clip(idx, 0, None)], 1.0)


def km_tail_fit(r, d):
    """KM with the largest residual counted as an event (Efron), so the estimate reaches 0 and E[eps | eps > r] exists."""
    d = np.asarray(d, bool).copy()
    d[np.argmax(r)] = True
    return km_fit(r, d)


def km_cond_mean(knots, S, r):
    """E_KM[eps | eps > r] = r + int_r^inf S(u) du / S(r) for a KM step function that reaches 0 (km_tail_fit); computed
    as sum_{knots > r} knot * mass / S(r). Where S(r) = 0 (r at or beyond the last knot) r itself is returned."""
    r = np.asarray(r, float)
    mass = -np.diff(np.r_[1.0, S])
    tail = np.r_[np.cumsum((knots * mass)[::-1])[::-1], 0.0]
    idx = np.searchsorted(knots, r, side="right")
    Sr = np.where(idx > 0, S[np.clip(idx - 1, 0, None)], 1.0)
    return np.where(Sr > 1e-12, tail[idx] / np.maximum(Sr, 1e-12), r)


def bj_targets(logy, d, loc, km):
    """Buckley-James responses Y* (numpy) given the training-residual KM `km` = (knots, S) from km_tail_fit."""
    r = logy - loc
    return np.where(d > 0, logy, loc + km_cond_mean(km[0], km[1], r))


def bj_loss(loc, y, d, km):
    """mean (Y* - loc)^2 with Y* fixed (computed from the detached location)."""
    logy = torch.log(y)
    ys = bj_targets(logy.numpy(), d.numpy(), loc.detach().numpy(), km)
    return ((torch.tensor(ys, dtype=loc.dtype) - loc) ** 2).mean()


def gehan_loss(loc, y, d, tau=0.05):
    """(1/n^2) sum_{i: d_i=1} sum_j tau * softplus((e_j - e_i)/tau), e = log y - loc (see the module docstring)."""
    e = torch.log(y) - loc
    diff = e[None, :] - e[:, None]                     # [i, j] = e_j - e_i
    sp = tau * F.softplus(diff / tau)
    n = len(e)
    return (sp * d.to(loc.dtype)[:, None]).sum() / (n * n)


def np_loss(model, eta, y, d):
    """The NP-AFT loss of FKANSurv `model` for linear predictor eta (location = model.mu + eta)."""
    loc = model.mu + eta
    if model.cfg["np_objective"] == "bj_km":
        return bj_loss(loc, y, d, model._np_km)
    return gehan_loss(loc, y, d, model.cfg["np_tau"])


@torch.no_grad()
def np_refresh(model):
    """Re-estimate the tail-corrected KM of the current training residuals (the BJ E-step)."""
    dd = model.data
    loc = (model.mu + model.net(dd["Xtr"], dd["Ztr"])).numpy()
    model._np_km = km_tail_fit(np.log(dd["ytr"].numpy()) - loc, dd["dtr"].numpy())


@torch.no_grad()
def np_finalise(model, train):
    """Training-residual KMs at the fitted location: plain (prediction) and tail-corrected (BJ loss)."""
    loc = (model.mu + model.net(model.data["Xtr"], model.data["Ztr"])).numpy()
    r = np.log(np.asarray(train["y"], float)) - loc
    model.np_knots, model.np_S = km_fit(r, train["d"])
    model._np_km = km_tail_fit(r, train["d"])
    model.np_residuals = r
    model.sigma = 1.0                              # no scale parameter: effect() returns -phi (log-time units)


def np_survival(model, e, times):
    """S(t | Z) = S_KM(log t - mu - eta) on a time grid; e: eta (B,), times (T,) -> (B, T)."""
    w = np.log(np.maximum(times, 1e-300))[None, :] - float(model.mu.detach()) - e[:, None]
    return km_eval(model.np_knots, model.np_S, w)


def NPKANAFT(objective="bj_km", kind="b2s", tau=0.05, **kw):
    """D112 ensemble (lr x wd tuned on the validation loss of the objective, 3 seeds) of the NP-KAN-AFT."""
    from .fkan import FKANEnsemble
    if objective not in NP_OBJECTIVES:
        raise ValueError(objective)
    return FKANEnsemble(kind=kind, loss="aft", family="np_aft", np_objective=objective, np_tau=tau, **kw)


def distill_and_replace(model, mu=10.0, max_abs_arg=50.0, max_iter=200, val_tol=0.0, return_edges=False):
    """Symbolic distillation of a fitted FKANSurv (any loss, including np_aft) with a non-degradation rollback guard.

    Stages 1-2 without PySR (max_abs_arg 50), guarded substitution, one fixed-mu distillation refit (D138: anchor = the
    spline model, safe=True, 200 L-BFGS iterations); for np_aft the KM of the residuals is re-estimated on the distilled
    model's residuals (FKANSurv._finalise). The validation loss of the spline model (before) and of the distilled model
    (after) are computed with the same function (heldout loss on the model's validation split; for bj_km each model's
    own training-residual KM). Rollback: if the distilled model has a non-finite prediction, loss or parameter, or
    loss_after > loss_before + val_tol, the spline model is returned unchanged.
    return_edges (Round 21, default off): also record each edge's stage, acceptance, formula and validation R^2.
    Returns (model, record)."""
    from .fkan import substitute_symbolic, symbolic_edges
    dd = model.data

    def vloss(m):
        with torch.no_grad():
            return float(m._loss(m.net(dd["Xva"], dd["Zva"]), dd["yva"], dd["dva"], dd["sva"]))

    before = vloss(model)
    recs = symbolic_edges(model, use_pysr=False, max_abs_arg=max_abs_arg)
    sub = copy.deepcopy(model)
    n_sub = substitute_symbolic(sub, recs, guard=True)
    sub._finalise(sub.train_)
    m = copy.deepcopy(sub).refit_symbolic(max_iter=max_iter, anchor=model, mu=mu, safe=True)
    after = vloss(m)
    with torch.no_grad():
        eta = m.net(dd["Xva"], dd["Zva"])
    finite = bool(np.isfinite(after) and torch.isfinite(eta).all()
                  and all(bool(torch.isfinite(q).all()) for q in m._params()))
    rollback = (not finite) or not (after <= before + val_tol)
    rec = dict(n_edges=len(recs), n_accepted=sum(r["result"].accepted for r in recs), n_substituted=n_sub,
               val_loss_before=before, val_loss_after=after, finite=finite, refit_fallback=bool(m.refit_fallback),
               rollback=bool(rollback), val_tol=val_tol)
    if return_edges:
        rec["edges"] = [dict(layer=r["layer"], i=r["i"], o=r["o"], share=r.get("share"), stage=r["result"].stage,
                           accepted=bool(r["result"].accepted), formula=r["result"].formula,
                           r2_val=r["result"].r2_val) for r in recs]
    return (model if rollback else m), rec
