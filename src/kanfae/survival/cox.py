"""Cox proportional hazards: exact Breslow and Efron negative log partial likelihoods, Breslow baseline (D112).

Unlike CoxKAN's published fast approximation, ties are handled exactly. Both functions are differentiable in eta
(torch) and accept optional strata (B3 landmark-stratified baselines): risk sets are formed within each stratum.
"""
from __future__ import annotations

import numpy as np
import torch


def _one_stratum(eta, time, event, ties):
    order = torch.argsort(time, descending=True, stable=True)
    eta, time, event = eta[order], time[order], event[order]
    # cumulative log-sum-exp over the risk set (subjects with time >= t), sorted by descending time
    lcse = torch.logcumsumexp(eta, 0)
    ut, inv = torch.unique(time, return_inverse=True)
    # risk-set log-sum at each unique time = value at the LAST (smallest-time) position of that tied block
    idx_last = torch.zeros(len(ut), dtype=torch.long)
    idx_last.scatter_reduce_(0, inv, torch.arange(len(time)), reduce="amax", include_self=False)
    log_risk = lcse[idx_last]                                       # (U,)
    ev = event.bool()
    d = torch.zeros(len(ut), dtype=eta.dtype).index_add_(0, inv[ev], torch.ones(int(ev.sum()), dtype=eta.dtype))
    sum_eta_ev = torch.zeros(len(ut), dtype=eta.dtype).index_add_(0, inv[ev], eta[ev])
    has = d > 0
    if ties == "breslow":
        ll = sum_eta_ev[has] - d[has] * log_risk[has]
        return -ll.sum(), int(ev.sum())
    # Efron: sum_{l=0}^{d-1} log(R - l/d * E), E = sum of exp(eta) over the tied events
    E = torch.zeros(len(ut), dtype=eta.dtype).index_add_(0, inv[ev], torch.exp(eta[ev]))
    # vectorised over events: event j in tied block u gets rank l_j in 0..d_u-1
    ue = inv[ev]
    srt = torch.argsort(ue, stable=True)
    ue_s = ue[srt]
    first = torch.ones_like(ue_s, dtype=torch.bool)
    first[1:] = ue_s[1:] != ue_s[:-1]
    start = torch.cummax(torch.where(first, torch.arange(len(ue_s)), torch.zeros_like(ue_s)), 0).values
    rank = (torch.arange(len(ue_s)) - start).to(eta.dtype)
    # log(R - l/d E) = log_risk + log(1 - (l/d) E/R), computed stably
    ratio = E[ue_s] * torch.exp(-log_risk[ue_s])
    terms = log_risk[ue_s] + torch.log1p(-(rank / d[ue_s]) * ratio)
    nll = -sum_eta_ev[has].sum() + terms.sum()
    return nll, int(ev.sum())


def cox_nll(eta, time, event, ties="efron", strata=None, reduction="mean"):
    """Negative log partial likelihood. reduction 'sum' or 'mean' (divided by the number of events)."""
    eta = torch.as_tensor(eta)
    time = torch.as_tensor(time, dtype=eta.dtype)
    event = torch.as_tensor(event)
    if ties not in ("efron", "breslow"):
        raise ValueError(ties)
    groups = [slice(None)] if strata is None else [torch.as_tensor(strata) == s for s in torch.unique(torch.as_tensor(strata))]
    total, n_ev = 0.0, 0
    for g in groups:
        v, k = _one_stratum(eta[g], time[g], event[g], ties)
        total, n_ev = total + v, n_ev + k
    return total if reduction == "sum" else total / max(n_ev, 1)


def breslow_baseline(eta, time, event, strata=None):
    """Breslow estimator of the baseline cumulative hazard. Returns {stratum: (event_times, Lambda0)}
    (stratum None when unstratified); Lambda0 is a right-continuous step function at the unique event times."""
    eta = np.asarray(eta, float)
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    out = {}
    keys = [None] if strata is None else list(np.unique(strata))
    for s in keys:
        m = np.ones(len(time), bool) if s is None else (np.asarray(strata) == s)
        t, e, r = time[m], event[m], np.exp(eta[m])
        ut = np.unique(t[e])
        dH = np.array([e[t == u].sum() / r[t >= u].sum() for u in ut])
        out[s] = (ut, np.cumsum(dH))
    return out


def cumhaz_at(baseline, times):
    ut, H = baseline
    idx = np.searchsorted(ut, np.asarray(times, float), side="right") - 1
    return np.where(idx >= 0, H[np.clip(idx, 0, None)], 0.0)


def cox_survival(eta, baseline, times):
    """S(t | x) = exp(-Lambda0(t) exp(eta)): (n,) x (T,) -> (n, T)."""
    H = cumhaz_at(baseline, times)
    return np.exp(-np.exp(np.asarray(eta, float))[:, None] * H[None, :])
