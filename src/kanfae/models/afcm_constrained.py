"""D168 Stage A, step A7: candidate repair of the Python AFCM optimiser (a NEW code path; kanfae.models.afcm, the
historical V3 and REML estimators, is not modified).

Two changes, each with a mathematical reason:
1. Identifiability imposed during fitting. For a Cox model any F_d(x, t) = c(t) (a function of t alone) shifts every
   linear predictor by the same constant and is not identified; the historical code keeps these directions and holds
   them with a 1e-6 ridge, which leaves the penalised Hessian nearly singular. Here the x-margin of each channel's
   tensor basis is centred over the training values (as mgcv's ti(..., mc = c(FALSE, TRUE)) does): with C_x = sum over
   training subjects and grid times of B_x(X_i(t_g)) and Z_x a basis of its null space, theta_d = (Z_x (x) I_t) beta_d.
   Because the B-spline basis sums to one, constant-in-x functions are excluded exactly, so the n_t unidentified
   directions are removed and no ridge is needed.
2. A second-order solver with a stationarity stopping rule. Damped Newton on the same objective as the historical code,
   mean negative Efron log partial likelihood + sum_d lam_x,d th' Kx th + lam_t,d th' Kt th, with the Hessian
   I/E + 2S (I the observed Cox information; Breslow form, exact when no event times are tied), Armijo backtracking,
   stop when max |gradient| < tol_grad and the Newton decrement g' H^-1 g / 2 < tol_dec.
With constrain=False the same solver is applied to the historical parametrisation (ridge kept), which re-converges the
historical objective and shows where L-BFGS stopped relative to its own optimum.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from ..survival.cox import breslow_baseline, cox_nll
from .afcm import cox_information


def _xcentre_null(proto, xg, d):
    Bx = proto._bx(d, xg[:, :, d])                                        # (n, G, n_x)
    C = Bx.reshape(-1, Bx.shape[-1]).sum(0)[None, :]                      # (1, n_x)
    q, _ = np.linalg.qr(C.T, mode="complete")
    return q[:, 1:]                                                       # (n_x, n_x - 1), orthonormal null space


def fit_newton(proto, xg, z, y, d, lam_x, lam_t, constrain=True, theta0=None, tol_grad=1e-6, tol_dec=1e-8,
               max_iter=200, ridge=None):
    """proto: an AFCM instance with the basis settings (e.g. AFCM(**V3 settings)); xg (n, G, D) training curves.
    lam_x, lam_t: scalars or per-channel sequences. theta0: optional (theta (D, n_x, n_t), gamma) warm start.
    Returns (fitted AFCM copy, record)."""
    torch.set_default_dtype(torch.float64)
    m = copy.deepcopy(proto)
    m.lam_x, m.lam_t = lam_x, lam_t
    m._set_ranges(xg)
    m.zm = z.mean(0)
    m.zs = np.where(z.std(0) > 0, z.std(0), 1.0)
    W = m.design(xg)                                                      # (n, D, P)
    n, D, P = W.shape
    Z = m._z(z, n)
    nx, nt = m.n_x, m.n_t
    Kx, Kt = (k.numpy() for k in m._penalty_mats())
    rg = (0.0 if constrain else m.ridge) if ridge is None else ridge
    blocks, S_blocks = [], []
    for k in range(D):
        T = np.kron(_xcentre_null(m, xg, k), np.eye(nt)) if constrain else np.eye(P)
        blocks.append(T)
        S_blocks.append(T.T @ (m._lam(lam_x, k) * Kx + m._lam(lam_t, k) * Kt + rg * np.eye(P)) @ T)
    widths = [b.shape[1] for b in blocks]
    X = np.hstack([W[:, k] @ blocks[k] for k in range(D)] + [Z])
    q = X.shape[1]
    S = np.zeros((q, q))
    o = 0
    for k in range(D):
        S[o:o + widths[k], o:o + widths[k]] = S_blocks[k]
        o += widths[k]
    E = float(np.sum(d))
    yt, dt, Xt, St = torch.tensor(y, dtype=torch.float64), torch.tensor(np.asarray(d, float)), torch.tensor(X), torch.tensor(S)

    def f_and_g(b):
        bt = torch.tensor(b, requires_grad=True)
        obj = cox_nll(Xt @ bt, yt, dt, "efron") + bt @ St @ bt
        obj.backward()
        return float(obj), bt.grad.numpy().copy()

    b = np.zeros(q)
    if theta0 is not None:
        th0, g0 = theta0
        o, parts = 0, []
        for k in range(D):
            parts.append(np.linalg.lstsq(blocks[k], np.asarray(th0[k]).reshape(-1), rcond=None)[0])
        b = np.concatenate(parts + [np.asarray(g0, float)])
    f, g = f_and_g(b)
    hist, conv = [], False
    for it in range(max_iter):
        H = cox_information(X, X @ b, y, d) / E + 2 * S
        step = np.linalg.solve(H, g)
        dec = float(g @ step) / 2
        hist.append(dict(iter=it, objective=f, grad_max_abs=float(np.max(np.abs(g))), decrement=dec))
        if np.max(np.abs(g)) < tol_grad and dec < tol_dec:
            conv = True
            break
        a = 1.0
        while True:
            bn = b - a * step
            fn, gn = f_and_g(bn)
            if np.isfinite(fn) and fn <= f - 1e-4 * a * float(g @ step):
                break
            a *= 0.5
            if a < 1e-12:
                break
        if a < 1e-12:
            hist.append(dict(iter=it, note="line search failed"))
            break
        b, f, g = bn, fn, gn
    o = 0
    theta = np.zeros((D, nx * nt))
    for k in range(D):
        theta[k] = blocks[k] @ b[o:o + widths[k]]
        o += widths[k]
    m.theta = theta.reshape(D, nx, nt)
    m.gamma = b[o:]
    m.D = D
    eta = X @ b
    m.baseline = breslow_baseline(eta, y, d)
    H = cox_information(X, eta, y, d) / E + 2 * S
    rec = dict(constrain=constrain, converged=conv, iterations=len([h for h in hist if "objective" in h]),
               objective=f, grad_max_abs=float(np.max(np.abs(g))),
               decrement=float(g @ np.linalg.solve(H, g)) / 2, cond_H=float(np.linalg.cond(H)), n_coef=int(q),
               ridge=rg, history_tail=hist[-3:])
    return m, rec


def stationarity(m, xg, z, y, d):
    """For a fitted AFCM (historical parametrisation, its own ridge): largest |penalised gradient|, Newton decrement
    and condition number of the penalised Hessian at the returned coefficients (as in the Round-25 diagnostic)."""
    torch.set_default_dtype(torch.float64)
    W = m.design(xg)
    n, D, P = W.shape
    Z = m._z(z, n)
    X = np.hstack([W.reshape(n, D * P), Z])
    b = np.concatenate([m.theta.reshape(-1), m.gamma])
    Kx, Kt = (k.numpy() for k in m._penalty_mats())
    S = np.zeros((X.shape[1], X.shape[1]))
    for k in range(D):
        sl = slice(k * P, (k + 1) * P)
        S[sl, sl] = m._lam(m.lam_x, k) * Kx + m._lam(m.lam_t, k) * Kt + m.ridge * np.eye(P)
    bt = torch.tensor(b, requires_grad=True)
    obj = cox_nll(torch.tensor(X) @ bt, torch.tensor(y, dtype=torch.float64), torch.tensor(np.asarray(d, float)), "efron") \
        + bt @ torch.tensor(S) @ bt
    obj.backward()
    g = bt.grad.numpy()
    H = cox_information(X, X @ b, y, d) / float(np.sum(d)) + 2 * S
    return dict(objective=float(obj), grad_max_abs=float(np.max(np.abs(g))),
                decrement=float(g @ np.linalg.solve(H, g)) / 2, cond_H=float(np.linalg.cond(H)))
