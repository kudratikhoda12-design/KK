"""Clamped cubic B-spline basis on [0, 1] (docs/design.md section 3.1)."""
from __future__ import annotations

import numpy as np


def bspline_basis_matrix(x, knots, degree, deriv=0):
    """Evaluate all B-splines B_{j,degree} (j = 0..n_knots-degree-2) and derivatives at x.

    Cox-de Boor recursion; the last non-degenerate interval is closed on the right so that the
    right end of a clamped knot vector is included.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(knots, dtype=float)
    n_basis = len(t) - degree - 1
    flat = x.reshape(-1)
    # degree-0 indicator functions on [t_j, t_{j+1})
    B = ((flat[:, None] >= t[None, :-1]) & (flat[:, None] < t[None, 1:])).astype(float)
    last = np.max(np.nonzero(t[:-1] < t[1:])[0])           # last interval with positive length
    B[flat == t[-1], :] = 0.0
    B[flat == t[-1], last] = 1.0

    def raise_degree(Bp, p):
        out = np.zeros((len(flat), len(t) - p - 1))
        for j in range(len(t) - p - 1):
            d1, d2 = t[j + p] - t[j], t[j + p + 1] - t[j + 1]
            left = (flat - t[j]) / d1 * Bp[:, j] if d1 > 0 else 0.0
            right = (t[j + p + 1] - flat) / d2 * Bp[:, j + 1] if d2 > 0 else 0.0
            out[:, j] = left + right
        return out

    def derivative(Bp, p):  # d/dx of degree-p splines from degree-(p-1) values Bp
        out = np.zeros((len(flat), len(t) - p - 1))
        for j in range(len(t) - p - 1):
            d1, d2 = t[j + p] - t[j], t[j + p + 1] - t[j + 1]
            a = p / d1 * Bp[:, j] if d1 > 0 else 0.0
            b = p / d2 * Bp[:, j + 1] if d2 > 0 else 0.0
            out[:, j] = a - b
        return out

    if deriv > degree:
        return np.zeros(x.shape + (n_basis,))
    levels = [B]
    for p in range(1, degree - deriv + 1):
        levels.append(raise_degree(levels[-1], p))
    # Start from the values of degree (degree - deriv), then apply
    # d^m B_{j,p} = p/(t_{j+p}-t_j) d^{m-1} B_{j,p-1} - p/(t_{j+p+1}-t_{j+1}) d^{m-1} B_{j+1,p-1}
    # for p = degree-deriv+1 .. degree; each step raises the degree and the derivative order by one.
    cur = levels[-1]
    for p in range(degree - deriv + 1, degree + 1):
        cur = derivative(cur, p)
    return cur.reshape(x.shape + (n_basis,))


class BSplineBasis:
    def __init__(self, n_basis: int = 12, degree: int = 3, domain=(0.0, 1.0)):
        if n_basis < degree + 1:
            raise ValueError("n_basis must be >= degree + 1")
        self.K, self.degree, self.domain = n_basis, degree, domain
        n_int = n_basis - degree - 1
        inner = np.linspace(domain[0], domain[1], n_int + 2)
        self.knots = np.r_[[domain[0]] * degree, inner, [domain[1]] * degree]
        self.breaks = inner

    def __call__(self, x, deriv=0):
        return bspline_basis_matrix(x, self.knots, self.degree, deriv)

    def _gauss_legendre(self, n_per_interval=64):
        z, w = np.polynomial.legendre.leggauss(n_per_interval)
        xs, ws = [], []
        for a, b in zip(self.breaks[:-1], self.breaks[1:]):
            xs.append(0.5 * (b - a) * z + 0.5 * (a + b))
            ws.append(0.5 * (b - a) * w)
        return np.concatenate(xs), np.concatenate(ws)

    def roughness(self):
        """R_{k,k'} = int phi_k''(t) phi_{k'}''(t) dt (Gauss-Legendre per knot interval; exact for polynomials)."""
        x, w = self._gauss_legendre()
        D2 = self(x, deriv=2)
        return (D2 * w[:, None]).T @ D2

    def gram(self):
        x, w = self._gauss_legendre()
        B = self(x)
        return (B * w[:, None]).T @ B

    def cross_gram(self, other: "BSplineBasis"):
        """J_{k,l} = int phi_k psi_l dt over the common domain (union of both break sets)."""
        brk = np.union1d(self.breaks, other.breaks)
        z, wq = np.polynomial.legendre.leggauss(64)
        xs = np.concatenate([0.5 * (b - a) * z + 0.5 * (a + b) for a, b in zip(brk[:-1], brk[1:])])
        ws = np.concatenate([0.5 * (b - a) * wq for a, b in zip(brk[:-1], brk[1:])])
        return (self(xs) * ws[:, None]).T @ other(xs)
