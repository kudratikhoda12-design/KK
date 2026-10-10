"""Symbolic regression of learned KAN shape functions (as in pykan's auto_symbolic).

For each learned univariate function phi(x) we try a small library of primitives f and fit
        phi(x) ~= a * f(b * x + c) + d
(b, c by grid search, a, d by least squares). The simplest primitive reaching R^2 >= r2_ok
is chosen, giving a closed-form log-hazard formula.
"""
from __future__ import annotations

import numpy as np

# (name, function, complexity) - lower complexity is preferred when fits are equally good
LIBRARY = [
    ("x", lambda z: z, 1),
    ("x^2", lambda z: z ** 2, 2),
    ("x^3", lambda z: z ** 3, 3),
    ("exp", np.exp, 3),
    ("tanh", np.tanh, 3),
    ("sin", np.sin, 4),
    ("log", lambda z: np.log(np.abs(z) + 1e-3), 4),
    ("sigmoid", lambda z: 1 / (1 + np.exp(-z)), 3),
]


def _affine_fit(fx, y):
    A = np.c_[fx, np.ones_like(fx)]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - pred) ** 2).sum() / ss if ss > 0 else 1.0
    return coef, r2


def fit_symbolic(x, y, r2_ok=0.98):
    x, y = np.asarray(x, float), np.asarray(y, float)
    results = []
    bs = np.r_[-np.geomspace(0.1, 3, 12)[::-1], np.geomspace(0.1, 3, 12)]
    cs = np.linspace(-3, 3, 25)
    for name, f, cx in LIBRARY:
        best = (-np.inf, None, 1.0, 0.0)
        if name in ("x",):
            grid = [(1.0, 0.0)]
        elif name in ("x^2", "x^3"):
            grid = [(1.0, c) for c in cs]
        else:
            grid = [(b, c) for b in bs for c in cs]
        with np.errstate(all="ignore"):
            for b, c in grid:
                fx = f(b * x + c)
                if not np.all(np.isfinite(fx)) or fx.std() < 1e-9:
                    continue
                coef, r2 = _affine_fit(fx, y)
                if r2 > best[0]:
                    best = (r2, coef, b, c)
        if best[1] is not None:
            results.append(dict(name=name, r2=float(best[0]), a=float(best[1][0]), d=float(best[1][1]),
                                b=float(best[2]), c=float(best[3]), complexity=cx))
    ok = [r for r in results if r["r2"] >= r2_ok]
    chosen = min(ok, key=lambda r: (r["complexity"], -r["r2"])) if ok else max(results, key=lambda r: r["r2"])
    return chosen, results


def to_string(fit, var="x", digits=3):
    a, b, c, d = (round(fit[k], digits) for k in "abcd")
    if fit["name"] == "x":
        inner = var
    else:
        inner = (var if b == 1 else f"{b}*{var}") + (f" {'+' if c >= 0 else '-'} {abs(c)}" if c != 0 else "")
    body = {
        "x": f"{var}", "x^2": f"({inner})^2", "x^3": f"({inner})^3", "exp": f"exp({inner})",
        "tanh": f"tanh({inner})", "sin": f"sin({inner})", "log": f"log|{inner}|",
        "sigmoid": f"sigmoid({inner})",
    }[fit["name"]]
    return f"{a}*{body} {'+' if d >= 0 else '-'} {abs(d)}"
