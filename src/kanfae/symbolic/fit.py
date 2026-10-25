"""Progressive symbolic approximation of learned univariate functions (KAN-AFT §3.2.3 and §4.1; CoxKAN §2.3.3).

Stage 1: linear  y = a x + b                         (accepted if validation R^2 > 0.99, CoxKAN)
Stage 2: library y = c f(a x + b) + d                 (affine form of KAN-AFT / Liu et al.; documented library below)
Stage 3: PySR symbolic regression, bounded budget     (if stages 1-2 fail)
A function without an accepted expression keeps its learned spline (never fabricated).

Fitting procedure (documented): grid search over (a, b) with closed-form least squares for (c, d), followed by a
Levenberg-Marquardt refinement of all four parameters (scipy.optimize.least_squares), with domain checks for
every candidate. R^2 is computed on held-out points that were not used to fit the parameters.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import least_squares

ACCEPT_R2 = 0.99


def _safe(fn):
    def g(u):
        with np.errstate(all="ignore"):
            return fn(u)
    return g


# name, function, complexity, domain predicate on u = a x + b. This library is our own documented choice (the
# CoxKAN 22-function library is in its supplementary material, which was not available to us).
LIBRARY = [
    ("x^2", _safe(lambda u: u ** 2), 2, lambda u: np.isfinite(u)),
    ("x^3", _safe(lambda u: u ** 3), 3, lambda u: np.isfinite(u)),
    ("x^4", _safe(lambda u: u ** 4), 4, lambda u: np.isfinite(u)),
    ("exp", _safe(np.exp), 2, lambda u: u < 50),
    ("log", _safe(np.log), 2, lambda u: u > 1e-6),
    ("sqrt", _safe(np.sqrt), 2, lambda u: u >= 0),
    ("1/x", _safe(lambda u: 1.0 / u), 2, lambda u: np.abs(u) > 1e-3),
    ("sin", _safe(np.sin), 2, lambda u: np.isfinite(u)),
    ("cos", _safe(np.cos), 2, lambda u: np.isfinite(u)),
    ("tanh", _safe(np.tanh), 2, lambda u: np.isfinite(u)),
    ("sigmoid", _safe(lambda u: 1 / (1 + np.exp(-u))), 2, lambda u: np.abs(u) < 50),
    ("arctan", _safe(np.arctan), 2, lambda u: np.isfinite(u)),
    ("abs", _safe(np.abs), 2, lambda u: np.isfinite(u)),
    ("gaussian", _safe(lambda u: np.exp(-u ** 2)), 3, lambda u: np.isfinite(u)),
    ("softplus", _safe(lambda u: np.logaddexp(0, u)), 2, lambda u: np.isfinite(u)),
]


def r2(y, yhat):
    ss = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / ss) if ss > 0 else float("nan")


@dataclass
class SymbolicResult:
    accepted: bool
    stage: str                    # "linear" | "library" | "pysr" | "none"
    formula: str
    params: dict
    complexity: int
    r2_fit: float
    r2_val: float
    r2_grid: float
    rmse_val: float
    maxabs_grid: float
    x_range: tuple
    candidates_tried: int = 0
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _affine_fit(f, dom, x, y, n_a=31, n_b=31):
    span = max(np.ptp(x), 1e-9)
    a_grid = np.r_[-np.geomspace(0.05, 20, n_a // 2) / span, np.geomspace(0.05, 20, n_a // 2) / span]
    best = None
    for a in a_grid:
        u0 = a * x
        b_grid = np.linspace(-3, 3, n_b) - np.mean(u0)
        for b in b_grid:
            u = u0 + b
            if not np.all(dom(u)):
                continue
            fu = f(u)
            if not np.all(np.isfinite(fu)) or np.ptp(fu) < 1e-12:
                continue
            A = np.column_stack([fu, np.ones_like(fu)])
            (c, d), *_ = np.linalg.lstsq(A, y, rcond=None)
            res = np.sum((A @ np.array([c, d]) - y) ** 2)
            if best is None or res < best[0]:
                best = (res, a, b, c, d)
    if best is None:
        return None
    _, a, b, c, d = best

    def resid(p):
        u = p[0] * x + p[1]
        with np.errstate(all="ignore"):
            out = p[2] * f(u) + p[3] - y
        return np.where(np.isfinite(out), out, 1e6)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = least_squares(resid, [a, b, c, d], method="lm", max_nfev=2000)
    p = sol.x
    if not np.all(dom(p[0] * x + p[1])) or not np.all(np.isfinite(f(p[0] * x + p[1]))):
        p = np.array([a, b, c, d])
    return p


def _predictor(name, f, p):
    return lambda z: p[2] * f(p[0] * np.asarray(z, float) + p[1]) + p[3]


def _evaluate(pred, x_fit, y_fit, x_val, y_val, x_grid, y_grid):
    with np.errstate(all="ignore"):
        pf, pv, pg = pred(x_fit), pred(x_val), pred(x_grid)
    if not (np.all(np.isfinite(pf)) and np.all(np.isfinite(pv)) and np.all(np.isfinite(pg))):
        return None
    return dict(r2_fit=r2(y_fit, pf), r2_val=r2(y_val, pv), r2_grid=r2(y_grid, pg),
                rmse_val=float(np.sqrt(np.mean((y_val - pv) ** 2))), maxabs_grid=float(np.max(np.abs(y_grid - pg))))


def fit_symbolic(x, y, func, seed=0, use_pysr=True, pysr_budget=None, val_frac=0.3, max_abs_arg=None):
    """x, y: samples of the learned function on its training inputs; func: callable giving the learned function
    exactly (used for a dense held-out grid over [min x, max x]). Returns SymbolicResult.
    max_abs_arg (D138, Round 16; default None = off): a library candidate c f(a x + b) + d whose inner argument
    |a x + b| exceeds this bound anywhere on [min x, max x] is rejected as degenerate (for example
    softplus(205 x + 384), a step disguised as a smooth formula)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_val = max(5, int(val_frac * len(x)))
    v, f_ = idx[:n_val], idx[n_val:]
    x_grid = np.linspace(x.min(), x.max(), 401)
    y_grid = np.asarray(func(x_grid), float)
    rng_x = (float(x.min()), float(x.max()))
    tried = 0
    # Stage 1: linear
    A = np.column_stack([x[f_], np.ones(len(f_))])
    (a1, b1), *_ = np.linalg.lstsq(A, y[f_], rcond=None)
    lin = lambda z: a1 * np.asarray(z) + b1  # noqa: E731
    ev = _evaluate(lin, x[f_], y[f_], x[v], y[v], x_grid, y_grid)
    tried += 1
    if ev and ev["r2_val"] > ACCEPT_R2:
        return SymbolicResult(True, "linear", f"{a1:.6g}*x + {b1:.6g}", {"a": a1, "b": b1}, 1, x_range=rng_x,
                              candidates_tried=tried, **ev)
    # Stage 2: library (accepted candidates: lowest complexity, then highest validation R^2)
    cands = []
    for name, f, comp, dom in LIBRARY:
        p = _affine_fit(f, dom, x[f_], y[f_])
        tried += 1
        if p is None:
            continue
        if max_abs_arg is not None and np.max(np.abs(p[0] * np.array(rng_x) + p[1])) > max_abs_arg:
            continue                                       # degenerate inner argument (D138)
        ev2 = _evaluate(_predictor(name, f, p), x[f_], y[f_], x[v], y[v], x_grid, y_grid)
        if ev2:
            cands.append((name, comp, p, ev2))
    acc = [c for c in cands if c[3]["r2_val"] > ACCEPT_R2]
    if acc:
        name, comp, p, ev2 = min(acc, key=lambda c: (c[1], -c[3]["r2_val"]))
        formula = f"{p[2]:.6g}*{name}({p[0]:.6g}*x + {p[1]:.6g}) + {p[3]:.6g}"
        return SymbolicResult(True, "library", formula, dict(zip("abcd", map(float, p)), f=name), comp + 2,
                              x_range=rng_x, candidates_tried=tried, **ev2)
    best_lib = max(cands, key=lambda c: c[3]["r2_val"]) if cands else None
    # Stage 3: PySR
    if use_pysr:
        try:
            res = _pysr_fit(x[f_], y[f_], x[v], y[v], x_grid, y_grid, pysr_budget or {}, seed)
            tried += 1
            if res is not None:
                res.x_range, res.candidates_tried = rng_x, tried
                if res.r2_val > ACCEPT_R2:
                    return res
                best_pysr = res
            else:
                best_pysr = None
        except Exception as exc:                       # recorded, never silently ignored
            best_pysr = SymbolicResult(False, "pysr", "", {}, 0, float("nan"), float("nan"), float("nan"),
                                       float("nan"), float("nan"), rng_x, tried, notes=f"PySR error: {exc}")
    else:
        best_pysr = None
    notes = ""
    if best_lib:
        notes += f"best library candidate {best_lib[0]} val R2={best_lib[3]['r2_val']:.4f}; "
    if best_pysr is not None and best_pysr.formula:
        notes += f"best PySR '{best_pysr.formula}' val R2={best_pysr.r2_val:.4f}; "
    return SymbolicResult(False, "none", "", {}, 0, float("nan"), float("nan"), float("nan"), float("nan"),
                          float("nan"), rng_x, tried, notes=notes + "learned spline retained",
                          extra={"best_library": None if not best_lib else {"f": best_lib[0], **best_lib[3]},
                                 "best_pysr": None if best_pysr is None else best_pysr.to_dict()})


def _pysr_fit(xf, yf, xv, yv, xg, yg, budget, seed):
    import pysr
    import sympy
    m = pysr.PySRRegressor(niterations=budget.get("niterations", 30), maxsize=budget.get("maxsize", 15),
                           binary_operators=["+", "-", "*", "/"],
                           unary_operators=["sin", "cos", "exp", "tanh", "square", "cube"],
                           populations=budget.get("populations", 15), timeout_in_seconds=budget.get("timeout", 120),
                           random_state=seed, deterministic=True, parallelism="serial", progress=False, verbosity=0,
                           model_selection="best")
    m.fit(xf[:, None], yf)
    expr = m.sympy()
    f = sympy.lambdify(sympy.Symbol("x0"), expr, "numpy")
    pred = lambda z: np.broadcast_to(np.asarray(f(np.asarray(z, float)), float), np.shape(z))  # noqa: E731
    ev = _evaluate(pred, xf, yf, xv, yv, xg, yg)
    if ev is None:
        return None
    comp = int(m.get_best()["complexity"])
    return SymbolicResult(ev["r2_val"] > ACCEPT_R2, "pysr", str(expr).replace("x0", "x"), {"sympy": str(expr)}, comp,
                          **ev, x_range=(0, 0), extra={"callable": None})
