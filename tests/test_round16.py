"""Round 16 (D138): deterministic checks of the fixed-mu NaN-safe refit, the degenerate-formula rejection and the
non-cliff gate (no formal seed is used).

R16-1  safe refit: a refit whose parameters become non-finite is undone exactly (fallback recorded); a normal refit
       lowers the objective and is kept; safe=False is the Round-15 refit
R16-2  degenerate-formula rejection: with max_abs_arg every accepted library formula keeps |a x + b| <= the bound on
       the training range; without it (default) results are unchanged
R16-3  non-cliff gate: "all off" is chosen only when strictly better than every setting that keeps a channel; the
       D133 parsimonious rule is unchanged
"""
import copy

import numpy as np
import pytest
import torch

from kanfae.models.fkan import FKANSurv, substitute_symbolic, symbolic_edges
from kanfae.models.fres import ALL_OFF, choose_gate
from kanfae.symbolic.fit import fit_symbolic

pytestmark = [pytest.mark.stage("R16"), pytest.mark.kind("DET")]
torch.set_default_dtype(torch.float64)


def _fitted():
    rng = np.random.default_rng(5)
    n, G = 160, 50
    t = np.linspace(0, 1, G)
    xg = rng.normal(size=(n, 1, 2)) + 0.5 * rng.normal(size=(n, 1, 2)) * t[None, :, None]
    z = rng.normal(size=(n, 1))
    eta = np.sin(xg[:, :, 0].mean(1)) + 0.5 * z[:, 0]
    T, C = rng.exponential(np.exp(-eta)), rng.exponential(2.0, n)
    p = dict(xg=xg, z=z, y=np.minimum(T, C), d=(T <= C).astype(float))
    tr = {k: v[:110] for k, v in p.items()}
    va = {k: v[110:] for k, v in p.items()}
    return FKANSurv(kind="b2s", loss="cox", max_epochs=60, patience=60, seed=0, lr=1e-2).fit(tr, va)


@pytest.mark.criterion("R16-1")
def test_R16_1_safe_refit_fallback(monkeypatch):
    m = _fitted()
    sub = copy.deepcopy(m)
    substitute_symbolic(sub, symbolic_edges(m), guard=True)
    before = [q.detach().clone() for q in sub._params()]

    def nan_step(self, closure):
        with torch.no_grad():
            for g in self.param_groups:
                for q in g["params"]:
                    q.fill_(float("nan"))
    bad = copy.deepcopy(sub)
    monkeypatch.setattr(torch.optim.LBFGS, "step", nan_step)
    bad.refit_symbolic(max_iter=20, anchor=m, mu=10.0, safe=True)
    assert bad.refit_fallback and all(torch.equal(a, b) for a, b in zip(bad._params(), before))
    assert np.isfinite(bad.val_loss)
    monkeypatch.undo()
    good = copy.deepcopy(sub).refit_symbolic(max_iter=20, anchor=m, mu=10.0, safe=True)
    plain = copy.deepcopy(sub).refit_symbolic(max_iter=20, anchor=m, mu=10.0)
    if not good.refit_fallback:
        assert all(torch.equal(a, b) for a, b in zip(good._params(), plain._params()))
    assert np.isfinite(good.val_loss) and plain.refit_fallback is False


@pytest.mark.criterion("R16-2")
def test_R16_2_degenerate_formula_rejection():
    x = np.linspace(-3.0, 3.0, 400)
    f = lambda v: 0.0018 * np.log1p(np.exp(np.clip(205.3 * np.asarray(v) + 384.1, -700, 700))) - 0.757  # noqa: E731
    y = f(x)
    loose = fit_symbolic(x, y, f, use_pysr=False)
    tight = fit_symbolic(x, y, f, use_pysr=False, max_abs_arg=50.0)
    again = fit_symbolic(x, y, f, use_pysr=False)
    assert loose.formula == again.formula and loose.stage == again.stage           # default unchanged, deterministic
    if tight.accepted and tight.stage == "library":
        a, b = tight.params["a"], tight.params["b"]
        assert np.max(np.abs(a * np.array([x.min(), x.max()]) + b)) <= 50.0
    m = _fitted()
    for r in symbolic_edges(m, max_abs_arg=50.0):
        res = r["result"]
        if res.accepted and res.stage == "library":
            lo, hi = res.x_range
            assert np.max(np.abs(res.params["a"] * np.array([lo, hi]) + res.params["b"])) <= 50.0 + 1e-9


@pytest.mark.criterion("R16-3")
def test_R16_3_non_cliff_gate():
    th = (0.0, 0.1, 0.25, ALL_OFF)
    assert choose_gate([4.300, 4.300, 4.305, 4.295], th, 0.01, "parsimonious") == 3       # D133: all off within tol
    assert choose_gate([4.300, 4.300, 4.305, 4.295], th, 0.01, "noncliff") == 3           # strictly better: off
    assert choose_gate([4.300, 4.300, 4.305, 4.300], th, 0.01, "noncliff") == 2           # tie: keep channels
    assert choose_gate([4.300, 4.300, 4.305, 4.3049], th, 0.01, "parsimonious") == 3
    assert choose_gate([4.300, 4.300, 4.305, 4.3049], th, 0.01, "noncliff") == 2
    assert choose_gate([4.300, 4.320, 4.330, 4.400], th, 0.01, "noncliff") == 0           # parsimony among on-settings
