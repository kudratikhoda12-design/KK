"""Round 15 post-hoc diagnosis (exploratory; does not change any registered result): why every distillation refit of
S-Add replicate 3515 became non-finite although its only restricted-domain formula (a log) was guarded. Re-runs the
registered pipeline for that seed deterministically and records, per mu, where the first non-finite value appears."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_round15 as D15  # noqa: E402
from kanfae.models.fkan import FKANEnsemble, FKANSurv, SymbolicEdge, select_lambda_theta, substitute_symbolic, symbolic_edges  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
ctx = D15.Ctx15("S-Add", 3515)
xg = ctx.front("bspline")["xg"]
tr, va, te = ctx.pack(ctx.tr, xg), ctx.pack(ctx.va, xg), ctx.pack(ctx.te, xg)
e = FKANEnsemble(kind="b2s", loss="cox").fit(tr, va)
sel, _ = select_lambda_theta(lambda l: FKANSurv(kind="b2s", loss="cox", lam=l, lr=e.lr, wd=e.wd, seed=0), tr, va)
recs = symbolic_edges(sel, use_pysr=False)
subg = copy.deepcopy(sel)
substitute_symbolic(subg, recs, guard=True)
subg._finalise(subg.train_)
out = dict(seed=3515, lr=e.lr, wd=e.wd, edges=[])
for lay_name in ("g", "h", "stat"):
    lay = getattr(subg.net, lay_name, None)
    if lay is None:
        continue
    for k, ed in lay.symbolic.items():
        out["edges"].append(dict(layer=lay_name, edge=list(k), f=ed.name, p=ed.p.detach().tolist(), guard=ed.guard))
res = {}
for mu in (0.01, 10.0):
    m = copy.deepcopy(subg)
    trace = []
    orig = m._loss

    def logged(eta, y, d, s=None, m=m, orig=orig):
        v = orig(eta, y, d, s)
        trace.append(dict(loss=float(v.detach()), eta_finite=bool(torch.isfinite(eta).all()),
                          eta_absmax=float(eta.detach().abs().max()) if torch.isfinite(eta).all() else None))
        return v
    m._loss = logged
    m.refit_symbolic(max_iter=200, anchor=sel, mu=mu)
    bad = [dict(layer=n, edge=list(k), f=ed.name, p=ed.p.detach().tolist()) for n in ("g", "h", "stat")
           if getattr(m.net, n, None) is not None for k, ed in getattr(m.net, n).symbolic.items()
           if not torch.isfinite(ed.p).all() or ed.p.detach().abs().max() > 1e3]
    spl = [n for n, q in m.net.named_parameters() if not torch.isfinite(q).all()]
    first = next((i for i, t in enumerate(trace) if not t["eta_finite"] or not np.isfinite(t["loss"])), None)
    res[str(mu)] = dict(n_evals=len(trace), first_nonfinite_eval=first, last_finite=trace[first - 1] if first else None,
                        trace_head=trace[:5], symbolic_params_bad_or_huge=bad, spline_params_nonfinite=spl,
                        val_loss=m.val_loss)
out["refits"] = res
(ROOT / "reports/round15/diag_3515.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str)[:4000])
