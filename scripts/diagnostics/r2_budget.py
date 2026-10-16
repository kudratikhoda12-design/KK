"""Diagnostic for the R2 failure: does a larger training budget fix it (under-optimisation) or not (structural)?"""
import sys, time
import numpy as np, torch
from kanfae.data.splits import split_groups
from kanfae.data.synthetic import SynthConfig, generate
from kanfae.evaluation.metrics import harrell_c
from kanfae.models.pipelines import NeuralAFT
torch.set_default_dtype(torch.float64)
obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=21))
tr, va, te = split_groups(obs.ids, 21, strata=obs.delta)
c_or = harrell_c(obs.y_obs[te], obs.delta[te], -ora.eta_true[te])
def probe(m):
    Ztr, Zte = m.latent(obs.subset(tr)), m.latent(obs.subset(te))
    A = np.column_stack([np.ones(len(Ztr)), Ztr]); B = np.column_stack([np.ones(len(Zte)), Zte])
    out = []
    for j in range(4):
        coef = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ ora.theta[tr, j]); p = B @ coef; y = ora.theta[te, j]
        out.append(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return np.round(out, 3)
for kind, cfg in [("fae_mlp", dict(max_epochs=400)), ("fae_mlp", dict(max_epochs=3000, patience=100)),
                  ("fae_mlp", dict(max_epochs=3000, patience=100, lr=1e-2)),
                  ("fae_mlp", dict(max_epochs=3000, patience=100, alpha=0.0)),
                  ("fae_kan", dict(max_epochs=3000, patience=100))]:
    t0 = time.time(); m = NeuralAFT(kind, **cfg).fit(obs.subset(tr), obs.subset(va), seed=0)
    c = harrell_c(obs.y_obs[te], obs.delta[te], m.predict_risk(obs.subset(te)))
    print(f"{kind} {cfg}: C={c:.4f} (oracle {c_or:.4f}) probe(a,b,A,w)={probe(m)} epochs={[r['epochs'] for r in m.info['runs']]} "
          f"{time.time()-t0:.1f}s", flush=True)
