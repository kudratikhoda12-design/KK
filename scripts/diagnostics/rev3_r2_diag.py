"""Diagnostics for revised-design R2 failure (gate data, seed 21 only; DIAGNOSTIC, not evidence for H1)."""
import time
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
    A = np.column_stack([np.ones(len(Ztr)), Ztr]); B = np.column_stack([np.ones(len(Zte)), Zte]); out = []
    for j in range(4):
        coef = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ ora.theta[tr, j]); y = ora.theta[te, j]
        out.append(1 - ((y - B @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return np.round(out, 3)
for cfg in [dict(), dict(alpha=0.0), dict(alpha=0.1), dict(r=8), dict(lr=1e-3)]:
    t0 = time.time(); m = NeuralAFT("fae_mlp", **cfg).fit(obs.subset(tr), obs.subset(va), seed=0)
    ctr = harrell_c(obs.y_obs[tr], obs.delta[tr], m.predict_risk(obs.subset(tr)))
    cte = harrell_c(obs.y_obs[te], obs.delta[te], m.predict_risk(obs.subset(te)))
    print(f"{cfg or 'revised default'}: train C={ctr:.4f} test C={cte:.4f} (oracle {c_or:.4f}, threshold {c_or-0.03:.4f}) "
          f"probe={probe(m)} epochs={[r['epochs'] for r in m.info['runs']]} {time.time()-t0:.0f}s", flush=True)
