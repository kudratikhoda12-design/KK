"""Diagnostic 2 for R2: capacity vs data (train vs test C), gate data only (seed 21)."""
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
for cfg in [dict(H=16, hidden=32), dict(H=32, hidden=64), dict(H=64, hidden=128), dict(H=32, hidden=64, r=8),
            dict(H=32, hidden=64, alpha=0.1)]:
    t0 = time.time()
    m = NeuralAFT("fae_mlp", max_epochs=3000, patience=100, **cfg).fit(obs.subset(tr), obs.subset(va), seed=0)
    ctr = harrell_c(obs.y_obs[tr], obs.delta[tr], m.predict_risk(obs.subset(tr)))
    cte = harrell_c(obs.y_obs[te], obs.delta[te], m.predict_risk(obs.subset(te)))
    print(f"{cfg}: train C={ctr:.4f} test C={cte:.4f} (oracle {c_or:.4f}) params={m.n_params} "
          f"epochs={[r['epochs'] for r in m.info['runs']]} {time.time()-t0:.1f}s", flush=True)
