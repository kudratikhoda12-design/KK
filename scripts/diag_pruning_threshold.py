"""D54 diagnostic: first-layer edge L1 scores of KAN-AFT (parametric) on the synthetic design of run_kanaft_symbolic.py."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kanfae.models.kan_aft import KANAFT  # noqa: E402
from run_kanaft_symbolic import TRUE_NAMES, simulate  # noqa: E402

OUT = ROOT / "reports" / "pruning_threshold_diag.jsonl"


def main():
    torch.set_default_dtype(torch.float64)
    OUT.write_text("")
    for seed in (201, 202, 203):
        Z, y, d, _ = simulate(2000, seed)
        idx = np.random.default_rng(seed).permutation(2000)
        tr, va = idx[:1200], idx[1200:1600]
        for lam in (0.0, 1e-3, 1e-2, 1e-1):
            m = KANAFT(loss="parametric", lam=lam, seed=0).fit(Z[tr], y[tr], d[tr], Z[va], y[va], d[va])
            with torch.no_grad():
                act = m.net.layers[0].edge_activations(m.X_train_std)[:, :, 0]
                l1 = act.abs().mean(0).numpy()
                l1c = (act - act.mean(0)).abs().mean(0).numpy()          # offset-free (centred) score
            rec = dict(seed=seed, lam=lam, edge_l1=dict(zip(TRUE_NAMES, l1.tolist())),
                       edge_l1_centred=dict(zip(TRUE_NAMES, l1c.tolist())), centred_null=float(l1c[3]),
                       theta_to_prune_null=float(l1[3]), min_true_edge_l1=float(l1[:3].min()),
                       separable=bool(l1[3] < l1[:3].min()))
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"seed {seed} lam {lam:g}: edge L1 {np.round(l1, 4)}; null/min-true ratio "
                  f"{l1[3] / l1[:3].min():.3f} | centred {np.round(l1c, 4)}", flush=True)


if __name__ == "__main__":
    main()
