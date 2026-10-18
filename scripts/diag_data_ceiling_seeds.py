"""D75 extension: parametric data-information ceiling on all 2000 individuals of fresh seeds 701-710 (no fitting across
individuals, so no train/test split is needed). Writes reports/diag_data_ceiling_seeds.jsonl."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kanfae.data.synthetic import SynthConfig, generate  # noqa: E402
from kanfae.evaluation.metrics import harrell_c  # noqa: E402
from diag_data_ceiling import parametric, true_eta  # noqa: E402

out = ROOT / "reports" / "diag_data_ceiling_seeds.jsonl"
out.write_text("")
for seed in range(701, 711):
    obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=seed))
    est = np.array([parametric(obs.times[i, obs.mask[i, :, 0]], obs.values[i, obs.mask[i, :, 0], 0],
                               obs.values[i, obs.mask[i, :, 0], 1]) for i in range(len(obs))])
    c_or = harrell_c(obs.y_obs, obs.delta, -ora.eta_true)
    c_pl = harrell_c(obs.y_obs, obs.delta, -true_eta(*est.T))
    # which parameter limits? plug in the truth for all but one estimated parameter
    lim = {}
    for j, nm in enumerate(("a", "b", "A", "omega")):
        th = ora.theta[:, :4].copy(); th[:, j] = est[:, j]
        lim[nm] = harrell_c(obs.y_obs, obs.delta, -true_eta(*th.T))
    rec = dict(seed=seed, oracle_C=c_or, threshold=c_or - 0.03, parametric_plugin_C=c_pl,
               gap_to_threshold=c_pl - (c_or - 0.03), only_this_parameter_estimated_C=lim)
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"seed {seed}: oracle {c_or:.4f} threshold {c_or - 0.03:.4f} parametric plug-in {c_pl:.4f} "
          f"(gap {c_pl - c_or + 0.03:+.4f}); only one parameter estimated: " +
          ", ".join(f"{k} {v:.4f}" for k, v in lim.items()), flush=True)
