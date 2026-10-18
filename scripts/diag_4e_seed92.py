"""D96 diagnostic: training-seed spread of the Revision-4e selection on gate seed 92, TRAIN/VALIDATION ONLY (the test
split is not used). Writes reports/diag_4e_seed92.json."""
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.data.synthetic import SynthConfig, generate  # noqa: E402
from kanfae.evaluation.ceiling import ceiling_C  # noqa: E402
from kanfae.evaluation.metrics import harrell_c  # noqa: E402
from kanfae.models.pipelines import NeuralAFT  # noqa: E402
from learnability_search import probe  # noqa: E402


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    cfg = json.loads((ROOT / "reports/learnability_selected_fa4e.json").read_text())["config"]
    obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=92))
    tr, va, _ = split_groups(obs.ids, 92, strata=obs.delta)            # the test split is not used
    c_or = harrell_c(obs.y_obs[va], obs.delta[va], -ora.eta_true[va])
    c_ceil = ceiling_C(obs, va)
    rows = []
    for seed in range(5):
        m = NeuralAFT("fae_mlp", **cfg).fit(obs.subset(tr), obs.subset(va), seed=seed)
        c = harrell_c(obs.y_obs[va], obs.delta[va], m.predict_risk(obs.subset(va)))
        pr = probe(m, obs, ora, tr, va)
        run = m.info["runs"][0]
        rows.append(dict(training_seed=seed, val_C=c, val_C_minus_ceiling=c - c_ceil, val_probe=pr,
                         val_probe_mean=float(np.mean(pr)), objective_stop_epoch=run["stop_epoch"],
                         risk_state_epoch=run.get("risk_state_epoch")))
        print(f"training seed {seed}: val C {c:.4f} (ceiling {c_ceil:.4f}, oracle {c_or:.4f}; C - ceiling "
              f"{c - c_ceil:+.4f}) probe {np.round(pr, 3)} epochs {run['stop_epoch']}/{run.get('risk_state_epoch')}",
              flush=True)
    cs = np.array([r["val_C"] for r in rows])
    out = dict(data_seed=92, split="validation", val_oracle_C=c_or, val_C_ceiling=c_ceil, rows=rows,
               val_C_mean=float(cs.mean()), val_C_sd=float(cs.std(ddof=1)), val_C_range=float(cs.max() - cs.min()))
    (ROOT / "reports/diag_4e_seed92.json").write_text(json.dumps(out, indent=1))
    print(f"val C mean {cs.mean():.4f} sd {cs.std(ddof=1):.4f} range {cs.max() - cs.min():.4f}")


if __name__ == "__main__":
    main()
