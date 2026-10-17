"""D65 diagnostic (validation only; seed-21 train/validation): does a compact spectral summary or stronger weight decay
remove the overfitting of the revision-4 front? Not a selection and not a gate. Writes reports/diag_fa_overfitting.jsonl."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.data.synthetic import SynthConfig, generate  # noqa: E402
from kanfae.evaluation.metrics import harrell_c  # noqa: E402
from kanfae.models.pipelines import NeuralAFT  # noqa: E402
from learnability_search import probe  # noqa: E402

OUT = ROOT / "reports" / "diag_fa_overfitting.jsonl"
SEL = json.loads((ROOT / "reports" / "learnability_selected_fa.json").read_text())["config"]
VARIANTS = {"compact_J17_r8": dict(SEL, fa_compact=True, fa_nfreq=17, r=8),
            "compact_J17_r12": dict(SEL, fa_compact=True, fa_nfreq=17, r=12),
            "compact_J33_r12": dict(SEL, fa_compact=True, fa_nfreq=33, r=12),
            "full_J17_r12_wd1e-3": dict(SEL, wd=1e-3)}


def main():
    torch.set_default_dtype(torch.float64)
    obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=21))
    tr, va, te = split_groups(obs.ids, 21, strata=obs.delta)          # te unused
    OUT.write_text("")
    for name, cfg in VARIANTS.items():
        for kind in ("fae_mlp", "fae_kan"):
            t0 = time.time()
            m = NeuralAFT(kind, **cfg).fit(obs.subset(tr), obs.subset(va), seed=0)
            rec = dict(variant=name, kind=kind, config=cfg,
                       train_C=harrell_c(obs.y_obs[tr], obs.delta[tr], m.predict_risk(obs.subset(tr))),
                       val_C=harrell_c(obs.y_obs[va], obs.delta[va], m.predict_risk(obs.subset(va))),
                       val_oracle_C=harrell_c(obs.y_obs[va], obs.delta[va], -ora.eta_true[va]),
                       val_probe=probe(m, obs, ora, tr, va), epochs=[r["epochs"] for r in m.info["runs"]],
                       block_params=m.info["block_params"], runtime_s=round(time.time() - t0, 1))
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{name:22s} {kind}: train C {rec['train_C']:.4f} val C {rec['val_C']:.4f} (oracle "
                  f"{rec['val_oracle_C']:.4f}) probe {np.round(rec['val_probe'], 2)} mean {np.mean(rec['val_probe']):.3f} "
                  f"epochs {rec['epochs']} {rec['runtime_s']}s", flush=True)


if __name__ == "__main__":
    main()
