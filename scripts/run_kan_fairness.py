"""D108 (Round 9): pre-registered KAN-fairness sensitivity study on P0 (replicates 1000-1004, n = 1000), single-threaded.
Arms (C4F otherwise; protocol 2.1 tuning lr x wd at seed 0 by validation IBS; primary = 3-seed ensemble on test):
  mlp         fae_mlp (C4F)                          kan         fae_kan (C4F; D31 decoder 16 -> 3 -> 36)
  kan_shared  fae_kan, Mode B: fae_mlp's MLP decoder  kan_unb     fae_kan, Mode A: single-layer KAN decoder 16 -> 36
  mlp_wide    fae_mlp with hidden 107 (encoder + decoder 10,966 parameters; Mode A's fae_kan has 10,932)
Strategies S2 (C4F joint) and S3 (reconstruction pre-training, then joint). Every record also carries the test
reconstruction error (runner.reconstruction_eval). Guarded by the 4f verdict. Records: $KANFAE_RESULTS."""
import argparse
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
from kanfae.experiments.runner import reconstruction_eval, tune_fit_evaluate_ensemble  # noqa: E402
from run_neural_synthetic import NN_GRID  # noqa: E402
from run_synthetic import SETTINGS  # noqa: E402

ARMS = {"mlp": ("fae_mlp", {}), "kan": ("fae_kan", {}), "kan_shared": ("fae_kan", {"dec_block": "mlp"}),
        "kan_unb": ("fae_kan", {"kan_dec": "single"}), "mlp_wide": ("fae_mlp", {"hidden": 107})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="comma list of arm:strategy, e.g. mlp:S2,kan:S3")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    v = json.loads((ROOT / "reports/revised_gate_verdict_fa4f.json").read_text())
    if v.get("verdict") != "CONFIRMED":
        sys.exit("BLOCKED (D57/D61): the conventional MLP has not passed the revised gate.")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    sel = json.loads((ROOT / "reports/learnability_selected_fa4f.json").read_text())
    tuned = {k: v for k, v in sel["config"].items() if k not in ("lr", "wd")}
    for cell in a.cells.split(","):
        arm, strategy = cell.split(":")
        model, extra = ARMS[arm]
        cfg = {**tuned, **extra, "strategy": strategy}
        for rep in range(a.reps):
            seed = 1000 + rep
            t0 = time.time()
            obs, ora = generate(SynthConfig(n=1000, seed=seed, **SETTINGS["P0"]))
            tr, va, te = split_groups(obs.ids, seed, strata=obs.delta)
            o_te = obs.subset(te)
            base = dict(dataset="synthetic", analysis="r9_fairness", setting="P0", replicate=rep, data_seed=seed,
                        split_seed=seed, n_train=len(tr), n_val=len(va), n_test=len(te),
                        censored_frac_test=float(1 - o_te.delta.mean()), data_kind="SYNTHETIC",
                        selected_config_id=sel["config_id"], arm=arm, strategy=strategy,
                        variant=json.dumps({"arm": arm, "strategy": strategy}, sort_keys=True))
            tune_fit_evaluate_ensemble(model, obs.subset(tr), obs.subset(va), o_te, base, ora.subset(te),
                                       np.linspace(0, ora.t_max, 101), grid=NN_GRID, k=int(sel["ensemble_seeds"]),
                                       extra_cfg=cfg, extra_eval=lambda m, s, o_te=o_te: reconstruction_eval(m, o_te))
            print(f"[{arm} {strategy} rep {rep}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
