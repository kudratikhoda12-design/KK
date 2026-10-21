"""Round 12, S-Int/P0 continuity (D113 as amended by D126): fae_kan_s3 vs fae_mlp_s3 on the Round-1 simulator setting
P0 (n 1,000) on fresh data seeds 3400-3419 (replicate r -> data and split seed 3400 + r). Configuration C4F (the 4f
selection; protocol 2.1 tuning of lr x wd at seed 0 by validation IBS; primary record = 3-seed ensemble on test),
strategy S3, exactly as the Round-9 fairness cells mlp:S3 and kan:S3 (scripts/run_kan_fairness.py). Records:
$KANFAE_RESULTS. Guarded by the 4f verdict. Analysis: TOST equivalence of survival-curve error, margin +-0.005."""
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
from run_kan_fairness import ARMS  # noqa: E402
from run_neural_synthetic import NN_GRID  # noqa: E402
from run_synthetic import SETTINGS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="mlp:S3,kan:S3")
    ap.add_argument("--reps", default="0-19")
    a = ap.parse_args()
    v = json.loads((ROOT / "reports/revised_gate_verdict_fa4f.json").read_text())
    if v.get("verdict") != "CONFIRMED":
        sys.exit("BLOCKED (D57/D61): the conventional MLP has not passed the revised gate.")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    sel = json.loads((ROOT / "reports/learnability_selected_fa4f.json").read_text())
    tuned = {k: v for k, v in sel["config"].items() if k not in ("lr", "wd")}
    lo, hi = map(int, a.reps.split("-"))
    for rep in range(lo, hi + 1):
        for cell in a.cells.split(","):
            arm, strategy = cell.split(":")
            model, extra = ARMS[arm]
            cfg = {**tuned, **extra, "strategy": strategy}
            seed = 3400 + rep
            t0 = time.time()
            obs, ora = generate(SynthConfig(n=1000, seed=seed, **SETTINGS["P0"]))
            tr, va, te = split_groups(obs.ids, seed, strata=obs.delta)
            o_te = obs.subset(te)
            base = dict(dataset="synthetic", analysis="r12_p0", setting="P0", replicate=rep, data_seed=seed,
                        split_seed=seed, n_train=len(tr), n_val=len(va), n_test=len(te),
                        censored_frac_test=float(1 - o_te.delta.mean()), data_kind="SYNTHETIC",
                        selected_config_id=sel["config_id"], arm=arm, strategy=strategy,
                        variant=json.dumps({"arm": arm, "strategy": strategy}, sort_keys=True))
            tune_fit_evaluate_ensemble(model, obs.subset(tr), obs.subset(va), o_te, base, ora.subset(te),
                                       np.linspace(0, ora.t_max, 101), grid=NN_GRID, k=int(sel["ensemble_seeds"]),
                                       extra_cfg=cfg, extra_eval=lambda m, s, o_te=o_te: reconstruction_eval(m, o_te))
            print(f"[P0 {arm} {strategy} rep {rep}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
