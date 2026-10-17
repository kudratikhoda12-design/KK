"""Neural experiments on synthetic data (only after the D41 confirmation passes; otherwise not run).

Analyses (configuration from the pre-registered selection file, fixed before these runs; revision 4: --selected
defaults to reports/learnability_selected_fa.json). D57/D61: refuses to run unless reports/revised_gate_verdict.json
records verdict CONFIRMED (revised gate + >= 4/5 fresh confirmations).
  H1        fae_kan vs fae_mlp, setting P0, 5 replicates, tuning rule (4 configs lr x wd, validation IBS), seeds 0-2
  H2        alpha in {0, 1} for fae_mlp and fae_kan (default config, no retuning), P0, 5 replicates x 3 seeds
  H3        fae_mlp vs flat_ae (and fae_kan), setting P-sparse, tuning rule as H1
  CONTROLS  strategies S1 / S2 / S3 for fae_mlp and fae_kan, P0, 5 replicates x seed 0
Results go to reports/experiment_results.jsonl via the shared runner (resumable).
"""
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
from kanfae.experiments.runner import tune_fit_evaluate  # noqa: E402
from run_synthetic import SETTINGS  # noqa: E402

NN_GRID = [{"lr": lr, "wd": wd} for lr in (1e-3, 3e-3) for wd in (1e-4, 1e-3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyses", default="H1,H2,H3,CONTROLS")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--selected", default="reports/learnability_selected_fa.json")
    args = ap.parse_args()
    verdict = ROOT / "reports" / "revised_gate_verdict.json"
    if not verdict.exists() or json.loads(verdict.read_text()).get("verdict") != "CONFIRMED":
        sys.exit("BLOCKED (D57/D61): the conventional MLP has not passed the revised gate; no neural hypothesis runs.")
    torch.set_default_dtype(torch.float64)
    sel = json.loads((ROOT / args.selected).read_text())
    base_cfg = {k: v for k, v in sel["config"].items() if k not in ("lr", "wd")}
    (ROOT / "configs/neural").mkdir(parents=True, exist_ok=True)
    (ROOT / "configs/neural/selected.json").write_text(json.dumps(sel, indent=1))
    plan = []
    for a in args.analyses.split(","):
        if a == "H1":
            plan += [("H1", "P0", m, None, base_cfg) for m in ("fae_mlp", "fae_kan")]
        elif a == "H2":
            plan += [("H2", "P0", m, [{}], {**base_cfg, "alpha": al}) for m in ("fae_mlp", "fae_kan") for al in (0.0, 1.0)]
        elif a == "H3":
            plan += [("H3", "P-sparse", m, None, base_cfg if m != "flat_ae" else {k: v for k, v in base_cfg.items()
                                                                            if k not in ("front", "n_maps", "pw_width")})
                     for m in ("fae_mlp", "fae_kan", "flat_ae")]
        elif a == "CONTROLS":
            plan += [("CONTROLS", "P0", m, [{}], {**base_cfg, "strategy": s}) for m in ("fae_mlp", "fae_kan")
                     for s in ("S1", "S2", "S3")]
    for analysis, setting, model, grid, cfg in plan:
        for rep in range(args.reps):
            seed = 1000 + rep
            t0 = time.time()
            obs, ora = generate(SynthConfig(n=1000, seed=seed, **SETTINGS[setting]))
            tr, va, te = split_groups(obs.ids, seed, strata=obs.delta)
            base = dict(dataset="synthetic", analysis=f"neural_{analysis}", setting=setting, replicate=rep,
                        data_seed=seed, split_seed=seed, n_train=len(tr), n_val=len(va), n_test=len(te),
                        censored_frac_test=float(1 - obs.subset(te).delta.mean()), data_kind="SYNTHETIC",
                        selected_config_id=sel["config_id"], variant=json.dumps({k: cfg.get(k) for k in
                                                                                 ("alpha", "strategy")}))
            tune_fit_evaluate(model, obs.subset(tr), obs.subset(va), obs.subset(te), base, ora.subset(te),
                              np.linspace(0, ora.t_max, 101), grid=grid if grid is not None else NN_GRID,
                              extra_cfg=cfg, holdout_test=True)
            print(f"[{analysis} {setting} {model} {cfg.get('alpha', '')} {cfg.get('strategy', '')} rep {rep}] "
                  f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
