"""Neural experiments on synthetic data (only after a pre-registered gate verdict is CONFIRMED; otherwise not run).

D57/D61 guard: refuses to run unless the verdict file that belongs to --selected records CONFIRMED.
Step 3 of Revision 4f (D104): configuration C4F = the 4f selection (nonlinear_joint head, ensemble of training seeds
0-2), applied symmetrically to fae_mlp, fae_kan and flat_ae. Every cell's primary record is the 3-seed ensemble
(runner.tune_fit_evaluate_ensemble); per-seed records are secondary. Replicates 1000-1004, n = 1000.
  H1         fae_kan vs fae_mlp, P0, tuned (lr x wd, validation IBS at seed 0)
  H2         alpha in {0, 1} x {fae_mlp, fae_kan}, P0, untuned
  H3         fae_mlp, fae_kan, flat_ae, P-sparse, tuned
  CONTROLS   strategy S1 / S3 x {fae_mlp, fae_kan}, P0 (S2 and survival-only are H2's cells)
  SECONDARY  P-c0, P-c60, P-V2, P-lognormal x {fae_mlp, fae_kan}, untuned
  AB1..AB8   protocol ablations, P0, untuned (cells shared with H2/CONTROLS are not refitted)
Results go to $KANFAE_RESULTS (one file per analysis group under reports/step3/; resumable).
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
from kanfae.experiments.runner import tune_fit_evaluate_ensemble  # noqa: E402
from run_synthetic import SETTINGS  # noqa: E402

NN_GRID = [{"lr": lr, "wd": wd} for lr in (1e-3, 3e-3) for wd in (1e-4, 1e-3)]
FRONT_KEYS = ("front", "n_maps", "pw_width", "fa_nfreq", "fa_fourier", "fa_windows", "fa_compact", "fa_balance")


def plan_for(a, C):
    """(analysis, setting, model, grid, cfg) cells of analysis a; C = the full selected configuration."""
    tuned = {k: v for k, v in C.items() if k not in ("lr", "wd")}
    flat = lambda c: {k: v for k, v in c.items() if k not in FRONT_KEYS}  # noqa: E731
    MK = ("fae_mlp", "fae_kan")
    if a == "H1":
        return [("H1", "P0", m, NN_GRID, tuned) for m in MK]
    if a == "H2":
        return [("H2", "P0", m, None, {**C, "alpha": al}) for m in MK for al in (0.0, 1.0)]
    if a == "H3":
        return [("H3", "P-sparse", m, NN_GRID, flat(tuned) if m == "flat_ae" else tuned) for m in MK + ("flat_ae",)]
    if a == "CONTROLS":
        return [("CONTROLS", "P0", m, None, {**C, "strategy": s}) for m in MK for s in ("S1", "S3")]
    if a == "SECONDARY":
        return [("SECONDARY", s, m, None, C) for s in ("P-c0", "P-c60", "P-V2", "P-lognormal") for m in MK]
    ab = {"AB1": [(m, {"alpha": al}) for m in MK for al in (0.1, 10.0)],
          "AB3": [("fae_linear", {})],
          "AB4": [(m, {"r": r}) for m in MK for r in (2, 4, 8)],
          "AB5": [("fae_kan", {"kan_grid": g}) for g in (3, 10)],
          "AB6": [("fae_kan", {"head": h}) for h in ("linear", "kan")],
          "AB7": [(m, {"family": fam}) for m in MK for fam in ("lognormal", "loglogistic")],
          "AB8": [("fae_mlp", {"n_basis": K}) for K in (8, 16)]}
    if a in ab:
        return [(a, "P0", m, None, {**C, **extra}) for m, extra in ab[a]]
    if a == "AB2":
        return []          # = CONTROLS (S1, S3) + H2 (S2 = alpha 1)
    if a == "NONE":
        return []
    raise ValueError(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyses", default="H1,H2,H3,CONTROLS")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--selected", default="reports/learnability_selected_fa4f.json")      # D101
    args = ap.parse_args()
    # the verdict that belongs to the selected configuration (D61: fa; D71: fa4b; ...; D101: fa4f)
    verdict = ROOT / "reports" / {"reports/learnability_selected_fa.json": "revised_gate_verdict.json",
                                  "reports/learnability_selected_fa4b.json": "revised_gate_verdict_fa4b.json",
                                  "reports/learnability_selected_fa4c.json": "revised_gate_verdict_fa4c.json",
                                  "reports/learnability_selected_fa4d.json": "revised_gate_verdict_fa4d.json",
                                  "reports/learnability_selected_fa4e.json": "revised_gate_verdict_fa4e.json",
                                  "reports/learnability_selected_fa4f.json": "revised_gate_verdict_fa4f.json"}.get(
        args.selected, "no_verdict_for_this_selection.json")
    if not verdict.exists() or json.loads(verdict.read_text()).get("verdict") != "CONFIRMED":
        sys.exit("BLOCKED (D57/D61): the conventional MLP has not passed the revised gate; no neural hypothesis runs.")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    sel = json.loads((ROOT / args.selected).read_text())
    k = int(sel.get("ensemble_seeds", 1))
    (ROOT / "configs/neural").mkdir(parents=True, exist_ok=True)
    (ROOT / "configs/neural/selected.json").write_text(json.dumps(sel, indent=1))
    plan = [c for a in args.analyses.split(",") for c in plan_for(a, sel["config"])]
    for analysis, setting, model, grid, cfg in plan:
        for rep in range(args.reps):
            seed = 1000 + rep
            t0 = time.time()
            obs, ora = generate(SynthConfig(n=1000, seed=seed, **SETTINGS[setting]))
            tr, va, te = split_groups(obs.ids, seed, strata=obs.delta)
            diff = {kk: cfg[kk] for kk in ("alpha", "strategy", "r", "kan_grid", "head", "family", "n_basis")
                    if kk in cfg and cfg[kk] != sel["config"].get(kk)}
            base = dict(dataset="synthetic", analysis=f"neural4f_{analysis}", setting=setting, replicate=rep,
                        data_seed=seed, split_seed=seed, n_train=len(tr), n_val=len(va), n_test=len(te),
                        censored_frac_test=float(1 - obs.subset(te).delta.mean()), data_kind="SYNTHETIC",
                        selected_config_id=sel["config_id"], variant=json.dumps(diff, sort_keys=True))
            tune_fit_evaluate_ensemble(model, obs.subset(tr), obs.subset(va), obs.subset(te), base, ora.subset(te),
                                       np.linspace(0, ora.t_max, 101), grid=grid, k=k, extra_cfg=cfg)
            print(f"[{analysis} {setting} {model} {json.dumps(diff)} rep {rep}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
