"""Stage E (unblocked part): synthetic experiments for the validated non-neural baselines.

Neural-autoencoder models are BLOCKED by the failed learnability gate R2 (decision D33) and are not run here.
Usage: .venv/bin/python scripts/run_synthetic.py [--settings P0,P-c0,...] [--reps 5]
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
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.data.synthetic import SynthConfig, generate  # noqa: E402
from kanfae.experiments.runner import KMNull, append_record, evaluate, run_id, tune_fit_evaluate, done_ids  # noqa: E402

SETTINGS = {
    "P0": dict(risk="V1", sampling="dense", censoring=0.3, family="weibull"),
    "P-c0": dict(risk="V1", sampling="dense", censoring=0.0, family="weibull"),
    "P-c60": dict(risk="V1", sampling="dense", censoring=0.6, family="weibull"),
    "P-sparse": dict(risk="V1", sampling="sparse", censoring=0.3, family="weibull", p_miss=0.1),
    "P-V2": dict(risk="V2", sampling="dense", censoring=0.3, family="weibull"),
    "P-lognormal": dict(risk="V1", sampling="dense", censoring=0.3, family="lognormal"),
}
MODELS = ["fpca_aft", "fpca_cox", "sum_aft", "sum_cox", "sum_rsf"]


class OracleModel:
    """Reference row: the true survival curves and true risk (not a comparator)."""
    name = "oracle_true"

    def __init__(self, ora, idx_map):
        self.ora, self.idx_map = ora, idx_map

    def prepare(self, obs, holdout=False):
        return None

    def predict_survival(self, obs, times, P=None):
        return self.ora.subset([self.idx_map[i] for i in obs.ids]).survival(times)

    def predict_risk(self, obs, P=None):
        return -self.ora.eta_true[[self.idx_map[i] for i in obs.ids]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    cfg_dir = ROOT / "configs" / "synthetic"; cfg_dir.mkdir(parents=True, exist_ok=True)
    for sname in args.settings.split(","):
        (cfg_dir / f"{sname}.json").write_text(json.dumps({"setting": sname, **SETTINGS[sname], "n": args.n,
                                                           "replicate_seeds": [1000 + r for r in range(args.reps)],
                                                           "models": MODELS + ["km_null", "oracle_true"]}, indent=2))
        for rep in range(args.reps):
            seed = 1000 + rep
            t0 = time.time()
            cfg = SynthConfig(n=args.n, seed=seed, **SETTINGS[sname])
            obs, ora = generate(cfg)
            tr, va, te = split_groups(obs.ids, seed, strata=obs.delta)
            o_tr, o_va, o_te, ora_te = obs.subset(tr), obs.subset(va), obs.subset(te), ora.subset(te)
            grid = np.linspace(0, ora.t_max, 101)
            base = dict(dataset="synthetic", analysis="synthetic_baselines", setting=sname, replicate=rep,
                        data_seed=seed, split_seed=seed, n_train=len(tr), n_val=len(va), n_test=len(te),
                        censored_frac_test=float(1 - o_te.delta.mean()), data_kind="SYNTHETIC")
            for name in MODELS + ["km_null"]:
                tune_fit_evaluate(name, o_tr, o_va, o_te, base, ora_te, grid,
                                  grid=[{}] if name == "km_null" else None, holdout_test=True)
            rid = run_id(**base, model="oracle_true", phase="final")
            if rid not in done_ids():
                om = OracleModel(ora, {i: k for k, i in enumerate(obs.ids)})
                rec = dict(base, run_id=rid, model="oracle_true", config="{}", seed=0, phase="final", status="ok",
                           n_params=0, fit_seconds=0.0, **evaluate(om, o_tr, o_te, None, ora_te, grid))
                append_record(rec)
            print(f"[{sname} rep {rep}] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
