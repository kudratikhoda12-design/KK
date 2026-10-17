"""D41 controlled learnability search for R2 recovery. Gate data (seed 21); selection on TRAIN/VALIDATION only.

Each fit is appended to reports/learnability_search.jsonl (resumable: completed config IDs are skipped).
Steps are greedy, one factor at a time; the selection rule is fixed in docs/decisions.md D41.
"""
import hashlib
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
from kanfae.evaluation.metrics import harrell_c  # noqa: E402
from kanfae.models.pipelines import DEFAULTS, NeuralAFT  # noqa: E402

OUT = ROOT / "reports" / "learnability_search.jsonl"
FRONT_ID = {"linear": "A0", "nonlinear": "A1", "energy": "A2", "tspline": "A3", "tsd": "A4"}


def probe(m, obs, ora, fit_idx, eval_idx):
    Zf, Ze = m.latent(obs.subset(fit_idx)), m.latent(obs.subset(eval_idx))
    A = np.column_stack([np.ones(len(Zf)), Zf]); B = np.column_stack([np.ones(len(Ze)), Ze])
    out = []
    for j in range(4):
        coef = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ ora.theta[fit_idx, j])
        y = ora.theta[eval_idx, j]
        out.append(float(1 - ((y - B @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
    return out


def cfg_id(kind, cfg):
    """ID from the EFFECTIVE configuration (defaults merged), so equivalent configurations share one ID.
    (Fix 2026-10-17: partial-dict hashing re-ran an identical model at step S2.)"""
    eff = {**DEFAULTS, **cfg}
    h = hashlib.sha1(json.dumps([kind, eff], sort_keys=True).encode()).hexdigest()[:8]
    return f"{FRONT_ID.get(cfg.get('front', 'nonlinear'), 'A?')}-{kind}-{h}"


def score(r):
    return (r["val_probe_mean"] >= 0.8, r["val_C"] if r["val_probe_mean"] >= 0.8 else r["val_C"] + 0.1 * r["val_probe_mean"])


def main():
    torch.set_default_dtype(torch.float64)
    obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=21))
    tr, va, te = split_groups(obs.ids, 21, strata=obs.delta)        # te is NOT used anywhere in this script
    done = {}
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            r = json.loads(l)
            if r.get("status") == "ok":
                done[(cfg_id(r["kind"], r["config"]), r["seed"])] = r

    def run(step, kind, cfg, seed=0):
        cid = cfg_id(kind, cfg)
        if (cid, seed) in done:
            return done[(cid, seed)]
        t0 = time.time()
        rec = dict(step=step, config_id=cid, kind=kind, config=cfg, seed=seed)
        try:
            m = NeuralAFT(kind, **cfg).fit(obs.subset(tr), obs.subset(va), seed=seed)
            rec.update(status="ok", train_C=harrell_c(obs.y_obs[tr], obs.delta[tr], m.predict_risk(obs.subset(tr))),
                       val_C=harrell_c(obs.y_obs[va], obs.delta[va], m.predict_risk(obs.subset(va))),
                       val_oracle_C=harrell_c(obs.y_obs[va], obs.delta[va], -ora.eta_true[va]),
                       val_probe=probe(m, obs, ora, tr, va), epochs=[r["epochs"] for r in m.info["runs"]],
                       n_params=m.n_params, block_params=m.info["block_params"],
                       z_std=[float(s) for s in m.latent(obs.subset(va)).std(0)])
            rec["val_probe_mean"] = float(np.mean(rec["val_probe"]))
        except Exception as exc:
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", val_C=float("nan"), val_probe_mean=float("nan"))
        rec["runtime_s"] = round(time.time() - t0, 1)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[(cid, seed)] = rec
        print(f"[{step}] {cid} {cfg} seed={seed}: val C={rec['val_C']:.4f} (oracle {rec.get('val_oracle_C', float('nan')):.4f}) "
              f"probe mean={rec['val_probe_mean']:.3f} train C={rec.get('train_C', float('nan')):.4f} "
              f"{rec['runtime_s']}s", flush=True)
        return rec

    def best(step, kind, cfgs):
        recs = [(run(step, kind, c), c) for c in cfgs]
        ok = [(r, c) for r, c in recs if r["status"] == "ok"]
        r, c = max(ok, key=lambda rc: score(rc[0]))       # max() keeps the first (simpler) on ties
        print(f"== {step} selected {r['config_id']} {c}", flush=True)
        return c

    base = {}
    c = best("S1-front", "fae_mlp", [dict(base, front=f) for f in ("linear", "nonlinear", "energy", "tspline")])
    c = best("S2-latent", "fae_mlp", [dict(c, r=r) for r in (4, 8)])
    c = best("S3-alpha", "fae_mlp", [dict(c, alpha=a) for a in (1.0, 0.1, 0.0)])
    c = best("S4-optim", "fae_mlp", [dict(c, optimizer="adam", lr=3e-3), dict(c, optimizer="adam", lr=1e-3),
                                     dict(c, optimizer="adamw", lr=3e-3)])
    c = best("S5-es", "fae_mlp", [dict(c, es_target="objective"), dict(c, es_target="nll")])
    for kind in ("fae_linear", "fae_mlp", "fae_kan"):            # information only (not used for selection)
        run("S6-encoder-info", kind, c)
    for s in (1, 2):
        run("S7-stability", "fae_mlp", c, seed=s)
    (ROOT / "reports" / "learnability_selected.json").write_text(json.dumps({"kind": "fae_mlp", "config": c,
                                                                            "config_id": cfg_id("fae_mlp", c)}, indent=1))
    print("SELECTED", json.dumps(c), flush=True)


if __name__ == "__main__":
    main()
