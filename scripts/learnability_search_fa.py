"""D61 pre-registered search for the revision-4 frequency-aware front ("fa"). Seed-21 TRAIN/VALIDATION only.

Steps (greedy, one factor at a time, D41 selection rule): F1 front {tspline (reference only), fa}; F2 options
{fa, fa+fourier, fa+windows}; F3 n_freq {17, 33}; F4 r {8, 12}; F5 alpha {1, 0.1}; F6 seeds 1, 2; F7 information
only: fae_linear and fae_kan. The selection is restricted to "fa" variants. Records: reports/learnability_search_fa.jsonl
(resumable); selection: reports/learnability_selected_fa.json.
"""
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
import learnability_search as L  # noqa: E402

L.FRONT_ID["fa"] = "A5"
OUT = ROOT / "reports" / "learnability_search_fa.jsonl"
BASE = dict(front="tspline", r=8, alpha=1.0, optimizer="adam", lr=3e-3, es_target="objective", max_epochs=3000,
            patience=100, H=32, hidden=64)


def main(steps=None, selected=None):
    """steps(run, best) -> selected config; default: the D61 steps below (revision 4). Revision 4b reuses this."""
    torch.set_default_dtype(torch.float64)
    obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=21))
    tr, va, te = split_groups(obs.ids, 21, strata=obs.delta)          # te is NOT used anywhere in this script
    done = {}
    OUT = globals()["OUT"]
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            r = json.loads(l)
            if r.get("status") == "ok":
                done[(L.cfg_id(r["kind"], r["config"]), r["seed"])] = r

    def run(step, kind, cfg, seed=0):
        cid = L.cfg_id(kind, cfg)
        if (cid, seed) in done:                                    # OUT is read at call time (4b reuses main)
            return done[(cid, seed)]
        t0 = time.time()
        rec = dict(step=step, config_id=cid, kind=kind, config=cfg, seed=seed)
        try:
            m = NeuralAFT(kind, **cfg).fit(obs.subset(tr), obs.subset(va), seed=seed)
            rec.update(status="ok", train_C=harrell_c(obs.y_obs[tr], obs.delta[tr], m.predict_risk(obs.subset(tr))),
                       val_C=harrell_c(obs.y_obs[va], obs.delta[va], m.predict_risk(obs.subset(va))),
                       val_oracle_C=harrell_c(obs.y_obs[va], obs.delta[va], -ora.eta_true[va]),
                       val_probe=L.probe(m, obs, ora, tr, va), epochs=[r["epochs"] for r in m.info["runs"]],
                       n_params=m.n_params, block_params=m.info["block_params"])
            rec["val_probe_mean"] = float(np.mean(rec["val_probe"]))
        except Exception as exc:
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", val_C=float("nan"),
                       val_probe_mean=float("nan"))
        rec["runtime_s"] = round(time.time() - t0, 1)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[(cid, seed)] = rec
        print(f"[{step}] {cid} {kind} {cfg} seed={seed}: val C={rec['val_C']:.4f} (oracle "
              f"{rec.get('val_oracle_C', float('nan')):.4f}) probe={np.round(rec.get('val_probe', []), 3)} "
              f"mean={rec['val_probe_mean']:.3f} epochs={rec.get('epochs')} {rec['runtime_s']}s", flush=True)
        return rec

    def best(step, cfgs, restrict_fa=True):
        recs = [(run(step, "fae_mlp", c), c) for c in cfgs]
        ok = [(r, c) for r, c in recs if r["status"] == "ok" and (not restrict_fa or c["front"] == "fa")]
        if not ok:
            raise RuntimeError(f"{step}: no successful fit")
        r, c = max(ok, key=lambda rc: L.score(rc[0]))
        print(f"== {step} selected {r['config_id']} {c}", flush=True)
        return c

    if steps is not None:
        c = steps(run, best)
        (selected).write_text(json.dumps({"kind": "fae_mlp", "config": c, "config_id": L.cfg_id("fae_mlp", c)}, indent=1))
        print("SELECTED", json.dumps(c), flush=True)
        return
    c = best("F1-front", [dict(BASE), dict(BASE, front="fa")])
    c = best("F2-options", [dict(c), dict(c, fa_fourier=True), dict(c, fa_windows=True)])
    c = best("F3-nfreq", [dict(c, fa_nfreq=17), dict(c, fa_nfreq=33)])
    c = best("F4-latent", [dict(c, r=8), dict(c, r=12)])
    c = best("F5-alpha", [dict(c, alpha=1.0), dict(c, alpha=0.1)])
    for s in (1, 2):
        run("F6-stability", "fae_mlp", c, seed=s)
    for kind in ("fae_linear", "fae_kan"):                              # information only (not used for selection)
        run("F7-encoder-info", kind, c)
    (ROOT / "reports" / "learnability_selected_fa.json").write_text(json.dumps(
        {"kind": "fae_mlp", "config": c, "config_id": L.cfg_id("fae_mlp", c)}, indent=1))
    print("SELECTED", json.dumps(c), flush=True)


if __name__ == "__main__":
    main()
