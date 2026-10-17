"""D46 search extension (S8 budget -> S9 front -> S10 capacity -> S11 stability). Validation only; same rule as D41."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import learnability_search as L  # noqa: E402


def main():
    import numpy as np  # noqa: F401
    import torch
    torch.set_default_dtype(torch.float64)
    start = json.loads((ROOT / "reports/learnability_selected.json").read_text())["config"]
    # reuse the search machinery by monkeypatching its step list
    obs, ora = L.generate(L.SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=21))
    tr, va, te = L.split_groups(obs.ids, 21, strata=obs.delta)       # te not used
    done = {}
    for l in L.OUT.read_text().splitlines():
        r = json.loads(l)
        if r.get("status") == "ok":
            done[(L.cfg_id(r["kind"], r["config"]), r["seed"])] = r

    def run(step, kind, cfg, seed=0):
        key = (L.cfg_id(kind, cfg), seed)
        if key in done:
            return done[key]
        import time
        t0 = time.time()
        rec = dict(step=step, config_id=key[0], kind=kind, config=cfg, seed=seed)
        try:
            m = L.NeuralAFT(kind, **cfg).fit(obs.subset(tr), obs.subset(va), seed=seed)
            rec.update(status="ok", train_C=L.harrell_c(obs.y_obs[tr], obs.delta[tr], m.predict_risk(obs.subset(tr))),
                       val_C=L.harrell_c(obs.y_obs[va], obs.delta[va], m.predict_risk(obs.subset(va))),
                       val_oracle_C=L.harrell_c(obs.y_obs[va], obs.delta[va], -ora.eta_true[va]),
                       val_probe=L.probe(m, obs, ora, tr, va), epochs=[r["epochs"] for r in m.info["runs"]],
                       n_params=m.n_params, block_params=m.info["block_params"])
            rec["val_probe_mean"] = float(np.mean(rec["val_probe"]))
        except Exception as exc:
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", val_C=float("nan"), val_probe_mean=float("nan"))
        rec["runtime_s"] = round(time.time() - t0, 1)
        with open(L.OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[key] = rec
        print(f"[{step}] {key[0]} {cfg} seed={seed}: val C={rec['val_C']:.4f} probe={np.round(rec.get('val_probe', []), 2)} "
              f"mean={rec['val_probe_mean']:.3f} train C={rec.get('train_C', float('nan')):.4f} epochs={rec.get('epochs')} "
              f"{rec['runtime_s']}s", flush=True)
        return rec

    def best(step, cfgs):
        recs = [(run(step, "fae_mlp", c), c) for c in cfgs]
        ok = [(r, c) for r, c in recs if r["status"] == "ok"]
        r, c = max(ok, key=lambda rc: L.score(rc[0]))
        print(f"== {step} selected {r['config_id']} {c}", flush=True)
        return c

    c = dict(start)
    c = best("S8-budget", [dict(c, max_epochs=3000, patience=100), dict(c, max_epochs=10000, patience=300)])
    c = best("S9-front", [dict(c, front="tspline"), dict(c, front="tsd")])
    c = best("S10-capacity", [dict(c, H=32, hidden=64), dict(c, H=64, hidden=128)])
    for s in (1, 2):
        run("S11-stability", "fae_mlp", c, seed=s)
    (ROOT / "reports/learnability_selected_ext.json").write_text(json.dumps(
        {"kind": "fae_mlp", "config": c, "config_id": L.cfg_id("fae_mlp", c)}, indent=1))
    print("SELECTED", json.dumps(c), flush=True)


if __name__ == "__main__":
    main()
