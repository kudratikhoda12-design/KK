"""D81 pre-registered Step-1 search (Revision 4c: training stability and stopping rule). Seed-21 TRAIN/VALIDATION
only; single-threaded (D79). Records: reports/learnability_search_fa4c.jsonl (resumable);
selection + Step-1 acceptance: reports/learnability_selected_fa4c.json.

Base = the Revision-4b selection (compact front, 17 freqs, r 12, head_wd 1e-2, Adam 3e-3, objective early stopping).
H1 spectral scaling {off, fa_balance}; H2 optimiser with gradient clipping 1.0 {Adam 1e-3, Adam 3e-3, AdamW 1e-3,
AdamW 3e-3} against the unclipped reference; H3 stopping {objective, nll, two_state_refit, two_checkpoint};
H4 seeds 1, 2; H5 information only: fae_linear, fae_kan.
Selection (owner amendment D80, prospective): among candidates with validation probe mean >= 0.78 (>= 0.80 under
two-state stopping) and, from H2 on, zero training-NLL jumps > 0.2 after epoch 300, maximise validation C; ties keep
the earlier (simpler) candidate. If none is eligible: maximise C + 0.1 * probe among the stable candidates (all
candidates if none is stable).
Step-1 acceptance (D81): on seeds 0, 1, 2 the selected configuration has zero jumps AND delivers risk predictions whose
validation NLL is within 0.01 of the minimum validation NLL of its joint run. Otherwise Step 1 FAILS and Step 2 is
not run.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.data.synthetic import SynthConfig, generate  # noqa: E402
from kanfae.evaluation.metrics import harrell_c  # noqa: E402
from kanfae.models.pipelines import DEFAULTS, NeuralAFT  # noqa: E402
from kanfae.survival.aft import aft_nll  # noqa: E402
from learnability_search import probe  # noqa: E402

OUT = ROOT / "reports" / "learnability_search_fa4c.jsonl"
SEL = ROOT / "reports" / "learnability_selected_fa4c.json"
BASE = dict(front="fa", fa_compact=True, fa_nfreq=17, r=12, alpha=1.0, optimizer="adam", lr=3e-3, wd=1e-4,
            head_wd=1e-2, es_target="objective", max_epochs=3000, patience=100, H=32, hidden=64)
TWO_STATE = ("two_state_refit", "two_checkpoint")


def cid(kind, cfg):
    eff = {**DEFAULTS, **cfg}
    return f"A6-{kind}-" + hashlib.sha1(json.dumps([kind, eff], sort_keys=True).encode()).hexdigest()[:8]


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=21))
    tr, va, te = split_groups(obs.ids, 21, strata=obs.delta)          # te is NOT used anywhere in this script
    done = {}
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            r = json.loads(l)
            done[(r["config_id"], r["seed"])] = r

    def run(step, kind, cfg, seed=0):
        k = (cid(kind, cfg), seed)
        if k in done:
            return done[k]
        t0 = time.time()
        m = NeuralAFT(kind, **cfg).fit(obs.subset(tr), obs.subset(va), seed=seed)
        joint = m.info["runs"][0]
        Tva = m._tensors(m.prepare(obs.subset(va)))
        _, eta, sigma, _ = m._forward(m.prepare(obs.subset(va)))
        delivered = float(aft_nll(Tva["y"], Tva["d"], torch.tensor(eta), torch.tensor(sigma), m.cfg["family"]))
        rec = dict(step=step, config_id=k[0], kind=kind, config=cfg, seed=seed,
                   train_C=harrell_c(obs.y_obs[tr], obs.delta[tr], m.predict_risk(obs.subset(tr))),
                   val_C=harrell_c(obs.y_obs[va], obs.delta[va], m.predict_risk(obs.subset(va))),
                   val_oracle_C=harrell_c(obs.y_obs[va], obs.delta[va], -ora.eta_true[va]),
                   val_probe=probe(m, obs, ora, tr, va), epochs=[r["epochs"] for r in m.info["runs"]],
                   jumps=joint["nll_jumps_gt02_after300"], max_train_nll_after300=joint["max_train_nll_after300"],
                   epoch_min_val_nll=joint["epoch_min_val_nll"], min_val_nll=joint["min_val_nll"],
                   stop_epoch=joint["stop_epoch"], delivered_val_nll=delivered,
                   risk_state_epoch=joint.get("risk_state_epoch"), balance=getattr(m, "balance_info", None),
                   runtime_s=round(time.time() - t0, 1))
        rec["val_probe_mean"] = float(np.mean(rec["val_probe"]))
        rec["stopping_conflict_resolved"] = bool(delivered <= rec["min_val_nll"] + 0.01)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[k] = rec
        print(f"[{step}] {k[0]} {kind} seed={seed} {({x: cfg[x] for x in cfg if cfg[x] != BASE.get(x)})}: val C "
              f"{rec['val_C']:.4f} (oracle {rec['val_oracle_C']:.4f}) probe {np.round(rec['val_probe'], 3)} mean "
              f"{rec['val_probe_mean']:.3f} | jumps {rec['jumps']} max-train-NLL {rec['max_train_nll_after300']:.2f} | "
              f"min-val-NLL epoch {rec['epoch_min_val_nll']} stop {rec['stop_epoch']} delivered-val-NLL "
              f"{delivered:.4f} vs min {rec['min_val_nll']:.4f} | {rec['runtime_s']}s", flush=True)
        return rec

    def best(step, cfgs, stability_filter=True):
        recs = [(run(step, "fae_mlp", c), c) for c in cfgs]
        thr = lambda c: 0.80 if c.get("es_mode") in TWO_STATE else 0.78  # noqa: E731
        stable = [(r, c) for r, c in recs if r["jumps"] == 0] if stability_filter else recs
        elig = [(r, c) for r, c in stable if r["val_probe_mean"] >= thr(c)]
        if elig:
            r, c = max(elig, key=lambda rc: rc[0]["val_C"])               # max() keeps the first on ties
            rule = "eligible: max val C"
        else:
            pool = stable or recs
            r, c = max(pool, key=lambda rc: rc[0]["val_C"] + 0.1 * rc[0]["val_probe_mean"])
            rule = "fallback: max C + 0.1 probe (" + ("stable" if stable else "all") + ")"
        print(f"== {step} selected {r['config_id']} ({rule})", flush=True)
        return c

    c = best("H1-scaling", [dict(BASE), dict(BASE, fa_balance=True)], stability_filter=False)
    c = best("H2-optimiser", [dict(c), dict(c, clip=1.0, lr=1e-3), dict(c, clip=1.0, lr=3e-3),
                              dict(c, clip=1.0, optimizer="adamw", lr=1e-3), dict(c, clip=1.0, optimizer="adamw", lr=3e-3)])
    c = best("H3-stopping", [dict(c), dict(c, es_target="nll"), dict(c, es_mode="two_state_refit"),
                             dict(c, es_mode="two_checkpoint")])
    recs = [run("H0-H3", "fae_mlp", c, 0)] + [run("H4-stability", "fae_mlp", c, s) for s in (1, 2)]
    for kind in ("fae_linear", "fae_kan"):
        run("H5-encoder-info", kind, c)
    acc = dict(jumps_zero_all_seeds=all(r["jumps"] == 0 for r in recs),
               stopping_conflict_resolved_all_seeds=all(r["stopping_conflict_resolved"] for r in recs),
               per_seed=[dict(seed=r["seed"], jumps=r["jumps"], delivered_val_nll=r["delivered_val_nll"],
                              min_val_nll=r["min_val_nll"], val_C=r["val_C"], val_probe_mean=r["val_probe_mean"])
                         for r in recs])
    acc["STEP1"] = "PASSED" if acc["jumps_zero_all_seeds"] and acc["stopping_conflict_resolved_all_seeds"] else "FAILED"
    SEL.write_text(json.dumps({"kind": "fae_mlp", "config": c, "config_id": cid("fae_mlp", c), "step1": acc}, indent=1))
    print("SELECTED", json.dumps(c), "| STEP 1", acc["STEP1"], flush=True)


if __name__ == "__main__":
    main()
