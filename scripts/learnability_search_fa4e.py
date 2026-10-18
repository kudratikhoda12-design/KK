"""D92 pre-registered Step-1 search (Revision 4e: representation capacity and reconstruction patience/weight;
two-tier multi-seed selection). Seed-21 TRAIN/VALIDATION only; single-threaded. Records:
reports/learnability_search_fa4e.jsonl (resumable); selection + Step-1 outcome: reports/learnability_selected_fa4e.json.

Base = 4d cand1 (two_checkpoint, min_epochs 300, lr 1e-3). Candidates (full factorial, tie order as listed):
r {12, 16} x patience {100, 300} x alpha {1.0, 2.0}, each on training seeds 0, 1, 2; cand0 is 4d cand1.
Tier 1: on all three seeds zero training-NLL jumps > 0.2 after epoch 300; every seed's validation probe mean >= 0.78
and omega probe >= 0.80; mean over seeds of the probe mean >= 0.80 -> highest mean validation C (then highest minimum
seed probe, then earlier). Tier 2 (only if Tier 1 is empty): the same with every seed's probe mean >= 0.775 -> highest
minimum seed probe (then highest mean probe, then highest mean validation C). Neither -> Step 1 FAILS.
"""
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
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

OUT = ROOT / "reports" / "learnability_search_fa4e.jsonl"
SEL = ROOT / "reports" / "learnability_selected_fa4e.json"
BASE = dict(front="fa", fa_compact=True, fa_nfreq=17, optimizer="adam", wd=1e-4, head_wd=1e-2, fa_balance=True, clip=1.0,
            es_target="objective", max_epochs=3000, H=32, hidden=64, es_mode="two_checkpoint", min_epochs=300, lr=1e-3)
CANDIDATES = [dict(BASE, r=r, patience=p, alpha=al)
              for r, p, al in itertools.product((12, 16), (100, 300), (1.0, 2.0))]
SEEDS = (0, 1, 2)


def cid(kind, cfg):
    eff = {**DEFAULTS, **cfg}
    return f"A7-{kind}-" + hashlib.sha1(json.dumps([kind, eff], sort_keys=True).encode()).hexdigest()[:8]


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

    def run(step, kind, cfg, seed):
        k = (cid(kind, cfg), seed)
        if k in done:
            return done[k]
        t0 = time.time()
        m = NeuralAFT(kind, **cfg).fit(obs.subset(tr), obs.subset(va), seed=seed)
        joint = m.info["runs"][0]
        P = m.prepare(obs.subset(va)); Tva = m._tensors(P)
        _, eta, sigma, _ = m._forward(P)
        delivered = float(aft_nll(Tva["y"], Tva["d"], torch.tensor(eta), torch.tensor(sigma), m.cfg["family"]))
        pr = probe(m, obs, ora, tr, va)
        rec = dict(step=step, config_id=k[0], kind=kind, config=cfg, seed=seed,
                   train_C=harrell_c(obs.y_obs[tr], obs.delta[tr], m.predict_risk(obs.subset(tr))),
                   val_C=harrell_c(obs.y_obs[va], obs.delta[va], m.predict_risk(obs.subset(va))),
                   val_oracle_C=harrell_c(obs.y_obs[va], obs.delta[va], -ora.eta_true[va]), val_probe=pr,
                   val_probe_mean=float(np.mean(pr)), jumps=joint["nll_jumps_gt02_after300"],
                   max_train_nll_after300=joint["max_train_nll_after300"], joint_epochs=joint["epochs"],
                   objective_stop_epoch=joint["stop_epoch"], risk_state_epoch=joint.get("risk_state_epoch"),
                   refit_stop_epoch=m.info["runs"][-1]["stop_epoch"] if m.info["runs"][-1].get("refit") else None,
                   delivered_val_nll=delivered, min_val_nll_joint=joint["min_val_nll"],
                   runtime_s=round(time.time() - t0, 1))
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[k] = rec
        print(f"[{step}] {k[0]} seed={seed} r {cfg['r']} patience {cfg['patience']} alpha {cfg['alpha']}: val C "
              f"{rec['val_C']:.4f} probe {np.round(pr, 3)} mean {rec['val_probe_mean']:.3f} jumps {rec['jumps']} | "
              f"objective stop {rec['objective_stop_epoch']} risk/refit epoch "
              f"{rec['risk_state_epoch'] if rec['risk_state_epoch'] is not None else rec['refit_stop_epoch']} | "
              f"{rec['runtime_s']}s", flush=True)
        return rec

    summary = []
    for i, c in enumerate(CANDIDATES):
        recs = [run(f"cand{i}", "fae_mlp", c, s) for s in SEEDS]
        row = dict(index=i, config=c, config_id=cid("fae_mlp", c),
                   mean_val_C=float(np.mean([r["val_C"] for r in recs])),
                   mean_probe=float(np.mean([r["val_probe_mean"] for r in recs])),
                   min_seed_probe=float(min(r["val_probe_mean"] for r in recs)),
                   min_seed_omega=float(min(r["val_probe"][3] for r in recs)),
                   max_jumps=int(max(r["jumps"] for r in recs)))
        common = row["max_jumps"] == 0 and row["mean_probe"] >= 0.80 and row["min_seed_omega"] >= 0.80
        row["tier1"] = bool(common and row["min_seed_probe"] >= 0.78)
        row["tier2"] = bool(common and row["min_seed_probe"] >= 0.775)
        summary.append(row)
        print(f"== cand{i} r {c['r']} patience {c['patience']} alpha {c['alpha']}: mean val C {row['mean_val_C']:.4f} "
              f"mean probe {row['mean_probe']:.3f} min seed probe {row['min_seed_probe']:.3f} min seed omega "
              f"{row['min_seed_omega']:.3f} max jumps {row['max_jumps']} -> tier1 {row['tier1']} tier2 {row['tier2']}",
              flush=True)
    t1 = [r for r in summary if r["tier1"]]
    t2 = [r for r in summary if r["tier2"]]
    out = {"candidates": summary, "STEP1": "PASSED" if (t1 or t2) else "FAILED"}
    best = None
    if t1:      # max() keeps the earliest on ties
        best, tier = max(t1, key=lambda r: (r["mean_val_C"], r["min_seed_probe"])), 1
    elif t2:
        best, tier = max(t2, key=lambda r: (r["min_seed_probe"], r["mean_probe"], r["mean_val_C"])), 2
    if best:
        out.update(kind="fae_mlp", config=best["config"], config_id=best["config_id"], selected_index=best["index"],
                   tier=tier)
        for kind in ("fae_linear", "fae_kan"):                     # information only
            run("info", kind, best["config"], 0)
    SEL.write_text(json.dumps(out, indent=1))
    print("STEP 1", out["STEP1"], "| selected", out.get("config_id"), "tier", out.get("tier"), flush=True)


if __name__ == "__main__":
    main()
