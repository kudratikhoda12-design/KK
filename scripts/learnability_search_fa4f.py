"""D100 pre-registered Step-1 search (Revision 4f: risk mapping x multi-seed risk averaging). Seed-21 TRAIN/VALIDATION
only; single-threaded. Records: reports/learnability_search_fa4f.jsonl (resumable; each fit stores its validation risk
scores so ensembles are recomputed exactly); selection + Step-1 outcome: reports/learnability_selected_fa4f.json.

Representation fixed at the 4e selection (r 16, patience 100, alpha 1, two_checkpoint, burn-in 300, lr 1e-3).
Factor 1, head_mode: linear (= 4e cand4) | nonlinear_joint (head "mlp", trained jointly) | nonlinear_refit (4e joint
run with the linear head, then a nonlinear head "mlp" refitted on the frozen latent). Factor 2, ensemble_seeds 1 | 3.
Each head mode is fitted on training seeds 0, 1, 2; ensemble_seeds 1 scores the mean of the three single-seed
validation C; ensemble_seeds 3 scores the validation C of the mean risk over seeds 0-2.
Eligible iff on seeds 0-2: zero training-NLL jumps > 0.2 after epoch 300; mean probe mean >= 0.80; every seed's probe
mean >= 0.78; every seed's omega >= 0.80. Select the highest validation C; ties: simpler head (order as listed), then
ensemble_seeds 3. No eligible candidate -> Step 1 FAILS.
"""
import hashlib
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

OUT = ROOT / "reports" / "learnability_search_fa4f.jsonl"
SEL = ROOT / "reports" / "learnability_selected_fa4f.json"
BASE = json.loads((ROOT / "reports/learnability_selected_fa4e.json").read_text())["config"]
HEAD_MODES = {"linear": {}, "nonlinear_joint": {"head": "mlp"}, "nonlinear_refit": {"head_refit": "mlp"}}
SEEDS = (0, 1, 2)


def cid(kind, cfg):
    eff = {**DEFAULTS, **cfg}
    return f"A7-{kind}-" + hashlib.sha1(json.dumps([kind, eff], sort_keys=True).encode()).hexdigest()[:8]


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=21))
    tr, va, te = split_groups(obs.ids, 21, strata=obs.delta)          # te is NOT used anywhere in this script
    yv, dv = obs.y_obs[va], obs.delta[va]
    done = {}
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            r = json.loads(l)
            done[(r["config_id"], r["seed"])] = r

    def run(step, kind, head_mode, cfg, seed):
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
        risk = m.predict_risk(obs.subset(va))
        rec = dict(step=step, config_id=k[0], kind=kind, head_mode=head_mode, config=cfg, seed=seed,
                   train_C=harrell_c(obs.y_obs[tr], obs.delta[tr], m.predict_risk(obs.subset(tr))),
                   val_C=harrell_c(yv, dv, risk), val_oracle_C=harrell_c(yv, dv, -ora.eta_true[va]), val_probe=pr,
                   val_probe_mean=float(np.mean(pr)), jumps=joint["nll_jumps_gt02_after300"],
                   joint_epochs=joint["epochs"], objective_stop_epoch=joint["stop_epoch"],
                   risk_state_epoch=joint.get("risk_state_epoch"),
                   refit_stop_epoch=m.info["runs"][-1]["stop_epoch"] if m.info["runs"][-1].get("refit") else None,
                   delivered_val_nll=delivered, n_params=m.n_params, val_risk=[float(x) for x in risk],
                   runtime_s=round(time.time() - t0, 1))
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[k] = rec
        print(f"[{step}] {k[0]} {kind} {head_mode} seed={seed}: val C {rec['val_C']:.4f} probe {np.round(pr, 3)} "
              f"mean {rec['val_probe_mean']:.3f} jumps {rec['jumps']} | objective stop {rec['objective_stop_epoch']} "
              f"risk state {rec['risk_state_epoch']} refit stop {rec['refit_stop_epoch']} | {rec['runtime_s']}s", flush=True)
        return rec

    def ens_C(recs):
        return harrell_c(yv, dv, np.mean([r["val_risk"] for r in recs], axis=0))

    summary = []
    for hi, (hm, extra) in enumerate(HEAD_MODES.items()):
        c = dict(BASE, **extra)
        recs = [run(f"head{hi}", "fae_mlp", hm, c, s) for s in SEEDS]
        elig = bool(max(r["jumps"] for r in recs) == 0 and np.mean([r["val_probe_mean"] for r in recs]) >= 0.80
                    and min(r["val_probe_mean"] for r in recs) >= 0.78 and min(r["val_probe"][3] for r in recs) >= 0.80)
        for ens in (1, 3):
            row = dict(head_mode=hm, head_index=hi, ensemble_seeds=ens, config=c, config_id=cid("fae_mlp", c),
                       val_C=float(np.mean([r["val_C"] for r in recs])) if ens == 1 else float(ens_C(recs)),
                       seed_val_C=[r["val_C"] for r in recs],
                       mean_probe=float(np.mean([r["val_probe_mean"] for r in recs])),
                       min_seed_probe=float(min(r["val_probe_mean"] for r in recs)),
                       min_seed_omega=float(min(r["val_probe"][3] for r in recs)),
                       max_jumps=int(max(r["jumps"] for r in recs)), eligible=elig)
            summary.append(row)
            print(f"== {hm} ensemble {ens}: val C {row['val_C']:.4f} (single seeds {np.round(row['seed_val_C'], 4)}) "
                  f"mean probe {row['mean_probe']:.3f} min seed probe {row['min_seed_probe']:.3f} min omega "
                  f"{row['min_seed_omega']:.3f} jumps {row['max_jumps']} -> eligible {elig}", flush=True)
    elig = [r for r in summary if r["eligible"]]
    out = {"candidates": summary, "STEP1": "PASSED" if elig else "FAILED"}
    if elig:     # highest C; ties: simpler head (lower index), then ensemble_seeds 3
        best = max(elig, key=lambda r: (r["val_C"], -r["head_index"], r["ensemble_seeds"]))
        out.update(kind="fae_mlp", config=best["config"], config_id=best["config_id"], head_mode=best["head_mode"],
                   ensemble_seeds=best["ensemble_seeds"])
        info = {}
        for kind in ("fae_linear", "fae_kan"):                     # information only, under the selected rule
            rs = [run("info", kind, best["head_mode"], best["config"], s) for s in range(best["ensemble_seeds"])]
            info[kind] = dict(val_C=float(ens_C(rs)) if len(rs) > 1 else rs[0]["val_C"],
                              probe_seed0=rs[0]["val_probe"])
        out["info"] = info
    SEL.write_text(json.dumps(out, indent=1))
    print("STEP 1", out["STEP1"], "| selected", out.get("head_mode"), "ensemble", out.get("ensemble_seeds"), flush=True)


if __name__ == "__main__":
    main()
