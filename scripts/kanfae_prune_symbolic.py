"""D104 (5): KAN-FAE pruning and symbolic extraction under C4F (nonlinear_joint head, 3-seed ensemble).

For each P0 replicate 1000-1004: fit fae_kan members (training seeds 0-2) with C4F (variant "plain") or C4F with the
KAN-AFT sparsity regularisation kan_lam = 1e-3 (variant "reg"). Test metrics of the ensemble (and reconstruction of
the objective-state members) before pruning, after uncentred pruning (theta 1e-2; reference) and after centred pruning
(theta 1e-2; D54 score, mean absorbed into the bias). On replicate 1000: symbolic stages 1-2 on every active edge of
the risk-state encoder of the uncentred-pruned members, with metrics after substitution; variant "plain" also runs
PySR (stage 3, 60 s) on the 10 highest-L1 edges of member 0 rejected by stages 1-2.
Records: reports/step3/kanfae_prune_symbolic.jsonl (metrics), reports/step3/kanfae_symbolic_edges.jsonl (edges).
"""
import argparse
import copy
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
from kanfae.experiments.runner import evaluate  # noqa: E402
from kanfae.models import fae_kan_tools as KT  # noqa: E402
from kanfae.models.ensemble import SeedEnsembleAFT  # noqa: E402
from kanfae.models.pipelines import NeuralAFT  # noqa: E402
from run_synthetic import SETTINGS  # noqa: E402

OUT = ROOT / "reports/step3/kanfae_prune_symbolic.jsonl"
EDGES = ROOT / "reports/step3/kanfae_symbolic_edges.jsonl"


def write(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")


def metrics(members, obs_tr, obs_te, ora_te, grid):
    ens = SeedEnsembleAFT.from_members(members)
    ev = evaluate(ens, obs_tr, obs_te, ens.prepare(obs_te, holdout=True), ora_te, grid)
    recs = []
    for m in members:
        T = m._tensors(m.prepare(obs_te))
        m.model.eval()
        with torch.no_grad():
            _, (_, rec) = m._objective(T, "rec")
        recs.append(float(rec))
    ev["reconstruction_test"] = float(np.mean(recs))
    ev["member_harrell_c"] = [float(evaluate(m, obs_tr, obs_te, m.prepare(obs_te, holdout=True))["harrell_c"])
                              for m in members]
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("plain", "reg"), default="plain")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--symbolic-rep", type=int, default=1000)
    a = ap.parse_args()
    verdict = json.loads((ROOT / "reports/revised_gate_verdict_fa4f.json").read_text())
    if verdict.get("verdict") != "CONFIRMED":
        sys.exit("BLOCKED (D57/D61): the conventional MLP has not passed the revised gate.")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    sel = json.loads((ROOT / "reports/learnability_selected_fa4f.json").read_text())
    C = dict(sel["config"], **({"kan_lam": 1e-3} if a.variant == "reg" else {}))
    k = int(sel["ensemble_seeds"])
    for rep in range(a.reps):
        seed = 1000 + rep
        obs, ora = generate(SynthConfig(n=1000, seed=seed, **SETTINGS["P0"]))
        tr, va, te = split_groups(obs.ids, seed, strata=obs.delta)
        otr, ova, ote, orte = obs.subset(tr), obs.subset(va), obs.subset(te), ora.subset(te)
        grid = np.linspace(0, ora.t_max, 101)
        t0 = time.time()
        members = [NeuralAFT("fae_kan", **C).fit(otr, ova, seed=s) for s in range(k)]
        base = dict(variant=a.variant, replicate=rep, data_seed=seed, config=C, block_params=members[0].info["block_params"],
                    active_edges={f"{sn}_{b}": KT.active_edges(getattr(net, b)) for sn, net in KT.states(members[0]).items()
                                  for b in ("encoder", "decoder")}, fit_seconds=round(time.time() - t0, 1))
        write(OUT, dict(base, stage="before", **metrics(members, otr, ote, orte, grid)))
        pruned = {}
        for method in ("uncentred", "centred"):
            ms = [copy.deepcopy(m) for m in members]
            info = [KT.prune_member(m, otr, method) for m in ms]
            write(OUT, dict(base, stage=f"pruned_{method}", prune_info=info, **metrics(ms, otr, ote, orte, grid)))
            pruned[method] = ms
        print(f"[{a.variant} rep {rep}] fitted + pruned in {time.time() - t0:.0f}s", flush=True)
        if seed != a.symbolic_rep:
            continue
        ms = pruned["uncentred"]
        n_acc, n_tot = 0, 0
        for j, m in enumerate(ms):
            x = KT.block_inputs(m.risk_model, KT.train_tensor(m, otr))["encoder"]
            t1 = time.time()
            recs = KT.symbolic_block(m.risk_model.encoder, x, use_pysr=False)
            for r in recs:
                write(EDGES, dict(variant=a.variant, replicate=rep, member=j, state="risk", block="encoder", stages="1-2", **r))
            n_acc += sum(r["accepted"] for r in recs); n_tot += len(recs)
            print(f"[{a.variant} rep {rep} member {j}] symbolic 1-2: {sum(r['accepted'] for r in recs)}/{len(recs)} "
                  f"accepted in {time.time() - t1:.0f}s", flush=True)
        write(OUT, dict(base, stage="symbolic_1_2", edges_fitted=n_tot, edges_accepted=n_acc,
                        **metrics(ms, otr, ote, orte, grid)))
        if a.variant == "plain":
            m = ms[0]
            rej = [json.loads(l) for l in EDGES.read_text().splitlines()]
            rej = sorted([r for r in rej if r["variant"] == "plain" and r["replicate"] == rep and r["member"] == 0
                          and not r["accepted"]], key=lambda r: -r["l1"])[:10]
            x = KT.block_inputs(m.risk_model, KT.train_tensor(m, otr))["encoder"]
            t1 = time.time()
            recs = KT.symbolic_block(m.risk_model.encoder, x, use_pysr=True, pysr_budget={"timeout": 60},
                                     edges={(r["layer"], r["i"], r["o"]) for r in rej})
            for r in recs:
                write(EDGES, dict(variant=a.variant, replicate=rep, member=0, state="risk", block="encoder", stages="1-3", **r))
            write(OUT, dict(base, stage="symbolic_1_3", pysr_edges=len(recs), pysr_accepted=sum(r["accepted"] for r in recs),
                            pysr_stage_accepted=sum(r["accepted"] and r["stage"] == "pysr" for r in recs),
                            **metrics(ms, otr, ote, orte, grid)))
            print(f"[plain rep {rep}] PySR on {len(recs)} edges: {sum(r['accepted'] for r in recs)} accepted "
                  f"in {time.time() - t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
