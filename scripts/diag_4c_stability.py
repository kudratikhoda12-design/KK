"""D83 diagnostics (seed-21 train/validation only): (i) full-budget (3000 epochs, no early stopping) training-NLL jumps
for the 4b base vs the 4c selection; (ii) thread-count spread of the 4c selection. Writes reports/diag_4c_stability.json."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHILD = r'''
import json, sys, torch
sys.path.insert(0, "%s/src")
from kanfae.data.splits import split_groups
from kanfae.data.synthetic import SynthConfig, generate
from kanfae.evaluation.metrics import harrell_c
from kanfae.models.pipelines import NeuralAFT
torch.set_default_dtype(torch.float64)
cfg = json.loads(sys.argv[1]); threads = int(sys.argv[2]); torch.set_num_threads(threads)
obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=21))
tr, va, te = split_groups(obs.ids, 21, strata=obs.delta)
m = NeuralAFT("fae_mlp", **cfg).fit(obs.subset(tr), obs.subset(va), seed=0)
r = m.info["runs"][0]
print(json.dumps(dict(threads=threads, val_C=harrell_c(obs.y_obs[va], obs.delta[va], m.predict_risk(obs.subset(va))),
                      epochs=r["epochs"], jumps=r["nll_jumps_gt02_after300"], max_train_nll_after300=r["max_train_nll_after300"])))
''' % ROOT


def child(cfg, threads):
    out = subprocess.run([sys.executable, "-c", CHILD, json.dumps(cfg), str(threads)], capture_output=True, text=True,
                         env={"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "PATH": "/usr/bin:/bin"})
    return json.loads(out.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    sel = json.loads((ROOT / "reports/learnability_selected_fa4c.json").read_text())["config"]
    base = json.loads((ROOT / "reports/learnability_selected_fa4b.json").read_text())["config"]
    res = {"full_budget": {}, "thread_spread": []}
    for name, cfg in (("4b base", base), ("4c selection", sel)):
        res["full_budget"][name] = child(dict(cfg, patience=10 ** 6, max_epochs=3000), 1)
        print(name, res["full_budget"][name], flush=True)
    for t in (1, 2, 4):
        res["thread_spread"].append(child(sel, t))
        print("threads", res["thread_spread"][-1], flush=True)
    (ROOT / "reports/diag_4c_stability.json").write_text(json.dumps(res, indent=1))
