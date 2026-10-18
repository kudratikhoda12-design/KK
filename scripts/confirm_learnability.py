"""D41 independent confirmation of the selected learnability configuration.

Replications on NEW data seeds 101-105 (same generator settings as the gate: N = 2000, sigma_y = 0.05, no censoring).
On each held-out TEST split: R2 criteria unchanged (C >= oracle C - 0.03; mean linear-probe R^2 over a, b, A, omega
>= 0.8). Confirmed if >= 4 of 5 replications pass (and the official gate passes separately). The configuration is read
from reports/learnability_selected.json (fixed before this script runs).
"""
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
from kanfae.models.pipelines import NeuralAFT  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from learnability_search import probe  # noqa: E402


def main():
    torch.set_default_dtype(torch.float64)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", nargs="?", default=None)
    ap.add_argument("--selected", default="reports/learnability_selected.json")
    ap.add_argument("--seeds", default="101,102,103,104,105")
    ap.add_argument("--tag", default="")
    ap.add_argument("--amended", action="store_true", help="D80/D82 criteria: C >= min(oracle-0.03, C_ceiling-0.015), "
                                                            "probe mean >= 0.78 and omega probe >= 0.80")
    ap.add_argument("--threads", type=int, default=None, help="torch threads (D79: 1 for deterministic fits)")
    ap.add_argument("--ceiling-margin", type=float, default=0.015, help="C_ceiling margin (D82: 0.015; D88: 0.020)")
    ap.add_argument("--probe-rule", choices=("4c", "4e", "4f"), default="4c",
                    help="amended probe part: 4c = mean >= 0.78 and omega >= 0.80 (D82); 4e = mean >= 0.75 and omega >= 0.75 "
                         "(D93); 4f = 0.78/0.80 for the seed-0 member and the member-averaged probe (D101)")
    ap.add_argument("--ensemble", type=int, default=0, help="D101: k > 0 fits the selection's seed ensemble of size k "
                                                           "(0 = the earlier single seed-0 fit)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    sel = json.loads((ROOT / a.selected).read_text())
    kind = a.kind or sel["kind"]
    out = ROOT / "reports" / f"learnability_confirmation_{kind}{a.tag}.jsonl"
    recs = []
    for seed in [int(x) for x in a.seeds.split(",")]:
        t0 = time.time()
        obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=seed))
        tr, va, te = split_groups(obs.ids, seed, strata=obs.delta)
        if a.ensemble:
            from kanfae.models.ensemble import SeedEnsembleAFT
            m = SeedEnsembleAFT(kind, ensemble_seeds=a.ensemble, **sel["config"]).fit(obs.subset(tr), obs.subset(va))
        else:
            m = NeuralAFT(kind, **sel["config"]).fit(obs.subset(tr), obs.subset(va), seed=0)
        c = harrell_c(obs.y_obs[te], obs.delta[te], m.predict_risk(obs.subset(te)))
        c_or = harrell_c(obs.y_obs[te], obs.delta[te], -ora.eta_true[te])
        pr = probe(m, obs, ora, tr, te)
        rec = dict(kind=kind, config_id=sel["config_id"], config=sel["config"], data_seed=seed, test_C=c, oracle_C=c_or,
                   threshold=c_or - 0.03, probe=pr, probe_mean=float(np.mean(pr)),
                   pass_C=bool(c >= c_or - 0.03), pass_probe=bool(np.mean(pr) >= 0.8),
                   runtime_s=round(time.time() - t0, 1),
                   block_params=(m.members[0] if a.ensemble else m).info["block_params"])
        if a.ensemble:
            pm = [probe(mem, obs, ora, tr, te) for mem in m.members]
            rec.update(ensemble_seeds=a.ensemble, probe_members=pm, probe_member_avg=list(np.mean(pm, axis=0)),
                       member_test_C=[harrell_c(obs.y_obs[te], obs.delta[te], mem.predict_risk(obs.subset(te)))
                                      for mem in m.members])
        if a.amended:
            from kanfae.evaluation.ceiling import ceiling_C
            c_ceil = ceiling_C(obs, te)
            rec.update(criteria=f"amended, C_ceiling margin {a.ceiling_margin}", C_ceiling=c_ceil, ceiling_margin=a.ceiling_margin,
                       threshold=min(c_or - 0.03, c_ceil - a.ceiling_margin),
                       pass_C_margin015=bool(c >= min(c_or - 0.03, c_ceil - 0.015)),
                       pass_C_margin020=bool(c >= min(c_or - 0.03, c_ceil - 0.020)))
            rec["pass_C"] = bool(c >= rec["threshold"])
            rec["pass_probe_078_080"] = bool(np.mean(pr) >= 0.78 and pr[3] >= 0.80)
            rec["pass_probe_075_075"] = bool(np.mean(pr) >= 0.75 and pr[3] >= 0.75)
            rec["pass_probe_original_080"] = bool(np.mean(pr) >= 0.80)
            rec["probe_rule"] = a.probe_rule
            if a.probe_rule == "4f":
                pa = rec.get("probe_member_avg", pr)
                rec["pass_probe_4f"] = bool(rec["pass_probe_078_080"] and np.mean(pa) >= 0.78 and pa[3] >= 0.80)
            rec["pass_probe"] = {"4e": rec["pass_probe_075_075"], "4c": rec["pass_probe_078_080"]}.get(
                a.probe_rule, rec.get("pass_probe_4f"))
        rec["pass"] = rec["pass_C"] and rec["pass_probe"]
        recs.append(rec)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"seed {seed}: C={c:.4f} (oracle {c_or:.4f}, threshold {rec['threshold']:.4f}"
              f"{', C_ceiling %.4f' % rec['C_ceiling'] if 'C_ceiling' in rec else ''}) probe={np.round(pr, 3)} "
              f"mean={np.mean(pr):.3f} pass={rec['pass']} [{rec['runtime_s']}s]", flush=True)
    n = sum(r["pass"] for r in recs)
    print(f"CONFIRMATION ({kind}): {n}/5 replications pass -> {'CONFIRMED' if n >= 4 else 'NOT CONFIRMED'}")


if __name__ == "__main__":
    main()
