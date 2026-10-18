"""D105: Step-3 real-data runs of fae_mlp and fae_kan under C4F (3-seed ensemble; protocol 2.1 tuning at seed 0).
PBC (landmark 730 days; E2 provisional primary, E1 sensitivity; 5 outer splits, seeds 0-4) and C-MAPSS FD001 Option B
(5 splits). Same loaders, splits and extra metrics as the round-1 scripts (run_pbc.py, run_cmapss.py). Guarded by the
4f verdict. Records: $KANFAE_RESULTS (reports/step3/real_*.jsonl)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kanfae.data.cmapss import informative_sensors, official_test_life, option_b, read_fd  # noqa: E402
from kanfae.data.pbc import landmark_dataset, read_pbcseq  # noqa: E402
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.evaluation import metrics as M  # noqa: E402
from kanfae.experiments.runner import tune_fit_evaluate_ensemble  # noqa: E402
from run_cmapss import rul_metrics  # noqa: E402
from run_neural_synthetic import NN_GRID  # noqa: E402
from run_pbc import censoring_covariates, last_values  # noqa: E402

MODELS = ("fae_mlp", "fae_kan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=("pbc", "cmapss"), required=True)
    ap.add_argument("--splits", type=int, default=5)
    a = ap.parse_args()
    v = json.loads((ROOT / "reports/revised_gate_verdict_fa4f.json").read_text())
    if v.get("verdict") != "CONFIRMED":
        sys.exit("BLOCKED (D57/D61): the conventional MLP has not passed the revised gate.")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    sel = json.loads((ROOT / "reports/learnability_selected_fa4f.json").read_text())
    k = int(sel["ensemble_seeds"])
    tuned = {kk: vv for kk, vv in sel["config"].items() if kk not in ("lr", "wd")}
    if a.data == "pbc":
        df = read_pbcseq(ROOT / "data" / "raw" / "pbc" / "data" / "pbc.rda")
        for endpoint in ("E2", "E1"):
            obs = landmark_dataset(df, endpoint)
            for sp in range(a.splits):
                t0 = time.time()
                tr, va, te = split_groups(obs.ids, sp, strata=obs.delta)
                o_tr, o_va, o_te = obs.subset(tr), obs.subset(va), obs.subset(te)
                Xtr_raw = np.hstack([last_values(o_tr), o_tr.u])
                ref = (np.nanmean(Xtr_raw, 0), np.where(np.nanstd(Xtr_raw, 0) > 0, np.nanstd(Xtr_raw, 0), 1.0))
                Gc = M.conditional_censoring_model(o_tr.y_obs, o_tr.delta, censoring_covariates(o_tr, ref))
                Xc_te = censoring_covariates(o_te, ref)
                base = dict(dataset="pbc", analysis=f"neural4f_pbc_{endpoint}", setting=f"pbc_{endpoint}", endpoint=endpoint,
                            replicate=sp, split_seed=sp, n_train=len(tr), n_val=len(va), n_test=len(te),
                            events_train=int(o_tr.delta.sum()), events_test=int(o_te.delta.sum()),
                            censored_frac_test=float(1 - o_te.delta.mean()), data_kind="REAL (right-censored)",
                            selected_config_id=sel["config_id"], variant="{}", n_channels=int(obs.values.shape[2]))

                def extra(m, seed, o_te=o_te, o_tr=o_tr, Gc=Gc, Xc_te=Xc_te):
                    times = M.ibs_time_grid(o_te.y_obs, o_tr.y_obs)
                    ibs_c, _, cap = M.ibs_conditional(m.predict_survival(o_te, times), times, o_te.y_obs, o_te.delta, Xc_te, Gc)
                    return {"ibs_conditional_ipcw": ibs_c, "ipcw_weight_cap": cap}
                for name in MODELS:
                    tune_fit_evaluate_ensemble(name, o_tr, o_va, o_te, base, grid=NN_GRID, k=k, extra_cfg=tuned,
                                               holdout_test=False, extra_eval=extra)
                print(f"[pbc {endpoint} split {sp}] {time.time() - t0:.0f}s", flush=True)
    else:
        tr_df, te_df, rul = read_fd(ROOT / "data" / "raw" / "cmapss", "FD001")
        units = np.array(sorted(tr_df.unit.unique()))
        life_test = official_test_life(te_df, rul)
        for sp in range(a.splits):
            t0 = time.time()
            i, j, l = split_groups(units, sp)
            u_tr, u_va, u_te = units[i], units[j], units[l]
            sens = informative_sensors(tr_df, u_tr)
            o_tr, o_va, o_te = (option_b(tr_df, u, sens) for u in (u_tr, u_va, u_te))
            o_off = option_b(te_df, sorted(te_df.unit.unique()), sens, life=life_test, last_only=True)
            grid = np.linspace(0, 2.0 * float(np.max(o_tr.y_obs)), 2001)
            base = dict(dataset="cmapss_FD001", analysis="neural4f_cmapss_B", setting="cmapss_B", replicate=sp, split_seed=sp,
                        n_train=len(o_tr), n_val=len(o_va), n_test=len(o_te), n_sensors=len(sens),
                        data_kind="REAL (simulated turbofan; RUL benchmark)", censored_frac_test=float(1 - o_te.delta.mean()),
                        selected_config_id=sel["config_id"], variant="{}", n_channels=int(o_tr.values.shape[2]))

            def extra(m, seed, o_te=o_te, o_off=o_off, grid=grid):
                out = rul_metrics(m, o_te, grid, "rul_test")
                if len(o_off):
                    out.update(rul_metrics(m, o_off, grid, "rul_official"))
                return out
            for name in MODELS:
                tune_fit_evaluate_ensemble(name, o_tr, o_va, o_te, base, grid=NN_GRID, k=k, extra_cfg=tuned,
                                           holdout_test=False, extra_eval=extra)
            print(f"[cmapss B split {sp}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
