"""Stage G: PBC sequential data, landmark 730 days, endpoints E1 and E2 as separate sensitivity analyses.

Neural autoencoders are BLOCKED (D33); validated baselines only. Neither endpoint is declared definitive (D9-R);
E3 (competing risks) is not implemented. Marginal-KM IPCW is primary but flagged as a limitation (D17); a
conditional (Cox censoring model) IPCW IBS is reported as a sensitivity analysis.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kanfae.data.pbc import describe, landmark_dataset, read_pbcseq  # noqa: E402
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.evaluation import metrics as M  # noqa: E402
from kanfae.experiments.runner import append_record, done_ids, run_id, tune_fit_evaluate  # noqa: E402

MODELS = ["fpca_aft", "fpca_cox", "sum_aft", "sum_cox", "sum_rsf", "km_null"]


def last_values(obs):
    D = obs.values.shape[2]
    out = np.zeros((len(obs), D))
    for i in range(len(obs)):
        for d in range(D):
            m = obs.mask[i, :, d]
            out[i, d] = obs.values[i, m, d][-1] if m.any() else np.nan
    return out


def censoring_covariates(obs, ref):
    X = np.hstack([last_values(obs), obs.u])
    mu, sd = ref
    X = (X - mu) / sd
    return np.nan_to_num(X)


def visit_process_descriptive(obs, endpoint):
    """Descriptive only: univariate Cox association between number of visits in the window and the outcome."""
    from lifelines import CoxPHFitter
    df = pd.DataFrame({"visits": obs.n_obs.astype(float), "T": obs.y_obs, "E": obs.delta.astype(int)})
    c = CoxPHFitter().fit(df, "T", "E")
    s = c.summary.loc["visits"]
    return dict(hr_per_visit=float(s["exp(coef)"]), hr_lo=float(s["exp(coef) lower 95%"]),
                hr_hi=float(s["exp(coef) upper 95%"]), p=float(s["p"]),
                visits_median=float(np.median(obs.n_obs)), visits_min=int(obs.n_obs.min()), visits_max=int(obs.n_obs.max()),
                frac_le2_visits=float(np.mean(obs.n_obs <= 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    df = read_pbcseq(ROOT / "data" / "raw" / "pbc" / "data" / "pbc.rda")
    (ROOT / "configs" / "pbc").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "logs" / "pbc_inspection.json").write_text(json.dumps(describe(df), indent=1))
    for endpoint in ("E1", "E2"):
        analysis = f"pbc_{endpoint}"
        obs = landmark_dataset(df, endpoint)
        (ROOT / "configs" / "pbc" / f"{analysis}.json").write_text(json.dumps(
            {"analysis": analysis, "endpoint": endpoint, "landmark_days": 730, "channels": list(obs.channel_names),
             "baseline_covariates": ["age", "sex(f=1)"], "split_seeds": list(range(args.splits)), "models": MODELS},
            indent=2))
        rid = run_id(analysis=analysis, kind="visit_descriptive")
        if rid not in done_ids():
            append_record(dict(run_id=rid, dataset="pbc", analysis=analysis, setting=analysis, model="visit_process",
                               phase="descriptive", status="ok", data_kind="REAL (right-censored)",
                               n_cohort=len(obs), events=int(obs.delta.sum()), **visit_process_descriptive(obs, endpoint)))
        for sp in range(args.splits):
            t0 = time.time()
            tr, va, te = split_groups(obs.ids, sp, strata=obs.delta)
            o_tr, o_va, o_te = obs.subset(tr), obs.subset(va), obs.subset(te)
            Xtr_raw = np.hstack([last_values(o_tr), o_tr.u])
            ref = (np.nanmean(Xtr_raw, 0), np.where(np.nanstd(Xtr_raw, 0) > 0, np.nanstd(Xtr_raw, 0), 1.0))
            Gc = M.conditional_censoring_model(o_tr.y_obs, o_tr.delta, censoring_covariates(o_tr, ref))
            Xc_te = censoring_covariates(o_te, ref)
            base = dict(dataset="pbc", analysis=analysis, setting=analysis, endpoint=endpoint, replicate=sp,
                        split_seed=sp, n_train=len(tr), n_val=len(va), n_test=len(te),
                        events_train=int(o_tr.delta.sum()), events_test=int(o_te.delta.sum()),
                        censored_frac_test=float(1 - o_te.delta.mean()), data_kind="REAL (right-censored)")

            def extra(m, seed, o_te=o_te, o_tr=o_tr, Gc=Gc, Xc_te=Xc_te):
                times = M.ibs_time_grid(o_te.y_obs, o_tr.y_obs)
                S = m.predict_survival(o_te, times)
                ibs_c, _, cap = M.ibs_conditional(S, times, o_te.y_obs, o_te.delta, Xc_te, Gc)
                return {"ibs_conditional_ipcw": ibs_c, "ipcw_weight_cap": cap}

            for name in MODELS:
                tune_fit_evaluate(name, o_tr, o_va, o_te, base, grid=[{}] if name == "km_null" else None,
                                  extra_eval=extra)
            print(f"[{analysis} split {sp}] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
