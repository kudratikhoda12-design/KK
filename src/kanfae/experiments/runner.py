"""Shared experiment machinery: tuning rule (protocol 2.1), evaluation (protocol 5), resumable result records."""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from ..data.schema import Observed
from ..evaluation import metrics as M
from ..models.pipelines import make_model

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(os.environ.get("KANFAE_RESULTS", ROOT / "reports" / "experiment_results.jsonl"))

GRIDS = {
    "fpca_aft": [{"n_components": r} for r in (2, 4, 8, 12)],
    "fpca_cox": [{"n_components": r, "penalizer": 0.01} for r in (2, 4, 8, 12)],
    "sum_aft": [{"l2": v} for v in (1e-3, 1e-2, 1e-1, 1.0)],
    "sum_cox": [{"penalizer": v} for v in (0.01, 0.1, 1.0, 10.0)],
    "sum_rsf": [{"min_samples_leaf": v} for v in (5, 10, 20, 40)],
}
STOCHASTIC = {"sum_rsf", "fae_mlp", "fae_kan", "fae_kan2", "fae_linear", "flat_ae"}


class KMNull:
    """Covariate-free Kaplan-Meier reference (D35)."""
    name = "km_null"
    n_params = 0

    def fit(self, train: Observed, val=None, seed=0):
        if not isinstance(train, Observed):
            raise TypeError("accepts only Observed")
        t0 = time.time()
        self.y, self.d = train.y_obs, train.delta
        self.fit_seconds = time.time() - t0
        return self

    def prepare(self, obs, holdout=False):
        return None

    def predict_survival(self, obs, times, P=None):
        return np.tile(M.km_curve(self.y, self.d, np.asarray(times, float))[None, :], (len(obs), 1))

    def predict_risk(self, obs, P=None):
        return np.zeros(len(obs))


def run_id(**kw):
    return hashlib.sha1(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:16]


def done_ids():
    if not RESULTS.exists():
        return set()
    return {json.loads(l)["run_id"] for l in RESULTS.read_text().splitlines() if l.strip()}


def append_record(rec):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")


def S_at_own_time(model, obs, P):
    """S_i(Y_i) for each individual (D-calibration). Times are passed sorted; column j is time y[order[j]]."""
    y = obs.y_obs
    order = np.argsort(y)
    S = model.predict_survival(obs, y[order], P=P)
    inv = np.argsort(order)
    return S[np.arange(len(y)), inv]


def evaluate(model, obs_tr, obs_te, P_te, oracle_te=None, curve_grid=None, horizons_q=(0.25, 0.5, 0.75)):
    y_tr, d_tr, y, d = obs_tr.y_obs, obs_tr.delta, obs_te.y_obs, obs_te.delta
    times = M.ibs_time_grid(y, y_tr)
    S = model.predict_survival(obs_te, times, P=P_te)
    ibs, _ = M.ibs_marginal(S, times, y, d, y_tr, d_tr)
    risk = model.predict_risk(obs_te, P=P_te)
    tau = float(np.quantile(y, 0.9))
    out = dict(ibs=ibs, uno_c=M.uno_c(y_tr, d_tr, y, d, risk, tau), harrell_c=M.harrell_c(y, d, risk),
               ibs_t_lo=float(times[0]), ibs_t_hi=float(times[-1]), uno_tau=tau)
    S_y = S_at_own_time(model, obs_te, P_te)
    chi2, p, _ = M.d_calibration(S_y, d)
    out.update(dcal_chi2=chi2, dcal_p=p)
    hz = np.quantile(y, horizons_q)
    pred, km = M.calibration_in_the_large(model.predict_survival(obs_te, hz, P=P_te), y, d, hz)
    for q, a, b, h in zip(horizons_q, pred, km, hz):
        out[f"cal_pred_q{int(q * 100)}"], out[f"cal_km_q{int(q * 100)}"], out[f"cal_t_q{int(q * 100)}"] = a, b, h
    if oracle_te is not None and curve_grid is not None:
        out["curve_error"] = M.survival_curve_error(model.predict_survival(obs_te, curve_grid, P=P_te),
                                                    oracle_te.survival(curve_grid), curve_grid)
    return out


def tune_fit_evaluate(name, obs_tr, obs_va, obs_te, base_record, oracle_te=None, curve_grid=None, grid=None,
                      seeds=(0, 1, 2), extra_cfg=None, holdout_test=False, skip_done=True, extra_eval=None):
    """Protocol 2.1: 4 configs at seed 0 selected by validation IBS; selected config re-run with seeds 1, 2."""
    grid = grid if grid is not None else GRIDS.get(name, [{}])
    extra_cfg = extra_cfg or {}
    have = done_ids() if skip_done else set()
    tuning = []
    for cfg in grid:
        rid = run_id(**base_record, model=name, cfg=cfg, seed=0, phase="tune")
        rec = dict(base_record, run_id=rid, model=name, config=json.dumps({**cfg, **extra_cfg}), seed=0, phase="tune")
        try:
            t0 = time.time()
            m = (KMNull() if name == "km_null" else make_model(name, **cfg, **extra_cfg)).fit(obs_tr, obs_va, seed=0)
            P_va = m.prepare(obs_va)
            yv, dv = obs_va.y_obs, obs_va.delta
            tv = M.ibs_time_grid(yv, obs_tr.y_obs)
            val_ibs = M.ibs_marginal(m.predict_survival(obs_va, tv, P=P_va), tv, yv, dv, obs_tr.y_obs, obs_tr.delta)[0]
            status = "ok" if np.isfinite(val_ibs) else "diverged"     # non-finite scores are never reported as ok
            rec.update(status=status, val_ibs=val_ibs, fit_seconds=getattr(m, "fit_seconds", time.time() - t0),
                       n_params=getattr(m, "n_params", None), fit_info=json.dumps(getattr(m, "info", {}), default=str)[:500])
            tuning.append((val_ibs, cfg, m))
        except Exception as exc:                                   # failed runs are recorded, never dropped
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-1500:])
        if rid not in have:
            append_record(rec)
    ok = [t for t in tuning if np.isfinite(t[0])]
    if not ok:
        return None
    best_ibs, best_cfg, best_model = min(ok, key=lambda t: t[0])
    results = []
    run_seeds = seeds if name in STOCHASTIC else (0,)
    for s in run_seeds:
        rid = run_id(**base_record, model=name, cfg=best_cfg, seed=s, phase="final")
        rec = dict(base_record, run_id=rid, model=name, config=json.dumps({**best_cfg, **extra_cfg}), seed=s,
                   phase="final", selected_val_ibs=best_ibs)
        try:
            m = best_model if s == 0 else (make_model(name, **best_cfg, **extra_cfg)).fit(obs_tr, obs_va, seed=s)
            P_te = m.prepare(obs_te, holdout=holdout_test)
            ev = evaluate(m, obs_tr, obs_te, P_te, oracle_te, curve_grid)
            rec.update(status="ok" if np.isfinite(ev["ibs"]) else "diverged", n_params=getattr(m, "n_params", None),
                       fit_seconds=getattr(m, "fit_seconds", None),
                       fit_info=json.dumps(getattr(m, "info", {}), default=str)[:500], **ev)
            if hasattr(m, "fpca") and holdout_test:
                rec.update(fpca_reconstruction(m, P_te))
            if extra_eval is not None:
                rec.update(extra_eval(m, s))
            results.append(rec)
        except Exception as exc:
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-1500:])
        if rid not in have:
            append_record(rec)
    return results


def median_from_curve(S, grid):
    """Predicted median residual time: first grid time with S <= 0.5 (linear interpolation). Curves that never
    reach 0.5 on the grid return grid[-1] and are flagged."""
    S = np.asarray(S, float)
    below = S <= 0.5
    has = below.any(axis=1)
    j = np.argmax(below, axis=1)
    j0 = np.clip(j - 1, 0, None)
    s0, s1 = S[np.arange(len(S)), j0], S[np.arange(len(S)), j]
    t0, t1 = grid[j0], grid[j]
    frac = np.where(s0 > s1, (s0 - 0.5) / np.maximum(s0 - s1, 1e-12), 0.0)
    med = np.where(j == 0, grid[0], t0 + frac * (t1 - t0))
    return np.where(has, med, grid[-1]), ~has


def fpca_reconstruction(m, P):
    """Reconstruction RMSE at held-out points (standardised scale) for FPCA with the selected r, and the
    smoother-only reference (all components)."""
    from ..functional.smoothing import evaluate as ev, individual_weighted_mse, observation_weighted_mse
    S = m.fpca.transform(P.c)
    c_hat = m.fpca.inverse_transform(S)
    out = {}
    for tag, cc in (("fpca", c_hat), ("smoother", P.c)):
        err2 = (P.values - ev(cc, P.Phi)) ** 2
        out[f"rec_rmse_ind_{tag}"] = float(np.sqrt(individual_weighted_mse(err2, P.eval_mask)))
        out[f"rec_rmse_obs_{tag}"] = float(np.sqrt(observation_weighted_mse(err2, P.eval_mask)))
    return out


def tune_fit_evaluate_ensemble(name, obs_tr, obs_va, obs_te, base_record, oracle_te=None, curve_grid=None, grid=None,
                               k=3, extra_cfg=None, holdout_test=True, skip_done=True, extra_eval=None):
    """D104 (Step 3, Revision 4f): protocol 2.1 tuning at seed 0 (validation IBS of the single seed-0 fit), then seeds
    1..k-1 of the selected configuration; the PRIMARY record is the k-seed ensemble (mean risk and survival over the
    members) on the test split, phase "ensemble"; per-seed records (phase "final") are secondary. grid=None -> [{}]."""
    from ..models.ensemble import SeedEnsembleAFT
    grid = grid if grid is not None else [{}]
    extra_cfg = extra_cfg or {}
    have = done_ids() if skip_done else set()
    tuning = []
    for cfg in grid:
        rid = run_id(**base_record, model=name, cfg=cfg, seed=0, phase="tune")
        rec = dict(base_record, run_id=rid, model=name, config=json.dumps({**extra_cfg, **cfg}), seed=0, phase="tune")
        try:
            t0 = time.time()
            m = make_model(name, **{**extra_cfg, **cfg}).fit(obs_tr, obs_va, seed=0)
            P_va = m.prepare(obs_va)
            tv = M.ibs_time_grid(obs_va.y_obs, obs_tr.y_obs)
            val_ibs = M.ibs_marginal(m.predict_survival(obs_va, tv, P=P_va), tv, obs_va.y_obs, obs_va.delta,
                                     obs_tr.y_obs, obs_tr.delta)[0]
            rec.update(status="ok" if np.isfinite(val_ibs) else "diverged", val_ibs=val_ibs,
                       val_harrell_c=M.harrell_c(obs_va.y_obs, obs_va.delta, m.predict_risk(obs_va, P=P_va)),
                       fit_seconds=getattr(m, "fit_seconds", time.time() - t0), n_params=getattr(m, "n_params", None),
                       block_params=json.dumps(getattr(m, "info", {}).get("block_params")),
                       fit_info=json.dumps(getattr(m, "info", {}), default=str)[:500])
            tuning.append((val_ibs, cfg, m))
        except Exception as exc:                                   # failed runs are recorded, never dropped
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-1500:])
        if rid not in have:
            append_record(rec)
    ok = [t for t in tuning if np.isfinite(t[0])]
    if not ok:
        return None
    best_ibs, best_cfg, best_model = min(ok, key=lambda t: t[0])
    members, results = [], []
    for s in range(k):
        rid = run_id(**base_record, model=name, cfg=best_cfg, seed=s, phase="final")
        rec = dict(base_record, run_id=rid, model=name, config=json.dumps({**extra_cfg, **best_cfg}), seed=s,
                   phase="final", selected_val_ibs=best_ibs)
        try:
            m = best_model if s == 0 else make_model(name, **{**extra_cfg, **best_cfg}).fit(obs_tr, obs_va, seed=s)
            members.append(m)
            ev = evaluate(m, obs_tr, obs_te, m.prepare(obs_te, holdout=holdout_test), oracle_te, curve_grid)
            rec.update(status="ok" if np.isfinite(ev["ibs"]) else "diverged", n_params=m.n_params,
                       fit_seconds=m.fit_seconds, fit_info=json.dumps(m.info, default=str)[:500], **ev)
            if extra_eval is not None:
                rec.update(extra_eval(m, s))
        except Exception as exc:
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-1500:])
        if rid not in have:
            append_record(rec)
        results.append(rec)
    rid = run_id(**base_record, model=name, cfg=best_cfg, seed=f"ens0-{k - 1}", phase="ensemble")
    rec = dict(base_record, run_id=rid, model=name, config=json.dumps({**extra_cfg, **best_cfg}), seed=f"ens0-{k - 1}",
               phase="ensemble", selected_val_ibs=best_ibs, ensemble_seeds=k)
    if len(members) == k:
        try:
            ens = SeedEnsembleAFT.from_members(members)
            ev = evaluate(ens, obs_tr, obs_te, ens.prepare(obs_te, holdout=holdout_test), oracle_te, curve_grid)
            rec.update(status="ok" if np.isfinite(ev["ibs"]) else "diverged", n_params=members[0].n_params, **ev)
            if extra_eval is not None:
                rec.update(extra_eval(ens, "ensemble"))
        except Exception as exc:
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-1500:])
    else:
        rec.update(status="failed", error=f"only {len(members)} of {k} members fitted")
    if rid not in have:
        append_record(rec)
    results.append(rec)
    return results
