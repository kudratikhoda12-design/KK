"""Round 13 (L36): real data, exploratory (D114 as amended by D128 and D129). Jobs (one record per model):

single  (dataset, split)  single landmark: PBC 730 d (E2 primary, E1 sensitivity), prothro 1 y, aids 6 months; whole
        history as the window; outer split split_groups(subject ids, split, strata = event), 60/20/20, splits 0-4.
        Curves: per-subject penalised B-spline smoothing (12 functions, held-out-point lambda on training subjects;
        missing channel values masked; a channel never observed takes the training mean curve) on the 50-point grid
        (primary front end); PACE (bandwidths 0.05-0.5) for the b2s_cox_pace sensitivity.
        Models: b2s_cox, b2s_aft, b2h_cox, b2h_aft, b1_fpca1_cox/aft, b1_sum_cox/aft, static_coxkan (last values + z),
        afcm_v3 (AFCM_D123, flagged not validated, D126/D128), fpca_cox, sum_cox, sum_rsf, lme2_cox, landmark_cox
        (lifelines on last values + z), b2s_cox_pace; then prune_symbolic on b2s_cox (D120 selection; stages 1-2 and
        PySR as D112; primary = substituted model without refit, secondary = refitted, D128).
        Features with missing values take training-split means. Every KAN model is the D112 ensemble; member seed 0
        is also reported.
        Metrics (test split): marginal-IPCW IBS, conditional-IPCW IBS (Cox censoring model on z, penaliser 0.1,
        weights capped at the 95th percentile), Harrell C, Uno C (tau = q90 of training residual times), D-cal p.
b3      (dataset, split)  stacked landmark supermodel (PBC E2: landmarks 1-4 y, horizon 5 y, window 2 y; prothro:
        0.5, 1, 2, 3 y, 5 y, 2 y; aids: 2, 6, 12 months, horizon 12 months, whole history): features = NaN-aware
        summaries of the window (absolute time units) + z + t_L; b3_coxkan (static KAN ensemble, landmark-stratified
        Cox, t_L an input edge) and b3_cox (lifelines, strata = landmark, penaliser 0.01). Per landmark on test rows:
        IPCW Brier at the horizon, time-dependent AUC at the horizon (cumulative/dynamic), Harrell C within the horizon.
cmapss  (split)  C-MAPSS FD001 Option B (30-cycle windows every 10 cycles, s/100 as covariate; split by engine):
        b2s_aft (Weibull KAN-AFT ensemble on B-spline curves of the informative sensors), b2s_aft_rc (the same ensemble,
        residual-calibrated), and Round 1's parametric sum_aft and fpca_aft. Metrics: IBS, Harrell C, Uno C, D-cal p.
stab    (dataset, model, split)  stability: 20 bootstrap resamples of the training subjects (seed 1000 split + b),
        single seed 0 at the split's tuned (lr, wd) for b2s_cox and b1_fpca1_cox on pbc_E2, prothro, aids. Effects saved
        on a common x-grid per channel (50 points between the 5th and 95th percentiles of the pooled observed values of
        the landmark cohort) at 50 grid times; npz files in reports/round13/stability/.
Records: $R13_OUT (jsonl). Failures are recorded with tracebacks. Resumable.
"""
import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kanfae.data import fsim  # noqa: E402
from kanfae.data.landmark import build_stacked  # noqa: E402
from kanfae.data.realdata import LOADERS, single_landmark  # noqa: E402
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.evaluation import metrics as M  # noqa: E402
from kanfae.functional.basis import BSplineBasis  # noqa: E402
from kanfae.functional.pace import PACE  # noqa: E402
from kanfae.functional.smoothing import curves_on_grid, pad_ragged, select_lambda  # noqa: E402
from kanfae.models.afcm import AFCM_D123, fit_afcm  # noqa: E402
from kanfae.models.fbaselines import GridFPCA, LinearCox, impute_train_mean, last_values, summaries  # noqa: E402
from kanfae.models.fkan import FKANEnsemble, FKANSurv, select_lambda_theta, substitute_symbolic, symbolic_edges  # noqa: E402
from kanfae.models.lme import LME2Cox  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
G = fsim.GRID
OUT = (ROOT / os.environ.get("R13_OUT", "reports/round13/dev/results_dev.jsonl")).resolve()
STAB_DIR = (ROOT / os.environ.get("R13_STAB", "reports/round13/stability")).resolve()
SINGLE = {"pbc_E2": 730.0, "pbc_E1": 730.0, "prothro": 1.0, "aids": 6.0}
B3 = {"pbc_E2": dict(landmarks=[365.25 * k for k in (1, 2, 3, 4)], horizon=5 * 365.25, window=2 * 365.25),
      "prothro": dict(landmarks=[0.5, 1.0, 2.0, 3.0], horizon=5.0, window=2.0),
      "aids": dict(landmarks=[2.0, 6.0, 12.0], horizon=12.0, window=12.0)}         # window 12 = the whole history
PACE_H = (0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5)
def _dev_longdata(seed=941):
    """Development data (mechanics dry run, D129): fsim S-Add sparse curves shaped like the aids records (months):
    history on [0, 6] months, residual time scaled by 6 months after the 6-month landmark; some values set missing."""
    from kanfae.data.realdata import LongData
    dat = fsim.generate("S-Add", n=400, seed=seed, sparse=True)
    rng = np.random.default_rng(seed)
    vals = []
    for v in dat.values:
        v = np.array(v, float)
        v[rng.random(v.shape) < 0.05] = np.nan
        vals.append(v)
    return LongData("dev_aids", np.arange(400), [t * 6.0 for t in dat.times], vals, dat.z, 6.0 + 6.0 * dat.y,
                    dat.delta.astype(bool), ("x1", "x2", "x3"), ("z1", "z2"), "months", dict(source="fsim dev"))


DEV = {"dev_aids": _dev_longdata}                                                  # development data sets (dry runs)


def _dev_cmapss_df(seed=942):
    """Synthetic table in the C-MAPSS FD001 format (unit, cycle, op settings, sensors) for the mechanics dry run."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, 41):
        life = int(rng.integers(130, 260))
        h = np.linspace(0, 1, life) ** 2 * rng.uniform(0.5, 1.5)
        for c in range(1, life + 1):
            r = dict(unit=u, cycle=c, os1=0.0, os2=0.0, os3=100.0)
            for k in range(1, 22):
                r[f"s{k}"] = (k % 3) * h[c - 1] + 0.1 * rng.normal() if k not in (1, 5, 10, 16, 18, 19) else 1.0
            rows.append(r)
    return pd.DataFrame(rows)


def _load(ds):
    return DEV[ds]() if ds in DEV else LOADERS[ds]()


# ----------------------------------------------------------------------------------------------- single landmark
class RealCtx:
    def __init__(self, ds, split):
        self.ds, self.split = ds, split
        lm = single_landmark(_load(ds), SINGLE[ds.replace("dev_", "")])
        self.lm = lm
        self.n, self.D = len(lm.ids), len(lm.channels)
        self.tr, self.va, self.te = split_groups(lm.ids, split, strata=lm.delta)
        self.y, self.d = lm.y, lm.delta
        self.y_tr, self.d_tr, self.y_te, self.d_te = lm.y[self.tr], lm.delta[self.tr], lm.y[self.te], lm.delta[self.te]
        self.tau = float(np.quantile(self.y_tr, 0.9))
        self.G_cond = M.conditional_censoring_model(self.y_tr, self.d_tr, lm.z[self.tr])
        self._fe, self._feat = {}, {}

    def front(self, name):
        if name in self._fe:
            return self._fe[name]
        lm = self.lm
        if name == "bspline":
            basis = BSplineBasis(12, 3, (0.0, 1.0))
            T, V, Mk, n_obs = pad_ragged([lm.times[i] for i in self.tr], [lm.values[i] for i in self.tr])
            lam, _ = select_lambda(T, V, Mk, n_obs, basis, seed=0)
            xg = curves_on_grid(lm.times, lm.values, G, basis, lam)
            seen = np.array([np.isfinite(v).any(0) for v in lm.values])                # (n, D)
            mean_curve = np.array([xg[self.tr][seen[self.tr, d], :, d].mean(0) for d in range(self.D)]).T   # (G, D)
            for d in range(self.D):
                xg[~seen[:, d], :, d] = mean_curve[:, d]
            self._fe[name] = dict(xg=xg, lam=lam, n_unseen=int((~seen).sum()))
        else:
            xg = np.zeros((self.n, len(G), self.D))
            for c in range(self.D):
                tt, vv = [], []
                for t, v in zip(lm.times, lm.values):
                    ok = np.isfinite(v[:, c]); tt.append(np.asarray(t)[ok]); vv.append(v[ok, c])
                p = PACE(seed=0, h_grid=PACE_H).fit([tt[i] for i in self.tr if len(tt[i])], [vv[i] for i in self.tr if len(tt[i])])
                xg[:, :, c] = p.mu[None, :] + p.scores(tt, vv) @ p.phi.T
            self._fe[name] = dict(xg=xg)
        return self._fe[name]

    def fpca(self, front="bspline"):
        if "fpca" not in self._feat:
            g = GridFPCA().fit(self.front(front)["xg"][self.tr])
            self._feat["fpca"] = (g.transform(self.front(front)["xg"]), g.phi, g.K)
        return self._feat["fpca"]

    def summ(self):
        if "summ" not in self._feat:
            self._feat["summ"] = impute_train_mean(summaries(self.lm.times, self.lm.values), self.tr)
        return self._feat["summ"]

    def lastv(self):
        if "last" not in self._feat:
            self._feat["last"] = impute_train_mean(last_values(self.lm.times, self.lm.values), self.tr)
        return self._feat["last"]

    def pack(self, idx, xg=None, F=None):
        z = self.lm.z[idx] if F is None else np.hstack([F[idx], self.lm.z[idx]])
        return dict(xg=None if xg is None else xg[idx], z=z, y=self.y[idx], d=self.d[idx].astype(float))


def real_metrics(ctx, S_fn, risk):
    tg = M.ibs_time_grid(ctx.y_te, ctx.y_tr)
    S_tg = S_fn(tg)
    Sy = np.diag(S_fn(ctx.y_te))
    Xc = ctx.lm.z[ctx.te]
    return dict(ibs=float(M.ibs_marginal(S_tg, tg, ctx.y_te, ctx.d_te, ctx.y_tr, ctx.d_tr)[0]),
                ibs_cond=float(M.ibs_conditional(S_tg, tg, ctx.y_te, ctx.d_te, Xc, ctx.G_cond)[0]),
                harrell_c=float(M.harrell_c(ctx.y_te, ctx.d_te, risk)),
                uno_c=float(M.uno_c(ctx.y_tr, ctx.d_tr, ctx.y_te, ctx.d_te, risk, ctx.tau)),
                dcal_p=float(M.d_calibration(Sy, ctx.d_te)[1]))


def run_ens(ctx, rec, kind, loss, xg=None, F=None, hidden=0):
    tr, va, te = ctx.pack(ctx.tr, xg, F), ctx.pack(ctx.va, xg, F), ctx.pack(ctx.te, xg, F)
    e = FKANEnsemble(kind=kind, loss=loss, hidden=hidden).fit(tr, va)
    rec.update(lr=e.lr, wd=e.wd, tuning=e.tuning, epochs=[len(m.history) for m in e.members])
    for tag, ms in (("metrics", e.members), ("metrics_seed0", e.members[:1])):
        rec[tag] = real_metrics(ctx, lambda tt, ms=ms: np.mean([m.predict_survival(te["xg"], te["z"], tt) for m in ms], 0),
                                np.mean([m.predict_risk(te["xg"], te["z"]) for m in ms], 0))
    return e


def lifelines_metrics(ctx, m, X):
    Xte = X[ctx.te]
    return real_metrics(ctx, lambda tt: m.predict_survival(Xte, tt), m.predict_risk(Xte))


def run_single_cell(ctx, model, rec):
    xg = ctx.front("bspline")["xg"]
    rec["front_info"] = dict(lam=ctx.front("bspline")["lam"], n_unseen=ctx.front("bspline")["n_unseen"])
    if model in ("b2s_cox", "b2s_aft", "b2h_cox", "b2h_aft"):
        kind, loss = model.split("_")
        run_ens(ctx, rec, kind, loss, xg=xg)
    elif model == "b2s_cox_pace":
        run_ens(ctx, rec, "b2s", "cox", xg=ctx.front("pace")["xg"])
    elif model in ("b1_fpca1_cox", "b1_fpca1_aft"):
        fp = ctx.fpca()
        run_ens(ctx, rec, "static", model.split("_")[-1], F=fp[0])
        rec["K"] = list(map(int, fp[2]))
    elif model in ("b1_sum_cox", "b1_sum_aft"):
        run_ens(ctx, rec, "static", model.split("_")[-1], F=ctx.summ())
    elif model == "static_coxkan":
        run_ens(ctx, rec, "static", "cox", F=ctx.lastv())
    elif model == "afcm_v3":
        m, recs = fit_afcm(ctx.pack(ctx.tr, xg), ctx.pack(ctx.va, xg), **AFCM_D123)
        te = ctx.pack(ctx.te, xg)
        rec.update(metrics=real_metrics(ctx, lambda tt: m.predict_survival(te["xg"], te["z"], tt),
                                        m.predict_risk(te["xg"], te["z"])), lam_x=float(m.lam_x), lam_t=float(m.lam_t),
                   note="AFCM flagged not validated (CK-6c FAIL; D126, D128)")
    elif model in ("fpca_cox", "sum_cox", "landmark_cox"):
        F = {"fpca_cox": lambda: ctx.fpca()[0], "sum_cox": ctx.summ, "landmark_cox": ctx.lastv}[model]()
        X = np.hstack([F, ctx.lm.z])
        m = LinearCox().fit(X[ctx.tr], ctx.y_tr, ctx.d_tr)
        rec.update(metrics=lifelines_metrics(ctx, m, X), coef=m.cph.params_.values.tolist())
    elif model == "sum_rsf":
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        X = np.hstack([ctx.summ(), ctx.lm.z])
        rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=10, max_features="sqrt", n_jobs=1, random_state=0)
        rsf.fit(X[ctx.tr], Surv.from_arrays(ctx.d_tr.astype(bool), ctx.y_tr))
        Sf, ut = rsf.predict_survival_function(X[ctx.te], return_array=True), rsf.unique_times_

        def S_fn(tt):
            idx = np.searchsorted(ut, np.asarray(tt, float), side="right") - 1
            return np.where(idx[None, :] >= 0, Sf[:, np.clip(idx, 0, None)], 1.0)
        rec["metrics"] = real_metrics(ctx, S_fn, rsf.predict(X[ctx.te]))
    elif model == "lme2_cox":
        m = LME2Cox(t_L=1.0).fit([ctx.lm.times[i] for i in ctx.tr], [ctx.lm.values[i] for i in ctx.tr], ctx.lm.z[ctx.tr],
                                 ctx.y_tr, ctx.d_tr)
        tt_, vv_, zz = [ctx.lm.times[i] for i in ctx.te], [ctx.lm.values[i] for i in ctx.te], ctx.lm.z[ctx.te]
        rec["metrics"] = real_metrics(ctx, lambda tt: m.predict_survival(tt_, vv_, zz, tt), m.predict_risk(tt_, vv_, zz))
    elif model == "prune_symbolic":
        run_prune_symbolic_real(ctx, rec, xg)
    else:
        raise ValueError(model)


def run_prune_symbolic_real(ctx, rec, xg):
    lw = tuned(ctx.ds, ctx.split, "b2s_cox")
    if lw is None:
        raise RuntimeError("no b2s_cox record for this split")
    tr, va, te = ctx.pack(ctx.tr, xg), ctx.pack(ctx.va, xg), ctx.pack(ctx.te, xg)
    sel, grid = select_lambda_theta(lambda l: FKANSurv(kind="b2s", loss="cox", lam=l, lr=lw[0], wd=lw[1], seed=0), tr, va)
    chosen = [g for g in grid if g.get("chosen")][0]
    rec.update(lr=lw[0], wd=lw[1], selected=sel.selected_channels(), chosen_lam=chosen["lam"],
               chosen_theta=chosen["theta"], grid=grid, channels=list(ctx.lm.channels))

    def ev(m):
        e = m.eta(te["xg"], te["z"])
        if not np.all(np.isfinite(e)):
            return dict(status="nonfinite")
        return real_metrics(ctx, lambda tt: m.predict_survival(te["xg"], te["z"], tt), m.predict_risk(te["xg"], te["z"]))

    rec["after_pruning"] = ev(sel)
    t0 = time.time()
    recs = symbolic_edges(sel, use_pysr=True, pysr_timeout=60, max_pysr=10, min_share=0.05)
    acc = [r for r in recs if r["result"].accepted]
    rec["symbolic"] = dict(n_edges=len(recs), n_accepted=len(acc), acceptance_rate=len(acc) / max(len(recs), 1),
                           mean_r2_val_accepted=float(np.mean([r["result"].r2_val for r in acc])) if acc else None,
                           stages={s: sum(r["result"].stage == s for r in recs) for s in ("linear", "library", "pysr", "none")},
                           edges=[dict(layer=r["layer"], i=r["i"], o=r["o"], share=r["share"], stage=r["result"].stage,
                                       accepted=r["result"].accepted, formula=r["result"].formula, r2_val=r["result"].r2_val,
                                       channel=(ctx.lm.channels[r["i"]] if r["layer"] in ("g", "phi") else None))
                                  for r in recs], seconds=round(time.time() - t0, 1))
    sub = copy.deepcopy(sel)
    n_sub = substitute_symbolic(sub, recs, stages=("linear", "library"))
    rec["symbolic"]["n_substituted"] = n_sub
    sub._finalise(sub.train_)
    rec["substituted_no_refit"] = ev(sub)                                           # primary (D128)
    e_p, e_s = sel.eta(te["xg"], te["z"]), sub.eta(te["xg"], te["z"])
    rec["fidelity_corr_risk"] = float(np.corrcoef(e_p, e_s)[0, 1]) if np.all(np.isfinite(e_s)) else None
    if n_sub:
        sub.refit_symbolic(max_iter=200)
        e_r = sub.eta(te["xg"], te["z"])
        rec["substituted_refit"] = ev(sub) if np.all(np.isfinite(e_r)) else dict(status="refit_nonfinite")
        rec["fidelity_corr_risk_refit"] = float(np.corrcoef(e_p, e_r)[0, 1]) if np.all(np.isfinite(e_r)) else None


SINGLE_MODELS = ("b2s_cox", "afcm_v3", "b2s_aft", "b2h_cox", "b2h_aft", "b1_fpca1_cox", "b1_fpca1_aft", "b1_sum_cox",
                 "b1_sum_aft", "static_coxkan", "fpca_cox", "sum_cox", "sum_rsf", "lme2_cox", "landmark_cox",
                 "b2s_cox_pace", "prune_symbolic")


# ----------------------------------------------------------------------------------------------------------- B3
def run_b3(ds, split, rec_base):
    ld = _load(ds)
    cfg = B3[ds.replace("dev_", "")]
    tr_s, va_s, te_s = split_groups(ld.ids, split, strata=ld.d)
    part = np.full(len(ld.ids), -1)
    part[tr_s], part[va_s], part[te_s] = 0, 1, 2
    st = build_stacked(ld.times, ld.values, ld.z, ld.y, ld.d, cfg["landmarks"], cfg["window"], cfg["horizon"])
    S = summaries(st.times_abs, st.values)
    rp = part[st.sid]
    F = impute_train_mean(S, rp == 0)
    lms = np.asarray(cfg["landmarks"])
    strata = np.searchsorted(lms, st.landmark)
    X = np.hstack([F, st.z, st.landmark[:, None]])
    out = []
    for model in ("b3_coxkan", "b3_cox"):
        rec = dict(rec_base, model=model, n_rows=int(len(st.y)), n_rows_test=int((rp == 2).sum()),
                   excluded_few_obs=st.meta["excluded_few_obs"])
        t0 = time.time()
        try:
            if model == "b3_coxkan":
                pk = lambda m: dict(xg=None, z=X[m], y=st.y[m], d=st.d[m].astype(float))  # noqa: E731
                e = FKANEnsemble(kind="static", loss="cox").fit(pk(rp == 0), pk(rp == 1), strata[rp == 0], strata[rp == 1])
                rec.update(lr=e.lr, wd=e.wd)
                te = rp == 2

                def surv_at(tt, mask=te):
                    return np.mean([m.predict_survival(None, X[mask], tt, strata=strata[mask]) for m in e.members], 0)
                risk = np.mean([m.predict_risk(None, X[te]) for m in e.members], 0)
            else:
                import pandas as pd
                from lifelines import CoxPHFitter
                cols = [f"f{j}" for j in range(X.shape[1])]
                df = pd.DataFrame(X, columns=cols).assign(T=st.y, E=st.d.astype(int), lm=strata)
                cph = CoxPHFitter(penalizer=0.01).fit(df[rp == 0], "T", "E", strata=["lm"])
                te = rp == 2
                dte = df[te]

                def surv_at(tt, dte=dte):
                    return cph.predict_survival_function(dte, times=tt).to_numpy().T
                risk = cph.predict_log_partial_hazard(dte).to_numpy()
            rec["per_landmark"] = b3_metrics(st, rp, strata, lms, cfg["horizon"], surv_at, risk)
            rec["status"] = "ok"
        except Exception as exc:
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])
        rec["seconds"] = round(time.time() - t0, 1)
        out.append(rec)
    return out


def b3_metrics(st, rp, strata, lms, w, surv_at, risk):
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv
    te = np.nonzero(rp == 2)[0]
    t_star = w * (1 - 1e-9)
    S_all = surv_at(np.array([t_star]))[:, 0]
    res = []
    for k, tl in enumerate(lms):
        mk = strata[te] == k
        trk = (rp == 0) & (strata == k)
        if mk.sum() < 5 or trk.sum() < 5:
            res.append(dict(landmark=float(tl), n_test=int(mk.sum()), note="too few rows"))
            continue
        y, d, r, Sk = st.y[te][mk], st.d[te][mk], risk[mk], S_all[mk]
        Gf = M.marginal_censoring_G(st.y[trk], st.d[trk])
        bs = float(M.brier_ipcw(Sk[:, None], [t_star], y, d, Gf(np.minimum(y, t_star)), Gf([t_star]))[0])
        t_auc = t_star * 0.999
        n_cases, n_ctrl = int(((y <= t_auc) & d).sum()), int((y > t_auc).sum())
        auc = None
        if n_cases >= 10 and n_ctrl >= 10:                                           # D129: AUC needs >= 10 of each
            try:
                auc = float(cumulative_dynamic_auc(Surv.from_arrays(st.d[trk].astype(bool), st.y[trk]),
                                                   Surv.from_arrays(d.astype(bool), y), r, [t_auc])[0][0])
            except Exception:
                auc = None
        res.append(dict(landmark=float(tl), n_test=int(mk.sum()), events=int(d.sum()), auc_cases=n_cases,
                        auc_controls=n_ctrl, brier=bs, td_auc=auc, harrell_c=float(M.harrell_c(y, d, r))))
    return res


# ------------------------------------------------------------------------------------------------------ C-MAPSS
def run_cmapss(split, rec_base):
    from kanfae.data.cmapss import informative_sensors, option_b, read_fd
    from kanfae.functional.smoothing import smooth
    from kanfae.models.pipelines import make_model
    df = _dev_cmapss_df() if rec_base["dataset"] == "dev_cmapss" else read_fd(ROOT / "data/raw/cmapss", "FD001")[0]
    units = np.sort(df.unit.unique())
    utr, uva, ute = split_groups(units, split)
    sensors = informative_sensors(df[df.unit.isin(units[utr])], units[utr])
    obs = {k: option_b(df, units[u], sensors) for k, u in (("tr", utr), ("va", uva), ("te", ute))}
    basis = BSplineBasis(12, 3, (0.0, 1.0))
    xg = {}
    for k, o in obs.items():
        c = smooth(o.times, o.values, o.mask, basis, lam=1e-4)
        xg[k] = np.einsum("ndk,gk->ngd", c, basis(G))
    pk = {k: dict(xg=xg[k], z=obs[k].u, y=obs[k].y_obs, d=obs[k].delta.astype(float)) for k in obs}
    y_tr, d_tr, y_te, d_te = obs["tr"].y_obs, obs["tr"].delta, obs["te"].y_obs, obs["te"].delta
    tau = float(np.quantile(y_tr, 0.9))
    tg = M.ibs_time_grid(y_te, y_tr)

    def metrics(S_fn, risk):
        return dict(ibs=float(M.ibs_marginal(S_fn(tg), tg, y_te, d_te, y_tr, d_tr)[0]),
                    harrell_c=float(M.harrell_c(y_te, d_te, risk)), uno_c=float(M.uno_c(y_tr, d_tr, y_te, d_te, risk, tau)),
                    dcal_p=float(M.d_calibration(np.diag(S_fn(y_te)), d_te)[1]))

    out = []
    for model in ("b2s_aft", "sum_aft", "fpca_aft"):
        rec = dict(rec_base, model=model, n_train=int(len(y_tr)), n_test=int(len(y_te)), sensors=list(sensors))
        t0 = time.time()
        try:
            if model == "b2s_aft":
                e = FKANEnsemble(kind="b2s", loss="aft").fit(pk["tr"], pk["va"])
                rec.update(lr=e.lr, wd=e.wd)
                te = pk["te"]
                S_fn = lambda tt: np.mean([m.predict_survival(te["xg"], te["z"], tt) for m in e.members], 0)  # noqa: E731
                risk = np.mean([m.predict_risk(te["xg"], te["z"]) for m in e.members], 0)
                rec["metrics"] = metrics(S_fn, risk)
                rc = dict(rec, model="b2s_aft_rc")
                for m in e.members:
                    m.cfg["residual_calibration"] = True
                    m._finalise(m.train_)
                rc["metrics"] = metrics(lambda tt: np.mean([m.predict_survival(te["xg"], te["z"], tt) for m in e.members], 0),
                                        risk)
                rc.update(status="ok", seconds=0.0, note="same fitted ensemble as b2s_aft, residual calibration")
                out.append(rc)
            else:                                                   # Round-1 protocol: grid tuned by validation IBS
                from kanfae.experiments.runner import GRIDS
                tv = M.ibs_time_grid(obs["va"].y_obs, y_tr)
                tun = []
                for cfg in GRIDS[model]:
                    mm = make_model(model, **cfg).fit(obs["tr"], obs["va"], seed=0)
                    tun.append((float(M.ibs_marginal(mm.predict_survival(obs["va"], tv), tv, obs["va"].y_obs,
                                                     obs["va"].delta, y_tr, d_tr)[0]), cfg, mm))
                vib, cfg, m = min(tun, key=lambda t: t[0])
                rec.update(config=cfg, tuning=[dict(val_ibs=t[0], **t[1]) for t in tun])
                rec["metrics"] = metrics(lambda tt: m.predict_survival(obs["te"], tt), m.predict_risk(obs["te"]))
            rec["status"] = "ok"
        except Exception as exc:
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])
        rec["seconds"] = round(time.time() - t0, 1)
        out.append(rec)
    return out


# ---------------------------------------------------------------------------------------------------- stability
def x_grids(lm):
    """50-point grid per channel between the 5th and 95th percentiles of the pooled observed values of the cohort."""
    out = []
    for d in range(len(lm.channels)):
        v = np.concatenate([np.asarray(x)[:, d] for x in lm.values])
        v = v[np.isfinite(v)]
        out.append(np.linspace(*np.percentile(v, [5, 95]), 50))
    return np.array(out)


def run_stab(ds, model, split, n_boot=20, rec_base=None):
    ctx = RealCtx(ds, split)
    lw = tuned(ds, split, model)
    if lw is None:
        raise RuntimeError(f"no {model} record for {ds} split {split}")
    xgr = x_grids(ctx.lm)
    xg = ctx.front("bspline")["xg"]
    fp = ctx.fpca() if model == "b1_fpca1_cox" else None
    eff = np.zeros((n_boot, ctx.D, xgr.shape[1], len(G)))
    for b in range(n_boot):
        rng = np.random.default_rng(1000 * split + b)
        trb = ctx.tr[rng.integers(0, len(ctx.tr), len(ctx.tr))]
        if model == "b2s_cox":
            m = FKANSurv(kind="b2s", loss="cox", lr=lw[0], wd=lw[1], seed=0).fit(ctx.pack(trb, xg), ctx.pack(ctx.va, xg))
            for d in range(ctx.D):
                eff[b, d] = m.effect(d, np.repeat(xgr[d][:, None], len(G), 1), G[None, :])
        else:
            F = fp[0]
            m = FKANSurv(kind="static", loss="cox", lr=lw[0], wd=lw[1], seed=0).fit(ctx.pack(trb, F=F), ctx.pack(ctx.va, F=F))
            with torch.no_grad():
                a = m.net.stat.edge_activations(m.data["Ztr"]).numpy()[:, :sum(fp[2]), 0]
            sl = np.array([np.polyfit(F[trb][:, k], a[:, k], 1)[0] for k in range(sum(fp[2]))])
            k0 = 0
            for d, K in enumerate(fp[2]):
                beta_t = fp[1][d] @ sl[k0:k0 + K]
                k0 += K
                eff[b, d] = xgr[d][:, None] * beta_t[None, :]
    STAB_DIR.mkdir(parents=True, exist_ok=True)
    f = STAB_DIR / f"{ds}_{model}_split{split}.npz"
    np.savez_compressed(f, effects=eff, x_grid=xgr, t_grid=G, channels=np.array(ctx.lm.channels))
    return dict(file=str(f.relative_to(ROOT)), n_boot=n_boot, lr=lw[0], wd=lw[1])


# ------------------------------------------------------------------------------------------------------ records
def tuned(ds, split, model):
    for f in sorted(OUT.parent.glob("results_*.jsonl")):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if (r.get("status") == "ok" and r.get("job") == "single" and r["dataset"] == ds and r["split"] == split
                    and r["model"] == model and "lr" in r):
                return r["lr"], r["wd"]
    return None


def _append(rec):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def done():
    out = set()
    for f in sorted(OUT.parent.glob("results_*.jsonl")):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if r.get("status") == "ok":
                out.add((r["job"], r.get("dataset"), r.get("split"), r["model"]))
    return out


def run_job(job, ds, split, model=None):
    have = done()
    base = dict(round="R13", job=job, dataset=ds, split=split)
    if job == "single":
        ctx = None
        for mo in SINGLE_MODELS:
            if (job, ds, split, mo) in have:
                continue
            if ctx is None:
                ctx = RealCtx(ds, split)
            rec = dict(base, model=mo, n_cohort=ctx.n, n_train=len(ctx.tr), n_test=len(ctx.te),
                       events_test=int(ctx.d_te.sum()), t_L=ctx.lm.t_L, channels=list(ctx.lm.channels))
            t0 = time.time()
            try:
                run_single_cell(ctx, mo, rec)
                rec["status"] = "ok"
            except Exception as exc:
                rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])
            rec["seconds"] = round(time.time() - t0, 1)
            _append(rec)
            print(f"[{job} {ds} split {split} {mo}] {rec['status']} {rec['seconds']}s", flush=True)
    elif job in ("b3", "cmapss"):
        if all((job, ds, split, m) in have for m in (("b3_coxkan", "b3_cox") if job == "b3" else
                                                       ("b2s_aft", "b2s_aft_rc", "sum_aft", "fpca_aft"))):
            return
        recs = run_b3(ds, split, base) if job == "b3" else run_cmapss(split, base)
        for r in recs:
            _append(r)
            print(f"[{job} {ds} split {split} {r['model']}] {r['status']} {r.get('seconds')}s", flush=True)
    elif job == "stab":
        if (job, ds, split, model) in have:
            return
        rec = dict(base, model=model)
        t0 = time.time()
        for _ in range(240):                                                       # wait for the main-run record
            if tuned(ds, split, model) is not None:
                break
            time.sleep(30)
        try:
            rec.update(run_stab(ds, model, split))
            rec["status"] = "ok"
        except Exception as exc:
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])
        rec["seconds"] = round(time.time() - t0, 1)
        _append(rec)
        print(f"[stab {ds} {model} split {split}] {rec['status']} {rec['seconds']}s", flush=True)


def jobs():
    J = [("single", ds, s, None) for s in range(5) for ds in ("pbc_E2", "prothro", "aids", "pbc_E1")]
    J += [("b3", ds, s, None) for s in range(5) for ds in ("pbc_E2", "prothro", "aids")]
    J += [("cmapss", "cmapss_FD001", s, None) for s in range(5)]
    J += [("stab", ds, s, m) for m in ("b2s_cox", "b1_fpca1_cox") for ds in ("pbc_E2", "prothro", "aids") for s in range(5)]
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    if a.dataset.startswith("dev_"):
        assert "dev" in OUT.parent.name, "development data sets write only to a development directory"
    else:
        assert "dev" not in str(OUT), "real data need a non-development R13_OUT"
    run_job(a.job, a.dataset, a.split, a.model)


if __name__ == "__main__":
    main()
