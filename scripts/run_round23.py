"""Round 23 (D160): where do functional KANs help, and can a blend keep the best of both?
Plan and rules: docs/round23_preregistration.md (committed before any formal job of this harness was run).

Jobs (one record per job; failures recorded with tracebacks; resumable):
  cm_<fd>_<k>      C-MAPSS FD002 / FD004 Option B (never analysed in this project before Round 23), outer engine splits
                   k = 0..9 (split_groups(units, k), 60/20/20). Six operating conditions: per-condition z-scoring of the
                   sensors (cmapss.condition_normalise, k-means on the training engines' settings), then the Round-13
                   Option-B pipeline (30-cycle windows every 10 cycles, u = s/100, target T - s, every window an event,
                   B-spline smoothing lam 1e-4). Models:
                     b2s_npaft            NP-KAN-AFT B2-S 3-seed ensemble (Round 20, rank_km, tau 0.05; D112 protocol)
                     landmark_cox_raw     Cox on the last raw sensor values + u
                     landmark_cox_smooth  Cox on the last value of the smoothed curves + u
                     landmark_cox         whichever of the two has the lower validation IBS (rule fixed in advance)
                     sum_aft, fpca_aft    Round-1 summary / FPCA AFT baselines (grid tuned on validation IBS)
                     stack                w * S_landmark + (1 - w) * S_npaft, w in {0, .25, .5, .75, 1} chosen on the
                                          validation IBS (ties -> the larger w, i.e. towards the simpler model)
  real_<ds>_<k>    ds in pbc_E2, pbc_E1, prothro, aids, renal, heart_valve; NEW outer partitions k = 5..24 (Rounds 13-22
                   used k = 0..4 only). The Round-13 single-landmark harness (RealCtx). Models: b2s_npaft, landmark_cox
                   (Cox on the last observed values + baseline covariates), stack (as above; w chosen on the validation
                   conditional IBS). renal only: b2s_npaft_sparse3, the same NP-KAN-AFT fitted after thinning every
                   history to three visits (first, middle and last observed visit in the window; front end refitted).
Output: $R23_OUT (jsonl).
"""
import dataclasses
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
os.environ.setdefault("R13_OUT", str(ROOT / "reports/round23/unused_r13.jsonl"))
os.environ.setdefault("R21_OUT", str(ROOT / "reports/round23/unused_r21.jsonl"))
import run_round13 as R13  # noqa: E402
import run_round21_real as R21  # noqa: E402
from kanfae.data.cmapss import condition_normalise, informative_sensors, option_b, read_fd  # noqa: E402
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.evaluation import metrics as M  # noqa: E402
from kanfae.functional.basis import BSplineBasis  # noqa: E402
from kanfae.functional.smoothing import smooth  # noqa: E402
from kanfae.models.fbaselines import LinearCox  # noqa: E402
from kanfae.models.fkan import FKANEnsemble  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
BASE = ROOT / "reports/round23"
OUT = Path(os.environ.get("R23_OUT", BASE / "dev/results_dev.jsonl"))
CLIN = ("pbc_E2", "pbc_E1", "prothro", "aids", "renal", "heart_valve")
CLIN_SPLITS = range(5, 25)
CM = ("FD002", "FD004")
CM_SPLITS = range(10)
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
N_COND = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}     # operating conditions (C-MAPSS readme); FD001/FD003 = dev mechanics only
NP = R21.FAMILY["npaft"]
ENS_KW = {}                                   # tests may shrink the budgets (never used by the formal jobs)
R13.SINGLE.update({"renal": 1.0, "heart_valve": 1.0})          # the Round-22 landmarks (D156), unchanged


def jobs():
    return [f"cm_{fd}_{k}" for k in CM_SPLITS for fd in CM] + [f"real_{ds}_{k}" for k in CLIN_SPLITS for ds in CLIN]


# ------------------------------------------------------------------------------------------------------- tools
def choose_w(score):
    """score(w) -> validation IBS of the blend. Returns (w, table). Ties (|diff| <= 1e-12) go to the larger w."""
    tab = [(float(score(w)), w) for w in W_GRID]
    best = min(s for s, _ in tab)
    w = max(w for s, w in tab if s <= best + 1e-12)
    return w, [dict(w=w_, val_ibs=s) for s, w_ in tab]


def blend(S_a, S_b, w):
    return lambda tt: w * S_a(tt) + (1.0 - w) * S_b(tt)


def blend_risk(S_fn, tg):
    """Risk score of a blended curve: minus its mean survival over the IBS time grid (higher = earlier events)."""
    return -np.mean(S_fn(tg), 1)


def thin(times, values, n_keep=3):
    """Keep n_keep observed visits per subject: first, last and evenly spaced in between (by visit index)."""
    out_t, out_v = [], []
    for t, v in zip(times, values):
        t, v = np.asarray(t, float), np.asarray(v, float)
        o = np.argsort(t, kind="stable")
        t, v = t[o], v[o]
        rows = np.nonzero(np.isfinite(v).any(1))[0]
        k = min(n_keep, len(rows))
        idx = rows[np.unique(np.round(np.linspace(0, len(rows) - 1, k)).astype(int))]
        out_t.append(t[idx]); out_v.append(v[idx])
    return out_t, out_v


def ens_np(tr, va):
    return FKANEnsemble(kind="b2s", **NP, **ENS_KW).fit(tr, va)


def ens_curves(e, P):
    return (lambda tt: np.mean([m.predict_survival(P["xg"], P["z"], tt) for m in e.members], 0),
            np.mean([m.predict_risk(P["xg"], P["z"]) for m in e.members], 0))


def finite(S_fn, y):
    return bool(np.all(np.isfinite(S_fn(np.array([np.median(y)])))))


# --------------------------------------------------------------------------------------------- clinical cohorts
def job_real(ds, k):
    ctx = R13.RealCtx(ds, k)
    xg, last, z = ctx.front("bspline")["xg"], ctx.lastv(), ctx.lm.z
    y, d = ctx.y, ctx.d.astype(float)
    pk = lambda idx, xg=xg: dict(xg=xg[idx], z=z[idx], y=y[idx], d=d[idx])  # noqa: E731
    P = {"tr": pk(ctx.tr), "va": pk(ctx.va), "te": pk(ctx.te)}
    nv = [int(np.isfinite(v).any(1).sum()) for v in ctx.lm.values]
    rec = dict(job="real", dataset=ds, split=k, n=int(ctx.n), n_train=len(ctx.tr), n_val=len(ctx.va), n_test=len(ctx.te),
               events_test=int(ctx.d_te.sum()), visits_median=float(np.median(nv)), visits_mean=float(np.mean(nv)),
               models={}, stack={})
    y_va, d_va = y[ctx.va], ctx.d[ctx.va].astype(bool)
    tgv = M.ibs_time_grid(y_va, ctx.y_tr)

    def val_ibs(S_fn):
        return M.ibs_conditional(S_fn(tgv), tgv, y_va, d_va, z[ctx.va], ctx.G_cond)[0]
    tg = M.ibs_time_grid(ctx.y_te, ctx.y_tr)
    # NP-KAN-AFT
    t0 = time.time()
    e = ens_np(P["tr"], P["va"])
    S_np, r_np = ens_curves(e, P["te"])
    S_np_va, _ = ens_curves(e, P["va"])
    rec["models"]["b2s_npaft"] = dict(R13.real_metrics(ctx, S_np, r_np), lr=e.lr, wd=e.wd, finite=finite(S_np, ctx.y_te),
                                      val_ibs_cond=float(val_ibs(S_np_va)), seconds=round(time.time() - t0, 1))
    # landmark Cox on the last observed values
    t0 = time.time()
    X = np.hstack([last, z])
    lc = LinearCox().fit(X[ctx.tr], ctx.y_tr, ctx.d_tr)
    S_lm = lambda tt: lc.predict_survival(X[ctx.te], tt)  # noqa: E731
    S_lm_va = lambda tt: lc.predict_survival(X[ctx.va], tt)  # noqa: E731
    rec["models"]["landmark_cox"] = dict(R13.real_metrics(ctx, S_lm, lc.predict_risk(X[ctx.te])), finite=finite(S_lm, ctx.y_te),
                                         val_ibs_cond=float(val_ibs(S_lm_va)), seconds=round(time.time() - t0, 1))
    # blend
    w, tab = choose_w(lambda w: val_ibs(blend(S_lm_va, S_np_va, w)))
    S_st = blend(S_lm, S_np, w)
    rec["models"]["stack"] = dict(R13.real_metrics(ctx, S_st, blend_risk(S_st, tg)), finite=finite(S_st, ctx.y_te), w=w)
    rec["stack"] = dict(w=w, table=tab)
    if ds == "renal":                                  # trajectory-thinning experiment (H-R23b)
        t0 = time.time()
        ctx_s = R13.RealCtx(ds, k)
        tt_, vv_ = thin(ctx_s.lm.times, ctx_s.lm.values, 3)
        ctx_s.lm = dataclasses.replace(ctx_s.lm, times=tt_, values=vv_)
        xg_s = ctx_s.front("bspline")["xg"]
        Ps = {n: pk(i, xg_s) for n, i in (("tr", ctx.tr), ("va", ctx.va), ("te", ctx.te))}
        es = ens_np(Ps["tr"], Ps["va"])
        S_s, r_s = ens_curves(es, Ps["te"])
        rec["models"]["b2s_npaft_sparse3"] = dict(R13.real_metrics(ctx, S_s, r_s), lr=es.lr, wd=es.wd,
                                                  finite=finite(S_s, ctx.y_te), seconds=round(time.time() - t0, 1),
                                                  visits_median=float(np.median([len(t) for t in tt_])))
    return rec


# ------------------------------------------------------------------------------------------------- C-MAPSS
def cm_data(k, fd):
    df = read_fd(ROOT / "data/raw/cmapss", fd)[0]
    units = np.sort(df.unit.unique())
    utr, uva, ute = split_groups(units, k)
    dn, info = condition_normalise(df, units[utr], n_conditions=N_COND[fd])
    sensors = informative_sensors(dn[dn.unit.isin(units[utr])], units[utr])
    obs = {n: option_b(dn, units[u], sensors) for n, u in (("tr", utr), ("va", uva), ("te", ute))}
    basis = BSplineBasis(12, 3, (0.0, 1.0))
    P = {}
    for n, o in obs.items():
        c = smooth(o.times, o.values, o.mask, basis, lam=1e-4)
        xg = np.einsum("ndk,gk->ngd", c, basis(R13.G))
        P[n] = dict(xg=xg, z=o.u, y=o.y_obs, d=o.delta.astype(float),
                    last=o.values[np.arange(len(o)), o.n_obs - 1, :], last_s=xg[:, -1, :])
    return P, sensors, obs, info, {n: len(u) for n, u in (("tr", utr), ("va", uva), ("te", ute))}


def job_cm(fd, k):
    P, sensors, obs, info, n_eng = cm_data(k, fd)
    y_tr, d_tr = P["tr"]["y"], P["tr"]["d"].astype(bool)
    y_va, d_va = P["va"]["y"], P["va"]["d"].astype(bool)
    y_te, d_te = P["te"]["y"], P["te"]["d"].astype(bool)
    tau = float(np.quantile(y_tr, 0.9))
    tg, tgv = M.ibs_time_grid(y_te, y_tr), M.ibs_time_grid(y_va, y_tr)

    def metric(S_fn, risk):                                  # the Round-13 C-MAPSS metrics
        return dict(ibs=float(M.ibs_marginal(S_fn(tg), tg, y_te, d_te, y_tr, d_tr)[0]),
                    harrell_c=float(M.harrell_c(y_te, d_te, risk)), uno_c=float(M.uno_c(y_tr, d_tr, y_te, d_te, risk, tau)),
                    dcal_p=float(M.d_calibration(np.diag(S_fn(y_te)), d_te)[1]))

    def val_ibs(S_fn):
        return float(M.ibs_marginal(S_fn(tgv), tgv, y_va, d_va, y_tr, d_tr)[0])
    rec = dict(job="cm", dataset=f"cmapss_{fd}_B", fd=fd, split=k, sensors=list(sensors), conditions=info,
               engines=n_eng, n_train=len(y_tr), n_val=len(y_va), n_test=len(y_te), models={}, stack={})
    # NP-KAN-AFT (fit cache: resumable after a container restart)
    R21.ENS_CACHE = BASE / "ckpt" / f"cm_{fd}_{k}_fits"
    t0 = time.time()
    e, reloaded = R21.cached("full_npaft", lambda: ens_np(P["tr"], P["va"]))
    S_np, r_np = ens_curves(e, P["te"])
    S_np_va, _ = ens_curves(e, P["va"])
    rec["models"]["b2s_npaft"] = dict(metric(S_np, r_np), lr=e.lr, wd=e.wd, finite=finite(S_np, y_te),
                                      val_ibs=val_ibs(S_np_va), seconds=round(time.time() - t0, 1), reloaded=reloaded)
    # landmark Cox: raw and smoothed last values; the one with the lower validation IBS is "landmark_cox"
    lms = {}
    for name, key in (("landmark_cox_raw", "last"), ("landmark_cox_smooth", "last_s")):
        t0 = time.time()
        Xs = {n: np.hstack([P[n][key], P[n]["z"]]) for n in ("tr", "va", "te")}
        lc = LinearCox().fit(Xs["tr"], y_tr, d_tr.astype(float))
        S_t = lambda tt, lc=lc, X=Xs["te"]: lc.predict_survival(X, tt)  # noqa: E731
        S_v = lambda tt, lc=lc, X=Xs["va"]: lc.predict_survival(X, tt)  # noqa: E731
        lms[name] = (S_t, S_v, lc.predict_risk(Xs["te"]))
        rec["models"][name] = dict(metric(S_t, lms[name][2]), finite=finite(S_t, y_te), val_ibs=val_ibs(S_v),
                                   seconds=round(time.time() - t0, 1))
    pick = min(lms, key=lambda n: rec["models"][n]["val_ibs"])
    S_lm, S_lm_va, r_lm = lms[pick]
    rec["models"]["landmark_cox"] = dict(rec["models"][pick], chosen=pick)
    # blend
    w, tab = choose_w(lambda w: val_ibs(blend(S_lm_va, S_np_va, w)))
    S_st = blend(S_lm, S_np, w)
    rec["models"]["stack"] = dict(metric(S_st, blend_risk(S_st, tg)), finite=finite(S_st, y_te), w=w, landmark=pick)
    rec["stack"] = dict(w=w, table=tab, landmark=pick)
    # Round-1 summary / FPCA AFT baselines (as Rounds 13 and 22)
    from kanfae.experiments.runner import GRIDS
    from kanfae.models.pipelines import make_model
    for model in ("sum_aft", "fpca_aft"):
        t0 = time.time()
        try:
            tun = []
            for cfg in GRIDS[model]:
                mm = make_model(model, **cfg).fit(obs["tr"], obs["va"], seed=0)
                tun.append((float(M.ibs_marginal(mm.predict_survival(obs["va"], tgv), tgv, y_va, d_va, y_tr, d_tr)[0]), cfg, mm))
            vib, cfg, m = min(tun, key=lambda t: t[0])
            rec["models"][model] = dict(metric(lambda tt, m=m: m.predict_survival(obs["te"], tt), m.predict_risk(obs["te"])),
                                        config=cfg, val_ibs=vib, seconds=round(time.time() - t0, 1))
        except Exception as exc:                       # a failed baseline is recorded, never refitted differently
            rec["models"][model] = dict(status="failed", error=f"{type(exc).__name__}: {exc}",
                                        traceback=traceback.format_exc()[-2000:], seconds=round(time.time() - t0, 1))
    return rec


# ----------------------------------------------------------------------------------------------------- driver
def run_job(name):
    t0 = time.time()
    R21.ENS_CACHE = None
    try:
        p = name.split("_")
        if p[0] == "cm":
            rec = job_cm(p[1], int(p[2]))
        elif p[0] == "real":
            rec = job_real("_".join(p[1:-1]), int(p[-1]))
        else:
            raise ValueError(name)
        rec["status"] = "ok"
    except Exception as e:
        rec = dict(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    rec.update(job_name=name, wall_seconds=round(time.time() - t0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    if rec["status"] == "ok" and R21.ENS_CACHE is not None:
        import shutil                                   # the job's record is written: its fit cache is not needed
        shutil.rmtree(R21.ENS_CACHE, ignore_errors=True)
    R21.ENS_CACHE = None
    return rec


def done(base=None):
    seen = set()
    for f in (base or OUT.parent).glob("results_*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip() and json.loads(line).get("status") == "ok":
                seen.add(json.loads(line)["job_name"])
    return seen


if __name__ == "__main__":
    for j in sys.argv[1:]:
        r = run_job(j)
        print(j, r["status"], r["wall_seconds"], r.get("error", ""), flush=True)
