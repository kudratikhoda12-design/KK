"""Round 24 (D163): a Bayesian joint model (JMbayes2) as the missing standard comparator on the clinical cohorts.
Plan and rules: docs/round24_preregistration.md (committed before any formal job of this harness was run).

Jobs real_<ds>_<k>: ds in pbc_E2, pbc_E1, prothro, aids, renal, heart_valve; the Round-23 outer partitions k = 5..24 and
the Round-23 single-landmark cohorts (scripts/run_round13.RealCtx with the Round-22 landmarks), so subjects, train /
validation / test sets, censoring model and metrics are identical to the Round-23 records. The joint model is fitted
in R (scripts/jm_fit.R, project-local R in .rlocal/) on the TRAINING subjects only:
  longitudinal  per marker, two forms: "lin" lme(value ~ t, random = ~ t | id) and "ns2" lme(value ~ ns(t, 2),
                random = ~ t | id), on each training subject's full observed history (all visits up to the
                event or censoring time; standard joint-model practice; no validation or test data)
  survival      Cox submodel on the baseline covariates with delayed entry at the landmark t_L (every cohort member
                is event-free at t_L)
  association   current value of every marker; JMbayes2 MCMC: 3 chains, 6500 iterations, 1500 burn-in, seed 2400 + k
Validation and test subjects: dynamic prediction from their history up to t_L (the same window the landmark models
use), conditional on being event-free at t_L. "jm" is the form with the lower validation conditional IBS (ties -> "lin",
the simpler form; a form whose fit failed or gave non-finite validation curves is not eligible); both forms are recorded.
The NP-KAN-AFT, landmark Cox and blend results are read from the Round-23 records (reports/round23/results_*.jsonl),
not refitted; test R24-2 checks on one partition that the cohort and landmark Cox reproduce those records exactly.
Output: $R24_OUT (jsonl).
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("R13_OUT", str(ROOT / "reports/round24/unused_r13.jsonl"))
import run_round13 as R13  # noqa: E402

R13.SINGLE.update({"renal": 1.0, "heart_valve": 1.0})          # the Round-22 landmarks (D156), as Round 23
BASE = ROOT / "reports/round24"
OUT = Path(os.environ.get("R24_OUT", BASE / "dev/results_dev.jsonl"))
RSCRIPT = ROOT / ".rlocal/env/bin/Rscript"
CLIN = ("pbc_E2", "pbc_E1", "prothro", "aids", "renal", "heart_valve")
CLIN_SPLITS = range(5, 25)
TIME_SCALE = {"pbc_E2": 365.25, "pbc_E1": 365.25}               # days -> years inside R (numerical scale only)
MCMC = dict(n_chains=3, n_iter=6500, n_burnin=1500)             # longer than the JMbayes2 defaults (3500 / 500): D163 §3
FORMS = ("lin", "ns2")
SEED0 = 2400
N_GRID = 201
R_TIMEOUT = 4 * 3600


def jobs():
    return [f"real_{ds}_{k}" for k in CLIN_SPLITS for ds in CLIN]


def jm_input(ctx, ld, seed, mcmc=None):
    """The JSON handed to scripts/jm_fit.R. Times are divided by TIME_SCALE (pbc: years)."""
    sc = TIME_SCALE.get(ctx.ds, 1.0)
    lm = ctx.lm
    pos = {g: i for i, g in enumerate(ld.ids)}
    t_L, W = lm.t_L, lm.window
    ch = [f"m{j + 1}" for j in range(len(lm.channels))]
    zn = [f"z{j + 1}" for j in range(lm.z.shape[1])]

    def rows(idx, upto_landmark):
        out = []
        for i in idx:
            g = lm.ids[i]
            t = np.asarray(ld.times[pos[g]], float)
            v = np.asarray(ld.values[pos[g]], float)
            if upto_landmark:
                keep = (t >= t_L - W) & (t <= t_L)
            else:
                keep = t <= ld.y[pos[g]]
            keep &= np.isfinite(v).any(1)
            for tt, vv in zip(t[keep], v[keep]):
                out.append(dict(id=int(i), t=tt / sc, **{c: (float(x) if np.isfinite(x) else None) for c, x in zip(ch, vv)}))
        return out

    def surv(idx, test):
        return [dict(id=int(i), entry=t_L / sc, time=(t_L if test else t_L + ctx.y[i]) / sc,
                     event=0 if test else int(ctx.d[i]), **{c: float(x) for c, x in zip(zn, lm.z[i])}) for i in idx]
    grid = np.linspace(0.0, float(np.max(ctx.y_tr)), N_GRID)     # JMbayes2 predicts up to the training follow-up only
    return dict(channels=ch, z_names=zn, t_L=t_L / sc, pred_grid=(grid / sc).tolist(), seed=int(seed), forms=list(FORMS),
                train_long=rows(ctx.tr, False), train_surv=surv(ctx.tr, False),
                val_long=rows(ctx.va, True), val_surv=surv(ctx.va, True),
                test_long=rows(ctx.te, True), test_surv=surv(ctx.te, True), **(mcmc or MCMC)), grid


def run_r(inp, timeout=R_TIMEOUT):
    with tempfile.TemporaryDirectory() as d:
        fi, fo = Path(d) / "in.json", Path(d) / "out.json"
        fi.write_text(json.dumps(inp))
        p = subprocess.run([str(RSCRIPT), str(ROOT / "scripts/jm_fit.R"), str(fi), str(fo)], capture_output=True, text=True,
                           timeout=timeout, env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        if not fo.exists():
            return dict(status="failed", stage="R", error=(p.stderr or p.stdout)[-3000:], returncode=p.returncode)
        return json.loads(fo.read_text())


def surv_fn(grid, S):
    """Test-subject conditional survival curves (rows) on `grid` (0 .. longest training follow-up) -> S_fn(tt) by
    linear interpolation, held flat after the last grid point (times beyond the training follow-up)."""
    S = np.asarray(S, float)

    def f(tt):
        tt = np.atleast_1d(np.asarray(tt, float))
        return np.vstack([np.interp(tt, grid, s) for s in S])
    return f


def _curves(part, idx, grid):
    assert part["ids"] == [int(i) for i in idx]
    S = np.array([[np.nan if v is None or v == "NA" else v for v in row] for row in part["S"]], float).reshape(len(idx), -1)
    return S, surv_fn(grid, S)


def job_real(ds, k, mcmc=None):
    ctx = R13.RealCtx(ds, k)
    ld = R13._load(ds)
    inp, grid = jm_input(ctx, ld, SEED0 + k, mcmc)
    rec = dict(job="real", dataset=ds, split=k, n=int(ctx.n), n_train=len(ctx.tr), n_val=len(ctx.va), n_test=len(ctx.te),
               events_test=int(ctx.d_te.sum()), train_long_rows=len(inp["train_long"]), test_long_rows=len(inp["test_long"]),
               seed=inp["seed"], mcmc=dict(mcmc or MCMC), models={})
    t0 = time.time()
    out = run_r(inp)
    rec["r_seconds_total"] = round(time.time() - t0, 1)
    if out.get("status") != "ok":
        for f in FORMS:
            rec["models"][f"jm_{f}"] = dict(status="failed", stage=out.get("stage"), error=out.get("error"))
        rec["models"]["jm"] = dict(status="failed", error="no form fitted")
        return rec
    rec["versions"] = out.get("versions")
    y_va, d_va = ctx.y[ctx.va], ctx.d[ctx.va].astype(bool)
    tgv, tg = R13.M.ibs_time_grid(y_va, ctx.y_tr), R13.M.ibs_time_grid(ctx.y_te, ctx.y_tr)
    for f in FORMS:
        o = out["forms"].get(f, {})
        if o.get("status") != "ok":
            rec["models"][f"jm_{f}"] = dict(status="failed", error=o.get("error"), seconds=o.get("seconds"))
            continue
        Sv, Sv_fn = _curves(o["val"], ctx.va, grid)
        St, St_fn = _curves(o["test"], ctx.te, grid)
        fin_v = bool(np.all(np.isfinite(Sv)))
        val = float(R13.M.ibs_conditional(Sv_fn(tgv), tgv, y_va, d_va, ctx.lm.z[ctx.va], ctx.G_cond)[0]) if fin_v else None
        rec["models"][f"jm_{f}"] = dict(R13.real_metrics(ctx, St_fn, -np.mean(St_fn(tg), 1)), val_ibs_cond=val,
                                         finite=bool(np.all(np.isfinite(St))), finite_val=fin_v,
                                         n_nan=int(np.isnan(St).sum()), rhat_alphas=o.get("rhat_alphas"),
                                         seconds=o.get("seconds"))
    pick = choose_form(rec["models"])
    rec["models"]["jm"] = (dict(rec["models"][f"jm_{pick}"], chosen=pick) if pick else
                           dict(status="failed", error="no eligible form"))
    return rec


def choose_form(models):
    """The form with the lower validation conditional IBS; ties (<= 1e-12) go to the earlier form in FORMS ("lin", the
    simpler one). A form without a validation score (failed fit or non-finite validation curves) is not eligible."""
    ok = [f for f in FORMS if models.get(f"jm_{f}", {}).get("val_ibs_cond") is not None]
    if not ok:
        return None
    best = min(models[f"jm_{f}"]["val_ibs_cond"] for f in ok)
    return next(f for f in ok if models[f"jm_{f}"]["val_ibs_cond"] <= best + 1e-12)


def run_job(name, mcmc=None):
    t0 = time.time()
    try:
        p = name.split("_")
        if p[0] != "real":
            raise ValueError(name)
        rec = job_real("_".join(p[1:-1]), int(p[-1]), mcmc)
        rec["status"] = "ok" if rec["models"]["jm"].get("status") != "failed" else "fit_failed"
    except Exception as e:
        rec = dict(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    rec.update(job_name=name, wall_seconds=round(time.time() - t0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    return rec


def done(base=None):
    """Jobs with a final record (ok, or a recorded fit failure: a failed fit is never refitted, D163)."""
    seen = set()
    for f in (base or OUT.parent).glob("results_*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip() and json.loads(line).get("status") in ("ok", "fit_failed"):
                seen.add(json.loads(line)["job_name"])
    return seen


if __name__ == "__main__":
    for j in sys.argv[1:]:
        r = run_job(j)
        print(j, r["status"], r["wall_seconds"], r.get("error", "") or r.get("models", {}).get("jm", {}).get("error", ""), flush=True)
