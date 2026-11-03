"""Round 21 (D153): real-data evaluation of the Round-20 NP-KAN-AFT and the two-half cross-confirmed anchor.
Plan and rules: docs/round21_preregistration.md (committed before any job of this harness was run on a full budget).

Jobs (one record per job; failures recorded with tracebacks; resumable):
  real_<ds>_<k>   ds in pbc_E2, pbc_E1, prothro, aids; outer split k = 0..4 (the Round-13 harness: RealCtx, single
                  landmark, B-spline front end, split_groups 60/20/20 stratified on the event, conditional/marginal IPCW
                  IBS, Harrell C, Uno C, D-calibration). Models (D112 3-seed B2-S ensembles, lr x wd tuned on the
                  split's validation set, as Rounds 13/15):
                    b2s_cox, b2s_waft (Weibull KAN-AFT = Round-13 b2s_aft), b2s_lnaft (log-normal), b2s_npaft
                    (NP-KAN-AFT, rank_km, tau 0.05);
                    b2s_anchor_{cox,lnaft,npaft}_cross (CrossConfirmRefitAnchor; halves h1/h2 = run_round18.split_val
                    (va, delta, k); margins locked by D150: Cox 0.02 absolute, log-normal 0 absolute, NP-AFT 0 relative;
                    refit = finetune; not confirmed -> the full-validation B2-S exactly, checked on the test split);
                    symbolic distillation diagnostic on the seed-0 member of b2s_npaft and b2s_cox
                    (fkan_surv.distill_and_replace: stages 1-2, guarded substitution, fixed mu = 10, NaN-safe refit,
                    validation-loss rollback).
  cmapss_<k>      C-MAPSS FD001 Option B as Round 13 (100 training engines split by engine, 30-cycle windows every 10
                  cycles, u = s/100, target T - s, every window an event): b2s_cox, b2s_waft, b2s_lnaft, b2s_npaft,
                  b2s_anchor_npaft_cross (halves = the validation engines split in two, generator (k, 21)).
Output: $R21_OUT (jsonl).
"""
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
os.environ.setdefault("R13_OUT", str(ROOT / "reports/round21/unused_r13.jsonl"))
import run_round13 as R13  # noqa: E402
import run_round18 as R18  # noqa: E402
from kanfae.evaluation import metrics as M  # noqa: E402
from kanfae.models.fkan import FKANEnsemble  # noqa: E402
from kanfae.models.fkan_surv import distill_and_replace  # noqa: E402
from kanfae.models.fres import CrossConfirmRefitAnchor  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
OUT = Path(os.environ.get("R21_OUT", ROOT / "reports/round21/dev/results_dev.jsonl"))
DATASETS = ("pbc_E2", "pbc_E1", "prothro", "aids")
SPLITS = range(5)
FAMILY = {"cox": dict(loss="cox"), "waft": dict(loss="aft", family="weibull"),
          "lnaft": dict(loss="aft", family="lognormal"),
          "npaft": dict(loss="aft", family="np_aft", np_objective="rank_km", np_tau=0.05)}
MARGIN = {"cox": (0.02, False), "lnaft": (0.0, False), "npaft": (0.0, True)}        # D150 (Round 20), locked
CROSS_REAL = ("cox", "lnaft", "npaft")
CROSS_CMAPSS = ("npaft",)
ENS_KW = {}                                          # tests may shrink the budgets (never used by the formal jobs)
CROSS_KW = {}


def jobs():
    return [f"real_{ds}_{k}" for k in SPLITS for ds in DATASETS] + [f"cmapss_{k}" for k in SPLITS]


def ens(f, tr, va):
    return FKANEnsemble(kind="b2s", **FAMILY[f], **ENS_KW).fit(tr, va)


def surv_risk(members, te, last=None):
    if last is None:
        return (lambda tt: np.mean([m.predict_survival(te["xg"], te["z"], tt) for m in members], 0),
                np.mean([m.predict_risk(te["xg"], te["z"]) for m in members], 0))
    return (lambda tt: members.predict_survival(te["xg"], te["last"], te["z"], tt),
            members.predict_risk(te["xg"], te["last"], te["z"]))


def dcal_chi2(S_fn, y, d):
    c2, p, _ = M.d_calibration(np.diag(S_fn(y)), d)
    return dict(dcal_chi2=float(c2), dcal_p_check=float(p))


def fit_family(f, P, metric, rec, cross=True, halves=None):
    """B2-S ensemble on the whole validation set (the unanchored model b2s_<f>), and, if cross, the cross-confirmed
    anchor with its h1/h2 reference ensembles. P: dict tr, va, h1, h2, te packs (with 'last')."""
    t0 = time.time()
    full = ens(f, P["tr"], P["va"])
    S_fn, risk = surv_risk(full.members, P["te"])
    rec["models"][f"b2s_{f}"] = dict(metric(S_fn, risk), **dcal_chi2(S_fn, P["te"]["y"], P["te"]["d"]),
                                     lr=full.lr, wd=full.wd, seconds=round(time.time() - t0, 1),
                                     finite=bool(np.all(np.isfinite(S_fn(np.array([np.median(P["te"]["y"])]))))))
    if not cross:
        return full, None
    t1 = time.time()
    h1, h2 = ens(f, P["tr"], P["h1"]), ens(f, P["tr"], P["h2"])
    t_ref = time.time() - t1
    margin, rel = MARGIN[f]
    t2 = time.time()
    m = CrossConfirmRefitAnchor(margin=margin, relative_margin=rel, **FAMILY[f], **CROSS_KW).fit(
        P["tr"], P["h1"], P["h2"], h1, h2, full)
    t_gate = time.time() - t2
    name = f"b2s_anchor_{f}_cross"
    S_c, r_c = surv_risk(m, P["te"], last=True)
    rec["models"][name] = dict(metric(S_c, r_c), **dcal_chi2(S_c, P["te"]["y"], P["te"]["d"]),
                               seconds=round(time.time() - t1, 1), seconds_refs=round(t_ref, 1),
                               seconds_gate=round(t_gate, 1))
    tg = M.ibs_time_grid(P["te"]["y"], P["tr"]["y"])
    gate = dict(confirmed=bool(m.confirmed), margin=margin, relative_margin=rel, dirs=m.dirs,
                traj_on_h1=m._alts[0][1].residual_on, last_on_h1=m._alts[0][1].last_on)
    if not m.confirmed:                                      # exact fallback check on the test split
        gate["fallback_max_abs_diff_S"] = float(np.max(np.abs(S_c(tg) - S_fn(tg))))
        gate["fallback_max_abs_diff_risk"] = float(np.max(np.abs(r_c - risk)))
    alt = m.refit_members() if not m.confirmed else None
    S_r, r_r = ((lambda tt: m.predict_survival(P["te"]["xg"], P["te"]["last"], P["te"]["z"], tt, alt),
                 m.predict_risk(P["te"]["xg"], P["te"]["last"], P["te"]["z"], alt)) if alt is not None else (S_c, r_c))
    rec["alternatives"][name] = dict(refit=metric(S_r, r_r))
    rec["gates"][name] = gate
    return full, m


def distill(f, full, P, metric, rec):
    t0 = time.time()
    m0 = full.members[0]
    S0, r0 = surv_risk([m0], P["te"])
    before = metric(S0, r0)
    m, dr = distill_and_replace(m0, return_edges=True)
    S1, r1 = surv_risk([m], P["te"])
    after = metric(S1, r1)
    rec["symbolic"][f"b2s_{f}"] = dict(dr, seed0_before=before, after=after,
                                       replacement_rate=dr["n_substituted"] / max(dr["n_edges"], 1),
                                       seconds=round(time.time() - t0, 1))


def job_real(ds, k):
    ctx = R13.RealCtx(ds, k)
    xg, last, z = ctx.front("bspline")["xg"], ctx.lastv(), ctx.lm.z
    y, d = ctx.y, ctx.d.astype(float)
    h1, h2 = R18.split_val(ctx.va, ctx.d.astype(int), k)
    pk = lambda idx: dict(xg=xg[idx], z=z[idx], y=y[idx], d=d[idx], last=last[idx])  # noqa: E731
    P = {"tr": pk(ctx.tr), "va": pk(ctx.va), "h1": pk(h1), "h2": pk(h2), "te": pk(ctx.te)}
    rec = dict(job="real", dataset=ds, split=k, channels=list(ctx.lm.channels), n=int(ctx.n), n_train=len(ctx.tr),
               n_val=len(ctx.va), n_h1=len(h1), n_h2=len(h2), n_test=len(ctx.te), events_test=int(ctx.d_te.sum()),
               models={}, alternatives={}, gates={}, symbolic={})

    def metric(S_fn, risk):
        return R13.real_metrics(ctx, S_fn, risk)
    fulls = {}
    for f in ("cox", "waft", "lnaft", "npaft"):
        fulls[f], _ = fit_family(f, P, metric, rec, cross=f in CROSS_REAL)
    for f in ("npaft", "cox"):
        distill(f, fulls[f], P, metric, rec)
    return rec


def cmapss_data(k):
    from kanfae.data.cmapss import informative_sensors, option_b, read_fd
    from kanfae.data.splits import split_groups
    from kanfae.functional.basis import BSplineBasis
    from kanfae.functional.smoothing import smooth
    df = read_fd(ROOT / "data/raw/cmapss", "FD001")[0]
    units = np.sort(df.unit.unique())
    utr, uva, ute = split_groups(units, k)
    sensors = informative_sensors(df[df.unit.isin(units[utr])], units[utr])
    obs = {n: option_b(df, units[u], sensors) for n, u in (("tr", utr), ("va", uva), ("te", ute))}
    basis = BSplineBasis(12, 3, (0.0, 1.0))
    P = {}
    for n, o in obs.items():
        c = smooth(o.times, o.values, o.mask, basis, lam=1e-4)
        xg = np.einsum("ndk,gk->ngd", c, basis(R13.G))
        lastv = o.values[np.arange(len(o)), o.n_obs - 1, :]
        P[n] = dict(xg=xg, z=o.u, y=o.y_obs, d=o.delta.astype(float), last=lastv, group=o.meta["group"])
    g = np.unique(P["va"]["group"])
    perm = np.random.default_rng([k, 21]).permutation(g)
    g1 = set(perm[len(perm) // 2:].tolist())
    in1 = np.array([x in g1 for x in P["va"]["group"]])
    for n, msk in (("h1", in1), ("h2", ~in1)):
        P[n] = {kk: v[msk] for kk, v in P["va"].items()}
    groups = {n: P[n].pop("group") for n in P}
    return P, sensors, groups


def job_cmapss(k):
    P, sensors, groups = cmapss_data(k)
    y_tr, d_tr = P["tr"]["y"], P["tr"]["d"].astype(bool)
    y_te, d_te = P["te"]["y"], P["te"]["d"].astype(bool)
    tau = float(np.quantile(y_tr, 0.9))
    tg = M.ibs_time_grid(y_te, y_tr)

    def metric(S_fn, risk):                                  # the Round-13 C-MAPSS metrics
        return dict(ibs=float(M.ibs_marginal(S_fn(tg), tg, y_te, d_te, y_tr, d_tr)[0]),
                    harrell_c=float(M.harrell_c(y_te, d_te, risk)), uno_c=float(M.uno_c(y_tr, d_tr, y_te, d_te, risk, tau)),
                    dcal_p=float(M.d_calibration(np.diag(S_fn(y_te)), d_te)[1]))
    rec = dict(job="cmapss", dataset="cmapss_FD001_B", split=k, sensors=list(sensors), n_train=len(y_tr),
               n_val=len(P["va"]["y"]), n_h1=len(P["h1"]["y"]), n_h2=len(P["h2"]["y"]), n_test=len(y_te),
               engines={n: int(len(np.unique(g))) for n, g in groups.items()},
               models={}, alternatives={}, gates={}, symbolic={})
    ck = OUT.parent / "ckpt" / f"cmapss_{k}.json"           # execution checkpoint (2026-11-03): one family at a time
    if ck.exists():
        rec = json.loads(ck.read_text())
    for f in ("cox", "waft", "lnaft", "npaft"):
        if f"b2s_{f}" in rec["models"]:
            continue                                         # fits are seeded per network and full-batch: skipping a
        fit_family(f, P, metric, rec, cross=f in CROSS_CMAPSS)   # finished family does not change the others
        ck.parent.mkdir(parents=True, exist_ok=True)
        ck.write_text(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    rec["resumed_from_checkpoint"] = bool(rec.get("resumed_from_checkpoint", False))
    return rec


def run_job(name):
    t0 = time.time()
    try:
        parts = name.split("_")
        rec = job_cmapss(int(parts[1])) if parts[0] == "cmapss" else job_real("_".join(parts[1:-1]), int(parts[-1]))
        rec["status"] = "ok"
    except Exception as e:
        rec = dict(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    rec.update(job_name=name, wall_seconds=round(time.time() - t0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    return rec


def done():
    seen = set()
    for f in OUT.parent.glob("results_lane*.jsonl"):
        for l in f.read_text().splitlines():
            if l.strip() and json.loads(l).get("status") == "ok":
                seen.add(json.loads(l)["job_name"])
    return seen


if __name__ == "__main__":
    for j in sys.argv[1:]:
        r = run_job(j)
        print(j, r["status"], r["wall_seconds"], r.get("error", ""), flush=True)
