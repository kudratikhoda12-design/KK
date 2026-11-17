"""D168 Stage A (development only; protocol docs/d168_afcm_reference_protocol.md, Section 4).

Datasets (historical development data only; no formal seed):
  ck22, ck914, ck966   the CK procedure (S-Add, DGP v2, n 2,000, Round-11 60/20/20 split, smoothed curves) on seeds
                       22 (CK-6c acceptance seed of Round 12b, later a development seed), 914 and 966 (development)
  mech11               the mechanics set: the same procedure with S-Add seed 11, n 400
Per dataset (one record in reports/d168/stageA/results.jsonl):
  A3  tied event times among training events and among all subjects
  A4  the mgcv reference AFCM (kanfae.models.afcm_mgcv, scripts/afcm_mgcv.R): identifiable and raw RISE of phi_1, phi_2
      at the true test curves (the existing metric, tests/test_round12_ck.shapes)
  A4  Python V3 (fit_afcm with validation, AFCM_D123, exactly the CK-6c call) and Python REML (fit_afcm
      selection="reml_per_channel", exactly the Round-22 afcm_reml call), refitted unchanged; their saved historical
      values are reported beside them where they exist
  A5  identifiability: max over grid times of |mean over training curves of F_d(t, X_i(t))| relative to the RMS of F_d,
      for every fit; and the identifiable relative difference between each Python surface and the reference surface
  A6  stationarity of the Python fits (largest penalised-gradient component, Newton decrement, condition number)
  A7  candidate repair (kanfae.models.afcm_constrained): at the V3 and at the REML smoothing parameters, the
      historical objective re-converged by Newton (unconstrained), and the constrained Newton fit from a cold start and
      warmed from the historical coefficients; acceptance: stopping rule reached and cold = warm (max |difference of the
      per-t-centred surfaces at the test points| < 1e-4)
Usage: python scripts/d168_stageA.py <dataset>   (A8 timing is scripts/d168_stageA_timing.py)
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
import run_round22 as R22  # noqa: E402
import test_round11 as R11  # noqa: E402
from test_round12_ck import shapes  # noqa: E402
from kanfae.data import fsim  # noqa: E402
from kanfae.models.afcm import AFCM, AFCM_D123, fit_afcm  # noqa: E402
from kanfae.models.afcm_constrained import fit_newton, stationarity  # noqa: E402
from kanfae.models.afcm_mgcv import AFCMReference  # noqa: E402

OUT = ROOT / "reports/d168/stageA/results.jsonl"
DATASETS = {"ck22": (22, 2000), "ck914": (914, 2000), "ck966": (966, 2000), "mech11": (11, 400)}
V3_KW = {k: v for k, v in AFCM_D123.items() if k not in ("lams_x", "lams_t")}


def data(seed, n):
    dat = fsim.generate("S-Add", n=n, seed=seed)
    tr, va, te = R11._split(n, seed)
    xg, lam = R11._smoothed(dat, tr)
    return dat, R11._pack(dat, xg, tr), R11._pack(dat, xg, va), R11._pack(dat, xg, te), dat.x_true_grid[te], lam


def ties(p):
    y, d = np.asarray(p["y"]), np.asarray(p["d"], bool)
    ev = y[d]
    return dict(tied_event_times=int(len(ev) - len(np.unique(ev))), tied_times_all=int(len(y) - len(np.unique(y))))


def ident(model, xg):
    n, G = xg.shape[:2]
    tt = np.broadcast_to(fsim.GRID[None, :], (n, G))
    out = []
    for d in range(xg.shape[2]):
        F = model.effect(d, xg[:, :, d], tt)
        out.append(float(np.max(np.abs(F.mean(0))) / max(np.sqrt(np.mean(F ** 2)), 1e-300)))
    return out


def vs_reference(model, ref, X):
    """Identifiable relative difference of a fitted surface from the reference surface, per channel (the reference
    plays the role of the truth in fsim.rise_identifiable)."""
    return {f"phi{d + 1}": fsim.rise_identifiable(lambda x, t, d=d: model.effect(d, x, t),
                                                  lambda x, t, d=d: ref.effect(d, x, t), X[:, :, d]) for d in (0, 1)}


def centred_maxdiff(a, b, X):
    n, G = X.shape[:2]
    tt = np.broadcast_to(fsim.GRID[None, :], (n, G))
    out = 0.0
    for d in range(X.shape[2]):
        fa, fb = a.effect(d, X[:, :, d], tt), b.effect(d, X[:, :, d], tt)
        fa, fb = fa - fa.mean(0, keepdims=True), fb - fb.mean(0, keepdims=True)
        out = max(out, float(np.max(np.abs(fa - fb))))
    return out


def historical(seed):
    h = {}
    for l in (ROOT / "reports/logs/round12_ck.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r.get("ck") == "CK-6c" and r.get("seed") == seed and r.get("run_id") == "round12b_ck_acceptance":
            h["v3_ck6c"] = r.get("shapes")
    f = ROOT / f"reports/round22/dev/results_afcmdev_{seed}.jsonl"
    if f.exists():
        h["reml_round22"] = json.loads(f.read_text().splitlines()[0])["afcm_reml"]["shapes"]
    return h


def a7(name, base, ptr, X, lam_x, lam_t):
    xg, z, y, d = ptr["xg"], ptr["z"], ptr["y"], ptr["d"]
    proto = AFCM(**V3_KW)
    out = {}
    t0 = time.time()
    mu, ru = fit_newton(proto, xg, z, y, d, lam_x, lam_t, constrain=False, theta0=(base.theta, base.gamma))
    out["unconstrained_newton_warm"] = dict(ru, shapes=shapes(mu, X), seconds=round(time.time() - t0, 1),
                                            centred_maxdiff_vs_historical=centred_maxdiff(mu, base, X))
    t0 = time.time()
    mc, rc = fit_newton(proto, xg, z, y, d, lam_x, lam_t, constrain=True)
    out["constrained_newton_cold"] = dict(rc, shapes=shapes(mc, X), seconds=round(time.time() - t0, 1))
    t0 = time.time()
    mw, rw = fit_newton(proto, xg, z, y, d, lam_x, lam_t, constrain=True, theta0=(base.theta, base.gamma))
    out["constrained_newton_warm"] = dict(rw, shapes=shapes(mw, X), seconds=round(time.time() - t0, 1))
    diff = centred_maxdiff(mc, mw, X)
    out["cold_warm_centred_maxdiff"] = diff
    out["accepted"] = bool(rc["converged"] and rw["converged"] and diff < 1e-4)
    return out, mc


def main(name):
    seed, n = DATASETS[name]
    rec = dict(dataset=name, seed=seed, n=n, role="development (historical data; not a formal seed)")
    t_all = time.time()
    try:
        dat, ptr, pva, pte, X, lam = data(seed, n)
        rec.update(n_train=len(ptr["y"]), n_test=len(pte["y"]), events_train=int(np.sum(ptr["d"])), smoothing_lambda=lam,
                   ties_train=ties(ptr), ties_all=dict(tied_times_all=int(len(dat.y) - len(np.unique(dat.y)))),
                   historical=historical(seed))
        t0 = time.time()
        ref = AFCMReference().fit(ptr, pte)
        rec["reference"] = dict(info=ref.info, seconds_wall=round(time.time() - t0, 1))
        if not ref.ok:
            raise RuntimeError(f"reference fit failed: {ref.info.get('error')}")
        rec["reference"].update(finite=ref.finite, shapes=shapes(ref, X), ident_rel=ident(ref, ptr["xg"]))
        fits = {}
        t0 = time.time()
        v3, _ = fit_afcm(ptr, pva, **AFCM_D123)
        fits["v3"] = (v3, time.time() - t0)
        t0 = time.time()
        rm, rrec = fit_afcm(ptr, None, selection="reml_per_channel", **AFCM_D123)
        fits["reml"] = (rm, time.time() - t0)
        for k, (m, secs) in fits.items():
            rec[k] = dict(lam_x=np.asarray(m.lam_x).tolist(), lam_t=np.asarray(m.lam_t).tolist(), seconds=round(secs, 1),
                          shapes=shapes(m, X), ident_rel=ident(m, ptr["xg"]), vs_reference=vs_reference(m, ref, X),
                          stationarity=stationarity(m, ptr["xg"], ptr["z"], ptr["y"], ptr["d"]))
        rec["reml"]["converged_fs"] = rrec.get("converged")
        for k, (m, _) in fits.items():
            res, mc = a7(name, m, ptr, X, m.lam_x, m.lam_t)
            res["constrained_cold_vs_reference"] = vs_reference(mc, ref, X)
            rec[f"a7_at_{k}_lambda"] = res
        rec["status"] = "ok"
    except Exception as e:
        rec.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
    rec["seconds_total"] = round(time.time() - t_all, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    print(name, rec["status"], rec["seconds_total"], rec.get("error", ""), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
