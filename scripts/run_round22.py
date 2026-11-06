"""Round 22 (D156): per-channel REML AFCM (CK-6d, H-R22a), untouched external longitudinal cohorts (H-R22b) and the
multi-law NP-KAN-AFT simulation (H-R22c). Plan and rules: docs/round22_preregistration.md (committed before any job of
this harness runs on a registered seed). One record per job, failures included (tracebacks kept); resumable.

Jobs:
  afcmdev_<s>         s in 22, 914, 966 (development gate, CK-6c procedure: S-Add, DGP v2, n 2,000, R11 split and
                      smoothing): afcm_reml (fit_afcm selection="reml_per_channel", AFCM_D123 basis/penalties/grid) and
                      afcm_reml_ens (fit_afcm_reml_bagged, B 3, seed 0); the LAML pilot is the Round-15 C2 model
                      (reproduction check against reports/round15/dev records).
  hr22a_<s>           s in 3000-3019 (H-R22a; only if the gate passed and CK-6d passed on seed 24): the GO variant on
                      the Round-12 S-Add replicate (run_round12.Ctx, n 1,000, B-spline front end, training data only);
                      shapes as Round 12 (shape_metrics: identifiable RISE of phi_1, phi_2 and their mean).
  hr22a_repro         Round-12 afcm_v3 refitted on replicate 3000, compared with its Round-12 record (harness check).
  sim_<law>_<s>       law in bimodal, studentt3, logistic05; s in 5000-5019 (H-R22c): fsim "S-Misspec-<law>", data
                      seed SeedSequence([s, k]) with k = 2, 3, 4, n 1,000, split_groups 60/20/20 stratified on the event
                      with split seed s, B-spline front end; D112 3-seed full-validation ensembles b2s_cox, b2s_waft,
                      b2s_lnaft, b2s_npaft (rank_km, tau 0.05); Round-12 survival metrics (oracle curve error, IBS, C,
                      D-calibration).
  real_<ds>_<k>       ds in renal, heart_valve; k = 0..4 (H-R22b): the Round-21 job (run_round21_real.job_real:
                      single landmark t_L = 1 year, the four B2-S families, the three cross-confirmed anchors, fixed-mu
                      distillation of the seed-0 NP-AFT and Cox members) plus the Round-13 single-landmark baselines
                      landmark_cox, fpca_cox, lme2_cox, sum_cox, sum_rsf (run_round13.run_single_cell).
  fd003_<k>_<part>    C-MAPSS FD003 Option B, engine split k = 0..4, part in cox, waft, lnaft, npaft (with the NP-AFT
                      cross-confirmed anchor, as Round 21 on FD001), base (landmark_cox on last values + u; the Round-13
                      C-MAPSS baselines sum_aft and fpca_aft). FD003 was analysed in Round 3 (D51) with other models: it
                      is not untouched (D156).
Output: $R22_OUT (jsonl).
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
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("R13_OUT", str(ROOT / "reports/round22/unused_r13.jsonl"))
os.environ.setdefault("R21_OUT", str(ROOT / "reports/round22/unused_r21.jsonl"))
import run_round12 as R12  # noqa: E402
import run_round13 as R13  # noqa: E402
import run_round21_real as R21  # noqa: E402
from kanfae.data import fsim  # noqa: E402
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.evaluation import metrics as M  # noqa: E402
from kanfae.models.afcm import AFCM_D123, fit_afcm, fit_afcm_reml_bagged  # noqa: E402
from kanfae.models.fbaselines import LinearCox  # noqa: E402
from kanfae.models.fkan import FKANEnsemble  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
OUT = Path(os.environ.get("R22_OUT", ROOT / "reports/round22/dev/results_dev.jsonl"))
BASE = ROOT / "reports/round22"
DEV_SEEDS = (22, 914, 966)
CK6D_SEED = 24
HR22A = range(3000, 3020)
SIM = range(5000, 5020)
LAWS = {"bimodal": 2, "studentt3": 3, "logistic05": 4}           # SeedSequence([s, k]) (D156)
REAL = ("renal", "heart_valve")
T_L = {"renal": 1.0, "heart_valve": 1.0}                        # years; fixed from descriptive counts (D156 §2)
FD003_PARTS = ("npaft", "lnaft", "waft", "cox", "base")
SPLITS = range(5)
GATE = BASE / "afcm_gate.json"                                   # written by analyze_round22.py gate (D156 rule)
CK6D = BASE / "ck6d.json"                                        # written by tests/test_round22_ck.py
R13.SINGLE.update(T_L)                                           # the Round-13 single-landmark table, new keys only


def jobs():
    """Queue order (wall time): the five long FD003 NP-AFT jobs first, then the other FD003 parts, the clinical jobs,
    the simulations; the AFCM jobs run outside the queue (afcm_jobs)."""
    J = [f"fd003_{k}_npaft" for k in SPLITS]
    J += [f"fd003_{k}_{p}" for p in FD003_PARTS[1:] for k in SPLITS]
    J += [f"real_{ds}_{k}" for k in SPLITS for ds in REAL]
    J += [f"sim_{law}_{s}" for s in SIM for law in LAWS]
    return J


def afcm_jobs(stage):
    return {"dev": [f"afcmdev_{s}" for s in DEV_SEEDS], "hr22a": ["hr22a_repro"] + [f"hr22a_{s}" for s in HR22A]}[stage]


# ------------------------------------------------------------------------------------------------------- Task 1: AFCM
def ck_data(seed, n=2000):
    """The CK-6c procedure's data (tests/test_round12b_ck.ck6c): S-Add, DGP v2, R11 split, smoothed curves."""
    import test_round11 as R11
    dat = fsim.generate("S-Add", n=n, seed=seed)
    tr, va, te = R11._split(n, seed)
    xg, lam = R11._smoothed(dat, tr)
    return dat, R11._pack(dat, xg, tr), dat.x_true_grid[te], lam


def shapes_ck(m, X):
    from test_round12_ck import shapes
    return shapes(m, X)


def fit_variant(variant, ptr):
    if variant == "afcm_reml":
        return fit_afcm(ptr, None, selection="reml_per_channel", **AFCM_D123)
    if variant == "afcm_reml_ens":
        return fit_afcm_reml_bagged(ptr, B=3, seed=0, **AFCM_D123)
    raise ValueError(variant)


def job_afcmdev(seed):
    dat, ptr, X, lam = ck_data(seed)
    rec = dict(job="afcmdev", seed=seed, n=2000, smoothing_lambda=lam)
    for v in ("afcm_reml", "afcm_reml_ens"):
        t0 = time.time()
        m, r = fit_variant(v, ptr)
        s = shapes_ck(m, X)
        rec[v] = dict(shapes=s, fit=r, seconds=round(time.time() - t0),
                      meets_ck6c_criterion=bool(s["phi1"]["rise_id"] < 0.15 and s["phi2"]["rise_id"] < 0.10))
    return rec


def gate_variant():
    if not GATE.exists():
        raise RuntimeError("the development gate has not been evaluated (D156)")
    g = json.loads(GATE.read_text())
    return g["go_variant"] if g["go"] else None


def job_hr22a(seed):
    v = gate_variant()
    if v is None or not CK6D.exists() or not json.loads(CK6D.read_text()).get("passed"):
        raise RuntimeError("H-R22a runs only after GO and a passed CK-6d (D156)")
    ctx = R12.Ctx("S-Add", seed - R12.SEED_BASE["S-Add"])
    xg = ctx.front("bspline")["xg"]
    t0 = time.time()
    m, r = fit_variant(v, ctx.pack(ctx.tr, xg))
    return dict(job="hr22a", seed=seed, variant=v, shape=R12.shape_metrics(ctx, lambda d, x, t: m.effect(d, x, t)),
                fit=r, seconds=round(time.time() - t0))


def job_hr22a_repro():
    """Harness check: the Round-12 afcm_v3 on replicate 3000 (validation-selected V3 grid), against its record."""
    ctx = R12.Ctx("S-Add", 0)
    xg = ctx.front("bspline")["xg"]
    m, _ = fit_afcm(ctx.pack(ctx.tr, xg), ctx.pack(ctx.va, xg), **AFCM_D123)
    now = R12.shape_metrics(ctx, lambda d, x, t: m.effect(d, x, t))
    ref = None
    for f in sorted((ROOT / "reports/round12").glob("results_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if (r.get("scenario") == "S-Add" and r.get("replicate") == 0 and r.get("model") == "afcm_v3"
                    and r.get("front") == "bspline" and r.get("status") == "ok"):
                ref = r["metrics"]["shape"]
    diff = None if ref is None else max(abs(now[f"phi{k}"]["rise_id"] - ref[f"phi{k}"]["rise_id"]) for k in (1, 2))
    return dict(job="hr22a_repro", seed=3000, shape=now, round12_shape=ref, max_abs_diff_rise_id=diff)


# ------------------------------------------------------------------------------------------- Task 3: simulations
class Ctx22(R12.Ctx):
    """Round-12 replicate context for the Round-22 error laws: data seed SeedSequence([s, k]), split seed s."""

    def __init__(self, law, seed):
        self.scen, self.rep, self.sub, self.seed = "S-Misspec", None, law, seed
        self.data_seed = R12.sub_seed(seed, LAWS[law])
        self.dat = fsim.generate(f"S-Misspec-{law}", R12.N, self.data_seed)
        self.tr, self.va, self.te = split_groups(np.arange(R12.N), seed, strata=self.dat.delta)
        d = self.dat
        self.y_tr, self.d_tr, self.y_te, self.d_te = d.y[self.tr], d.delta[self.tr], d.y[self.te], d.delta[self.te]
        self.t_max = float(np.quantile(self.y_tr, 0.9))
        self.curve_grid = np.linspace(0.0, self.t_max, 101)
        self.S_true = d.subset(self.te).survival(self.curve_grid)
        self.X_te = d.x_true_grid[self.te]
        self._fe, self._feat = {}, {}


SIM_FAMILIES = ("cox", "waft", "lnaft", "npaft")


def job_sim(law, seed):
    ctx = Ctx22(law, seed)
    xg = ctx.front("bspline")["xg"]
    tr, va, te = (ctx.pack(i, xg) for i in (ctx.tr, ctx.va, ctx.te))
    rec = dict(job="sim", law=law, scenario=f"S-Misspec-{law}", seed=seed, data_seed=int(ctx.data_seed),
               n_train=len(ctx.tr), n_val=len(ctx.va), n_test=len(ctx.te), censored_frac=float(1 - ctx.dat.delta.mean()),
               models={})
    for f in SIM_FAMILIES:
        t0 = time.time()
        e = FKANEnsemble(kind="b2s", **R21.FAMILY[f], **R21.ENS_KW).fit(tr, va)
        S_fn = lambda tt, e=e: np.mean([m.predict_survival(te["xg"], te["z"], tt) for m in e.members], 0)  # noqa: E731
        risk = np.mean([m.predict_risk(te["xg"], te["z"]) for m in e.members], 0)
        met = R12.survival_metrics(ctx, S_fn, risk)
        rec["models"][f"b2s_{f}"] = dict(met, lr=e.lr, wd=e.wd, seconds=round(time.time() - t0, 1),
                                        finite=bool(np.all(np.isfinite(S_fn(ctx.curve_grid)))))
    return rec


# ------------------------------------------------------------------------------------------ Task 2: external data
BASELINES = ("landmark_cox", "fpca_cox", "lme2_cox", "sum_cox", "sum_rsf")


def job_real(ds, k):
    rec = R21.job_real(ds, k)
    rec["round"] = 22
    ctx = R13.RealCtx(ds, k)
    rec["baselines"] = {}
    for b in BASELINES:
        t0 = time.time()
        sub = {}
        try:
            R13.run_single_cell(ctx, b, sub)
            sub["status"] = "ok"
        except Exception as exc:
            sub.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])
        sub["seconds"] = round(time.time() - t0, 1)
        rec["baselines"][b] = sub
    rec["cohort"] = dict(t_L=R13.SINGLE[ds.replace("dev_", "")], n=int(ctx.n), events=int(ctx.d.sum()), n_all=int(ctx.lm.meta["n_all"]),
                         unseen_per_channel=[int(sum((~np.isfinite(v[:, c])).all() for v in ctx.lm.values))
                                             for c in range(ctx.D)],
                         obs_per_subject_median=float(np.median([len(t) for t in ctx.lm.times])))
    return rec


def cmapss_data(k, fd="FD003"):
    """run_round21_real.cmapss_data with the subset as a parameter (FD001 there): Round-13 Option B."""
    from kanfae.data.cmapss import informative_sensors, option_b, read_fd
    from kanfae.functional.basis import BSplineBasis
    from kanfae.functional.smoothing import smooth
    df = read_fd(ROOT / "data/raw/cmapss", fd)[0]
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
    return P, sensors, groups, obs


def job_fd003(k, part):
    P, sensors, groups, obs = cmapss_data(k)
    y_tr, d_tr = P["tr"]["y"], P["tr"]["d"].astype(bool)
    y_te, d_te = P["te"]["y"], P["te"]["d"].astype(bool)
    tau = float(np.quantile(y_tr, 0.9))
    tg = M.ibs_time_grid(y_te, y_tr)

    def metric(S_fn, risk):                                  # the Round-13 C-MAPSS metrics
        return dict(ibs=float(M.ibs_marginal(S_fn(tg), tg, y_te, d_te, y_tr, d_tr)[0]),
                    harrell_c=float(M.harrell_c(y_te, d_te, risk)), uno_c=float(M.uno_c(y_tr, d_tr, y_te, d_te, risk, tau)),
                    dcal_p=float(M.d_calibration(np.diag(S_fn(y_te)), d_te)[1]))
    rec = dict(job="fd003", dataset="cmapss_FD003_B", split=k, part=part, sensors=list(sensors), n_train=len(y_tr),
               n_val=len(P["va"]["y"]), n_h1=len(P["h1"]["y"]), n_h2=len(P["h2"]["y"]), n_test=len(y_te),
               engines={n: int(len(np.unique(g))) for n, g in groups.items()},
               models={}, alternatives={}, gates={}, symbolic={})
    if part in R21.FAMILY:
        R21.ENS_CACHE = BASE / "ckpt" / f"fd003_{k}_{part}_fits"      # resumable after a container restart
        R21.fit_family(part, P, metric, rec, cross=part == "npaft")
        return rec
    X = {n: np.hstack([P[n]["last"], P[n]["z"]]) for n in ("tr", "te")}
    t0 = time.time()
    lc = LinearCox().fit(X["tr"], y_tr, d_tr.astype(float))
    S_fn = lambda tt: lc.predict_survival(X["te"], tt)  # noqa: E731
    rec["models"]["landmark_cox"] = dict(metric(S_fn, lc.predict_risk(X["te"])), seconds=round(time.time() - t0, 1))
    from kanfae.experiments.runner import GRIDS
    from kanfae.models.pipelines import make_model
    for model in ("sum_aft", "fpca_aft"):                  # Round-1 protocol: grid tuned by validation IBS (Round 13)
        t0 = time.time()
        tv = M.ibs_time_grid(obs["va"].y_obs, y_tr)
        tun = []
        for cfg in GRIDS[model]:
            mm = make_model(model, **cfg).fit(obs["tr"], obs["va"], seed=0)
            tun.append((float(M.ibs_marginal(mm.predict_survival(obs["va"], tv), tv, obs["va"].y_obs, obs["va"].delta,
                                             y_tr, d_tr)[0]), cfg, mm))
        vib, cfg, m = min(tun, key=lambda t: t[0])
        rec["models"][model] = dict(metric(lambda tt, m=m: m.predict_survival(obs["te"], tt), m.predict_risk(obs["te"])),
                                    config=cfg, tuning=[dict(val_ibs=t[0], **t[1]) for t in tun],
                                    seconds=round(time.time() - t0, 1))
    return rec


# ----------------------------------------------------------------------------------------------------------- driver
def run_job(name):
    t0 = time.time()
    try:
        p = name.split("_")
        if p[0] == "afcmdev":
            rec = job_afcmdev(int(p[1]))
        elif name == "hr22a_repro":
            rec = job_hr22a_repro()
        elif p[0] == "hr22a":
            rec = job_hr22a(int(p[1]))
        elif p[0] == "sim":
            rec = job_sim(p[1], int(p[2]))
        elif p[0] == "real":
            rec = job_real("_".join(p[1:-1]), int(p[-1]))
        elif p[0] == "fd003":
            rec = job_fd003(int(p[1]), p[2])
        else:
            raise ValueError(name)
        rec["status"] = "ok"
    except Exception as e:
        rec = dict(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    rec.update(job_name=name, wall_seconds=round(time.time() - t0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    if rec["status"] == "ok" and name.startswith("fd003_") and R21.ENS_CACHE is not None:
        import shutil                                       # the job's record is written: its fit cache is not needed
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
