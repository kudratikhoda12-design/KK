"""D52: KAN-AFT (additive, parametric Weibull) vs linear Weibull AFT on PBC landmark covariates, with pruning and
symbolic extraction. Writes reports/kanaft_pbc.jsonl (one record per endpoint x split)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kanfae.data.pbc import landmark_dataset, read_pbcseq  # noqa: E402
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.evaluation import metrics as M  # noqa: E402
from kanfae.models.kan_aft import KANAFT  # noqa: E402
from kanfae.survival.linear_aft import fit_linear_aft, predict_eta, predict_survival  # noqa: E402
from run_pbc import censoring_covariates, last_values  # noqa: E402

OUT = ROOT / "reports" / "kanaft_pbc.jsonl"


class LinAFT:
    def __init__(self, l2):
        self.l2 = l2

    def fit(self, X, y, d):
        self.mu, self.sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1.0)
        self.f = fit_linear_aft((X - self.mu) / self.sd, y, d, "weibull", l2=self.l2, compute_se=False)
        return self

    def predict_survival(self, X, t):
        return predict_survival(self.f, (X - self.mu) / self.sd, t)

    def predict_risk(self, X):
        return -predict_eta(self.f, (X - self.mu) / self.sd)


def metrics(m, Xte, yte, dte, ytr, dtr, Xc_te, Gc):
    times = M.ibs_time_grid(yte, ytr)
    S = m.predict_survival(Xte, times)
    ibs, _ = M.ibs_marginal(S, times, yte, dte, ytr, dtr)
    ibs_c, _, _ = M.ibs_conditional(S, times, yte, dte, Xc_te, Gc)
    r = m.predict_risk(Xte)
    return dict(ibs=ibs, ibs_conditional=ibs_c, uno_c=M.uno_c(ytr, dtr, yte, dte, r, float(np.quantile(yte, 0.9))),
                harrell_c=M.harrell_c(yte, dte, r))


def val_ibs(m, Xva, yva, dva, ytr, dtr):
    t = M.ibs_time_grid(yva, ytr)
    return M.ibs_marginal(m.predict_survival(Xva, t), t, yva, dva, ytr, dtr)[0]


def main():
    torch.set_default_dtype(torch.float64)
    df = read_pbcseq(ROOT / "data" / "raw" / "pbc" / "data" / "pbc.rda")
    OUT.write_text("")
    for endpoint in ("E1", "E2"):
        obs = landmark_dataset(df, endpoint)
        names = list(obs.channel_names) + ["age", "sex(f=1)"]
        for sp in range(5):
            t0 = time.time()
            tr, va, te = split_groups(obs.ids, sp, strata=obs.delta)
            o_tr, o_va, o_te = obs.subset(tr), obs.subset(va), obs.subset(te)
            raw = {k: np.hstack([last_values(o), o.u]) for k, o in (("tr", o_tr), ("va", o_va), ("te", o_te))}
            mu = np.nanmean(raw["tr"], 0)
            n_imp = {k: int(np.isnan(v).sum()) for k, v in raw.items()}
            X = {k: np.where(np.isnan(v), mu, v) for k, v in raw.items()}
            ref = (mu, np.where(np.nanstd(raw["tr"], 0) > 0, np.nanstd(raw["tr"], 0), 1.0))
            Gc = M.conditional_censoring_model(o_tr.y_obs, o_tr.delta, censoring_covariates(o_tr, ref))
            Xc_te = censoring_covariates(o_te, ref)
            ytr, dtr, yva, dva, yte, dte = o_tr.y_obs, o_tr.delta, o_va.y_obs, o_va.delta, o_te.y_obs, o_te.delta
            ev = lambda m: metrics(m, X["te"], yte, dte, ytr, dtr, Xc_te, Gc)  # noqa: E731
            rec = dict(endpoint=endpoint, split=sp, n=[len(tr), len(va), len(te)], events_test=int(dte.sum()),
                       covariates=names, imputed_counts=n_imp)
            lin = [(val_ibs(m, X["va"], yva, dva, ytr, dtr), l2, m) for l2 in (1e-3, 1e-2, 1e-1)
                   for m in [LinAFT(l2).fit(X["tr"], ytr, dtr)]]
            v, l2, mlin = min(lin, key=lambda t: t[0])
            rec["linear"] = dict(l2=l2, val_ibs=v, test=ev(mlin), beta=mlin.f.beta.tolist())
            kan = [(val_ibs(m, X["va"], yva, dva, ytr, dtr), lam, m) for lam in (0.0, 1e-3, 1e-2)
                   for m in [KANAFT(loss="parametric", lam=lam, seed=0).fit(X["tr"], ytr, dtr, X["va"], yva, dva)]]
            v, lam, m = min(kan, key=lambda t: t[0])
            rec["kan"] = dict(lam=lam, val_ibs=v, tuning=[(float(a), b) for a, b, _ in kan], test=ev(m))
            rec["prune"] = m.prune(1e-2)
            rec["pruned_mask"] = m.net.layers[0].mask[:, 0].tolist()
            rec["kan_pruned_test"] = ev(m)
            syms = m.symbolic(use_pysr=(sp == 0), pysr_budget={"niterations": 30, "timeout": 120})
            for s in syms:
                s["covariate"] = names[s["edge"][0]]
                s["edge"] = list(s["edge"])
            rec["symbolic"] = syms
            rec["kan_symbolic_test"] = ev(m)
            rec["runtime_s"] = round(time.time() - t0, 1)
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")
            k = rec["kan"]["test"]; li = rec["linear"]["test"]
            print(f"[{endpoint} split {sp}] linear IBS {li['ibs']:.4f} C {li['uno_c']:.3f} | KAN(lam={lam}) IBS "
                  f"{k['ibs']:.4f} C {k['uno_c']:.3f} -> pruned {rec['kan_pruned_test']['ibs']:.4f} -> symbolic "
                  f"{rec['kan_symbolic_test']['ibs']:.4f}; accepted {sum(s['accepted'] for s in syms)}/{len(syms)}: "
                  + "; ".join(f"{s['covariate']}: {s['stage']} {s['formula'][:50]} (R2val {s['r2_val']:.4f})"
                              for s in syms) + f" [{rec['runtime_s']}s]", flush=True)


if __name__ == "__main__":
    main()
