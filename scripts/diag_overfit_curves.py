"""D70 overfitting diagnosis for Revision 4b (seed-21 TRAIN/VALIDATION only; nothing here touches a test split).

For three configurations (A3 reference; revision-4 selection "fa" J17 r12; compact summary J17 r12), records:
  curves:   per epoch, train and validation NLL and reconstruction loss (the joint objective = NLL + alpha * rec);
            every 25 epochs, train and validation Harrell C of eta
  stopping: the epoch at which (a) the joint validation objective, (b) the validation NLL is minimal
  features: magnitude of the front's two paths (A3 tensor-spline vs spectral) on train/validation; tails of the
            standardised spectral features
  latent:   per-parameter linear-probe R^2 on train (in-sample) vs validation; head weight norm
Writes reports/diag_overfit_curves.json.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kanfae.data.splits import split_groups  # noqa: E402
from kanfae.data.synthetic import SynthConfig, generate  # noqa: E402
from kanfae.evaluation.metrics import harrell_c  # noqa: E402
from kanfae.functional.layers import FrequencyAwareFunctional  # noqa: E402
from kanfae.models.pipelines import NeuralAFT  # noqa: E402

A3 = dict(front="tspline", r=8, alpha=1.0, optimizer="adam", lr=3e-3, es_target="objective", max_epochs=3000,
          patience=100, H=32, hidden=64)
CONFIGS = {"A3_reference": A3,
           "fa_J17_r12 (revision-4 selection)": dict(A3, front="fa", r=12, fa_nfreq=17),
           "fa_compact_J17_r12": dict(A3, front="fa", r=12, fa_nfreq=17, fa_compact=True)}


class Traced(NeuralAFT):
    """Identical training; additionally records per-epoch train/val NLL and reconstruction, and C every 25 epochs.
    Runs the full max_epochs budget (no early stop) so that both stopping rules can be located afterwards."""

    def _run(self, Ttr, Tva, parts, params):
        Opt = torch.optim.AdamW if self.cfg["optimizer"] == "adamw" else torch.optim.Adam
        opt = Opt(params, lr=self.cfg["lr"], weight_decay=self.cfg["wd"])
        self.curve = []
        for ep in range(self.cfg["max_epochs"]):
            self.model.train()
            opt.zero_grad()
            loss, (tn, tr) = self._objective(Ttr, parts)
            loss.backward()
            opt.step()
            self.model.eval()
            with torch.no_grad():
                vl, (vn, vr) = self._objective(Tva, parts)
                row = dict(ep=ep, train_nll=float(tn), train_rec=float(tr), val_nll=float(vn), val_rec=float(vr),
                           val_obj=float(vl))
                if ep % 25 == 0:
                    _, _, et, _ = self.model(Ttr["inp"], Ttr["u"])
                    _, _, ev, _ = self.model(Tva["inp"], Tva["u"])
                    row["train_C"] = harrell_c(Ttr["y"].numpy(), Ttr["d"].numpy().astype(bool), -et.numpy())
                    row["val_C"] = harrell_c(Tva["y"].numpy(), Tva["d"].numpy().astype(bool), -ev.numpy())
            self.curve.append(row)
        return {"parts": parts, "epochs": len(self.curve), "best_val": np.nan, "diverged": False}


def probe_r2(Zf, tf, Ze, te):
    A = np.column_stack([np.ones(len(Zf)), Zf]); B = np.column_stack([np.ones(len(Ze)), Ze])
    coef = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ tf)
    r2 = lambda y, p: float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())  # noqa: E731
    return r2(tf, A @ coef), r2(te, B @ coef)


def main():
    torch.set_default_dtype(torch.float64)
    obs, ora = generate(SynthConfig(n=2000, censoring=0.0, sigma_y=0.05, seed=21))
    tr, va, te = split_groups(obs.ids, 21, strata=obs.delta)          # te unused
    out = {}
    for name, cfg in CONFIGS.items():
        m = Traced("fae_mlp", **cfg).fit(obs.subset(tr), obs.subset(va), seed=0)
        cur = m.curve
        e_obj = int(np.argmin([r["val_obj"] for r in cur]))
        e_nll = int(np.argmin([r["val_nll"] for r in cur]))
        res = {"config": cfg, "epoch_min_val_objective": e_obj, "epoch_min_val_nll": e_nll,
               "val_nll_at_obj_stop": cur[e_obj]["val_nll"], "val_nll_min": cur[e_nll]["val_nll"],
               "curve": cur[::5]}
        # final-state (after the full budget) feature and latent diagnostics
        Ptr, Pva = m.prepare(obs.subset(tr)), m.prepare(obs.subset(va))
        Ttr, Tva = m._tensors(Ptr), m._tensors(Pva)
        fl = m.model.flin
        with torch.no_grad():
            for lab, T in (("train", Ttr), ("val", Tva)):
                c = T["inp"]
                base = super(FrequencyAwareFunctional, fl).forward(c) if isinstance(fl, FrequencyAwareFunctional) \
                    else fl(c)
                res[f"{lab}_A3_path_rms"] = float(base.pow(2).mean().sqrt())
                if isinstance(fl, FrequencyAwareFunctional):
                    sv = fl.norm(fl.spectral_vector(c))
                    res[f"{lab}_spectral_path_rms"] = float(fl.spec(sv).pow(2).mean().sqrt())
                    res[f"{lab}_spectral_feature_absmax"] = float(sv.abs().max())
                    res[f"{lab}_spectral_feature_frac_abs_gt3"] = float((sv.abs() > 3).double().mean())
        res["head_weight_norm"] = float(m.model.head.net.weight.norm()) if hasattr(m.model.head, "net") and \
            hasattr(m.model.head.net, "weight") else None
        Zt, Zv = m.latent(obs.subset(tr)), m.latent(obs.subset(va))
        res["probe_train_vs_val"] = {p: probe_r2(Zt, ora.theta[tr, j], Zv, ora.theta[va, j])
                                     for j, p in enumerate(("a", "b", "A", "omega"))}
        out[name] = res
        cs = [r for r in cur if "train_C" in r]
        best_c = max(cs, key=lambda r: r["val_C"])
        print(f"{name}: min val objective at epoch {e_obj}, min val NLL at epoch {e_nll}; val NLL at objective stop "
              f"{res['val_nll_at_obj_stop']:.4f} vs min {res['val_nll_min']:.4f}; best val C {best_c['val_C']:.4f} at "
              f"epoch {best_c['ep']} (train C {best_c['train_C']:.4f}); final train/val C {cs[-1]['train_C']:.4f}/"
              f"{cs[-1]['val_C']:.4f}; paths train/val A3 {res['train_A3_path_rms']:.3f}/{res['val_A3_path_rms']:.3f} "
              f"spectral {res.get('train_spectral_path_rms', float('nan')):.3f}/{res.get('val_spectral_path_rms', float('nan')):.3f}",
              flush=True)
        print("   probe (train in-sample, val):", {k: tuple(round(x, 3) for x in v) for k, v in res["probe_train_vs_val"].items()},
              flush=True)
    (ROOT / "reports" / "diag_overfit_curves.json").write_text(json.dumps(out, default=float))


if __name__ == "__main__":
    main()
