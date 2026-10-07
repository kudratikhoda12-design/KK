#!/usr/bin/env python3
"""Stage 15: interpretability of the primary model (development data only).

The model is trained on the training window of the second-to-last development fold (F6) and explained on
F6's own validation block (2025-06-01 .. 2025-08-31), so permutation importance is measured out of sample.
  * LightGBM gain importance
  * permutation importance (single features, and whole feature groups permuted jointly), metric = AUC drop
    and log-loss increase, 3 repeats, seed fixed
  * SHAP values via LightGBM's exact TreeSHAP (pred_contrib) on 20,000 validation rows
  * standardised logistic-regression coefficients (logit_E_all) for a linear view
Importance is predictive association, not causation.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ethof import models as M  # noqa: E402
from ethof import validation as V  # noqa: E402
from ethof.config import Paths, load_config  # noqa: E402
from ethof.features import FEATURE_SETS, GROUPS  # noqa: E402
from ethof.plotting import PALETTE, plt, save  # noqa: E402

TAB = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures" / "interpret"
GROUP_OF = {f: g for g, fs in GROUPS.items() for f in fs}
GCOL = {"price": PALETTE["grey"], "volume": PALETTE["green"], "flow": PALETTE["blue"], "book": PALETTE["red"],
        "funding_oi": PALETTE["purple"]}


def main() -> None:
    cfg = load_config()
    rs = cfg["research"]
    H = rs["primary_horizon"]
    paths = Paths(cfg["data_dir"])
    dev = V.development_only(pl.read_parquet(paths.processed / "research" / "ETHUSDT-research.parquet"),
                             rs["holdout"]["start"])
    wf = rs["walk_forward"]
    folds = V.walk_forward_folds(wf["start"], wf["first_validation"], wf["end_exclusive"], wf["block_months"],
                                 timedelta(days=wf["embargo_days"]))
    fo = folds[-2]
    feats = FEATURE_SETS["E_all"]
    tr, va = V.split(dev, fo, H, stride=wf["train_stride"])
    tr = tr.filter(pl.col(f"fwd_ret_{H}").is_not_null()); va = va.filter(pl.col(f"fwd_ret_{H}").is_not_null())
    Xtr, ytr = tr.select(feats).to_numpy().astype(np.float32), tr[f"up_{H}"].to_numpy()
    Xva, yva = va.select(feats).to_numpy().astype(np.float32), va[f"up_{H}"].to_numpy()
    params = json.loads((TAB / "lgbm_selected_params.json").read_text())
    model = M.lightgbm(**params).fit(Xtr, ytr)
    p0 = model.predict_proba(Xva)[:, 1]
    auc0, ll0 = roc_auc_score(yva, p0), log_loss(yva, p0)
    out = {"fold": fo.as_dict(), "n_train": len(ytr), "n_valid": len(yva), "auc": auc0, "log_loss": ll0}

    gain = model.booster_.feature_importance("gain")
    imp = pl.DataFrame({"feature": feats, "group": [GROUP_OF[f] for f in feats], "gain": gain,
                        "gain_share": gain / gain.sum()})

    rng = np.random.default_rng(42)
    sub = rng.choice(len(yva), size=min(200_000, len(yva)), replace=False)
    Xs, ys = Xva[sub], yva[sub]
    ps = model.predict_proba(Xs)[:, 1]
    base_auc, base_ll = roc_auc_score(ys, ps), log_loss(ys, ps)

    def perm(cols):
        da, dl = [], []
        for _ in range(3):
            Xp = Xs.copy()
            idx = rng.permutation(len(ys))
            Xp[:, cols] = Xs[idx][:, cols]          # joint permutation keeps within-group structure
            pp = model.predict_proba(Xp)[:, 1]
            da.append(base_auc - roc_auc_score(ys, pp)); dl.append(log_loss(ys, pp) - base_ll)
        return float(np.mean(da)), float(np.std(da)), float(np.mean(dl))

    prow = []
    for j, f in enumerate(feats):
        a, s, l = perm([j])
        prow.append({"feature": f, "perm_auc_drop": a, "perm_auc_drop_sd": s, "perm_logloss_increase": l})
    imp = imp.join(pl.DataFrame(prow), on="feature")
    grow = []
    for g, fs in GROUPS.items():
        a, s, l = perm([feats.index(f) for f in fs])
        grow.append({"group": g, "n_features": len(fs), "perm_auc_drop": a, "perm_auc_drop_sd": s,
                     "perm_logloss_increase": l})
    gimp = pl.DataFrame(grow).sort("perm_auc_drop", descending=True)

    # SHAP (exact TreeSHAP from LightGBM)
    ssub = sub[:20_000]
    contrib = model.booster_.predict(Xva[ssub], pred_contrib=True)[:, :-1]
    imp = imp.join(pl.DataFrame({"feature": feats, "mean_abs_shap": np.abs(contrib).mean(axis=0)}), on="feature")
    imp = imp.sort("perm_auc_drop", descending=True)
    imp.write_csv(TAB / "interpret_feature_importance.csv")
    gimp.write_csv(TAB / "interpret_group_permutation.csv")
    gshap = imp.group_by("group").agg(pl.col("mean_abs_shap").sum(), pl.col("gain_share").sum()).sort("mean_abs_shap", descending=True)
    gshap.write_csv(TAB / "interpret_group_shap_gain.csv")

    # logistic coefficients (standardised inputs)
    lr = M.logistic(C=rs["models"]["logit_C"]).fit(Xtr, ytr)
    coefs = lr[-1].coef_[0][: len(feats)]
    pl.DataFrame({"feature": feats, "group": [GROUP_OF[f] for f in feats], "coef_std": coefs}) \
      .sort(pl.col("coef_std").abs(), descending=True).write_csv(TAB / "interpret_logit_coefficients.csv")

    # SHAP dependence for the main flow / book features
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    for a, f in zip(ax, ("ofi_1", "dist_vwap_15", "obi_1pct", "ret_15")):
        j = feats.index(f)
        a.scatter(Xva[ssub, j], contrib[:, j], s=2, alpha=0.25, color=GCOL[GROUP_OF[f]])
        a.set_xlabel(f); a.set_title(f"SHAP dependence: {f}")
        lo, hi = np.nanquantile(Xva[ssub, j], [0.005, 0.995]); a.set_xlim(lo, hi)
    ax[0].set_ylabel("SHAP (log-odds of UP)")
    save(fig, FIG / "shap_dependence.png")

    top = imp.head(20)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].barh(top["feature"][::-1], top["perm_auc_drop"][::-1] * 1e3, xerr=top["perm_auc_drop_sd"][::-1] * 1e3,
               color=[GCOL[g] for g in top["group"][::-1]])
    ax[0].set_xlabel("AUC drop when permuted (x 1e-3)"); ax[0].set_title("Permutation importance (top 20, out of sample)")
    ax[1].bar(gimp["group"], gimp["perm_auc_drop"] * 1e3, yerr=gimp["perm_auc_drop_sd"] * 1e3,
              color=[GCOL[g] for g in gimp["group"]])
    ax[1].set_ylabel("AUC drop (x 1e-3)"); ax[1].set_title("Feature groups permuted jointly")
    save(fig, FIG / "permutation_importance.png")
    out.update(base_auc_subsample=base_auc)
    (TAB / "interpret_summary.json").write_text(json.dumps(out, indent=1, default=str))
    with pl.Config(tbl_rows=30, tbl_width_chars=200):
        print(out); print(imp.head(20)); print(gimp); print(gshap)


if __name__ == "__main__":
    main()
