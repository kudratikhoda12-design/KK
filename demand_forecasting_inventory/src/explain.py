"""Phase 21 - SHAP explanations of the final (train+validation) XGBoost models.

* SHAP TreeExplainer on the TEST-period origins of the final model (explanation only - nothing is tuned on it).  At most
  ``config.XGB_SHAP_MAX_ROWS`` rows are used (seeded random sample, documented in the report).
* SHAP values are *additive attributions of the model's prediction* (in units of the H-day demand sum):
  positive value -> the feature pushes the prediction above the average prediction (base value); negative -> below.
  They describe the MODEL, not causal effects in the real world.
* Correctness check: sum(SHAP) + base value == model prediction (max error reported).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

from . import config as C
from . import viz
from .features import FEATURES, FEATURE_GROUPS, dataset_for_horizon, rows_for_origins
from .forecast_core import Context

GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}


def load_final_model(H: int) -> xgb.XGBRegressor:
    m = xgb.XGBRegressor()
    m.load_model(C.OUT_MOD / f"xgb_H{H}_final_train_val.json")
    return m


def compute_shap(ctx: Context, H: int) -> dict:
    model = load_final_model(H)
    ds = dataset_for_horizon(ctx, H)
    rows = rows_for_origins(ds, ctx.split.all_origins("test")).reset_index(drop=True)
    n = min(C.XGB_SHAP_MAX_ROWS, len(rows))
    sample = rows.sample(n=n, random_state=C.SEED).sort_values(["series_idx", "origin"]).reset_index(drop=True)
    X = sample[FEATURES]
    ex = shap.TreeExplainer(model)
    sv = np.asarray(ex.shap_values(X))
    base = float(np.atleast_1d(ex.expected_value)[0])
    pred = model.predict(X)
    err = float(np.abs(sv.sum(axis=1) + base - pred).max())
    return {"H": H, "model": model, "X": X, "meta": sample[["series_idx", "origin", "date", "target"]], "shap": sv, "base": base,
            "pred": pred, "additivity_max_abs_error": err, "n_rows_total": len(rows), "n_rows_used": n}


def importance_table(res: dict) -> pd.DataFrame:
    X, sv = res["X"], res["shap"]
    rows = []
    for j, f in enumerate(FEATURES):
        x = X[f].to_numpy()
        corr = np.corrcoef(x, sv[:, j])[0, 1] if np.ptp(x) > 0 and np.ptp(sv[:, j]) > 0 else np.nan
        rows.append({"H": res["H"], "feature": f, "group": GROUP_OF[f], "mean_abs_shap": float(np.abs(sv[:, j]).mean()),
                     "mean_shap": float(sv[:, j].mean()), "corr(feature value, shap)": float(corr)})
    df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["share_of_total"] = df["mean_abs_shap"] / df["mean_abs_shap"].sum()
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def pick_individual_cases(res: dict, ctx: Context) -> pd.DataFrame:
    """Three explanation cases chosen by RULE (not by outcome): the largest prediction, the median prediction, and the
    first sampled origin whose forecast window contains Christmas."""
    meta, pred, X = res["meta"], res["pred"], res["X"]
    cases = {"largest prediction": int(np.argmax(pred)),
             "median prediction": int(np.argsort(pred)[len(pred) // 2])}
    xmas = np.flatnonzero(X["christmas_in_window"].to_numpy() > 0)
    cases["Christmas in forecast window"] = int(xmas[0]) if len(xmas) else int(np.argmin(pred))
    rows = []
    for name, i in cases.items():
        rows.append({"case": name, "row": i, "series_id": ctx.ids[int(meta.loc[i, "series_idx"])], "origin_date": str(pd.Timestamp(meta.loc[i, "date"]).date()),
                     "prediction": float(pred[i]), "actual": float(meta.loc[i, "target"]) if pd.notna(meta.loc[i, "target"]) else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_importance_bar(res: dict, imp: pd.DataFrame, name: str) -> str:
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["mean_abs_shap"], color=viz.BLUE, height=0.62)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    for y, (v, f) in enumerate(zip(top["mean_abs_shap"], top["feature"])):
        ax.text(v, y, f"  {v:.2f}", va="center", ha="left", fontsize=8, color=viz.INK2)
    ax.set_xlabel(f"mean |SHAP value| (units of the {res['H']}-day demand sum)")
    first = imp.iloc[0]
    viz.title_block(fig, f"Which features drive the {res['H']}-day XGBoost forecast most?",
                    f"Top feature: {first['feature']} ({100 * first['share_of_total']:.0f}% of total mean |SHAP|); "
                    f"top 3 together = {100 * imp.head(3)['share_of_total'].sum():.0f}%. "
                    f"{res['n_rows_used']:,} test-period origins sampled.")
    return viz.save(fig, name)


def plot_summary(res: dict, name: str) -> str:
    plt.close("all")
    fig = plt.figure(figsize=(8.6, 6.2))
    shap.summary_plot(res["shap"], res["X"], feature_names=FEATURES, max_display=14, show=False, plot_size=None, cmap=viz.DIV)
    ax = plt.gca()
    ax.set_xlabel("SHAP value  (negative = pushes the forecast down, positive = pushes it up)", color=viz.INK2)
    ax.grid(False)
    fig = plt.gcf()
    viz.title_block(fig, f"How do feature values move the {res['H']}-day forecast?",
                    "Each dot is one forecast origin; colour = feature value (blue low, red high). Attributions describe the model, not causal effects.")
    return viz.save(fig, name)


def plot_dependence(res: dict, imp: pd.DataFrame, name: str, k: int = 3) -> str:
    feats = [f for f in imp["feature"].head(k)]
    fig, axes = plt.subplots(1, k, figsize=(4.2 * k, 3.8))
    X, sv = res["X"], res["shap"]
    for ax, f in zip(np.atleast_1d(axes), feats):
        j = FEATURES.index(f)
        x = X[f].to_numpy()
        ax.scatter(x, sv[:, j], s=7, color=viz.BLUE, alpha=0.35, linewidths=0)
        ax.axhline(0, color=viz.AXIS, lw=0.8)
        ax.set_xlabel(f)
        ax.set_ylabel("SHAP value")
    viz.title_block(fig, f"How does the {res['H']}-day forecast respond to its top 3 features?",
                    "Dependence plots (descriptive of the fitted model; interactions blur single-feature curves).")
    return viz.save(fig, name)


def plot_individual(res: dict, case: pd.Series, name: str, ctx: Context) -> str:
    i = int(case["row"])
    sv_i = res["shap"][i]
    expl = shap.Explanation(values=sv_i, base_values=res["base"], data=res["X"].iloc[i].to_numpy(), feature_names=FEATURES)
    plt.close("all")
    shap.plots.waterfall(expl, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8.6, 5.2)
    act = "" if pd.isna(case["actual"]) else f"; actual = {case['actual']:.0f}"
    viz.title_block(fig, f"Why this forecast? {case['case']}",
                    f"{case['series_id']}, origin {case['origin_date']}: predicted {case['prediction']:.1f} units over {res['H']} days{act}. "
                    f"Base value {res['base']:.1f}; bars are additive attributions.", top=0.99)
    return viz.save(fig, name, rect=(0, 0, 1, 0.88))


def run_shap(ctx: Context, horizons=(7, 28), make_plots: bool = True) -> dict:
    out = {"importance": [], "cases": [], "checks": []}
    for H in horizons:
        res = compute_shap(ctx, H)
        imp = importance_table(res)
        out["importance"].append(imp)
        out["checks"].append({"H": H, "rows_used": res["n_rows_used"], "rows_available": res["n_rows_total"],
                              "additivity_max_abs_error": res["additivity_max_abs_error"], "base_value": res["base"]})
        cases = pick_individual_cases(res, ctx)
        cases.insert(0, "H", H)
        out["cases"].append(cases)
        if make_plots:
            plot_importance_bar(res, imp, f"shap_importance_H{H}.png")
            plot_summary(res, f"shap_summary_H{H}.png")
            plot_dependence(res, imp, f"shap_dependence_H{H}.png")
            if H == horizons[0]:
                for k, (_, c) in enumerate(cases.iterrows(), start=1):
                    plot_individual(res, c, f"shap_individual_H{H}_{k}.png", ctx)
    imp_all = pd.concat(out["importance"], ignore_index=True)
    imp_all.to_csv(C.OUT_TAB / "shap_importance.csv", index=False)
    pd.concat(out["cases"], ignore_index=True).to_csv(C.OUT_TAB / "shap_individual_cases.csv", index=False)
    pd.DataFrame(out["checks"]).to_csv(C.OUT_TAB / "shap_checks.csv", index=False)
    return {"importance": imp_all, "cases": pd.concat(out["cases"], ignore_index=True), "checks": pd.DataFrame(out["checks"])}
