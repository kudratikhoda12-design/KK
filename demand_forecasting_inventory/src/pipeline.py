"""Stage functions of the end-to-end pipeline (called by run_project.py).

Order: env -> acquire -> inspect -> prepare -> eda -> forecast -> evaluate -> explain -> uncertainty -> inventory -> audit -> reports.
Each stage reads what earlier stages wrote to data/processed and outputs/tables, so a stage can be re-run on its own.

Protocol reminder (see reports/validation_strategy.md): everything that is *chosen* (moving-average window, ETS variant,
SARIMA structure, XGBoost hyper-parameters, sigma_L, headline Policy-B model, interval method) is chosen on VALIDATION data;
TEST metrics are computed once the choices are frozen and are only reported.
"""
from __future__ import annotations

import time
import traceback
from contextlib import contextmanager

import pandas as pd

from . import config as C

FIGURE_ERRORS: list[str] = []
TIMINGS: dict[str, float] = {}


@contextmanager
def timed(name: str):
    t0 = time.time()
    print(f"\n=== [{name}] ===", flush=True)
    yield
    TIMINGS[name] = time.time() - t0
    print(f"=== [{name}] done in {time.time() - t0:.1f}s ===", flush=True)


def safe_fig(label: str, fn, *a, **k):
    """Run a plotting function; a plotting bug must never destroy numerical results, but it is recorded and surfaced."""
    try:
        return fn(*a, **k)
    except Exception as exc:                                         # noqa: BLE001
        FIGURE_ERRORS.append(f"{label}: {type(exc).__name__}: {exc}")
        print(f"  !! figure '{label}' failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


# --------------------------------------------------------------------------------------
# Stages 0-4
# --------------------------------------------------------------------------------------
def stage_env():
    from . import env_check
    env_check.write_environment_report()


def stage_acquire(force: bool = False):
    from . import data_acquisition as da
    ver = da.acquire(force=force)
    da.write_source_report(ver)


def stage_inspect():
    from . import data_inspection as di
    di.inspect_all()


def stage_prepare():
    from . import data_prep as dp
    dp.prepare()


def stage_eda():
    from . import eda
    from .forecast_core import load_context
    eda.run_eda(load_context())


# --------------------------------------------------------------------------------------
# Stage 5: fit every forecasting model (validation phase + test phase)
# --------------------------------------------------------------------------------------
def stage_forecast():
    from . import baselines as bl, stat_models as sm, xgb_models as xm
    from .forecast_core import ForecastBook, load_context
    ctx = load_context()
    book = ForecastBook(ctx)
    bl.add_baselines(book)
    ets_meta = sm.run_ets(book)
    ets_meta.to_csv(C.OUT_TAB / "ets_fit_log.csv", index=False)
    n_fb = int(ets_meta["event"].notna().sum()) if "event" in ets_meta else 0
    print(f"  ETS: {n_fb} fallback event(s) (HoltWinters -> Holt -> SES)")
    cands, sel = sm.run_sarima(book)
    cands.to_csv(C.OUT_TAB / "sarima_candidates.csv", index=False)
    sel.to_csv(C.OUT_TAB / "sarima_selection.csv", index=False)
    print(f"  SARIMA: chosen structures {sel['chosen'].value_counts().to_dict()}; fallbacks val/test = "
          f"{int((sel['fallback_val'] != '').sum())}/{int((sel['fallback_test'] != '').sum())}")
    sm.stationarity_table(ctx).to_csv(C.OUT_TAB / "stationarity_tests.csv", index=False)
    xg = xm.run_xgboost(book)
    xg["log"].to_csv(C.OUT_TAB / "xgb_tuning_log.csv", index=False)
    xg["chosen"].to_csv(C.OUT_TAB / "xgb_chosen_params.csv", index=False)
    book.save()


def load_book():
    from . import evaluation as ev
    from .forecast_core import ForecastBook, load_context
    ctx = load_context()
    book = ForecastBook(ctx).load()
    ma_tab = ev.select_ma_alias(book)
    return ctx, book, ma_tab


# --------------------------------------------------------------------------------------
# Stage 6: accuracy tables, DM tests, residual analysis
# --------------------------------------------------------------------------------------
def stage_evaluate():
    from . import evaluation as ev, plots, residuals as rs
    from . import eda
    ctx, book, ma_tab = load_book()
    ma_tab.to_csv(C.OUT_TAB / "ma_window_selection.csv", index=False)
    models = ev.MODELS_FINAL
    pooled_val = ev.metrics_pooled(book, "val", models, C.HORIZONS)
    pooled_test = ev.metrics_pooled(book, "test", models, C.HORIZONS)
    by_series = pd.concat([ev.metrics_by_series(book, "val", models), ev.metrics_by_series(book, "test", models)], ignore_index=True)
    by_series.to_csv(C.OUT_TAB / "forecast_results.csv", index=False)
    pooled_val.to_csv(C.OUT_TAB / "forecast_summary_validation.csv", index=False)
    pooled_test.to_csv(C.OUT_TAB / "forecast_summary_test.csv", index=False)
    # --- final table: pooled TEST metrics at the three primary horizons
    fin = pooled_test[pooled_test["H"].isin(C.PRIMARY_HORIZONS)][["model", "H", "MAE", "RMSE", "sMAPE", "WAPE"]].rename(
        columns={"model": "Model", "H": "Horizon"})
    fin["Family"] = fin["Model"].map(ev.MODEL_FAMILY)
    fin.sort_values(["Horizon", "WAPE"]).to_csv(C.OUT_TAB / "final_forecasting_results.csv", index=False)
    # --- best model per horizon (validation = what could be chosen ex ante; test = what is reported)
    rows = []
    for H in C.HORIZONS:
        v = pooled_val[pooled_val["H"] == H].sort_values("WAPE")
        t = pooled_test[pooled_test["H"] == H].sort_values("WAPE")
        vr = pooled_val[pooled_val["H"] == H].sort_values("RMSE")
        tr = pooled_test[pooled_test["H"] == H].sort_values("RMSE")
        rows.append({"H": H, "primary_horizon": H in C.PRIMARY_HORIZONS, "best_by_validation_WAPE": v.iloc[0]["model"], "val_WAPE": v.iloc[0]["WAPE"],
                     "best_by_test_WAPE": t.iloc[0]["model"], "test_WAPE": t.iloc[0]["WAPE"], "runner_up_test_WAPE": t.iloc[1]["model"],
                     "runner_up_WAPE": t.iloc[1]["WAPE"], "best_by_validation_RMSE": vr.iloc[0]["model"], "best_by_test_RMSE": tr.iloc[0]["model"],
                     "validation_choice_equals_test_best": v.iloc[0]["model"] == t.iloc[0]["model"]})
    best = pd.DataFrame(rows)
    best.to_csv(C.OUT_TAB / "best_model_by_horizon.csv", index=False)
    print(best[["H", "best_by_validation_WAPE", "best_by_test_WAPE", "test_WAPE"]].to_string(index=False))
    # --- Diebold-Mariano (TEST)
    panels, pers = [], []
    for ref in ("XGBoost", "SARIMA"):
        p, q = ev.dm_summary(book, "test", ref=ref)
        p["ref"] = ref
        q["ref"] = ref
        panels.append(p), pers.append(q)
    pd.concat(panels, ignore_index=True).to_csv(C.OUT_TAB / "dm_tests_panel.csv", index=False)
    pd.concat(pers, ignore_index=True).to_csv(C.OUT_TAB / "dm_tests_per_series.csv", index=False)
    # --- residual analysis
    sarima_sel = pd.read_csv(C.OUT_TAB / "sarima_selection.csv")
    ets_meta = pd.read_csv(C.OUT_TAB / "ets_fit_log.csv")
    rtab, rarrays = rs.stat_model_residuals(ctx, sarima_sel, ets_meta)
    rtab.to_csv(C.OUT_TAB / "residuals_statistical_models.csv", index=False)
    rs.stat_residual_summary(rtab).to_csv(C.OUT_TAB / "residuals_statistical_models_summary.csv", index=False)
    xres = {}
    for H in C.PRIMARY_HORIZONS:
        r = rs.xgboost_residuals(book, H)
        xres[H] = r
        for key in ("by_level", "by_tier", "by_month", "by_weekday"):
            r[key].to_csv(C.OUT_TAB / f"residuals_xgboost_H{H}_{key}.csv", index=False)
    pd.DataFrame([{"H": H, **r["overall"]} for H, r in xres.items()]).to_csv(C.OUT_TAB / "residuals_xgboost_overall.csv", index=False)
    rs.error_concentration(book, models, C.PRIMARY_HORIZONS).to_csv(C.OUT_TAB / "error_concentration.csv", index=False)
    rs.zero_forecast_shares(book, models, C.PRIMARY_HORIZONS).to_csv(C.OUT_TAB / "zero_forecast_shares.csv", index=False)
    # --- figures
    rep = eda.representative_series(ctx)
    safe_fig("forecast_vs_actual", plots.fig_forecast_vs_actual, book, rep)
    safe_fig("model_comparison", plots.fig_model_comparison, pooled_test, pooled_val)
    safe_fig("error_comparison", plots.fig_error_comparison, pooled_test)
    for H in (7, 28):
        safe_fig(f"series_level_wape_H{H}", plots.fig_series_level_wape, by_series[by_series["phase"] == "test"], H)
        safe_fig(f"residuals_xgb_H{H}", plots.fig_residuals_xgb, xres[H], H)
    safe_fig("residuals_stat", plots.fig_residuals_stat, rarrays, rep)


# --------------------------------------------------------------------------------------
# Stage 7: SHAP
# --------------------------------------------------------------------------------------
def stage_explain():
    from . import explain
    from .forecast_core import load_context
    ctx = load_context()
    out = explain.run_shap(ctx, horizons=(7, 28))
    for H, g in out["importance"].groupby("H"):
        print(f"  H={H} top features:", ", ".join(f"{r.feature} ({100 * r.share_of_total:.0f}%)" for r in g.head(5).itertuples()))
    print("  additivity check:", out["checks"].to_dict("records"))


# --------------------------------------------------------------------------------------
# Stage 8: uncertainty
# --------------------------------------------------------------------------------------
def stage_uncertainty():
    from . import evaluation as ev, plots, uncertainty as un
    from . import eda
    ctx, book, _ = load_book()
    models = ev.MODELS_FINAL
    sig = un.sigma_table(book, models)
    sig.to_csv(C.OUT_TAB / "sigma_table.csv", index=False)
    norm = un.normality_table(book, models)
    norm.to_csv(C.OUT_TAB / "uncertainty_normality.csv", index=False)
    calib = un.split_half_calibration(book, models)
    calib.to_csv(C.OUT_TAB / "uncertainty_calibration_validation.csv", index=False)
    method, summ = un.choose_interval_method(calib)
    summ.to_csv(C.OUT_TAB / "uncertainty_method_choice.csv", index=False)
    print(f"  interval method chosen on validation (split-half): {method}\n{summ.to_string(index=False)}")
    cov = un.test_coverage(book, models, method)
    cov.to_csv(C.OUT_TAB / "uncertainty_coverage_test.csv", index=False)
    rep = eda.representative_series(ctx)
    arr = un.interval_arrays(book, "XGBoost", 7, method, 0.90)
    safe_fig("prediction_interval_H7", plots.fig_prediction_interval, arr, rep, ctx, 7, "XGBoost", method, 0.90)
    arr28 = un.interval_arrays(book, "SARIMA", 28, method, 0.80)
    safe_fig("prediction_interval_H28", plots.fig_prediction_interval, arr28, rep, ctx, 28, "SARIMA", method, 0.80)
    safe_fig("coverage", plots.fig_coverage, cov)
    safe_fig("qq_xgb", plots.fig_qq_standardised, book, "XGBoost", 7)
    return method


# --------------------------------------------------------------------------------------
# Stage 9: inventory simulation and analyses
# --------------------------------------------------------------------------------------
def stage_inventory():
    from . import evaluation as ev, inventory_analysis as ia, inventory_experiments as ie, plots, robustness as rb
    from . import eda
    ctx, book, _ = load_book()
    models = ev.MODELS_FINAL
    pooled_val = pd.read_csv(C.OUT_TAB / "forecast_summary_validation.csv")
    pooled_test = pd.read_csv(C.OUT_TAB / "forecast_summary_test.csv")
    sig = pd.read_csv(C.OUT_TAB / "sigma_table.csv")
    best_B = ia.best_models_by_validation(pooled_val, models)
    print("  headline Policy B model per lead time (lowest validation WAPE at H = L):", best_B)
    pd.DataFrame([{"lead_time": L, "model": m, "selection_rule": "lowest pooled VALIDATION WAPE at horizon H = L"} for L, m in best_B.items()]).to_csv(
        C.OUT_TAB / "headline_policy_B_models.csv", index=False)
    keep = {(p, 7, 0.95, "gaussian") for p in {ie.POLICY_A, best_B[7]}}
    ie.eoq_cycle_sanity(ctx).to_csv(C.OUT_TAB / "eoq_cycle_sanity_train.csv", index=False)
    runs, ledgers = ie.run_experiments(ctx, book, models, sig, keep_ledgers_for=keep)
    runs.to_csv(C.OUT_TAB / "inventory_runs_all.csv", index=False)
    print(f"  simulated {len(runs):,} (policy x lead time x service level x series) runs")
    # ---- headline tables
    fin = ia.final_inventory_table(runs, best_B)
    fin.to_csv(C.OUT_TAB / "final_inventory_results.csv", index=False)
    sens = ia.sensitivity_table(runs, best_B)
    sens.to_csv(C.OUT_TAB / "sensitivity_headline_policies.csv", index=False)
    opt = ia.optimal_service_level(sens)
    opt.to_csv(C.OUT_TAB / "sensitivity_optimal_service_level.csv", index=False)
    ia.factor_importance(sens).to_csv(C.OUT_TAB / "sensitivity_factor_importance.csv", index=False)
    pair = ia.paired_table(runs, best_B)
    pair.to_csv(C.OUT_TAB / "inventory_paired_bootstrap_B_vs_A.csv", index=False)
    # also B_xgboost and B_best-by-test for completeness of the accuracy-vs-cost story
    pair_all = pd.DataFrame([ia.paired_bootstrap(runs, m, L, SL, "MEDIUM") for m in models for L in C.LEAD_TIMES for SL in C.SERVICE_LEVELS])
    pair_all.to_csv(C.OUT_TAB / "inventory_paired_bootstrap_all_models_MEDIUM.csv", index=False)
    # ---- every-model results (aggregated) for transparency
    allm = pd.concat([ie.aggregate(runs, ["policy", "method", "L", "SL"], scenario=s) for s in C.SCENARIOS], ignore_index=True)
    allm.to_csv(C.OUT_TAB / "inventory_all_policies_aggregated.csv", index=False)
    # ---- accuracy vs business value
    avc, avc_sum = ia.accuracy_vs_cost(runs, pooled_val, pooled_test, models)
    avc.to_csv(C.OUT_TAB / "accuracy_vs_cost_detail.csv", index=False)
    avc_sum.to_csv(C.OUT_TAB / "accuracy_vs_cost_summary.csv", index=False)
    # ---- robustness
    fbg = rb.forecast_by_group(book, "test")
    fbg.to_csv(C.OUT_TAB / "robustness_forecast_by_group.csv", index=False)
    rb.best_by_group(fbg).to_csv(C.OUT_TAB / "robustness_best_model_by_group.csv", index=False)
    ibg = rb.inventory_by_group(runs, best_B)
    ibg.to_csv(C.OUT_TAB / "robustness_inventory_by_group.csv", index=False)
    # ---- ledger of the headline configuration (simulation audit trail)
    frames = []
    for (pol, L, SL, meth, s), led in ledgers.items():
        cp = ie.prepare_inputs(ctx)["cost_params"][s]
        df = led.to_frame(dates=ctx.dates[ctx.split.val_end:], h_day=cp["h_day"], order_cost=cp["order_cost"], stockout_cost_unit=cp["p_MEDIUM"])
        df.insert(0, "series_id", ctx.ids[s]); df.insert(1, "policy", pol); df.insert(2, "lead_time", L); df.insert(3, "service_level", SL)
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(C.OUT_TAB / "inventory_ledger_headline_L7_SL95.csv", index=False)
    # ---- figures
    rep = eda.representative_series(ctx)
    polB = best_B[7]
    for tier in ("high", "medium"):
        s = int(rep[rep["tier"] == tier]["series_idx"].iloc[0])
        safe_fig(f"trajectory_{tier}", plots.fig_inventory_trajectory, ledgers, ctx, s, ie.POLICY_A, polB, 7, 0.95)
    safe_fig("stockout_timeline", plots.fig_stockout_timeline, ledgers, ctx, ie.POLICY_A, polB, 7, 0.95)
    safe_fig("service_vs_cost", plots.fig_service_vs_cost, sens, "MEDIUM")
    safe_fig("cost_vs_fill_rate", plots.fig_cost_vs_fill_rate, sens, "MEDIUM")
    safe_fig("total_cost_components", plots.fig_total_cost_components, runs, 7, 0.95, "MEDIUM")
    safe_fig("sensitivity_heatmaps", plots.fig_sensitivity_heatmaps, sens)
    safe_fig("accuracy_vs_cost", plots.fig_accuracy_vs_cost, avc, avc_sum, 0.95, "MEDIUM")
    safe_fig("robustness", plots.fig_robustness, fbg, ibg)
    print(opt[opt["policy_group"] == "B_forecast_driven"][["L", "scenario", "cost_minimising_SL"]].to_string(index=False))


# --------------------------------------------------------------------------------------
# Stage 10: leakage audit (needs models + sigma table)
# --------------------------------------------------------------------------------------
def stage_audit():
    from . import leakage_audit as la
    ctx, book, _ = load_book()
    tuning = pd.read_csv(C.OUT_TAB / "xgb_tuning_log.csv") if (C.OUT_TAB / "xgb_tuning_log.csv").exists() else None
    sig = pd.read_csv(C.OUT_TAB / "sigma_table.csv") if (C.OUT_TAB / "sigma_table.csv").exists() else None
    df = la.run_audit(ctx, book=book, sigma_df=sig, tuning_log=tuning)
    n_fail = int((~df["passed"]).sum())
    print(f"  leakage audit: {len(df) - n_fail}/{len(df)} checks passed")
    if n_fail:
        raise RuntimeError(f"LEAKAGE AUDIT FAILED ({n_fail} checks) - fix and re-run before using any result:\n{df[~df['passed']]}")


STAGES = {
    "env": stage_env, "acquire": stage_acquire, "inspect": stage_inspect, "prepare": stage_prepare, "eda": stage_eda,
    "forecast": stage_forecast, "evaluate": stage_evaluate, "explain": stage_explain, "uncertainty": stage_uncertainty,
    "inventory": stage_inventory, "audit": stage_audit,
}
