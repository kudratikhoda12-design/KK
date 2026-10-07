"""README, methodology / results summaries and the final audit checklist (Phase 42).

The final audit ticks a box only when a programmatic check on the produced files succeeds, and re-derives three headline numbers
independently (from raw forecast arrays and from the daily ledger) to guard against transcription or pipeline errors.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .interview import Numbers
from .reporting import Facts, PRIMARY, md, sgn

ROOT = C.ROOT


# ======================================================================================
# independent re-derivations
# ======================================================================================
def rederive_wape_xgboost_h28(f: Facts) -> tuple[bool, str]:
    from .forecast_core import load_context, truth_sums
    ctx = load_context()
    z = np.load(C.DATA_PROC / "forecasts.npz")
    arr = z["test|XGBoost"]                                           # (series, origins, horizons)
    H = 28
    origins = ctx.split.eval_origins("test", H)
    first = ctx.split.all_origins("test")[0]
    pred = arr[:, origins - first, C.HORIZONS.index(H)]
    y = truth_sums(ctx.Y, H)[:, origins]
    wape = 100 * np.abs(y - pred).sum() / y.sum()
    table = f.wape("XGBoost", 28)
    return abs(wape - table) < 1e-6, f"recomputed {wape:.6f} vs table {table:.6f}"


def rederive_costs_from_ledger(f: Facts) -> tuple[bool, str]:
    led = f.ledger
    out = []
    ok = True
    for pol, group in (("A_historical", "A_historical"), (f.headB[7], "B_forecast_driven")):
        g = led[led.policy == pol]
        tot = g.total_cost.sum()
        tab = float(f.inv(group, 7, 0.95, "MEDIUM")["Total_Cost"])
        fill = g.fulfilled_demand.sum() / g.demand.sum()
        tabf = float(f.inv(group, 7, 0.95, "MEDIUM")["Fill_Rate"])
        ok &= abs(tot - tab) < 1e-6 and abs(fill - tabf) < 1e-9
        out.append(f"{pol}: ledger total cost {tot:.3f} vs table {tab:.3f}; fill rate {fill:.5f} vs {tabf:.5f}")
    return bool(ok), "; ".join(out)


# ======================================================================================
# final audit
# ======================================================================================
_PLACEHOLDER = re.compile(r"\{(?:N|f|self|sp|best|ratio|conc7)\.[A-Za-z_]|\{[A-Za-z_]+\[['\"]")
_MISSING = re.compile(r"\b(?:nan|NaN|None|inf)\b")


def prose_defects() -> list[str]:
    """Unrendered template placeholders or missing-value tokens in the narrative of the generated documents (tables and code blocks are skipped)."""
    bad = []
    for p in [C.REPORTS / n for n in ("final_report.md", "interview_guide.md", "project_story_and_resume.md", "results_summary.md", "methodology_summary.md")] + [ROOT / "README.md"]:
        if not p.exists():
            continue
        in_code = False
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code or line.startswith("|"):
                continue
            if _PLACEHOLDER.search(line) or _MISSING.search(line):
                bad.append(f"{p.name}:{i}")
    return bad


def _exists(*names) -> bool:
    return all((p if isinstance(p, Path) else ROOT / p).exists() for p in names)


def build_final_audit(f: Facts) -> pd.DataFrame:
    rep = (C.REPORTS / "final_report.md").read_text()
    tests_txt = (C.REPORTS / "test_results.txt").read_text() if (C.REPORTS / "test_results.txt").exists() else ""
    inv_cols = set(f.inv_final.columns)
    models_test = set(f.test.model)
    figs = {p.name for p in C.OUT_FIG.glob("*.png")}
    sm = [p for p in (ROOT / "src").glob("*.py")]
    compile_ok = subprocess.run([sys.executable, "-m", "compileall", "-q", str(ROOT / "src")], capture_output=True).returncode == 0
    markers = ("TO" + "DO", "FIX" + "ME")
    todo = [p.name for p in sm if any(t in p.read_text() for t in markers)]
    r1 = rederive_wape_xgboost_h28(f)
    r2 = rederive_costs_from_ledger(f)
    rows = []

    def chk(section, item, ok, evidence):
        rows.append({"section": section, "item": item, "done": bool(ok), "evidence": evidence})

    ver = f.ver
    chk("DATA", "Dataset source documented", _exists(C.REPORTS / "data_source.md") and "Hugging Face" in (C.REPORTS / "data_source.md").read_text(), "reports/data_source.md (mirrors, 401 on Kaggle, hashes)")
    chk("DATA", "Raw data preserved", all(_exists(C.DATA_RAW / n) for n in C.RAW_FILES.values()) and bool(ver[["hash_matches_primary", "hash_matches_crosscheck"]].all().all()),
        "raw CSVs untouched; SHA-256 equal to both mirrors' published hashes (raw_data_verification.csv)")
    chk("DATA", "Data dictionary created", _exists(C.REPORTS / "data_dictionary.md"), "reports/data_dictionary.md (meaning, dtype, use, leakage risk)")
    chk("DATA", "Cleaning documented", "PROBLEM" in (C.REPORTS / "data_cleaning.md").read_text(), "reports/data_cleaning.md (PROBLEM -> DIAGNOSIS -> ACTION -> REASON)")
    chk("DATA", "Zero-demand issue analyzed", _exists(C.REPORTS / "zero_sales_analysis.md") and "may underestimate true demand" in rep, "reports/zero_sales_analysis.md; limitation sentence in final report")
    chk("FORECASTING", "Target defined", "Target_H" in rep, "final_report section 7")
    chk("FORECASTING", "Train/validation/test chronological", bool(f.audit[f.audit.check.str.contains("chronological")].status.eq("PASS").all()), "leakage audit: split is chronological; reports/validation_strategy.md")
    chk("FORECASTING", "Leakage audit passed", bool(f.audit.status.eq("PASS").all()), f"{int(f.audit.status.eq('PASS').sum())}/{len(f.audit)} checks PASS (reports/leakage_audit.md)")
    chk("FORECASTING", "Naive implemented", "Naive" in models_test, "forecast_summary_test.csv")
    chk("FORECASTING", "Seasonal Naive implemented", "SeasonalNaive" in models_test, "forecast_summary_test.csv")
    chk("FORECASTING", "Moving Average implemented (7/14/28, chosen on validation)", "MovingAverage" in models_test and len(f.ma) == 12, "ma_window_selection.csv")
    chk("FORECASTING", "ETS implemented/evaluated", {"SES", "Holt", "HoltWinters"} <= models_test, "SES, Holt, Holt-Winters in the results; ets_fit_log.csv")
    chk("FORECASTING", "SARIMA evaluated", "SARIMA" in models_test and _exists(C.OUT_TAB / "sarima_candidates.csv"), "5 candidates per series; sarima_selection.csv")
    chk("FORECASTING", "XGBoost implemented", "XGBoost" in models_test and _exists(C.OUT_MOD / "xgb_H7_final_train_val.json"), "outputs/models/xgb_H*_*.json")
    chk("FORECASTING", "Hyperparameters selected correctly (validation only)", not any(c.startswith("test_") for c in f.xgb_log.columns) and bool(f.audit[f.audit.check.str.contains("tuning")].status.eq("PASS").all()),
        "xgb_tuning_log.csv holds validation metrics only; rows cut before validation")
    chk("FORECASTING", "Metrics calculated (MAE, RMSE, sMAPE, WAPE)", _exists(C.OUT_TAB / "forecast_results.csv", C.OUT_TAB / "final_forecasting_results.csv") and {"MAE", "RMSE", "sMAPE", "WAPE"} <= set(rd_cols("final_forecasting_results.csv")),
        "per model x product x horizon (forecast_results.csv) and pooled test table")
    chk("FORECASTING", "Models compared (best per horizon, DM tests)", _exists(C.OUT_TAB / "best_model_by_horizon.csv", C.OUT_TAB / "dm_tests_panel.csv"), "best_model_by_horizon.csv; dm_tests_panel.csv")
    chk("EXPLAINABILITY", "SHAP completed", _exists(C.OUT_TAB / "shap_importance.csv") and "shap_summary_H7.png" in figs and "shap_importance_H7.png" in figs, "TreeExplainer, additivity checked; summary/bar/dependence figures")
    chk("EXPLAINABILITY", "Feature importance analyzed", "## 14. SHAP" in rep and bool(f.shap_chk.additivity_max_abs_error.max() < 1e-3), "final_report section 14")
    chk("EXPLAINABILITY", "Individual predictions explained (>= 3)", {"shap_individual_H7_1.png", "shap_individual_H7_2.png", "shap_individual_H7_3.png"} <= figs, "3 waterfall plots")
    chk("UNCERTAINTY", "Forecast uncertainty estimated", _exists(C.OUT_TAB / "sigma_table.csv", C.OUT_TAB / "uncertainty_normality.csv"), "sigma_table.csv (validation RMSE), normality diagnostics")
    chk("UNCERTAINTY", "Prediction intervals", _exists(C.OUT_TAB / "uncertainty_coverage_test.csv") and "unc_01_prediction_interval_H7.png" in figs, "80/90% intervals, honest test coverage")
    chk("INVENTORY", "Safety stock", "SS = z" in rep or "SS       = z" in rep, "formula + z-values in final report; tests/test_inventory.py")
    chk("INVENTORY", "ROP", "ROP_t" in rep, "final_report section 16")
    chk("INVENTORY", "Inventory policy (ROP + EOQ, order-up-to)", "EOQ" in rep, "src/inventory.py")
    chk("INVENTORY", "Baseline policy (A)", "A_historical" in set(f.inv_final.Policy), "final_inventory_results.csv")
    chk("INVENTORY", "Simulation (daily ledger)", _exists(C.OUT_TAB / "inventory_ledger_headline_L7_SL95.csv"), "ledger with begin/demand/fulfilled/stockout/orders/costs")
    chk("INVENTORY", "Stockouts", {"Stockouts", "Stockout_Units"} <= inv_cols, "final_inventory_results.csv")
    chk("INVENTORY", "Service level (cycle, fill rate, in-stock)", {"Fill_Rate", "Cycle_Service_Level", "In_Stock_Day_Rate"} <= inv_cols, "final_inventory_results.csv")
    chk("INVENTORY", "Holding / ordering / stockout / total cost", {"Holding_Cost", "Ordering_Cost", "Stockout_Cost", "Total_Cost"} <= inv_cols, "final_inventory_results.csv (3 penalty scenarios)")
    chk("ROBUSTNESS", "Service-level sensitivity", _exists(C.OUT_TAB / "sensitivity_headline_policies.csv") and "inv_03_service_level_vs_cost.png" in figs, "sensitivity tables + figures")
    chk("ROBUSTNESS", "Lead-time sensitivity", bool(set(f.sens.L) == set(C.LEAD_TIMES)), "L = 3, 7, 14 days")
    chk("ROBUSTNESS", "Cost sensitivity", bool(set(f.sens.scenario) == set(C.SCENARIOS)), "LOW / MEDIUM / HIGH stockout penalty")
    chk("ROBUSTNESS", "Product-level robustness (tiers, intermittent vs regular)", _exists(C.OUT_TAB / "robustness_forecast_by_group.csv", C.OUT_TAB / "robustness_inventory_by_group.csv"), "robustness_*.csv; per-product metrics in forecast_results.csv")
    passed_line = [l for l in tests_txt.splitlines() if "passed" in l]
    chk("QUALITY", "Unit tests (all pass)", bool(passed_line) and "failed" not in tests_txt.lower().split("passed")[-1] and "error" not in (passed_line[-1].lower() if passed_line else ""), passed_line[-1].strip() if passed_line else "tests not run yet (python run_project.py --only tests)")
    chk("QUALITY", "Leakage tests", _exists(ROOT / "tests" / "test_models.py") and "test_perturbation_test_has_power" in (ROOT / "tests" / "test_models.py").read_text(), "tests/test_models.py (incl. negative control)")
    chk("QUALITY", "Code cleaned", compile_ok and not todo, "all modules compile; no " + "TO" + "DO/FIX" + "ME markers" + (f" (found in {todo})" if todo else ""))
    chk("QUALITY", "README", _exists(ROOT / "README.md"), "README.md")
    chk("QUALITY", "Requirements (pinned)", "==" in (ROOT / "requirements.txt").read_text(), "requirements.txt")
    defects = prose_defects()
    chk("QUALITY", "Generated documents free of unrendered placeholders / missing-value tokens", not defects, "checked final_report, interview_guide, project story, summaries, README" + (f"; FOUND: {defects[:5]}" if defects else ""))
    chk("QUALITY", "Final report", len(rep.splitlines()) > 300, "reports/final_report.md (24 sections)")
    chk("QUALITY", "Interview guide", _exists(C.REPORTS / "interview_guide.md"), "reports/interview_guide.md")
    chk("QUALITY", "Resume bullets and stories", _exists(C.REPORTS / "project_story_and_resume.md"), "reports/project_story_and_resume.md")
    chk("NO FABRICATION", "Independent re-derivation: XGBoost 28-day test WAPE", r1[0], r1[1])
    chk("NO FABRICATION", "Independent re-derivation: headline total cost and fill rate from the ledger", r2[0], r2[1])
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT_TAB / "final_audit.csv", index=False)
    L = ["# Final audit (Phase 42)", "",
         f"**{int(df.done.sum())} of {len(df)} checks pass.** Each box is ticked only if a programmatic check on the produced files succeeded (`src/docs.py`).", ""]
    for sec, g in df.groupby("section", sort=False):
        L.append(f"## {sec}")
        L.append("")
        for r in g.itertuples():
            L.append(f"- [{'x' if r.done else ' '}] {r.item} - {r.evidence}")
        L.append("")
    (C.REPORTS / "final_audit.md").write_text("\n".join(L) + "\n")
    return df


def rd_cols(name: str) -> list[str]:
    return list(pd.read_csv(C.OUT_TAB / name, nrows=1).columns)


# ======================================================================================
# methodology + results summaries
# ======================================================================================
def build_methodology_summary(f: Facts) -> None:
    sp = f.split
    L = ["# Methodology summary", "",
         "**Question.** Do accurate, explainable demand forecasts lead to better inventory decisions (service vs holding cost)?", "",
         "| step | what was done |", "|---|---|",
         "| Data | M5 Walmart (30,490 series x 1,941 days) from two SHA-256-verified public mirrors; sample of 18 item-store series chosen with TRAIN information only (store per state, listing eligibility, percentile demand tiers, seeded draws) |",
         f"| Split | chronological {sp.date_range('train')[0].date()}..{sp.date_range('train')[1].date()} / {sp.date_range('validation')[0].date()}..{sp.date_range('validation')[1].date()} / {sp.date_range('test')[0].date()}..{sp.date_range('test')[1].date()}; daily rolling origins |",
         "| Target | Target_H(t) = sum of demand over t+1..t+H, H = 7, 14, 28 (H = 3 supplementary for the 3-day lead time) |",
         "| Baselines | naive, seasonal naive, moving average (window by validation), Croston-SBA |",
         "| Statistical | SES, Holt (damped), Holt-Winters additive s = 7 (AICc), SARIMA 5 candidates s = 7 (validation) with ADF/KPSS/ACF/PACF |",
         "| ML | direct XGBoost per horizon, global model, 25 leakage-safe features (history / calendar-of-origin / known-ahead calendar counts), 24-config validation search |",
         "| Leakage control | features at origin; training rows cut before the next period; perturbation tests (40 checks) incl. negative control |",
         "| Explainability | TreeSHAP on the final models, additivity verified, 3 rule-selected individual explanations |",
         "| Uncertainty | sigma_L = validation RMSE of the H-day sum; Gaussian vs empirical intervals chosen by split-half calibration; test coverage reported |",
         "| Inventory | (s, S) policy: ROP = mu_L + z sigma_L, Q = EOQ; order arrives at start of day t+L+1; lost sales; hypothetical costs |",
         "| Policies | A: trailing 28-day mean and sqrt(L) x trailing std; B: forecast model + validation sigma; identical lead times, costs, period and initial inventory |",
         "| Evaluation | MAE, RMSE, sMAPE, WAPE; Diebold-Mariano (HAC, HLN, Holm); paired bootstrap over series for costs; fill rate, cycle service level, in-stock days, turnover |",
         "| Sensitivity | 3 penalty scenarios x 3 service levels x 3 lead times; accuracy-vs-cost rank correlation; tier and regularity subgroups |", ""]
    (C.REPORTS / "methodology_summary.md").write_text("\n".join(L) + "\n")


def build_results_summary(f: Facts) -> None:
    N = Numbers(f)
    pt = f.test[f.test.H.isin(PRIMARY)].pivot(index="model", columns="H", values="WAPE")
    pt.columns = [f"{c}-day WAPE %" for c in pt.columns]
    pt = pt.sort_values(pt.columns[-1]).reset_index().rename(columns={"model": "Model"})
    inv = f.inv_final[(f.inv_final.Cost_Scenario == "MEDIUM") & (f.inv_final.Service_Level == 0.95)][["Policy", "Forecast_Model", "Lead_Time_days", "Stockouts", "Fill_Rate", "Cycle_Service_Level", "Average_Inventory", "Holding_Cost", "Ordering_Cost", "Stockout_Cost", "Total_Cost"]]
    L = ["# Results summary", "", "*Test-period results; costs hypothetical.*", "", "## Forecasting (pooled test WAPE, lower is better)", "", md(pt), "",
         "Best by horizon: " + ", ".join(f"{H}-day: {N.best[H]['best_by_test_WAPE']} ({N.best[H]['test_WAPE']:.1f}%)" for H in PRIMARY) + ".", "",
         "## Inventory (MEDIUM penalty, 95% target)", "", md(inv, floatfmt=".3f"), "",
         f"B vs A total cost: {sgn(N.g[3])} (L=3), {sgn(N.g[7])} (L=7), {sgn(N.g[14])} (L=14); cheaper in {N.n_cheaper}/{N.n_cfg} grid cells; bootstrap intervals include zero in {N.n_cfg - N.n_ci_excl0}/{N.n_cfg}.", "",
         f"Accuracy vs cost: lowest-WAPE model is the cheapest in {N.same_best}/{len(f.avc_s)} cells; mean Spearman rho (excl. Naive) LOW {N.rho['LOW']:+.2f}, MEDIUM {N.rho['MEDIUM']:+.2f}, HIGH {N.rho['HIGH']:+.2f}.", ""]
    (C.REPORTS / "results_summary.md").write_text("\n".join(L) + "\n")


# ======================================================================================
# README
# ======================================================================================
FIG_DESC = {
    "eda_01_daily_rolling.png": "daily demand, rolling mean/std", "eda_02_weekly_monthly.png": "weekly / monthly demand", "eda_03_distribution.png": "demand distribution",
    "eda_04_dow_monthly_seasonality.png": "weekday and monthly seasonality", "eda_05_acf_pacf.png": "ACF / PACF", "eda_09_acf_pacf_seasonally_differenced.png": "ACF / PACF after seasonal differencing",
    "eda_06_intermittency_map.png": "ADI-CV2 intermittency map", "eda_07_price_variation.png": "price variation", "eda_08_quarterly_index_heatmap.png": "trends / regime changes heatmap",
    "fc_01_forecast_vs_actual.png": "forecast vs actual (test)", "fc_02_model_comparison_wape.png": "model comparison (WAPE)", "fc_03_error_comparison_skill.png": "error comparison vs moving average",
    "fc_04_series_level_wape_H7.png": "per-series WAPE (7 days)", "fc_04_series_level_wape_H28.png": "per-series WAPE (28 days)",
    "resid_01_statistical_models.png": "residual diagnostics, ETS/SARIMA", "resid_02_xgboost_H7.png": "XGBoost residuals (7-day)", "resid_02_xgboost_H28.png": "XGBoost residuals (28-day)",
    "shap_summary_H7.png": "SHAP summary", "shap_importance_H7.png": "SHAP importance", "shap_dependence_H7.png": "SHAP dependence", "shap_individual_H7_1.png": "SHAP individual explanation (1 of 3)",
    "unc_01_prediction_interval_H7.png": "prediction interval", "unc_02_coverage_calibration.png": "interval calibration", "unc_03_residual_normality_XGBoost_H7.png": "residual normality (Q-Q)",
    "inv_01_trajectory_FOODS_2_347_TX_2.png": "inventory trajectory A vs B", "inv_02_stockout_periods.png": "stockout periods", "inv_03_service_level_vs_cost.png": "service level vs cost",
    "inv_04_cost_vs_fill_rate.png": "cost vs fill rate", "inv_05_total_cost_components.png": "total cost comparison", "inv_06_sensitivity_heatmaps.png": "sensitivity heatmaps",
    "inv_07_accuracy_vs_cost.png": "accuracy vs business value", "robust_01_tiers_and_regularity.png": "robustness by tier / regularity",
}


def _n_tests() -> str:
    t = (C.REPORTS / "test_results.txt")
    if t.exists():
        import re
        m = re.search(r"(\d+) passed", t.read_text())
        if m:
            return m.group(1)
    return "many"


def _runtime_txt() -> str:
    p = C.OUT_TAB / "pipeline_runtime.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[~d.stage.isin(["acquire"])]
        return f"about {d.seconds.sum() / 60:.0f} minutes on 4 cores without the download (last full run)"
    return "a few minutes on 4 cores"


def build_readme(f: Facts) -> None:
    N = Numbers(f)
    pt = f.test[f.test.H.isin(PRIMARY)].pivot(index="model", columns="H", values="WAPE")
    pt.columns = [f"{c}-day" for c in pt.columns]
    pt = pt.sort_values(pt.columns[-1]).reset_index().rename(columns={"model": "Test WAPE %"})
    figs = [(n, d) for n, d in FIG_DESC.items() if (C.OUT_FIG / n).exists()]
    L = ["# Explainable Demand Forecasting & Inventory Optimization", "",
         "**Forecast -> decision -> business impact.** Can accurate, explainable retail demand forecasts be converted into better inventory decisions by balancing stockout risk against holding cost? "
         "An M.Tech (Quality, Reliability & Operations Research) portfolio project on the M5 Walmart data: time-series baselines, exponential smoothing, SARIMA, XGBoost + SHAP, forecast uncertainty, a reorder-point/EOQ policy and a daily inventory simulation - "
         "with a leakage audit and honest reporting (including results that go *against* the 'ML wins' narrative).", "",
         "## Headline results (all computed; costs are hypothetical)", "",
         f"* **Accuracy (test, pooled WAPE):** best {N.best[7]['best_by_test_WAPE']} {N.best[7]['test_WAPE']:.1f}% (7-day), {N.best[14]['best_by_test_WAPE']} {N.best[14]['test_WAPE']:.1f}% (14-day), {N.best[28]['best_by_test_WAPE']} {N.best[28]['test_WAPE']:.1f}% (28-day); "
         + ("the leading models are statistically hard to separate (Diebold-Mariano, Holm-adjusted)." if N.xgb_never_better else "see the Diebold-Mariano tests (Holm-adjusted) for which differences are significant."),
         f"* **SHAP:** 7/14/28-day rolling means carry {N.roll[7]:.0f}% (7-day) / {N.roll[28]:.0f}% (28-day) of XGBoost's attribution.",
         f"* **Inventory:** forecast-driven policy vs historical baseline: {sgn(N.g[3])} / {sgn(N.g[7])} / {sgn(N.g[14])} total cost at lead times 3 / 7 / 14 days (95% target, medium penalty); "
         f"cheaper in {N.n_cheaper}/{N.n_cfg} grid cells but bootstrap intervals include zero in {N.n_cfg - N.n_ci_excl0}/{N.n_cfg}; "
         + (f"{100 * N.ma_gain7 / N.g[7]:.0f}% of the L=7 change comes from validation-based sigma alone (same point forecast as the baseline)." if N.same_pf7 else f"a moving-average forecast with validation-based sigma gives {sgn(N.ma_gain7)} at L=7."),
         f"* **Accuracy != business value:** the lowest-WAPE model is cheapest in only {N.same_best}/{len(f.avc_s)} cells (mean rank correlation LOW {N.rho['LOW']:+.2f}, MEDIUM {N.rho['MEDIUM']:+.2f}, HIGH {N.rho['HIGH']:+.2f}).", "",
         md(pt), "",
         "Full write-up: [`reports/final_report.md`](reports/final_report.md). Interview preparation: [`reports/interview_guide.md`](reports/interview_guide.md), [`reports/project_story_and_resume.md`](reports/project_story_and_resume.md).", "",
         "## Quick start", "",
         "```bash\npip install -r requirements.txt\npython run_project.py            # whole pipeline, {RUNTIME} (first run downloads ~450 MB)\npython run_project.py --from forecast     # resume\npython run_project.py --only evaluate     # one stage\npytest -q                                   # unit tests (no data download needed for most)\n```", "",
         "Stages: `env, acquire, inspect, prepare, eda, forecast, evaluate, explain, uncertainty, inventory, audit, tests, reports`. Seeds are fixed (`config.SEED = 42`); results reproduce exactly on the same package versions (see `requirements.txt`)." + (" A clean-copy rerun is documented in [`reports/reproducibility_check.md`](reports/reproducibility_check.md)." if (C.REPORTS / "reproducibility_check.md").exists() else ""), "",
         "**Data prerequisite.** The official Kaggle files need credentials; the pipeline instead downloads the four original files from two public Hugging Face mirrors and verifies their SHA-256 hashes against both (see `reports/data_source.md`). "
         "If the files are already in `data/raw/`, nothing is downloaded. Raw data are never modified (and are git-ignored because of their size).", "",
         "## Repository layout", "",
         "```\ndemand_forecasting_inventory/\n+-- data/raw/            original M5 files (not committed)\n+-- data/processed/      panel.parquet, selected_series.csv, forecasts.npz\n+-- src/                 pipeline code (see below)\n+-- tests/               {NTESTS} unit / leakage tests\n+-- notebooks/           01_results_walkthrough.ipynb\n+-- outputs/figures|tables|models\n+-- reports/             final_report.md, interview_guide.md, validation_strategy.md, leakage_audit.md, data_cleaning.md, eda.md, ...\n+-- run_project.py       end-to-end runner\n+-- requirements.txt\n```", "",
         "| module | role |", "|---|---|",
         "| `config.py` | all constants, seeds, hypothetical cost assumptions |", "| `data_acquisition.py`, `data_loading.py`, `data_inspection.py`, `data_prep.py`, `subset_selection.py` | download + verification, wide-to-long, inspection, cleaning log, zero-sales analysis, TRAIN-only sampling |",
         "| `splits.py`, `forecast_core.py` | chronological split, origins, true H-day sums, `ForecastBook` |", "| `baselines.py`, `stat_models.py`, `features.py`, `xgb_models.py` | forecasting models and leakage-safe features |",
         "| `evaluation.py`, `metrics.py`, `residuals.py`, `explain.py`, `uncertainty.py` | accuracy tables, Diebold-Mariano, residuals, SHAP, intervals |",
         "| `inventory.py`, `inventory_experiments.py`, `inventory_analysis.py`, `robustness.py` | EOQ/ROP, daily simulator, experiments, sensitivity, accuracy-vs-cost |",
         f"| `leakage_audit.py` | {len(f.audit)} automated leakage checks incl. perturbation tests |", "| `eda.py`, `plots.py`, `viz.py` | figures (validated colour tokens, one axis, question-style titles) |", "| `reporting.py`, `interview.py`, `docs.py` | all reports generated from result tables |", "",
         "## Design decisions (why not something fancier?)", "",
         "* **Chronological everything**, a frozen protocol, test evaluated once; choices on validation; refit on train+validation.",
         "* **Direct multi-horizon XGBoost on raw units** (explainable SHAP in demand units); no deep learning - 18 series do not justify it and explainability matters.",
         "* **Same target for every model** (H-day sums) = the lead-time demand the inventory policy consumes.",
         "* **Hypothetical costs are labelled as such everywhere;** conclusions are checked across three penalty scenarios.", "",
         "## Limitations (details in the final report)", "",
         f"Observed sales may underestimate true demand during stockouts; costs/lead times are assumptions; 18 series from {len(f.log['stores'])} stores; one {f.split.n_days - f.split.val_end}-day test period that was harder than validation for {'every' if N.ratio.r.min() > 1 else 'most'} model; bootstrap/DM intervals are wide.", "",
         "## Figures", "", "| figure | shows |", "|---|---|"] + [f"| `outputs/figures/{n}` | {d} |" for n, d in figs] + ["",
         "## Citation / data licence", "Makridakis, Spiliotis & Assimakopoulos (2022), *The M5 competition: Background, organization, and implementation*, International Journal of Forecasting 38(4). M5 data are provided by the organisers for research use.", ""]
    text = "\n".join(L) + "\n"
    text = text.replace("{RUNTIME}", _runtime_txt()).replace("{NTESTS}", _n_tests())
    (ROOT / "README.md").write_text(text)


def build_all() -> None:
    from . import interview, reporting
    f = Facts()
    reporting.build_validation_strategy(f)
    reporting.build_final_report(f)
    interview.build_interview_guide(f)
    interview.build_project_story(f)
    build_methodology_summary(f)
    build_results_summary(f)
    build_readme(f)
    build_final_audit(f)                      # last: it inspects the files written above
    print("reports written:", ", ".join(sorted(p.name for p in C.REPORTS.glob("*.md"))))
