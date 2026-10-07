# Explainable Demand Forecasting & Inventory Optimization

**Forecast -> decision -> business impact.** Can accurate, explainable retail demand forecasts be converted into better inventory decisions by balancing stockout risk against holding cost? An M.Tech (Quality, Reliability & Operations Research) portfolio project on the M5 Walmart data: time-series baselines, exponential smoothing, SARIMA, XGBoost + SHAP, forecast uncertainty, a reorder-point/EOQ policy and a daily inventory simulation - with a leakage audit and honest reporting (including results that go *against* the 'ML wins' narrative).

## Headline results (all computed; costs are hypothetical)

* **Accuracy (test, pooled WAPE):** best Holt 37.6% (7-day), XGBoost 32.8% (14-day), XGBoost 29.8% (28-day); the leading models are statistically hard to separate (Diebold-Mariano, Holm-adjusted).
* **SHAP:** 7/14/28-day rolling means carry 74% (7-day) / 73% (28-day) of XGBoost's attribution.
* **Inventory:** forecast-driven policy vs historical baseline: -2.5% / -8.5% / -13.9% total cost at lead times 3 / 7 / 14 days (95% target, medium penalty); cheaper in 21/27 grid cells but bootstrap intervals include zero in 27/27; 86% of the L=7 change comes from validation-based sigma alone (same point forecast as the baseline).
* **Accuracy != business value:** the lowest-WAPE model is cheapest in only 2/27 cells (mean rank correlation LOW +0.47, MEDIUM -0.08, HIGH -0.26).

| Test WAPE %   |   7-day |   14-day |   28-day |
|:--------------|--------:|---------:|---------:|
| XGBoost       |   38.46 |    32.79 |    29.85 |
| SARIMA        |   38.01 |    33.71 |    31.56 |
| MovingAverage |   39.42 |    34.48 |    32.15 |
| Holt          |   37.57 |    34.22 |    32.73 |
| HoltWinters   |   37.70 |    34.35 |    32.85 |
| SES           |   37.60 |    34.31 |    32.86 |
| Croston       |   40.66 |    36.04 |    33.63 |
| SeasonalNaive |   40.91 |    39.90 |    40.33 |
| Naive         |   70.42 |    70.88 |    72.82 |

Full write-up: [`reports/final_report.md`](reports/final_report.md). Interview preparation: [`reports/interview_guide.md`](reports/interview_guide.md), [`reports/project_story_and_resume.md`](reports/project_story_and_resume.md).

## Quick start

```bash
pip install -r requirements.txt
python run_project.py            # whole pipeline, about 3 minutes on 4 cores without the download (last full run) (first run downloads ~450 MB)
python run_project.py --from forecast     # resume
python run_project.py --only evaluate     # one stage
pytest -q                                   # unit tests (no data download needed for most)
```

Stages: `env, acquire, inspect, prepare, eda, forecast, evaluate, explain, uncertainty, inventory, audit, tests, reports`. Seeds are fixed (`config.SEED = 42`); results reproduce exactly on the same package versions (see `requirements.txt`). A clean-copy rerun is documented in [`reports/reproducibility_check.md`](reports/reproducibility_check.md).

**Data prerequisite.** The official Kaggle files need credentials; the pipeline instead downloads the four original files from two public Hugging Face mirrors and verifies their SHA-256 hashes against both (see `reports/data_source.md`). If the files are already in `data/raw/`, nothing is downloaded. Raw data are never modified (and are git-ignored because of their size).

## Repository layout

```
demand_forecasting_inventory/
+-- data/raw/            original M5 files (not committed)
+-- data/processed/      panel.parquet, selected_series.csv, forecasts.npz
+-- src/                 pipeline code (see below)
+-- tests/               76 unit / leakage tests
+-- notebooks/           01_results_walkthrough.ipynb
+-- outputs/figures|tables|models
+-- reports/             final_report.md, interview_guide.md, validation_strategy.md, leakage_audit.md, data_cleaning.md, eda.md, ...
+-- run_project.py       end-to-end runner
+-- requirements.txt
```

| module | role |
|---|---|
| `config.py` | all constants, seeds, hypothetical cost assumptions |
| `data_acquisition.py`, `data_loading.py`, `data_inspection.py`, `data_prep.py`, `subset_selection.py` | download + verification, wide-to-long, inspection, cleaning log, zero-sales analysis, TRAIN-only sampling |
| `splits.py`, `forecast_core.py` | chronological split, origins, true H-day sums, `ForecastBook` |
| `baselines.py`, `stat_models.py`, `features.py`, `xgb_models.py` | forecasting models and leakage-safe features |
| `evaluation.py`, `metrics.py`, `residuals.py`, `explain.py`, `uncertainty.py` | accuracy tables, Diebold-Mariano, residuals, SHAP, intervals |
| `inventory.py`, `inventory_experiments.py`, `inventory_analysis.py`, `robustness.py` | EOQ/ROP, daily simulator, experiments, sensitivity, accuracy-vs-cost |
| `leakage_audit.py` | 40 automated leakage checks incl. perturbation tests |
| `eda.py`, `plots.py`, `viz.py` | figures (validated colour tokens, one axis, question-style titles) |
| `reporting.py`, `interview.py`, `docs.py` | all reports generated from result tables |

## Design decisions (why not something fancier?)

* **Chronological everything**, a frozen protocol, test evaluated once; choices on validation; refit on train+validation.
* **Direct multi-horizon XGBoost on raw units** (explainable SHAP in demand units); no deep learning - 18 series do not justify it and explainability matters.
* **Same target for every model** (H-day sums) = the lead-time demand the inventory policy consumes.
* **Hypothetical costs are labelled as such everywhere;** conclusions are checked across three penalty scenarios.

## Limitations (details in the final report)

Observed sales may underestimate true demand during stockouts; costs/lead times are assumptions; 18 series from 3 stores; one 292-day test period that was harder than validation for every model; bootstrap/DM intervals are wide.

## Figures

| figure | shows |
|---|---|
| `outputs/figures/eda_01_daily_rolling.png` | daily demand, rolling mean/std |
| `outputs/figures/eda_02_weekly_monthly.png` | weekly / monthly demand |
| `outputs/figures/eda_03_distribution.png` | demand distribution |
| `outputs/figures/eda_04_dow_monthly_seasonality.png` | weekday and monthly seasonality |
| `outputs/figures/eda_05_acf_pacf.png` | ACF / PACF |
| `outputs/figures/eda_09_acf_pacf_seasonally_differenced.png` | ACF / PACF after seasonal differencing |
| `outputs/figures/eda_06_intermittency_map.png` | ADI-CV2 intermittency map |
| `outputs/figures/eda_07_price_variation.png` | price variation |
| `outputs/figures/eda_08_quarterly_index_heatmap.png` | trends / regime changes heatmap |
| `outputs/figures/fc_01_forecast_vs_actual.png` | forecast vs actual (test) |
| `outputs/figures/fc_02_model_comparison_wape.png` | model comparison (WAPE) |
| `outputs/figures/fc_03_error_comparison_skill.png` | error comparison vs moving average |
| `outputs/figures/fc_04_series_level_wape_H7.png` | per-series WAPE (7 days) |
| `outputs/figures/fc_04_series_level_wape_H28.png` | per-series WAPE (28 days) |
| `outputs/figures/resid_01_statistical_models.png` | residual diagnostics, ETS/SARIMA |
| `outputs/figures/resid_02_xgboost_H7.png` | XGBoost residuals (7-day) |
| `outputs/figures/resid_02_xgboost_H28.png` | XGBoost residuals (28-day) |
| `outputs/figures/shap_summary_H7.png` | SHAP summary |
| `outputs/figures/shap_importance_H7.png` | SHAP importance |
| `outputs/figures/shap_dependence_H7.png` | SHAP dependence |
| `outputs/figures/shap_individual_H7_1.png` | SHAP individual explanation (1 of 3) |
| `outputs/figures/unc_01_prediction_interval_H7.png` | prediction interval |
| `outputs/figures/unc_02_coverage_calibration.png` | interval calibration |
| `outputs/figures/unc_03_residual_normality_XGBoost_H7.png` | residual normality (Q-Q) |
| `outputs/figures/inv_01_trajectory_FOODS_2_347_TX_2.png` | inventory trajectory A vs B |
| `outputs/figures/inv_02_stockout_periods.png` | stockout periods |
| `outputs/figures/inv_03_service_level_vs_cost.png` | service level vs cost |
| `outputs/figures/inv_04_cost_vs_fill_rate.png` | cost vs fill rate |
| `outputs/figures/inv_05_total_cost_components.png` | total cost comparison |
| `outputs/figures/inv_06_sensitivity_heatmaps.png` | sensitivity heatmaps |
| `outputs/figures/inv_07_accuracy_vs_cost.png` | accuracy vs business value |
| `outputs/figures/robust_01_tiers_and_regularity.png` | robustness by tier / regularity |

## Citation / data licence
Makridakis, Spiliotis & Assimakopoulos (2022), *The M5 competition: Background, organization, and implementation*, International Journal of Forecasting 38(4). M5 data are provided by the organisers for research use.

