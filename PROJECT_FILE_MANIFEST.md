# Project file manifest

Every file in the repository and in the delivery ZIP (163 files). Generated from the git file list, so it matches the package exactly. Raw and processed market data are **not** included (licence and size); `data/README.md` and `reproduction/README.md` explain how to rebuild them, and `reproduction/checksums/` lists the hash of every input file.

**Columns.** *Type*: source code, test, script, configuration, notebook, report, documentation or output. *Needed to reproduce*: whether a fresh reproduction of the market study needs the file. *Generated*: whether the file is produced by code in this repository.

## Source code, scripts, tests, configuration and documents

| File | Purpose | Type | Needed to reproduce | Generated |
|---|---|---|---|---|
| `.gitattributes` | Excludes delivery archives from `git archive` | Configuration | No | No |
| `.gitignore` | Keeps raw/processed data, caches and virtual environments out of git | Configuration | No | No |
| `.vscode/extensions.json` | VS Code: recommended Python and Jupyter extensions | Configuration | No | No |
| `.vscode/settings.json` | VS Code: interpreter path, pytest discovery, import paths | Configuration | No | No |
| `PROJECT_FILE_MANIFEST.md` | This file: every file, its role and the project tree | Documentation | No | Yes (from git file list) |
| `PROJECT_NOTES.md` | Licence, data-licensing and use notes | Documentation | No | No |
| `README.md` | Project overview, quick start for reviewers, results summary, repository map | Documentation | Guide | No |
| `credit_module/Credit_Risk_Project_Report.pdf` | Earlier credit-risk project: original report (unchanged) | Report | No | No |
| `credit_module/README.md` | Earlier credit-risk project: headline results and evidence status | Documentation | No | No |
| `data/README.md` | Data source, columns, units, sizes, download and build commands, licence | Documentation | Yes | No |
| `notebooks/01_credit_risk.ipynb` | Earlier credit-risk project: original notebook (unchanged; needs LendingClub data) | Notebook | No | No |
| `notebooks/02_market_data_health.ipynb` | Displays saved outputs of the data health stage (RUN_STAGE = False by default) | Notebook | No | Yes (make_notebooks.py; executed) |
| `notebooks/03_market_eda.ipynb` | Displays saved outputs of the EDA stage (RUN_STAGE = False by default) | Notebook | No | Yes (make_notebooks.py; executed) |
| `notebooks/04_hypothesis_features.ipynb` | Displays saved outputs of the hypotheses and features stage (RUN_STAGE = False by default) | Notebook | No | Yes (make_notebooks.py; executed) |
| `notebooks/05_modeling.ipynb` | Displays saved outputs of the models stage (RUN_STAGE = False by default) | Notebook | No | Yes (make_notebooks.py; executed) |
| `notebooks/06_walk_forward_validation.ipynb` | Displays saved outputs of the walk-forward validation stage (RUN_STAGE = False by default) | Notebook | No | Yes (make_notebooks.py; executed) |
| `notebooks/07_backtesting.ipynb` | Displays saved outputs of the backtesting stage (RUN_STAGE = False by default) | Notebook | No | Yes (make_notebooks.py; executed) |
| `notebooks/08_risk_analysis.ipynb` | Displays saved outputs of the risk analysis stage (RUN_STAGE = False by default) | Notebook | No | Yes (make_notebooks.py; executed) |
| `notebooks/09_stress_testing.ipynb` | Displays saved outputs of the stress testing stage (RUN_STAGE = False by default) | Notebook | No | Yes (make_notebooks.py; executed) |
| `notebooks/10_robustness.ipynb` | Displays saved outputs of the robustness stage (RUN_STAGE = False by default) | Notebook | No | Yes (make_notebooks.py; executed) |
| `pyproject.toml` | pytest configuration (test path, import path) | Configuration | Yes (tests) | No |
| `reports/Final_Gravia_Project_Report.docx` | Formal project report — editable | Report | No | Yes (from .md) |
| `reports/Final_Gravia_Project_Report.md` | Formal project report — source | Report | No | No |
| `reports/Final_Gravia_Project_Report.pdf` | Formal project report — for sending | Report | No | Yes (from .docx) |
| `reports/Gravia_Project_One_Page_Summary.html` | One-page summary — source | Report | No | No |
| `reports/Gravia_Project_One_Page_Summary.pdf` | One-page summary — for sending | Report | No | Yes (from .html) |
| `reports/TECHNICAL_REVIEWER_README.md` | Where to look, by time available and by topic | Documentation | No | No |
| `reports/final_report.md` | Detailed working report (27 sections + addendum) | Report | No | No |
| `reports/gravia_mapping_and_cv.md` | Job-description mapping and checked CV entries | Report | No | No |
| `reports/interview_prep_concepts.md` | Interview preparation: concepts | Interview notes | No | No |
| `reports/interview_prep_project.md` | Interview preparation: project questions | Interview notes | No | No |
| `reports/module_connection.md` | How the credit and market modules map onto each other | Report | No | No |
| `reports/phase1_dataset_selection.md` | Dataset research and choice of Binance Vision | Report | No | No |
| `reports/phase2_data_acquisition.md` | Data acquisition record | Report | No | No |
| `reports/pipeline_validation.md` | Synthetic negative/positive control results (method only) | Report | No | No |
| `reports/project_walkthrough.md` | Short plain-English walkthrough of the market study | Report | No | No |
| `reports/research_design_preregistration.md` | Pre-registered design (written before any data) and deviation log §12 | Report | Yes (defines the study) | No |
| `reports/results.json` | Every headline number, by stage (audit, features, validation, test, risk, stress, robustness, ETH) | Output | No | Yes (pipeline) |
| `reports/selection.json` | The frozen validation selection, with reason and code commit | Output | Yes (test stage reads it) | Yes (validation stage) |
| `reports/spot_fee_check.json` | Pre-registered spot-fee check result and metadata | Output | No | Yes (run_spot_fee_check.py) |
| `reports/teaching/Final_Completion_Report.docx` | Completion checklist, verification, reproducibility (docx) | Interview documentation | No | Yes (from .md) |
| `reports/teaching/Final_Completion_Report.md` | Completion checklist, verification, reproducibility (md) | Interview documentation | No | No |
| `reports/teaching/Final_Completion_Report.pdf` | Completion checklist, verification, reproducibility (pdf) | Interview documentation | No | Yes (from .md) |
| `reports/teaching/Final_Interview_Readiness_Audit.docx` | Evidence and claims audit (docx) | Interview documentation | No | Yes (from .md) |
| `reports/teaching/Final_Interview_Readiness_Audit.md` | Evidence and claims audit (md) | Interview documentation | No | No |
| `reports/teaching/Final_Interview_Readiness_Audit.pdf` | Evidence and claims audit (pdf) | Interview documentation | No | Yes (from .md) |
| `reports/teaching/Project_Walkthrough_Raw_Data_to_Verdict.docx` | Teaching / interview-defence walkthrough traced to the code (docx) | Interview documentation | No | Yes (from .md) |
| `reports/teaching/Project_Walkthrough_Raw_Data_to_Verdict.md` | Teaching / interview-defence walkthrough traced to the code (md) | Interview documentation | No | No |
| `reports/teaching/Project_Walkthrough_Raw_Data_to_Verdict.pdf` | Teaching / interview-defence walkthrough traced to the code (pdf) | Interview documentation | No | Yes (from .md) |
| `reproduction/README.md` | Step-by-step reproduction recipe with expected outputs | Documentation | Yes | No |
| `reproduction/checksums/BTCUSDT_1m_spot_manifest.csv` | URL, size and SHA-256 of the 110 BTCUSDT files used | Reference data (hashes only) | Recommended | Yes (download log) |
| `reproduction/checksums/BTCUSDT_fundingRate_manifest.csv` | URL, size and SHA-256 of the 81 funding files used | Reference data (hashes only) | Recommended | Yes (download log) |
| `reproduction/checksums/ETHUSDT_1m_spot_manifest.csv` | URL, size and SHA-256 of the 110 ETHUSDT files used | Reference data (hashes only) | Recommended | Yes (download log) |
| `requirements.txt` | Python dependencies (minimum and tested versions) | Configuration | Yes | No |
| `scripts/download_data.sh` | Shell-only alternative downloader with SHA-256 checks | Script | Optional | No |
| `scripts/make_notebooks.py` | Generates notebooks 02–10 from one specification | Script | No | No |
| `scripts/make_report_figures.py` | Three report figures drawn from saved tables only | Script | Report only | No |
| `scripts/pipeline_controls.py` | Synthetic negative/positive pipeline controls behind reports/pipeline_validation.md | Script | No | No |
| `scripts/report_tools/build_docx.js` | Markdown → Word converter used for the reports (Node.js + docx) | Report tooling | No | No |
| `scripts/run_pipeline.py` | Staged runner: audit, eda, features, validation, test, risk, stress, robustness, second_asset | Script | Yes | No |
| `scripts/run_spot_fee_check.py` | Pre-registered spot-fee variant: re-prices the frozen test positions at 12 bp/side, no funding | Script | Yes (that check) | No |
| `scripts/smoke_test_synthetic.py` | Runs every stage on 4.8M synthetic minutes in a temporary folder (code check, no results) | Script / test | No | No |
| `scripts/verify_checksums.py` | Compares local downloads with the study's reference SHA-256 manifests | Script | Recommended | No |
| `src/__init__.py` | Makes `src` a package | Source code | Yes | No |
| `src/backtest.py` | Positions from probabilities, event-level backtest with turnover costs and funding, buy-and-hold, capacity check | Source code | Yes | No |
| `src/check_data_access.py` | Network probe: can this machine reach Binance Vision (and verify one checksum)? | Helper script | Optional | No |
| `src/config.py` | Every frozen research parameter: paths, periods, horizon, lags, model grid, selection rule, thresholds, costs, statistics settings | Source code / configuration | Yes | No |
| `src/data.py` | Download with SHA-256 verification and manifest; per-file ms/µs timestamp detection; typed loading (no cleaning); funding loader; CLI `python -m src.data` | Source code | Yes | No |
| `src/eda.py` | Exploratory analysis: return distributions, autocorrelation, volume vs returns, intraday profile, regimes; EDA figures; shared plot style | Source code | Yes | No |
| `src/features.py` | 16 closed-candle features, hourly decision grid, target with execution lag, regime labels, leakage perturbation check, timing table | Source code | Yes | No |
| `src/models.py` | Base rate, logistic regression pipeline, LightGBM; walk-forward prediction; metrics, calibration, bootstrap AUC CI, HAC log-loss test, importance, PSI | Source code | Yes | No |
| `src/pipeline.py` | Hypothesis tests, validation stage and frozen selection, single test stage, robustness variants, permutation test | Source code | Yes | No |
| `src/plots.py` | Result figures (equity, calibration, decile returns, Monte Carlo, bar charts) | Source code | Yes | No |
| `src/quality.py` | Data-health audit (18 checks), gap and phase-shift analysis, outlier scan, regular 1-minute grid with documented treatments | Source code | Yes | No |
| `src/risk.py` | Performance metrics, Sharpe CI, PSR/DSR, VaR/ES (historical, normal, Student-t), Kupiec test, block-bootstrap Monte Carlo | Source code | Yes | No |
| `src/stats.py` | Stationary bootstrap, Newey–West (HAC) mean test, block-bootstrap Spearman test, Holm correction | Source code | Yes | No |
| `src/stress_test.py` | Cost, slippage, volatility, signal-flip, regime, worst-day and gap stress scenarios | Source code | Yes | No |
| `src/validation.py` | Purged expanding walk-forward folds (development and test) and the experiment log | Source code | Yes | No |
| `tests/conftest.py` | Synthetic data generator used only by tests | Test | Yes (tests) | No |
| `tests/test_backtest.py` | Cost accounting: constant long, flips, gaps, no-trade region | Test | Yes (tests) | No |
| `tests/test_data_quality.py` | Timestamp units, µs dates, audit detection, grid without filling, resolution-safe close time, funding window, phase-shifted candles | Test | Yes (tests) | No |
| `tests/test_features_leakage.py` | Hourly grid, closed-candle alignment, execution lag, future perturbation, no filling, backward windows | Test | Yes (tests) | No |
| `tests/test_pipeline_controls.py` | End-to-end negative (random walk) and positive (injected momentum) controls | Test | Yes (tests) | No |
| `tests/test_risk.py` | Drawdown, VaR/ES, Kupiec, Deflated Sharpe, PSI | Test | Yes (tests) | No |
| `tests/test_validation.py` | Chronological purged folds; experiment-log key collisions | Test | Yes (tests) | No |

## Output tables (`reports/tables/`, 65 files)

All produced by `scripts/run_pipeline.py` (or `run_spot_fee_check.py`) from the Binance data; none is needed to reproduce the study, all are needed to check it.

| File | Purpose |
|---|---|
| `BTCUSDT_audit_dq_table.csv` | BTCUSDT data-health audit |
| `BTCUSDT_audit_file_summary.csv` | BTCUSDT data-health audit |
| `BTCUSDT_audit_gaps_top25.csv` | BTCUSDT data-health audit |
| `BTCUSDT_audit_missing_by_year.csv` | BTCUSDT data-health audit |
| `BTCUSDT_audit_monthly.csv` | BTCUSDT data-health audit |
| `BTCUSDT_audit_outliers_top50.csv` | BTCUSDT data-health audit |
| `BTCUSDT_audit_treatments.csv` | BTCUSDT data-health audit |
| `ETHUSDT_audit_dq_table.csv` | ETHUSDT data-health audit |
| `ETHUSDT_audit_file_summary.csv` | ETHUSDT data-health audit |
| `ETHUSDT_audit_gaps_top25.csv` | ETHUSDT data-health audit |
| `ETHUSDT_audit_missing_by_year.csv` | ETHUSDT data-health audit |
| `ETHUSDT_audit_monthly.csv` | ETHUSDT data-health audit |
| `ETHUSDT_audit_outliers_top50.csv` | ETHUSDT data-health audit |
| `ETHUSDT_audit_treatments.csv` | ETHUSDT data-health audit |
| `ETHUSDT_feature_summary.csv` | ETHUSDT features, leakage and hypothesis tests |
| `ETHUSDT_feature_timing_table.csv` | ETHUSDT features, leakage and hypothesis tests |
| `ETHUSDT_h1_h2_univariate_dev.csv` | ETHUSDT features, leakage and hypothesis tests |
| `ETHUSDT_h1_ic_by_block_dev.csv` | ETHUSDT features, leakage and hypothesis tests |
| `ETHUSDT_h3_regime_comparison_dev.csv` | ETHUSDT features, leakage and hypothesis tests |
| `ETHUSDT_h3_regime_ic_ci_dev.csv` | ETHUSDT features, leakage and hypothesis tests |
| `ETHUSDT_h3_regime_ic_dev.csv` | ETHUSDT features, leakage and hypothesis tests |
| `ETHUSDT_leakage_perturbation_real_data.csv` | ETHUSDT features, leakage and hypothesis tests |
| `eda_dependence_dev.csv` | Exploratory analysis (development data for anything predictive) |
| `eda_extreme_hours.csv` | Exploratory analysis (development data for anything predictive) |
| `eda_intraday_dev.csv` | Exploratory analysis (development data for anything predictive) |
| `eda_regimes_by_year.csv` | Exploratory analysis (development data for anything predictive) |
| `eda_return_distribution.csv` | Exploratory analysis (development data for anything predictive) |
| `eda_volume_return_dev.csv` | Exploratory analysis (development data for anything predictive) |
| `experiment_log.csv` | Every configuration evaluated, including failures |
| `exploratory_polymarket_event_deciles.csv` | EXPLORATORY, not pre-registered; producing code not in the repository; no prediction-market prices used |
| `feature_summary.csv` | Feature summary and timing table |
| `feature_timing_table.csv` | Feature summary and timing table |
| `h1_h2_univariate_dev.csv` | Hypothesis H1/H2 tests on development data |
| `h1_ic_by_block_dev.csv` | Hypothesis H1/H2 tests on development data |
| `h3_regime_comparison_dev.csv` | Hypothesis H3 (regime) tests on development data |
| `h3_regime_ic_ci_dev.csv` | Hypothesis H3 (regime) tests on development data |
| `h3_regime_ic_dev.csv` | Hypothesis H3 (regime) tests on development data |
| `leakage_perturbation_real_data.csv` | Leakage perturbation test on real data |
| `pipeline_controls.csv` | Synthetic control runs (method only) |
| `risk_bootstrap_mc.csv` | VaR/ES, Kupiec, Monte Carlo |
| `risk_kupiec.csv` | VaR/ES, Kupiec, Monte Carlo |
| `risk_var_es_daily.csv` | VaR/ES, Kupiec, Monte Carlo |
| `robustness_spot_fee_check.csv` | Robustness variants and the spot-fee check |
| `robustness_variants.csv` | Robustness variants and the spot-fee check |
| `second_asset_ETHUSDT_benchmarks.csv` | ETHUSDT replication with the frozen design |
| `second_asset_ETHUSDT_calibration.csv` | ETHUSDT replication with the frozen design |
| `second_asset_ETHUSDT_per_block.csv` | ETHUSDT replication with the frozen design |
| `stress_costs.csv` | Stress scenarios |
| `stress_gap_shock.csv` | Stress scenarios |
| `stress_regime_trend.csv` | Stress scenarios |
| `stress_regime_vol.csv` | Stress scenarios |
| `stress_signal_degradation.csv` | Stress scenarios |
| `stress_volatility.csv` | Stress scenarios |
| `stress_worst_btc_days.csv` | Stress scenarios |
| `test_backtest_sample_rows.csv` | Untouched test period: benchmarks, per-block, calibration, confusion, drift, importance, sample backtest rows |
| `test_benchmarks.csv` | Untouched test period: benchmarks, per-block, calibration, confusion, drift, importance, sample backtest rows |
| `test_calibration.csv` | Untouched test period: benchmarks, per-block, calibration, confusion, drift, importance, sample backtest rows |
| `test_confusion.csv` | Untouched test period: benchmarks, per-block, calibration, confusion, drift, importance, sample backtest rows |
| `test_drift.csv` | Untouched test period: benchmarks, per-block, calibration, confusion, drift, importance, sample backtest rows |
| `test_importance.csv` | Untouched test period: benchmarks, per-block, calibration, confusion, drift, importance, sample backtest rows |
| `test_per_block.csv` | Untouched test period: benchmarks, per-block, calibration, confusion, drift, importance, sample backtest rows |
| `validation_baselines.csv` | Walk-forward validation: folds, models, signal grid, baselines |
| `validation_folds.csv` | Walk-forward validation: folds, models, signal grid, baselines |
| `validation_models.csv` | Walk-forward validation: folds, models, signal grid, baselines |
| `validation_signal_grid.csv` | Walk-forward validation: folds, models, signal grid, baselines |

## Figures (`figures/`, 13 files)

| File | Purpose |
|---|---|
| `eda_acf_dev.png` | EDA figure |
| `eda_hourly_return_qq.png` | EDA figure |
| `eda_intraday_dev.png` | EDA figure |
| `eda_price_volatility.png` | EDA figure |
| `report_architecture.png` | Formal-report figure (from saved tables) |
| `report_drift.png` | Formal-report figure (from saved tables) |
| `report_walk_forward.png` | Formal-report figure (from saved tables) |
| `risk_mc_one_year.png` | Monte Carlo figure |
| `robustness_variants.png` | Robustness figure |
| `stress_costs.png` | Stress figure |
| `test_calibration.png` | Test-period result figure |
| `test_decile_returns.png` | Test-period result figure |
| `test_equity.png` | Test-period result figure |

## Not included (by design)

| Path | Why | How to recreate |
|---|---|---|
| `data/raw/` | Binance archives: CC BY-NC-SA licence, ~0.45 GB | `python -m src.data download …` (see `reproduction/README.md`) |
| `data/interim/`, `data/processed/` | Derived from the raw archives | `python -m src.data build …`, `python scripts/run_pipeline.py --stage all` |
| LendingClub data for the credit notebook | Kaggle licence, account required | Download `accepted_2007_to_2018Q4.csv.gz` from Kaggle |
| `.venv/`, `__pycache__/`, `.pytest_cache/` | Local environment and caches | `python -m venv .venv && pip install -r requirements.txt` |

## Complete project tree

```
.gitattributes
.gitignore
.vscode/
    extensions.json
    settings.json
PROJECT_FILE_MANIFEST.md
PROJECT_NOTES.md
README.md
credit_module/
    Credit_Risk_Project_Report.pdf
    README.md
data/
    README.md
figures/
    eda_acf_dev.png
    eda_hourly_return_qq.png
    eda_intraday_dev.png
    eda_price_volatility.png
    report_architecture.png
    report_drift.png
    report_walk_forward.png
    risk_mc_one_year.png
    robustness_variants.png
    stress_costs.png
    test_calibration.png
    test_decile_returns.png
    test_equity.png
notebooks/
    01_credit_risk.ipynb
    02_market_data_health.ipynb
    03_market_eda.ipynb
    04_hypothesis_features.ipynb
    05_modeling.ipynb
    06_walk_forward_validation.ipynb
    07_backtesting.ipynb
    08_risk_analysis.ipynb
    09_stress_testing.ipynb
    10_robustness.ipynb
pyproject.toml
reports/
    Final_Gravia_Project_Report.docx
    Final_Gravia_Project_Report.md
    Final_Gravia_Project_Report.pdf
    Gravia_Project_One_Page_Summary.html
    Gravia_Project_One_Page_Summary.pdf
    TECHNICAL_REVIEWER_README.md
    final_report.md
    gravia_mapping_and_cv.md
    interview_prep_concepts.md
    interview_prep_project.md
    module_connection.md
    phase1_dataset_selection.md
    phase2_data_acquisition.md
    pipeline_validation.md
    project_walkthrough.md
    research_design_preregistration.md
    results.json
    selection.json
    spot_fee_check.json
    tables/
        BTCUSDT_audit_dq_table.csv
        BTCUSDT_audit_file_summary.csv
        BTCUSDT_audit_gaps_top25.csv
        BTCUSDT_audit_missing_by_year.csv
        BTCUSDT_audit_monthly.csv
        BTCUSDT_audit_outliers_top50.csv
        BTCUSDT_audit_treatments.csv
        ETHUSDT_audit_dq_table.csv
        ETHUSDT_audit_file_summary.csv
        ETHUSDT_audit_gaps_top25.csv
        ETHUSDT_audit_missing_by_year.csv
        ETHUSDT_audit_monthly.csv
        ETHUSDT_audit_outliers_top50.csv
        ETHUSDT_audit_treatments.csv
        ETHUSDT_feature_summary.csv
        ETHUSDT_feature_timing_table.csv
        ETHUSDT_h1_h2_univariate_dev.csv
        ETHUSDT_h1_ic_by_block_dev.csv
        ETHUSDT_h3_regime_comparison_dev.csv
        ETHUSDT_h3_regime_ic_ci_dev.csv
        ETHUSDT_h3_regime_ic_dev.csv
        ETHUSDT_leakage_perturbation_real_data.csv
        eda_dependence_dev.csv
        eda_extreme_hours.csv
        eda_intraday_dev.csv
        eda_regimes_by_year.csv
        eda_return_distribution.csv
        eda_volume_return_dev.csv
        experiment_log.csv
        exploratory_polymarket_event_deciles.csv
        feature_summary.csv
        feature_timing_table.csv
        h1_h2_univariate_dev.csv
        h1_ic_by_block_dev.csv
        h3_regime_comparison_dev.csv
        h3_regime_ic_ci_dev.csv
        h3_regime_ic_dev.csv
        leakage_perturbation_real_data.csv
        pipeline_controls.csv
        risk_bootstrap_mc.csv
        risk_kupiec.csv
        risk_var_es_daily.csv
        robustness_spot_fee_check.csv
        robustness_variants.csv
        second_asset_ETHUSDT_benchmarks.csv
        second_asset_ETHUSDT_calibration.csv
        second_asset_ETHUSDT_per_block.csv
        stress_costs.csv
        stress_gap_shock.csv
        stress_regime_trend.csv
        stress_regime_vol.csv
        stress_signal_degradation.csv
        stress_volatility.csv
        stress_worst_btc_days.csv
        test_backtest_sample_rows.csv
        test_benchmarks.csv
        test_calibration.csv
        test_confusion.csv
        test_drift.csv
        test_importance.csv
        test_per_block.csv
        validation_baselines.csv
        validation_folds.csv
        validation_models.csv
        validation_signal_grid.csv
    teaching/
        Final_Completion_Report.docx
        Final_Completion_Report.md
        Final_Completion_Report.pdf
        Final_Interview_Readiness_Audit.docx
        Final_Interview_Readiness_Audit.md
        Final_Interview_Readiness_Audit.pdf
        Project_Walkthrough_Raw_Data_to_Verdict.docx
        Project_Walkthrough_Raw_Data_to_Verdict.md
        Project_Walkthrough_Raw_Data_to_Verdict.pdf
reproduction/
    README.md
    checksums/
        BTCUSDT_1m_spot_manifest.csv
        BTCUSDT_fundingRate_manifest.csv
        ETHUSDT_1m_spot_manifest.csv
requirements.txt
scripts/
    download_data.sh
    make_notebooks.py
    make_report_figures.py
    pipeline_controls.py
    report_tools/
        build_docx.js
    run_pipeline.py
    run_spot_fee_check.py
    smoke_test_synthetic.py
    verify_checksums.py
src/
    __init__.py
    backtest.py
    check_data_access.py
    config.py
    data.py
    eda.py
    features.py
    models.py
    pipeline.py
    plots.py
    quality.py
    risk.py
    stats.py
    stress_test.py
    validation.py
tests/
    conftest.py
    test_backtest.py
    test_data_quality.py
    test_features_leakage.py
    test_pipeline_controls.py
    test_risk.py
    test_validation.py
```
