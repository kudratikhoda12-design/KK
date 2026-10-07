# Reproducibility check (clean-copy rerun)

*Static record of a one-off verification, not produced by `run_project.py`. It documents that the committed results can be regenerated from the code and the raw data alone.*

## Procedure

On 2026-10-07 the code (`src/`, `tests/`, `notebooks/`, `run_project.py`, `requirements.txt`) and the four raw M5 files were copied into an **empty directory** (no `data/processed`, `outputs` or `reports`) and `python run_project.py` was run end to end (all 13 stages, 3.2 minutes on 4 cores, same seeds: `config.SEED = 42`). Every file under `outputs/tables`, `outputs/figures`, `outputs/models`, `reports` and `data/processed` was then compared with the corresponding file of the main run (forecast arrays bit-wise, the parquet panel by content, everything else byte-wise; CSV files that differ are compared column by column). The raw files were copied from the main run, so the acquisition stage re-verified their SHA-256 hashes against both mirrors' published values but did not download them again (the download path itself was exercised for `calendar.csv` in an earlier clean-copy rerun). The README and the executed notebook are not part of the comparison. After the full rerun, only the text of the report generators was edited; re-running just the `reports` stage in the clean copy with the final generators reproduced the committed reports exactly (the table below is that final comparison).

Environment: Python 3.13.16, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1, xgboost 3.4.1, shap 0.52.0, matplotlib 3.11.2.

## Result

**138 files compared: 132 identical, 6 different** (files present in only one run: 1 in the main run (`reports/reproducibility_check.md`, this record itself), 0 in the rerun).

| file | kind of difference | detail |
|---|---|---|
| `outputs/tables/final_audit.csv` | wall-clock / machine-state only | columns: `evidence` |
| `outputs/tables/pipeline_runtime.csv` | wall-clock / machine-state only (seconds per stage) |  |
| `outputs/tables/xgb_tuning_log.csv` | wall-clock / machine-state only | columns: `seconds` |
| `reports/environment.md` | wall-clock / machine-state only (value quoted in text) | 1 line(s) differ (e.g. `* CPU cores: 4; RAM: 15.7 GB; free disk: 28.7 GB` vs `* CPU cores: 4; RAM: 15.7 GB; free disk: 28.2 GB`) |
| `reports/final_audit.md` | wall-clock / machine-state only (value quoted in text) | 1 line(s) differ (e.g. `- [x] Unit tests (all pass) - 76 passed in 7.56s` vs `- [x] Unit tests (all pass) - 76 passed in 7.75s`) |
| `reports/test_results.txt` | wall-clock / machine-state only (value quoted in text) | 1 line(s) differ (e.g. `76 passed in 7.56s` vs `76 passed in 7.75s`) |

## Interpretation

All forecasts, metrics, test statistics, SHAP values, simulation results, figures, models and reports that depend on data or randomness are reproduced exactly. The remaining differences are run-specific values only: wall-clock seconds (per pipeline stage, per XGBoost tuning configuration, the pytest duration line and the sentences that quote it) and the free-disk-space figure in `environment.md`.

