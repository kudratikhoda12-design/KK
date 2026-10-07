# Reproducing the market study

This page is the step-by-step recipe. Everything here was run on Python 3.11 (Linux). Commands are run from the repository root.

## 0. What you can and cannot reproduce

| Component | Reproducible from public data? | Needs |
|---|---|---|
| Market study: BTCUSDT audit, features, validation, test, backtest, risk, stress, robustness | **Yes** | Binance Vision download (~235 MB) |
| ETHUSDT replication | **Yes** | Binance Vision download (~219 MB) |
| Funding in the cost model | **Yes** | Binance Vision funding files (0.1 MB) |
| Credit-risk logistic model (`notebooks/01_credit_risk.ipynb`) | Only with the original data | LendingClub `accepted_2007_to_2018Q4.csv.gz` from Kaggle (account required) |
| Credit-risk LightGBM challenger | **No** | Its code is not in the repository |

## 1. Environment (≈ 2 minutes)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                  # expect: 26 passed
```

The tests use only synthetic data, so they run before any download.

## 2. Data (≈ 0.45 GB download)

```bash
python src/check_data_access.py                            # optional: can this machine reach Binance Vision?
python -m src.data download --symbol BTCUSDT --funding     # 110 monthly kline files + 81 funding files
python -m src.data build    --symbol BTCUSDT               # -> data/interim/BTCUSDT_1m_raw.parquet
python -m src.data download --symbol ETHUSDT
python -m src.data build    --symbol ETHUSDT
python scripts/verify_checksums.py                         # expect: 110/110, 110/110, 81/81, 0 mismatches
```

Each file is checked against Binance's `.CHECKSUM` when it is downloaded (a failing file is saved as `*.zip.bad` and never used), and logged in `data/raw/.../manifest.csv`. `verify_checksums.py` then compares every local file with `reproduction/checksums/`, the hashes of the exact files used for the published results. Downloads resume: re-running skips files already present. On Linux/macOS, `bash scripts/download_data.sh BTCUSDT 2017-08 2026-09` is a shell-only alternative for the BTC klines.

## 3. Pipeline (≈ 15 minutes in total)

```bash
python scripts/run_pipeline.py --stage all            # audit, eda, features, validation, test, risk, stress, robustness
python scripts/run_pipeline.py --stage second_asset   # ETHUSDT with the frozen BTC design
python scripts/run_spot_fee_check.py                  # pre-registered spot-fee variant (re-prices frozen positions)
python scripts/make_report_figures.py                 # three report figures, drawn from saved tables only
```

Individual stages can be run with `--stage audit`, `--stage eda`, and so on. The test stage refuses to run unless `reports/selection.json` exists.

## 4. Expected key outputs

| File | Expected value |
|---|---|
| `reports/results.json` → `BTCUSDT_audit.schema.n_rows` | 4,789,279 |
| `reports/results.json` → `BTCUSDT_audit.missing_minutes` | 30,163 |
| `reports/results.json` → `features.leakage_features_changed` | 0 (of 24 checks) |
| `reports/selection.json` | `"model": "lgbm"`, `"k": 1.5`, `"mode": "long_flat"` |
| `reports/results.json` → `test.classification.auc` | 0.5448 (CI 0.5383–0.5516) |
| `reports/results.json` → `test.performance.sharpe` | −5.841 |
| `reports/tables/robustness_variants.csv` | 23 rows, every `net_sharpe` < 0 |
| `reports/results.json` → `second_asset_ETHUSDT.classification.auc` | 0.5456 |
| `reports/tables/robustness_spot_fee_check.csv` | spot row net Sharpe −9.856 |

All random processes use seed 42 and LightGBM runs in deterministic mode, so a re-run on the same files should reproduce these values; tiny floating-point differences across library versions are possible.

**Verified on 7 October 2026.** The delivery ZIP was extracted into an empty folder, installed into a new virtual environment with `pip install -r requirements.txt` (Python 3.11.15, the tested versions), and steps 1 and 3 were run on the study's Binance files. Result: 26/26 tests passed; all 670 numeric values in `reports/results.json`, `reports/selection.json` and all 63 pipeline tables were **identical** to the committed ones; the experiment log matched row for row except for its timestamps. Notebooks `02`–`10` also executed without error.

## 5. Two things to know before re-running

1. **Re-running creates a new experiment record.** `--stage validation` starts a fresh `reports/tables/experiment_log.csv`, and each `--stage test` adds a TEST row. The evidence that the original test was run exactly once lives in the git history (`selection.json` committed in `231634d` before the test results in `bd2963f`). To keep the original record, work on a copy of the repository.
2. **To check the code without touching any result,** run the synthetic end-to-end test, which writes only to a temporary folder:

   ```bash
   python scripts/smoke_test_synthetic.py            # about 7 minutes; all stages on 4.8M synthetic minutes
   ```

## 6. Rebuilding the reports (optional)

The Word versions of the reports are generated from their Markdown sources with `scripts/report_tools/build_docx.js` (Node.js + the `docx` npm package), and the PDFs from the Word files with LibreOffice:

```bash
npm install docx
DOC_H1_BREAK=0 DOC_CONTENTS_FIRST='' node scripts/report_tools/build_docx.js reports/Final_Gravia_Project_Report.md reports/Final_Gravia_Project_Report.docx
soffice --headless --convert-to pdf --outdir reports reports/Final_Gravia_Project_Report.docx
```

The one-page summary PDF is printed from `reports/Gravia_Project_One_Page_Summary.html` with any browser ("Print → Save as PDF", A4, no margins/headers).
