# Quantitative Market Signal Research & Portfolio Risk Analysis

**Kudrati Khoda** · M.Tech Quality, Reliability & Operations Research, Indian Statistical Institute, Kolkata

> **In one sentence:** on 4.8 million Binance BTCUSDT one-minute candles, the direction of the next hour turned out to be *statistically* predictable out of sample (AUC 0.545), but the edge is about **0.19 basis points per trade side against ~7 bp of cost**, so the strategy is **not tradable**. The project is about measuring that gap honestly.

| At a glance | |
|---|---|
| Data | Binance spot BTCUSDT 1-minute klines, 2017-08-17 → 2026-09-30, **4,789,279 candles**, 110/110 files checksum-verified |
| Question | Does information at time *t* predict the next 60-minute direction well enough to survive trading costs? |
| Design | Pre-registered in git **before** any data was downloaded; one untouched test run (2024-01 → 2026-09) |
| Prediction | Test AUC **0.545** (95% CI 0.538–0.552); log-loss gain HAC p = 0.004; same on ETHUSDT (0.546) |
| Economics | Net **−92.2%** vs buy-and-hold +97.6%; net Sharpe **−5.84**; break-even cost **0.19 bp/side** vs 7 bp assumed |
| Robustness | 23 kill tests: AUC 0.534–0.548 in all; **net negative in all 23** |
| Verdict (pre-registered rule) | **WEAK / INCONCLUSIVE**: a real but economically worthless signal. Do not deploy. |

---

## 1. Motivation

My first project was **credit risk** (LendingClub loans): predict probability of default, validate it out of time, turn it into expected loss, and stress the portfolio's tail. It taught two lessons: a model can rank well and still be mis-calibrated, and tail risk is driven by the assumption you are least sure about.

I extended the same discipline to a **fast, adversarial market**. In markets the main danger is a backtest that looks good only because of look-ahead leakage, overfitting or ignored costs. So the goal here was not "find a profitable strategy" but **"test a trading idea the way a sceptical quant would, and report what survives."**

## 2. Dataset

| Item | Value |
|---|---|
| Source | Binance Vision public archive (`data.binance.vision`), spot **BTCUSDT 1-minute klines**, CC BY-NC-SA 4.0 (non-commercial research) |
| Range | 2017-08-17 04:00 → 2026-09-30 23:59 UTC (110 monthly files, 235 MB) |
| Size | **4,789,279 candles** on a 4,797,840-minute grid |
| Also used | USD-M perpetual funding rates (81 files, for costs); ETHUSDT (110 files, replication) |

**Why Binance?** Official, free, complete history since 2017, a SHA-256 checksum for every file, order-flow fields (taker-buy volume) that most free sources lack, and accessible from India (see `reports/phase1_dataset_selection.md`).

**Why 1-minute data for an hourly decision?** To control timing exactly (features use only candles closed by *t*; the trade fills at the open of *t* + 1 min) and to build features at many scales (1 min to 7 days).

**Data-quality audit** (`src/quality.py`, `reports/tables/BTCUSDT_audit_*.csv`):

| Check | Result | Action |
|---|---|---|
| Checksums | 110 / 110 verified | — |
| Duplicates / conflicting rows / impossible candles | 0 / 0 / 0 | No rows removed |
| Missing minutes | 30,163 (0.63%) in 33 gaps; **0 in 2022 and 2024–2026** | Left missing, never forward-filled |
| **Timestamp units** | Binance switched from **milliseconds to microseconds in 2025** (89 ms files, 21 µs files) | Unit detected per file from the number of digits; refuses to guess |
| **Phase-shifted candles** | 21,602 candles in Dec 2017 / Feb 2018 start **20.8 s or 14.8 s after the minute** (same in ETH) | Treated as missing: snapping them to the minute would leak up to 20.8 s of future data into "closed" features |
| Extreme moves | 408 minutes with a > 3% move or range (COVID crash, May 2021, FTX, Aug 2024 …) | Kept: they are real risk |

## 3. Why Binance instead of direct Polymarket data?

Polymarket-style hourly "Bitcoin Up or Down" markets were part of the **motivation**: according to their published market rules (found in my Phase 1 research, not verified in this repository), they settle on the Binance BTC/USDT 1-hour candle, so the underlying Binance market is where any directional signal would have to come from. But:

* Binance provides **long, reproducible, checksum-verified** history (2017–2026). According to Polymarket's API documentation (Phase 1 research), fine-grained price history is kept only for a short time, and the hourly markets are recent.
* Polymarket access is blocked in India (government order reported in May 2026), and a project that needs a VPN is neither reproducible nor appropriate.
* **Direct Polymarket order-book, fill or price analysis was not part of this experiment.**

So this is **quantitative research on the underlying market**, not a Polymarket trading strategy.

## 4. Research question

> **Can information available at time *t* predict the direction of the next 60-minute BTCUSDT return out of sample, well enough to survive realistic trading costs?**

## 5. Methodology (all fixed in advance)

The full design, every parameter (`src/config.py`) and the final classification rule were committed to git **before any data was downloaded** (`reports/research_design_preregistration.md`; every later change is in its deviation log).

* **Timing:** decide on the hour using candles closed by *t* → buy at the open of *t* + 1 min → exit at the open of *t* + 61 min. One decision per hour, so labels never overlap.
* **Target:** y = 1 if ln(Open(*t*+61) / Open(*t*+1)) > 0.
* **16 features:** past returns (1–60 min), momentum (4 h, 24 h, distance from the 24 h mean), realised volatility (60 min, 24 h, their ratio, relative range), volume and order flow (relative volume, volume change, taker-buy imbalance), and close location within the last hour's range. A rolling feature needs ≥ 90% of its window present.
* **Leakage audit:** at 24 real decision times, everything at or after *t* was randomised: **0 of 16 features changed** (every target did).
* **Models:** base rate; logistic regression (scaled, L2, C ∈ {0.01, 0.1, 1}); two small LightGBMs.
* **Validation:** purged, expanding walk-forward over **10 half-year folds (2019–2023)**. Never random K-fold.
* **Model-selection rule:** LightGBM replaces logistic regression only if mean AUC gain > 0.005 **and** lower log-loss than LR in ≥ 70% of folds. Result: +0.0064 and exactly 7/10 → LightGBM, **marginally**.
* **Signal:** long if P(up) > 0.5 + 1.5 × (sd of training predictions), else flat (chosen on validation only).
* **Final test:** **2024-01-01 → 2026-09-30, untouched, run once** with six-monthly refits. The selection was committed before the test ran; the experiment log has exactly one TEST row.

## 6. Results (prediction)

| | Value | Source |
|---|---|---|
| Validation AUC (pooled, 2019–2023) | LightGBM **0.565**, logistic regression **0.557** | `validation_models.csv` |
| **Test AUC** | **0.545**, block-bootstrap 95% CI **0.538–0.552** | `results.json` |
| Log-loss gain vs base rate | Newey–West (HAC) t = 2.65, one-sided **p = 0.004** | `results.json` |
| Label-permutation test (200 shuffles, logistic regression) | real AUC 0.540 vs null 95th pct 0.523 → **p = 0.005** | `results.json` |
| Accuracy | 53.5% vs 50.5% base rate | `results.json` |
| Calibration | Over-confident: top decile predicts 60.0% up, 56.4% observed | `test_calibration.csv` |
| What it learned | **Short-term reversal**: after a rise, the next hour is slightly more likely to be weaker (14/16 features significant on development data after Holm correction) | `h1_h2_univariate_dev.csv` |
| ETH replication (frozen design, no re-tuning) | AUC **0.546** (0.538–0.554) | `second_asset_ETHUSDT_*` |
| Regime decay | AUC 0.567 (2024H2) → 0.532 (2026H2) as volatility features drifted (PSI up to 3.5) | `test_per_block.csv`, `test_drift.csv` |

## 7. Trading economics (untouched test, 2024-01 → 2026-09)

| | Frozen strategy | Buy and hold |
|---|---|---|
| Cost assumption | **7 bp per side = 14 bp round trip** (5 bp Binance taker fee, sourced; 1 bp half-spread + 1 bp slippage, **assumed**) + actual funding | same |
| Gross return (sum of hourly returns) | +7.0% | — |
| Costs paid | 256.8% of notional | ≈ 0 |
| **Net cumulative return** | **−92.2%** | **+97.6%** |
| **Net Sharpe** (95% CI) | **−5.84** (−6.94, −4.77) | 0.76 |
| Gross Sharpe | 0.16 | 0.76 |
| Max drawdown | −92.3% (never recovered) | −53.7% |
| Time in market / turnover | 9.1% of hours; 667 round trips a year | 100% |
| **Break-even cost** | **0.19 bp per side** (gross return ÷ turnover) | — |

**Why it fails:** the mean next-hour return across predicted-probability deciles is only −1.3 to +1.9 bp, against a 14 bp round trip. Statistical significance measures *certainty* that the effect is not zero, not its *size*.

![Test period: strategy vs buy-and-hold](figures/test_equity.png)

## 8. Robustness and risk

* **23 post-hoc kill tests** (thresholds, feature-group removal, model swap, execution lag 0/5 min, decisions at :15/:30/:45, 15/30-min horizons): AUC 0.534–0.548 in all, **net Sharpe negative in all 23**. Reported, never used to change the selection.
* **Stress tests:** only the zero-cost case is positive (Sharpe +0.16); ×2 and ×3 costs, +2/+5 bp slippage, every volatility and trend regime, and random signal flips all lose; a −10% gap while long is 2.8× the daily 99% VaR.
* **Pre-registered spot-fee check** (10 bp fee, long/flat; run late on 7 Oct 2026 by re-pricing the frozen positions): net Sharpe −9.86. Changes nothing above.
* **VaR / ES** (daily): historical 99% VaR **3.53%** vs 2.18% under normality (fat tails); ES99 4.54%; rolling VaR passes Kupiec (p = 0.42 at 95%, 0.60 at 99%).
* **Monte Carlo:** stationary block bootstrap of the strategy's own daily returns, 5,000 one-year paths: none profitable, median −60% (a replay of the test period, not a market forecast).
* **ETH replication:** AUC 0.546, net Sharpe −3.70 (lost even before costs).

## 9. Conclusion

**Prediction was statistically better than chance, but the economic edge was far too small to overcome transaction costs.** BTCUSDT shows a real, robust short-term reversal; it is worth about 1–2 bp per hour, an order of magnitude below a taker's cost. By the pre-registered rule the result is **WEAK / INCONCLUSIVE** (statistical evidence without positive net performance). The strategy should not be deployed.

## 10. Future extension (not done)

Use historical **prediction-market prices, fills / order book and settlement data** to test whether the same directional signal creates a tradable edge in a market that pays on direction rather than size. This requires a legally accessible, reproducible historical dataset and a new pre-registration; **it has not been done in this project.**

---

## Module A — the original credit-risk project

`credit_module/` and `notebooks/01_credit_risk.ipynb` (unchanged originals): logistic-regression PD model on LendingClub loans validated out of time, decile calibration, vintage PSI, ECL = PD × LGD × EAD, one-factor Monte Carlo VaR/ES and stress tests. **Evidence status:** the code is present, but the LendingClub file is not redistributed and the notebook has no saved outputs, so the numbers (e.g. 2017 AUC ≈ 0.70) come from the original report; the LightGBM credit challenger reported there is **not reproducible** from this repository (no code). Details: [`credit_module/README.md`](credit_module/README.md). How the two modules connect: [`reports/module_connection.md`](reports/module_connection.md).

## Repository map

| # | What | Where |
|---|---|---|
| 1 | Original credit-risk module | `credit_module/`, `notebooks/01_credit_risk.ipynb` |
| 2 | Market-signal research code | `src/` (config, data, quality, eda, features, validation, models, backtest, risk, stress_test, stats, pipeline, plots), `scripts/run_pipeline.py` |
| 3 | Dataset-selection research | `reports/phase1_dataset_selection.md`, `reports/phase2_data_acquisition.md` |
| 4 | Research pre-registration | `reports/research_design_preregistration.md` §1–11, `src/config.py` |
| 5 | Deviation log | `reports/research_design_preregistration.md` §12 |
| 6 | Validation results | `reports/tables/validation_*.csv`, `reports/selection.json` (frozen selection) |
| 7 | Final test results | `reports/results.json` → `test`; `reports/tables/test_*.csv` |
| 8 | Backtest results | `reports/tables/test_benchmarks.csv`, `test_backtest_sample_rows.csv`, `figures/test_equity.png` |
| 9 | Risk analysis | `reports/tables/risk_*.csv`, `stress_*.csv`, `figures/risk_mc_one_year.png` |
| 10 | Robustness results | `reports/tables/robustness_variants.csv`, `robustness_spot_fee_check.csv`, `figures/robustness_variants.png` |
| 11 | ETH replication | `reports/tables/second_asset_ETHUSDT_*.csv`, `ETHUSDT_*.csv`, `results.json` → `second_asset_ETHUSDT` |
| 12 | Final interview documentation | `reports/teaching/` (walkthrough, readiness audit, final completion report; Markdown, Word, PDF), `reports/gravia_mapping_and_cv.md` |
| — | Every trial, including failures | `reports/tables/experiment_log.csv` (44 rows, 1 TEST row) |
| — | Full research report | `reports/final_report.md` |
| — | Tests | `tests/` (26 tests incl. synthetic negative/positive pipeline controls) |

## Reproduce

Tested with Python 3.11 and the package versions noted in `requirements.txt`. All randomness is seeded (`RANDOM_SEED = 42`; LightGBM runs in deterministic mode). Raw data is not committed (licence and size); every downloaded file is checked against Binance's SHA-256 and logged in `data/raw/.../manifest.csv`. Full instructions: [`data/README.md`](data/README.md).

```bash
pip install -r requirements.txt
python -m pytest -q                                            # 26 tests, ~30 s

# Data (about 0.45 GB download in total)
python -m src.data download --symbol BTCUSDT --funding         # 110 klines + funding files, checksum-verified
python -m src.data build --symbol BTCUSDT                      # -> data/interim/BTCUSDT_1m_raw.parquet
python -m src.data download --symbol ETHUSDT                   # for the ETH replication
python -m src.data build --symbol ETHUSDT

# Pipeline (stages: audit, eda, features, validation, test, risk, stress, robustness)
python scripts/run_pipeline.py --stage all                     # ~10 min
python scripts/run_pipeline.py --stage second_asset            # ETH replication, ~4 min
python scripts/run_spot_fee_check.py                           # pre-registered spot-fee check
```

**Expected key outputs:** `reports/selection.json` → LightGBM config 1, k = 1.5, long_flat; `reports/results.json` → test AUC 0.5448, net Sharpe −5.841; `reports/tables/robustness_variants.csv` → 23 rows, all net Sharpe < 0.

**Important:** the validation stage resets the experiment log and each test run adds a TEST row, so re-running them on the real data creates a new record rather than reproducing the original one. Notebooks `02`–`10` only display saved outputs (`RUN_STAGE = False`). To check that the code runs end to end without touching any result, use `python scripts/smoke_test_synthetic.py` (synthetic data, temporary folder).

## Key documents

| Document | Content |
|---|---|
| [`reports/final_report.md`](reports/final_report.md) | Full research report (27 sections + addendum) |
| [`reports/teaching/Final_Completion_Report.md`](reports/teaching/Final_Completion_Report.md) | Completion checklist, verification, reproducibility, CV bullets, claims, interview answers |
| [`reports/teaching/Final_Interview_Readiness_Audit.md`](reports/teaching/Final_Interview_Readiness_Audit.md) | Credit-module evidence status, selection-rule wording, claims audit |
| [`reports/teaching/Project_Walkthrough_Raw_Data_to_Verdict.md`](reports/teaching/Project_Walkthrough_Raw_Data_to_Verdict.md) | Complete teaching walkthrough traced to the code |
| [`reports/research_design_preregistration.md`](reports/research_design_preregistration.md) | Frozen design and deviation log |
| [`reports/pipeline_validation.md`](reports/pipeline_validation.md) | Synthetic negative and positive controls (method only) |
| [`reports/gravia_mapping_and_cv.md`](reports/gravia_mapping_and_cv.md) | Job-description mapping and CV entry |

Market data © Binance, via Binance Vision under CC BY-NC-SA 4.0, used for non-commercial academic research only.
