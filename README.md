# Credit Risk Modelling & Quantitative Market Research

**Kudrati Khoda**, M.Tech Quality, Reliability & Operations Research, Indian Statistical Institute, Kolkata

Two connected modules share one quantitative framework: *predict, validate out of time, quantify the tail, stress, and distrust the result until it survives.*

| Module | Question | Status |
|---|---|---|
| **A. Credit risk** (existing project, unchanged) | Who defaults, how much could the loan portfolio lose, and how sensitive is that to assumptions? | Complete. See [`credit_module/`](credit_module/README.md) |
| **B. Market research** (extension) | Does information at time *t* predict the next-hour BTCUSDT return well enough to survive costs out of sample? | **Code, tests and pre-registered design complete. Waiting for real data**; see [Status](#status) |

> **Results policy.** No number in this repository about BTCUSDT will be reported until it is computed from the real Binance data. Synthetic data appears only in tests and the smoke test, labelled as such.

---

## 1. Problem (Module B)

Can information available at time *t* predict the **direction of the BTCUSDT return over the next 60 minutes** well enough to be (a) statistically distinguishable from no skill out of sample, and (b) profitable after realistic trading costs? "No robust signal" is an acceptable answer.

## 2. Motivation

Module A showed that a model can rank risk yet still be mis-calibrated out of time, and that tail risk depends heavily on assumptions. Module B applies the same discipline to a fast, adversarial, 24/7 market, where the main danger is a backtest that looks good only because of leakage, overfitting or ignored costs.

## 3. Data

* **Source:** Binance Vision public archive (`data.binance.vision`), BTCUSDT spot 1-minute klines, 2017-08 to 2026-09.
* **Size:** 110 monthly files, about 4.8M candles.
* **Integrity:** every file is SHA-256-verified, with a provenance manifest.
* **Licence:** CC BY-NC-SA 4.0; used only for non-commercial academic research; raw data not committed.
* **Instructions:** [`data/README.md`](data/README.md). **Why this dataset:** [`reports/phase1_dataset_selection.md`](reports/phase1_dataset_selection.md).

## 4. Data health

The audit (`src/quality.py`) covers:
* schema and dtypes;
* timestamp unit (ms → µs from 2025), alignment and ordering;
* missing minutes and gap lengths;
* OHLC consistency, and VWAP inside [low, high];
* volume validity;
* exact and conflicting duplicates;
* extreme moves, each with evidence on whether it is genuine;
* stale prices;
* coverage changes over time.

The treatment rules were fixed in advance. Gaps are never filled. Extreme but consistent moves are kept and flagged. *Findings: pending the real-data run.*

## 5. Methodology (pre-registered)

The full design is in [`reports/research_design_preregistration.md`](reports/research_design_preregistration.md) and [`src/config.py`](src/config.py). It was committed **before** any data was seen, so the git history is the evidence against data snooping.

* **Timing:** decide at *t* using candles closed by *t*; execute at the open of *t*+1 min; exit at *t*+61 min. Decisions are hourly, so labels do not overlap.
* **Target:** Y = 1 if ln(P_exit / P_entry) > 0. P&L uses the actual return, so magnitude still matters.
* **Hypotheses:** H1 momentum/reversal, H2 volume/order flow, H3 regime dependence, H4 a model beats single features, H5 net economic value.
* **Features (16, scale-free):**
  - returns over 1, 5, 15, 30 and 60 min;
  - 4 h and 24 h momentum, and distance from the 24 h moving average;
  - realised volatility (60 min, 24 h), their ratio, and the relative range;
  - relative volume, volume change and taker-buy imbalance;
  - close location within the last hour's range.
* **Leakage audit:** a timing table, plus a perturbation test that randomises everything at or after *t* and requires every feature at *t* to stay unchanged. It runs in the tests and on the real data.

## 6. Models

* **Baselines:** base rate (equivalent to buy-and-hold), sign of the 1 h return, sign of the 24 h return.
* **Logistic regression:** C ∈ {0.01, 0.1, 1}.
* **LightGBM:** two conservative configurations.
* **Simplicity rule:** LightGBM is used only if it beats logistic regression by > 0.005 AUC **and** has lower log-loss in ≥ 70% of folds.

## 7. Validation

* **Never random K-fold.**
* **Walk-forward:** an expanding window, 10 six-month validation blocks (2019H1–2023H2), purging of labels that overlap a block.
* **Final test:** 2024-01 onwards, run **once** with the frozen selection (`reports/selection.json`) and refitted every 6 months.

## 8. Backtesting

An event-level backtest (`src/backtest.py`). Each row records decision time, execution time, entry and exit prices, position, turnover, gross return, cost and net return. Benchmarks: buy-and-hold and 1 h momentum.

## 9. Costs

| Component | Per side | Status |
|---|---|---|
| Fee (USD-M perpetual taker) | 5 bp | sourced |
| Half-spread | 1 bp | assumed |
| Slippage | 1 bp | assumed |

That totals 14 bp per round trip, charged on every change in position. Stress scenarios: ×2 and ×3 costs, +2 and +5 bp slippage, and spot fees.

## 10. Risk

* **VaR / ES:** 1-day horizon at 95% and 99%; historical, normal and Student-t.
* **Kupiec VaR backtest:** mirrors the PD backtest in Module A.
* **Block-bootstrap Monte Carlo:** one-year outcomes and drawdowns.
* **Also reported:** maximum drawdown, Sharpe, Sortino, Calmar, Probabilistic and Deflated Sharpe.

## 11. Robustness: trying to kill the signal

Every variant runs after the frozen result is recorded:
* sub-periods and regimes;
* thresholds;
* costs;
* execution lag 0/1/5 min;
* decision offsets of :15, :30 and :45;
* feature-group removal;
* model swap;
* horizons of 15 and 30 min;
* label permutation;
* signal-flip degradation;
* ETHUSDT with the frozen design.

All experiments are logged, including failures.

## 12. Results

*Pending the real-data run.* The pipeline-control results (synthetic, about the method, not about BTC) are in `reports/pipeline_validation.md`.

## 13. Limitations (known in advance)

* No historical order book, so spread and slippage are assumptions, stress-tested.
* Spot prices stand in for the perpetual; basis and funding are not modelled unless the funding files are added.
* Results cover a single venue.
* A 60-minute horizon cannot capture faster microstructure effects.

## 14. Conclusion

*Pending the real-data run.* The classification rule (STRONG / MODERATE / WEAK / NO ROBUST SIGNAL) is fixed in the pre-registration, §10.

---

## Status

| Phase | State |
|---|---|
| 1 Dataset selection | Done |
| 2 Acquisition | Code done; **blocked**: this environment cannot reach `data.binance.vision` ([details](reports/phase2_data_acquisition.md)) |
| 3–20 Audit → robustness | Implemented, unit-tested, smoke-tested at full size on synthetic data |
| 21–25 Conclusions, CV, interview prep | Written after the real-data run |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q                                   # 25 tests incl. synthetic pipeline controls
python -m src.data download && python -m src.data build
python scripts/run_pipeline.py --stage all            # or execute notebooks 02-10 in order
```

## Repository layout

```
├── credit_module/      Module A: original report + provenance notes
├── data/README.md      how to obtain the data (data itself not committed)
├── notebooks/          01 credit (original) | 02 data health … 10 robustness
├── src/                config, data, quality, eda, features, validation, models,
│                       backtest, risk, stress_test, stats, pipeline, plots
├── scripts/            run_pipeline.py, download_data.sh, smoke_test_synthetic.py, make_notebooks.py
├── tests/              unit tests + synthetic negative/positive pipeline controls
├── reports/            phase reports, pre-registration, tables/, final report (after the run)
└── figures/
```
