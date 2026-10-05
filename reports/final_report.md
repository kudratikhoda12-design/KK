# Credit Risk Modelling & Quantitative Market Signal Research: Final Report

**Author:** Kudrati Khoda, M.Tech QR&OR, Indian Statistical Institute, Kolkata
**Date:** 5 October 2026
**Code state:** design frozen in `b312d58` (before any data); selection frozen in `231634d` (before the test); results in `bd2963f`.

---

## 1. Executive summary

* **Question.** Can information available at time *t* predict the direction of the next 60-minute BTCUSDT return well enough to (a) beat no-skill out of sample and (b) make money after realistic costs?
* **Data.** 4.79M real Binance 1-minute candles (2017-08 → 2026-09). All 110 files were checksum-verified. The research design was pre-registered before the data was downloaded.
* **Statistical answer: yes.** There is a small but robust short-term **reversal** effect. On the untouched 2024–2026 test period:
  * AUC was **0.545** (95% CI 0.538–0.552), and log-loss beat the base rate (p = 0.004).
  * A label-permutation test gave p = 0.005.
  * AUC stayed above 0.53 in all 23 robustness variants and in every test half-year.
  * The same frozen pipeline gave AUC 0.546 on ETHUSDT.
* **Economic answer: no.** The frozen strategy lost **−92% net** (net Sharpe **−5.8**, 95% CI −6.9 to −4.8) while buy-and-hold made +98%.
  * Its gross edge is **0.19 bp per unit of turnover**, against costs of **7 bp per side**.
  * Every one of the 23 robustness variants lost money after costs.
* **Pre-registered classification: WEAK / INCONCLUSIVE**, meaning "statistical evidence on test without positive net performance". In plain words: *the pattern is real, but it is far too small to trade on Binance at retail costs.*
* **Most useful lesson.** Direction is predictable, but size is not. A model can be right 53.5% of the time and still have conditional mean returns of only −1.3 to +1.9 bp, against a 14 bp round-trip cost. That gap between statistical significance and economic value is the main finding.

## 2. Existing module: credit risk (unchanged)

The original project (`credit_module/`, `notebooks/01_credit_risk.ipynb`) built a full credit-risk workflow on LendingClub loans:
* PD models: logistic regression, AUC 0.700 on 2017 out-of-time loans;
* calibration by risk decile: PD was under-predicted in every decile;
* vintage backtesting and PSI;
* expected credit loss of about $494.7M from PD × LGD (93.3%) × EAD;
* one-factor Monte Carlo VaR/ES: VaR99 $68.5M at ρ = 0.15;
* stress and dependence tests: VaR99 ranges from $30.5M to $108.4M as ρ goes from 0 to 0.5.

That module is **not modified** by this work. Two reproducibility gaps found during integration are recorded in `credit_module/README.md`: the LightGBM code is missing from the notebook, and the notebook has no saved outputs.

## 3. Why extend to market data

The credit module taught two lessons:
* a model can rank well and still be mis-calibrated out of time;
* tail results depend heavily on assumptions.

A fast 24/7 market is a harder test of the same discipline. Labels arrive every hour, the noise is enormous, and the main danger is a backtest that looks good only because of leakage, overfitting or ignored costs. This is also the environment Gravia works in.

## 4. Dataset

| Item | Value |
|---|---|
| Source | Binance Vision public archive, spot **BTCUSDT 1-minute klines** (CC BY-NC-SA 4.0, non-commercial research use) |
| Files | 110 monthly archives (2017-08 → 2026-09), 235 MB; **110 / 110 SHA-256 checksums verified**; manifest in `data/raw/.../manifest.csv` |
| Rows | **4,789,279** candles from 2017-08-17 04:00 to 2026-09-30 23:59 UTC; 4,797,840 minutes on a complete grid |
| Columns | open time, OHLC, volume, close time, quote volume, number of trades, taker-buy base and quote volume |
| Storage | Typed parquet of 356 MB; 517 MB in memory |
| Also used | USD-M funding rates (81 files, verified) for the cost model; ETHUSDT (110 files, verified) for the second-asset test |
| Why this source | See `reports/phase1_dataset_selection.md`: official, free, bulk download with checksums, legal to access from India, and the settlement source of Polymarket's hourly BTC markets |

## 5. Data-health audit

| Data check | Result | Severity | Action |
|---|---|---|---|
| Schema | 12 expected columns in every file, no header rows, no unexpected columns | info | none |
| Timestamp units | 89 files in **ms**, 21 files (2025-01 → 2026-09) in **µs**, as Binance documents | info | Unit detected per file and tested |
| **Timestamp alignment** | **21,602 rows off the minute grid**: 20,401 candles at :20.799 s past the minute (2017-12-04 → 12-18) and 1,201 at :14.789 s (2018-02-08 → 02-10). Same windows in ETHUSDT, so an exchange-wide event | medium | **Treated as missing** (pre-registered rule; see below) |
| Truncated candles | 18 candles whose close time is not open + 59.999 s, mostly at the edges of outages and restarts (e.g. 2023-03-24, when Binance halted spot trading); one closes before it opens (2020-12-21) | low | Kept; reported |
| Missing 1-min intervals | **30,163 of 4,797,840 grid minutes (0.629%)** in 33 gaps: 1 phase-shifted stretch, 1 mixed outage + phase shift (Feb 2018), and 31 outages or archive gaps (e.g. 10 h on 2018-06-26 and on 2019-05-15) | medium | Left missing, never filled |
| Missing by year | 2017 10.6%, 2018 0.98%, 2019 0.34%, 2020 0.24%, 2021 0.19%, 2023 0.015%; **2022, 2024, 2025, 2026: 0** | — | Test period is complete |
| Duplicate rows / timestamps | 0 / 0 | info | — |
| Conflicting duplicates | 0 | info | — |
| Missing cells | 0 | info | — |
| Invalid OHLC | 0 (high < low, high < open or close, low > open or close, price ≤ 0, VWAP outside [low, high]: all 0) | info | — |
| Invalid volume | 0 negative; 0 taker-buy > volume; 0 volume without trades | info | — |
| Zero-volume minutes | 24,003, all with zero trades (mostly thin 2017–18) | low | Kept and flagged: no trading is real |
| Extreme 1-min moves | 408 moves or ranges > 3%, matching known events: 2020-03-12/13 COVID crash, 2021-05-19, 2022-11 FTX, 2024-08-05, 2025-10-10 liquidation cascade. 23,906 "locally extreme" minutes, 22,322 of them in 2017 | medium | **Kept**; listed in `BTCUSDT_audit_outliers_top50.csv` |
| Suspect (thin + full reversal) | 6,517, **all in 2017** (bid-ask bounce on a few trades); **0 from 2018 onward** | low | Kept (training data only) |
| Stale prices | Longest unchanged-close run 81 min; 32 runs ≥ 30 min (early years) | low | Reported |

**Cleaning performed.** None was needed beyond placing the data on a regular 1-minute grid. No duplicates, conflicts or impossible candles were found, so the pre-registered treatments removed 0 rows.

**Problems deliberately not fixed.**
1. *Phase-shifted candles.* Snapping 06:00:20.8 to 06:00 would let features see up to 20.8 s of the future. Snapping forward would be a new rule invented after seeing the data. The pre-registered rule (the grid uses open time) treats them as missing. The cost is about 2.3 weeks of 2017–18 training data; validation and test are unaffected.
2. *Thin 2017 microstructure.* It is real trading, kept as-is, and it sits in training data only.
3. *No historical order book.* The spread cannot be measured, so it stays an assumption and is stressed in §17 and §21.

**Two diagnostic bugs found and fixed during the audit:**
* The gap denominator had counted off-grid rows as extra minutes.
* The outlier z-score divided by a zero median move in illiquid 2017.

Neither changes any treatment (deviation log, pre-registration §12).

## 6. Exploratory data analysis

Predictive relationships use **development data only** (before 2024).

| Question | Answer (measured) |
|---|---|
| Are returns normal? | No. Hourly excess kurtosis is **38**; 0.92% of hours lie beyond 4σ (normal: 0.006%). The worst hour was −20.1% (2020-03-12 10:00); the worst day −50.3% (log return). |
| Are hourly returns autocorrelated? | Slightly negative: lag 1 **−0.047**, lag 2 −0.033, lag 24 −0.045 (Ljung–Box p ≈ 0). That is weak reversal. |
| Does volatility cluster? | Strongly: the autocorrelation of \|return\| is **0.31** at lag 1 and still 0.22 at 24 h. |
| Volume vs returns? | Same-hour volume vs \|return\|: Spearman 0.41. Volume now vs the next hour's \|return\|: 0.18. Volume now vs the next hour's *direction*: 0.012. **Volume predicts size, not direction.** |
| Intraday pattern? | Volatility and volume peak at 13–16 UTC (US open; volume about 1.4× average) and bottom out at 03–05 UTC. |
| Regimes? | Mean hourly volatility fell from 1.28% (2017) to about 0.4% (2023, 2025, 2026). **The test period is a calmer market**: the mean absolute 60-minute move was 32.4 bp in test vs 49.0 bp in development. |

Figures: `eda_price_volatility.png`, `eda_hourly_return_qq.png`, `eda_acf_dev.png`, `eda_intraday_dev.png`.

## 7. Research question (pre-registered)

> Can information available at *t* predict the sign of the BTCUSDT return from *t* + 1 min to *t* + 61 min, out of sample, well enough to be profitable after realistic costs?

## 8. Hypotheses and results (development data)

| ID | Hypothesis | Result | Verdict |
|---|---|---|---|
| H1 | Past returns predict the next-hour return (momentum or reversal) | Spearman −0.046 (5 min) to −0.074 (30 min), −0.067 (4 h); `clv_60m` −0.076. All Holm-significant. The 30-min and 4-h correlations are **negative in all 13 half-years**, 2017H2–2023H2 | **Supported: reversal** |
| H2 | Volume and order flow carry information | Taker-buy imbalance −0.036 (reversal), relative volume +0.012 (significant but tiny), volume change 0.0005 (n.s.) | **Partly supported** (order flow yes, volume barely) |
| H3 | Relationships differ by regime | Only **2 of 32** regime comparisons have non-overlapping CIs; about 1.6 are expected by chance. Reversal is about −0.07 in every regime | **Not supported** |
| H4 | A model beats single features | Validation AUC: LR 0.557, LightGBM 0.565, vs sign-of-past-return rules at 0.453 / 0.475 (i.e. inverted) | **Supported (predictively)** |
| H5 | Net economic value on test | Net Sharpe −5.84 (CI −6.94 to −4.77) | **Rejected** |

Overall, 14 of 16 features are significant after Holm correction. Statistical significance here is easy: 55,000 observations detect a correlation of 0.02. Economic significance is the hard part.

## 9. Target definition

* Decision at *t* (on the hour), using candles closed by *t*.
* Entry at the open of *t* + 1 min; exit at the open of *t* + 61 min.
* **Y = 1 if ln(exit / entry) > 0.**

Decisions are hourly, so labels never overlap. There are 79,964 decisions; 78,522 are usable and 1,442 (1.8%) were dropped for missing data. The development up-rate is 50.78%.

Classification was chosen over regression because one-hour returns are heavy-tailed (kurtosis 38); least squares would be driven by a few crash hours. Magnitude still enters through the P&L and the decile return analysis.

## 10. Feature engineering

16 scale-free features in 5 groups:
* **returns:** 1, 5, 15, 30 and 60 min;
* **momentum:** 4 h and 24 h returns, distance from the 24 h moving average;
* **volatility:** realised volatility over 60 min and 24 h, their ratio, relative range;
* **volume / flow:** relative volume, volume change, taker-buy imbalance;
* **candle:** close location within the last hour's range.

Definitions and the timing table are in `reports/tables/feature_timing_table.csv`. Gain importance of the model that traded the final test block: **`ret_4h` 19.7%, `ret_60m` 15.9%, `clv_60m` 12.0%, `ret_15m` 10.5%**. The reversal features dominate, consistent with H1.

## 11. Leakage audit

* **Timing table:** every feature's latest input is the candle closing at *t*; execution happens one minute later.
* **Perturbation test on real data:** at 24 random decision times (2018–2026), everything at or after *t* was replaced with random values. **0 of 16 features changed**, while all 24 targets did change, which proves the test reached the future data. The same result held for ETHUSDT.
* **Training-only fitting:** scaling is fitted inside the training window, and training labels overlapping each evaluation block are purged.
* **Synthetic controls** (`reports/pipeline_validation.md`, method only, not BTC): 40 random walks gave mean AUC 0.498 and a 5.0% false-positive rate; 20 of 20 injected signals were detected.

## 12. Baselines

| Baseline | Validation net Sharpe | Validation AUC | Test net Sharpe |
|---|---|---|---|
| Buy-and-hold (= base rate) | **1.01** | 0.50 | **0.76** |
| Sign of last 1 h return | −10.12 | 0.453 | −14.48 |
| Sign of last 24 h return | −1.95 | 0.475 | — |

The momentum rules have AUC *below* 0.5, which is the reversal effect seen from the other side.

## 13. Models

| Model (validation, 2019–2023) | AUC (pooled) | Mean fold AUC | Log-loss | Rank corr. | Log-loss better than base in |
|---|---|---|---|---|---|
| Base rate | 0.496 | 0.500 | 0.69308 | — | — |
| Logistic regression C = 0.01 | 0.557 | 0.560 | 0.68828 | 0.081 | 9 / 10 folds |
| LightGBM (7 leaves, 200 trees) | **0.565** | **0.566** | 0.68720 | 0.087 | 9 / 10 folds |
| LightGBM (15 leaves, 400 trees) | 0.563 | 0.563 | 0.68856 | 0.082 | 9 / 10 folds |

The regularisation level of logistic regression makes almost no difference (AUC 0.557 for every C).

## 14. Time-series validation

* **Development:** an expanding window, 10 six-month validation blocks (2019H1–2023H2), training from 2017-08-17 with purging. Fold sizes range from 12,043 to 51,451 training rows and about 4,400 evaluation rows (`validation_folds.csv`).
* **Final test:** 2024-01-01 → 2026-09-30, 24,095 hours. The model was refitted every 6 months on all earlier data with frozen settings, and **run once**. The experiment log contains exactly one TEST row.

## 15. Signal construction

* **Rule:** long if p̂ > 0.5 + k·s and flat otherwise (the long/flat mode won), where s is the spread of the model's training predictions.
* **Grid:** k ∈ {0, 0.25, 0.5, 1, 1.5} × {long/short, long/flat}, selected on validation.
* **All 10 grid points lost money in validation** (pooled net Sharpe −3.9 to −7.2; 0% of folds positive), while gross Sharpe was positive (0.6–1.6). The pre-registered rule picked the least-bad option: **k = 1.5, long/flat** (validation net Sharpe −3.88, gross 0.70).
* The pre-registered usefulness test, "beats the best baseline net", **already failed in validation** (−3.88 vs buy-and-hold +1.01).

**Cost-implied threshold.** A long trade only pays if (2p − 1)·E|r| > 14 bp. With E|r| = 32.4 bp in the test period, that needs **p > 0.716**. The model's most confident decile averaged p = 0.600.

## 16. Backtesting

The event-level backtest (`reports/tables/test_backtest_sample_rows.csv`) records, for each decision: decision time, execution time (+1 min), entry and exit price, position, previous position, turnover, gross return, cost, funding and net return. Costs are charged on |Δposition| only. Buy-and-hold pays one entry and one exit.

## 17. Transaction costs

| Component | Per side | Status |
|---|---|---|
| Fee: Binance USD-M taker, regular tier | 5 bp | **sourced** (2026 schedule) |
| Half-spread | 1 bp | **assumed** (no historical book) |
| Slippage | 1 bp | **assumed** |
| Funding (USD-M, actual history) | applied to positions held over funding times; −1.5% in total | **observed** |

* **Gross vs costs on test:** sum of gross returns +7.0%; sum of costs **256.8%** of notional (667 round trips a year).
* **Break-even cost: 0.19 bp per side**, about 37 times below the base assumption, and below any realistic taker fee.
* **Capacity:** a $100k order would be 12% of the execution minute's median traded value (79% at the 95th percentile), so the 1 bp slippage assumption is if anything optimistic at that size.

## 18. Performance (untouched test, 2024-01 → 2026-09)

| | Frozen strategy | Buy and hold | Momentum 1 h |
|---|---|---|---|
| Net Sharpe (95% CI) | **−5.84** (−6.94, −4.77) | 0.76 | −14.48 |
| Gross Sharpe | 0.16 | 0.76 | −0.94 |
| Annualised return | −60.4% | +28.1% | −99.9% |
| Cumulative return | **−92.2%** | **+97.6%** | −100% |
| Annualised volatility | 15.7% | — | — |
| Sortino | −7.06 | 1.08 | −18.98 |
| Max drawdown | −92.3% (never recovered) | −53.7% | −100% |
| Exposure | 9.1% of hours | 100% | 100% |
| Round trips / year | 667 | ≈0 | 4,585 |
| Win rate (hours in market) | 49.3% | 50.5% | 38.3% |
| Average win / loss | +0.32% / −0.42% | +0.32% / −0.32% | — |
| Profit factor | 0.74 | 1.03 | 0.62 |
| Probabilistic / Deflated Sharpe | 0.00 / ≈0 (13 trials) | — | — |

**Predictive metrics on test:**
* AUC 0.5448 [0.5383, 0.5516]; log-loss 0.6911 vs base rate (gain 0.0020, HAC t = 2.65, p = 0.004); Brier 0.249.
* Accuracy 53.5% (base rate 50.5%); precision 0.535; recall 0.591; F1 0.562.
* Confusion matrix: [[5,698, 6,232], [4,982, 7,183]].

**By half-year:**

| Half-year | AUC | Net Sharpe | Gross Sharpe |
|---|---|---|---|
| 2024H1 | 0.558 | −5.76 | 0.34 |
| 2024H2 | 0.567 | −3.38 | 2.53 |
| 2025H1 | 0.536 | −6.65 | −0.55 |
| 2025H2 | 0.535 | −8.27 | −1.77 |
| 2026H1 | 0.534 | −6.23 | −0.62 |
| 2026H2 (Jul–Sep) | 0.532 | −6.19 | 1.66 |

* **Calibration** (`test_calibration.png`): the model is over-confident. The top decile predicts 60.0% up and 56.4% is observed; the bottom decile predicts 40.1% and 43.8% is observed. The ranking is right, as in the credit model, but the probabilities are mis-scaled.
* **Decile returns** (`test_decile_returns.png`): mean forward returns span only −1.3 to +1.9 bp across deciles, against a 14 bp round trip.
* **Drift (PSI vs development):** volatility features shifted strongly as the market calmed. `log_rv_24h` PSI was 0.40 → 1.39 (2025H2) → 3.51 (2026H2), and taker imbalance PSI rose from 0.04 to 0.37. The fall in AUC from 0.56 to 0.53 coincides with this drift.

## 19. Risk (daily strategy returns, test period, 1-day horizon)

| | VaR 95% | ES 95% | VaR 99% | ES 99% |
|---|---|---|---|---|
| Strategy, historical | 1.72% | 2.81% | **3.53%** | **4.54%** |
| Strategy, normal | 1.62% | 1.96% | 2.18% | 2.47% |
| Strategy, Student-t | n/a: degenerate fit (df = 1.2) because most days have little exposure | | | |
| Buy-and-hold, historical | 3.65% | 5.11% | 6.14% | 7.59% |
| Buy-and-hold, Student-t (df = 3.6) | 3.78% | 5.87% | 6.88% | 9.93% |

* **Fat tails:** the normal assumption understates the strategy's 99% VaR by about 38% (2.18% vs 3.53%).
* **Kupiec test** (rolling 250-day historical VaR): 95% level, 33 exceptions vs 37.7 expected (p = 0.42); 99% level, 9 vs 7.5 (p = 0.60). The VaR model is **not rejected**, the same idea as the PD backtest in Module A.
* **Max drawdown:** −92.3%, from the 2024-01-02 peak to the 2026-09-28 trough, never recovered.

## 20. Monte Carlo (stationary block bootstrap of daily returns, 5,000 one-year paths)

The block bootstrap is used instead of the credit module's Gaussian copula. Strategy returns are observed directly, so resampling 10-day blocks keeps their real fat tails and volatility clustering without assuming a distribution.

Results:
* **P(loss over one year) = 100%**;
* median one-year return −60.0%;
* one-year VaR95 70.2% and ES95 72.3%;
* median maximum drawdown −60.4% (5th percentile −70.3%).

Limitation: the bootstrap can only replay regimes that occurred in the sample.

## 21. Stress testing

| Scenario | Strategy impact (net Sharpe; cumulative) | Risk impact | Conclusion |
|---|---|---|---|
| Base (7 bp/side, excl. funding) | −5.81; −92.1% | Daily VaR99 3.5%; max drawdown −92.2% | Fails |
| Costs × 0 (gross) | **+0.16; +3.8%** | VaR99 3.0%; max drawdown −25.9% | The only profitable case; edge ≈ 0 |
| Costs × 2 | −11.40; −99.4% | VaR99 3.9% | Fails worse |
| Costs × 3 | −16.33; −99.95% | VaR99 4.3% | Fails worse |
| +2 / +5 bp extra slippage | −7.46 / −9.86 | VaR99 3.6% / 3.8% | Fails |
| Volatility × 2 (same signals) | −2.85; −92.3% | VaR99 doubles to 6.5%; worst day −13.8% | Larger moves dilute fixed costs but double tail risk |
| Signal degradation, 10 / 25 / 50% flipped | −6.02 / −6.31 / −6.58 (0% of 200 sims positive) | — | Little margin to lose |
| Market shock: historical worst BTC days | On 2024-08-05 the strategy lost 7.2% vs BTC 6.9% (it was long). Mostly flat on the others (2026-02-05: BTC −11.9%, strategy −0.5%) | A dip-buying strategy can catch falling knives | Exposure is small but badly timed when it hits |
| Market shock: hypothetical gap −10 / −20 / −30% while long | Long 9.1% of the time; loss = gap | A −10% gap is 2.8× daily VaR99 | VaR understates gap risk |
| Regimes | High / mid / low volatility −4.78 / −5.48 / −7.75; trending / range −4.89 / −6.44 | — | No regime rescues it; worst in calm markets |

## 22. Robustness (post-hoc, test period; reported, never used for selection)

| Variant | AUC | Net Sharpe | Gross Sharpe |
|---|---|---|---|
| Frozen: k = 1.5, long/flat | 0.545 | −5.81 | 0.16 |
| k = 0, long/short (always in market) | 0.545 | −8.82 | **2.05** |
| Other thresholds (8 variants) | 0.545 | −5.28 to −10.14 | 0.20 to 1.83 |
| Without returns / momentum / volatility / volume / candle features | 0.537–0.545 | −4.84 to −6.02 | 0.30–0.66 |
| Model swapped to logistic regression | 0.540 | −4.35 | 1.41 |
| Execution lag 0 / 5 min | 0.548 / 0.540 | −4.57 / −6.31 | 1.41 / 0.35 |
| Decisions at :15 / :30 / :45 | 0.534–0.539 | −4.43 to −4.99 | 0.56–1.22 |
| Horizon 15 / 30 min | 0.538 / 0.536 | −17.27 / −10.10 | 2.47 / 1.26 |
| **ETHUSDT, same frozen pipeline** | **0.546** [0.538, 0.554] | **−3.70** | −0.52 |

**23 of 23 variants have negative net Sharpe. 23 of 23 have AUC > 0.53.**

## 23. Trying to kill the signal

| Attack | Did the *predictive* signal survive? | Did the *economic* signal survive? |
|---|---|---|
| Untouched test period | Yes (AUC 0.545, p = 0.004) | **No** |
| Label permutation (200 shuffles, logistic regression) | Yes: real AUC 0.540 vs null mean 0.501 and null 95th percentile 0.523; **p = 0.005** | — |
| Every half-year | Yes (AUC 0.532–0.567) | No (net negative in 6 / 6; gross positive in only 3 / 6) |
| Execution delay 5 min | Mostly (AUC 0.540) | No (gross falls 1.41 → 0.35: part of the edge decays within minutes) |
| Shifted hour boundary | Yes | No |
| Other horizons | Yes | No |
| Feature-group removal | Yes | No |
| Second asset (ETH) | Yes | No (even gross is negative) |
| Multiple testing | 44 logged experiments; Holm correction across 16 features; Deflated Sharpe ≈ 0 | — |

**Conclusion of the attack:** the statistical signal could not be killed. The economic signal was dead before the first test. Validation had already shown negative net Sharpe in every configuration.

## 24. Model comparison: simplicity vs complexity

* **Validation:** LightGBM beat logistic regression by +0.0064 AUC, and its log-loss was better in exactly 7 of 10 folds. That is the threshold itself, so the pre-registered simplicity rule passed only just.
* **Test:** LightGBM still had a slightly higher AUC (0.545 vs 0.540), but logistic regression lost less money (net −4.35 vs −5.81, gross 1.41 vs 0.16).
* **Lesson:** a borderline AUC gain did not translate into better trading. If I re-ran the study with a fresh design, I would require the challenger to win on the *economic* validation metric as well. That change is recorded as a lesson, **not applied retroactively**.

## 25. Limitations

* **No historical order book.** Spread and slippage are assumptions. This doesn't change the conclusion, because the break-even cost of 0.19 bp is below the fee alone.
* **Spot prices stand in for perpetual execution.** The basis is not modelled; funding is included from actual history.
* **One venue (Binance) and two assets.**
* **One-hour horizon only**, plus 15 and 30 minutes as robustness checks. Slower designs (daily) or maker-order execution were not tested.
* **Volatility features are raw, not volatility-normalised**, so they drift (PSI > 1) when the market regime changes.
* **2017–18 training data** includes thin-market microstructure and 2.3 weeks treated as missing.
* **The 6-month validation block structure** and the k-grid were fixed in advance; other reasonable choices exist.

## 26. Final conclusion

| Question | Answer |
|---|---|
| 1. Is there evidence of a predictive signal? | **Yes.** A short-term reversal: past 5 min–4 h returns, closing near the hour's high and taker-buy pressure all predict a slightly lower next hour. |
| 2. Is it statistically meaningful? | **Yes.** Test AUC CI [0.538, 0.552]; HAC p = 0.004; permutation p = 0.005; stable sign across 13 development half-years. |
| 3. Does ML improve over a simple baseline? | **Yes for prediction** (AUC 0.545 vs 0.5; single-sign rules are inverted). **No for money**: buy-and-hold beat every model variant. LightGBM vs logistic regression: marginal and mixed. |
| 4. Does it survive out-of-sample testing? | **Predictively yes** (test period and ETH). AUC decays from 0.56 to 0.53 as features drift. |
| 5. Does it survive transaction costs? | **No.** Break-even is 0.19 bp per side vs 7 bp assumed; net Sharpe −5.84. |
| 6. Is the strategy economically useful? | **No**, as a Binance taker strategy. |
| 7. Is it robust? | The *statistical* result is robust. The *economic* result is robustly negative: 23 of 23 variants lose. |
| 8. Major limitations | §25: no order book, single venue, raw volatility features, one horizon family. |
| 9. What next? | (a) **Exploratory, not pre-registered:** binary prediction-market contracts pay on direction, which is what the model predicts, not on size, which it can't. On the test period, its top-decile Up calls on the Polymarket hourly settlement event (Binance 1h close ≥ open) were right **56.6%** of the time (CI 54.6–58.5%), and its bottom-decile Down calls 55.9%, vs a hypothetical break-even of 51.75% for a 0.50 contract under Polymarket's crypto fee. **Polymarket prices were never observed**; its market makers see the same Binance data; liquidity is thin; and access from India is blocked. So this is a hypothesis for a forward test on post-September-2026 data, not a result. (b) Volatility-normalised features to reduce drift. (c) Lower-turnover designs, e.g. daily horizon or trading only extreme deciles. (d) Maker execution, which needs order-book data. |

### Final classification (pre-registered rule, §10): **WEAK / INCONCLUSIVE**

That is "statistical evidence on test without positive net performance". Stated plainly: **a real, robust but economically worthless predictive signal; no tradable edge for a Binance taker at realistic costs.**

## 27. Gravia relevance

See `reports/gravia_mapping_and_cv.md` for the requirement-by-requirement mapping. In short, the project shows the full loop the JD describes: question → data → hypothesis → test → model → backtest → kill tests → a defensible conclusion. The conclusion is negative, which is the point of having skepticism toward backtests.
