# Research Design: Pre-registration (Market Module)

**Written:** 5 October 2026, **before any BTCUSDT data was downloaded or inspected.**
This environment could not reach `data.binance.vision`; see `phase2_data_acquisition.md`.
**Frozen parameters:** `src/config.py`. The git history of this file and that one is the evidence that the design came before the data.
**Rule:** any change made after seeing data is recorded in the deviation log (§12) with its reason. Nothing is changed silently.

Why pre-register? The biggest risk in a backtest is not a coding bug. It is the researcher trying many variants and reporting the one that worked (data snooping). Fixing the design first keeps the number of hidden trials at zero.

---

## 1. Research question

> **Can information available at time *t* predict the direction of the BTCUSDT return over the next 60 minutes well enough to be (a) statistically distinguishable from no skill out-of-sample, and (b) profitable after realistic trading costs?**

The answer is not assumed to be yes. "No robust signal" is a valid and expected outcome, given that BTCUSDT is one of the most liquid and most-watched markets in the world.

## 2. Timing: the most important definitions

All times are UTC. Let *t* be a decision time on the hourly grid (HH:00).

| Quantity | Definition |
|---|---|
| Information set at *t* | All 1-minute candles with `open_time ≤ t − 1 min`, i.e. candles already **closed** by *t* |
| Prediction time | *t* |
| Execution time | *t* + 1 min (`EXECUTION_LAG_MIN = 1`): a market order fills at the **open** of the candle starting at *t* + 1 min |
| Exit time | *t* + 1 + 60 min, at that candle's open |
| Target return | r = ln( Open(*t*+61) / Open(*t*+1) ) |
| Label | Y = 1 if r > 0, else 0 (ties count as 0) |
| Sampling | One decision per hour, so holding periods do **not** overlap and labels never overlap |

**Why a 1-minute execution lag?** The candle closing at *t* is the last information we have. Trading at that same price would assume zero latency and expose the test to bid-ask bounce: the last print bounces between bid and ask, which creates fake mean reversion. Waiting one minute is conservative and cheap at a 60-minute horizon. Lags of 0 and 5 minutes are robustness checks.

**Why 60 minutes as the primary horizon?**
1. **Costs.** A fixed cost per trade (14 bp round trip in the base case) is a smaller fraction of a typical 60-minute move than of a 15-minute move. Volatility grows roughly with √h, so a 1-hour move is about twice a 15-minute move.
2. **Clean statistics.** Hourly decisions with a 1-hour hold give non-overlapping labels. That means no overlapping-label correlation and no inflated sample size.
3. **Sample size.** About 80,000 hourly observations overall and about 24,000 in the untouched test period.
4. **Relevance.** Polymarket's hourly BTC Up/Down markets settle on this same 1-hour Binance BTC/USDT candle.

Only 15 and 30 minutes are tested as alternatives. No other horizon is ever tried.

**Why classification of direction rather than regression of return?** One-hour returns are heavy-tailed. Least-squares regression would be dominated by a few crash hours. Direction plus a probability is robust, works for both logistic regression and LightGBM, and maps directly to a position. Magnitude still enters through the economic evaluation: P&L uses the actual return, and we report the mean forward return by predicted-probability decile and the rank correlation between the score and the return. A model that wins often but small, and loses rarely but big, will be exposed by the P&L.

## 3. Hypotheses (fixed in advance)

| ID | Hypothesis | Feature(s) | Supported if (development data) | Rejected if | Possible false positive |
|---|---|---|---|---|---|
| **H1** Momentum/reversal | Past returns over 1 min–24 h carry information about the next-hour return, in either direction | `ret_*`, `ma_dist_24h` | Spearman rank correlation with r significantly ≠ 0 after Holm correction across all 16 features, same sign in most validation blocks | Not significant, or the sign flips between blocks | Bid-ask bounce (controlled by the 1-min lag); a few crash hours (rank correlation is outlier-robust) |
| **H2** Volume / order flow | Abnormal volume and taker-buy imbalance carry information | `rel_volume_60m`, `volume_chg_60m`, `taker_imb_60m` | As H1 | As H1 | Volume regime breaks (e.g. fee promotions), flagged in the audit |
| **H3** Regime dependence | H1/H2 relationships differ between high- and low-volatility, and trending vs range-bound, regimes | Rank correlation within regimes | Correlation differs across regimes with non-overlapping bootstrap CIs | CIs overlap | Too few observations in some regimes |
| **H4** Combination | A multivariate model beats the best single-feature baseline out-of-sample | LR / LightGBM | Higher validation AUC and log-loss improvement over the base rate (HAC test) | No improvement | Overfitting (controlled by walk-forward) |
| **H5** Economic value | The frozen strategy earns positive net returns after base costs on the untouched test set | Backtest | Net Sharpe > 0 with a bootstrap 95% CI excluding 0 | CI includes 0, or the mean is negative | Luck in one period (we report sub-periods, the Deflated Sharpe Ratio and a permutation test) |

**Use of the data during research.** Every analysis that relates features to future returns (return autocorrelation, rank correlations, conditional means) uses **development data only** (before 2024-01-01). Data from 2024 onwards is used only in the data-health audit and descriptive price and volatility plots, until the final test.

## 4. Features (16, all scale-free, all using closed candles only)

| Group | Feature | Definition (window ends at the candle closing at *t*) | Why it might predict |
|---|---|---|---|
| Returns | `ret_1m`, `ret_5m`, `ret_15m`, `ret_30m`, `ret_60m` | ln C(*t*−1) − ln C(*t*−1−k) | Short-term continuation or reversal (liquidity, overreaction) |
| Momentum | `ret_4h`, `ret_24h` | as above, k = 240 and 1,440 | Intraday trend persistence |
| Momentum | `ma_dist_24h` | ln C − ln(mean close over the last 1,440 min) | Stretch away from the recent average, i.e. mean reversion |
| Volatility | `log_rv_60m` | ln √(Σ of 1-min r² over 60 min) | Volatility level; risk premium or regime |
| Volatility | `log_rv_24h` | ln √(Σ of 1-min r² over 1,440 min / 24), in hourly units | Slower volatility level |
| Volatility | `vol_ratio` | `log_rv_60m − log_rv_24h` | Volatility shock relative to the recent norm |
| Volatility | `range_rel_60m` | ln(max H / min L over 60 min) / exp(`log_rv_24h`) | Size of the last hour's range relative to normal |
| Volume | `rel_volume_60m` | ln(volume over 60 min / mean 60-min volume over the last 7 days) | Attention or information arrival |
| Volume | `volume_chg_60m` | ln(volume over the last 60 min / volume over the 60 min before) | Acceleration of activity |
| Order flow | `taker_imb_60m` | (2 × taker-buy volume − volume) / volume over 60 min | Aggressive buying or selling pressure |
| Candle | `clv_60m` | (C − min L) / (max H − min L) over 60 min | Where the last hour closed within its own range |

Rules:
* A feature is NaN unless at least 90% of the minutes in its window exist; such observations are dropped, never forward-filled.
* Scaling is fitted inside the training window only, through an sklearn `Pipeline`.
* The leakage audit perturbs all data at or after *t* and requires every feature at *t* to stay identical (`tests/test_features_leakage.py`, also run on the real data).

Regime labels (for H3 and robustness only):
* **High/low volatility:** `log_rv_24h` above or below its trailing 90-day 67th or 33rd percentile, computed from the past only.
* **Trending:** |`ret_24h`| > one daily sigma (√24 × hourly realised volatility).

## 5. Validation design

```
2017-08 ---- train (expanding) ---->| val 6m | → refit → | val 6m | → ... → | val 2023H2 |   DEVELOPMENT
                                                                          frozen design
2024-01 -> | test 6m | refit | test 6m | refit | ... | 2026-09 |          UNTOUCHED TEST
```

* **Development (walk-forward):** the first validation block is 2019H1. Training always uses all data before the block start (expanding window), with **purging**: training rows whose label window ends after the block start are dropped. That gives 10 validation blocks (2019H1–2023H2).
* **All selection happens on development validation blocks:** LR regularisation C ∈ {0.01, 0.1, 1}, two LightGBM configurations, threshold k, and long/short vs long/flat.
* **Simplicity rule:** LightGBM replaces logistic regression only if its mean validation AUC is ≥ 0.005 higher **and** its log-loss is lower in ≥ 70% of folds.
* **Final test (2024-01-01 to the end of data):** the frozen pipeline is refitted every 6 months on all data before each block (purged). No parameter, feature, threshold or rule changes. It is run **once**.
* **Random K-fold is never used.** It would put 2024 information into a model judged on 2022, and it would ignore volatility clustering, so performance would be overstated.

## 6. Baselines

| Baseline | Prediction | Trading rule |
|---|---|---|
| Base rate | Constant p = training-window up-rate | Always long if p > 0.5, otherwise flat (this is buy-and-hold) |
| Momentum 1h | Score = `ret_60m` | Long if > 0, short if < 0 |
| Momentum 24h | Score = `ret_24h` | Long if > 0, short if < 0 |
| Zero return (regression view) | E[r] = 0 | Never trades |

A model counts as useful only if it beats the base rate on log-loss and AUC **and** beats the best baseline on net economic performance in validation.

## 7. Signal construction

With p̂ the model probability at *t* and s the standard deviation of the model's predictions on its training window:

* **Long** if p̂ > 0.5 + k·s
* **Short** if p̂ < 0.5 − k·s (long/short mode only)
* **Flat** otherwise: this is the no-trade region

The neutral point is 0.5, because P(up) = 50% means no expected direction. Scaling by s is needed because regularised models squeeze probabilities into a narrow band (e.g. 0.49–0.52), so a fixed absolute threshold would mean nothing.

The grid is k ∈ {0, 0.25, 0.5, 1.0, 1.5} × {long/short, long/flat}. The selected configuration maximises the mean validation **net** Sharpe at base costs, among configurations averaging at least 100 trades per year. Every grid point's validation result is reported. Position size is a constant 1× notional (no leverage). The holding period is 60 minutes, and the position is re-decided every hour: an unchanged position costs nothing; a change pays costs on |Δposition|.

**Cost-implied threshold (for interpretation).** A trade is only worth taking if the expected move exceeds the round-trip cost: (2p − 1)·E|r| > c. For example, with c = 0.14% and a hypothetical E|r| of 0.4%, that needs p > 0.675. Whether a model ever produces such confidence is itself evidence.

## 8. Execution and cost model

| Component | Base value (per side) | Status |
|---|---|---|
| Fee | 0.05%: Binance USD-M perpetual, regular-tier taker | **Sourced** (2026 schedule; earlier schedules differed) |
| Half-spread | 0.01% | **Assumption**: no historical order book in the dataset |
| Slippage | 0.01% | **Assumption**: small notional, a market order at the next minute's open |
| **Base total** | **0.07% per side = 0.14% round trip** | |
| Stress | ×2, ×3 costs; +2 bp and +5 bp extra slippage | **Sensitivity scenarios** |
| Spot alternative | 0.10% fee per side, long/flat only | **Sourced**, reported as a robustness check |

Instrument assumption: positions are taken in the BTCUSDT perpetual (which allows shorting), with returns measured on **spot** prices. The perpetual–spot basis and funding payments are a stated limitation; funding is added as a sensitivity check if the funding-rate files are downloaded. The capacity check compares the assumed trade size with observed 1-minute volume.

## 9. Statistical tests

| Question | Test |
|---|---|
| Are returns autocorrelated, and does volatility cluster? (development data) | Ljung–Box test on hourly returns and on \|returns\| |
| Univariate H1/H2 | Spearman rank correlation with a stationary block bootstrap p-value (mean block 1 week); **Holm** correction across 16 features |
| Out-of-sample AUC > 0.5? | Block-bootstrap 95% CI |
| Better than the base rate? | Per-observation log-loss difference; mean tested with a Newey–West (HAC) standard error |
| Is the strategy Sharpe > 0? | Block-bootstrap CI, Probabilistic Sharpe Ratio, **Deflated Sharpe Ratio** using the true number of configurations tried (from the experiment log) |
| Could a skill-less model do this? | Permutation test: shuffle training labels, re-fit, and compare the out-of-sample AUC with the real one (200 permutations, LR) |
| Does VaR work? | Kupiec proportion-of-failures test on rolling one-day VaR (mirrors the PD backtest in the credit module) |

The significance level is 5%. Every configuration evaluated is written to `reports/tables/experiment_log.csv`, including failures.

## 10. Final classification (decided in advance)

| Class | Criteria, all on the **untouched test period** |
|---|---|
| **STRONG EVIDENCE** | AUC CI lower bound > 0.5 **and** HAC log-loss improvement p < 0.05 **and** net Sharpe CI lower bound > 0 at base costs **and** net Sharpe > 0 at 2× costs **and** positive in ≥ 2/3 of half-years **and** Deflated Sharpe > 0.95 |
| **MODERATE EVIDENCE** | Statistical criteria met **and** net Sharpe point estimate > 0 at base costs, but the CI includes 0 or the 2× cost / sub-period checks fail |
| **WEAK / INCONCLUSIVE** | Statistical evidence in validation that is not confirmed on test, **or** statistical evidence on test without positive net performance |
| **NO ROBUST SIGNAL** | No statistically significant out-of-sample predictive ability on the test period |

## 11. "Try to kill the signal" checklist (run even if the result is negative)

Sub-periods (half-years); volatility and trend regimes; thresholds around the chosen k; costs ×0, ×1, ×2, ×3; extra slippage; execution lag 0/1/5 min; decision offsets of :15/:30/:45 past the hour (is the hour boundary special?); removing each feature group; LR vs LightGBM vs baselines; horizons of 15 and 30 min; the label-permutation test; random signal degradation (10/25/50% of signals flipped); and, after BTC is frozen, the identical pipeline on ETHUSDT. Each result is reported whether or not it helps.

## 12. Deviation log

| Date | Change | Reason | Effect on results |
|---|---|---|---|
| (none yet) | | | |
