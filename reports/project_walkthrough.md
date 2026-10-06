# Project Walkthrough: From Raw Data to Final Answer

**Project:** Credit Risk Modelling & Quantitative Market Signal Research (market module)
**Author:** Kudrati Khoda, M.Tech QR&OR, ISI Kolkata
**Purpose of this file:** explain, in simple English, every step of the market module:
- **what** we did;
- **why** we did it;
- **how** we did it (the exact method);
- **what we got** (the real numbers).

All numbers come from the project's saved outputs (`reports/tables/`, `reports/results.json`). Nothing here is estimated or invented.

---

## Contents

1. The big picture
2. Step 1: Choosing the data
3. Step 2: Downloading the raw data
4. Step 3: What the raw data looks like
5. Step 4: Data-health audit (checking the raw data)
6. Step 5: Cleaning and building the 1-minute grid
7. Step 6: Exploring the data (EDA)
8. Step 7: Writing the plan before looking (pre-registration)
9. Step 8: Building the target (what we predict)
10. Step 9: Feature engineering (what the model sees)
11. Step 10: The final modelling table, before any model is fitted
12. Step 11: Checking for leakage
13. Step 12: Testing the hypotheses one feature at a time
14. Step 13: Splitting time correctly
15. Step 14: Baselines and models
16. Step 15: Choosing the model and the trading rule
17. Step 16: The final test (run once)
18. Step 17: The backtest (turning predictions into money)
19. Step 18: Risk analysis
20. Step 19: Stress testing
21. Step 20: Robustness ("trying to kill the signal")
22. Step 21: Second asset (ETHUSDT)
23. Step 22: Final answer
24. Glossary
25. Where to find everything

---

## 1. The big picture

**The question:** can information we already have at time *t* predict whether Bitcoin's price goes **up or down in the next 60 minutes**? And if yes, can we make money from it **after paying trading costs**?

**The pipeline (each box is a step below):**

```
Raw Binance files ─► audit ─► clean 1-min grid ─► EDA ─► plan written in advance
      ─► target + 16 features ─► leakage check ─► hypothesis tests
      ─► walk-forward validation (choose model + rule) ─► ONE final test
      ─► backtest with costs ─► risk ─► stress ─► robustness ─► ETH check ─► answer
```

**The final answer in one line:** the prediction is **real** (test AUC 0.545) but **too small to trade**. The edge is 0.19 basis points per trade, while trading costs are 7 basis points, so the strategy lost 92% while simply holding Bitcoin made 98%.

---

## 2. Step 1: Choosing the data

**What we did.** We compared four possible data sources and chose the official **Binance Vision archive** (BTCUSDT spot, 1-minute candles).

| Option | Main problem |
|---|---|
| Polymarket | Blocked in India (May 2026), no official history of the order book, fine-grained prices kept for only a few weeks |
| Kalshi | Reportedly also being blocked in India; settles on a paid index, so we could not check its outcomes |
| Yahoo Finance (stocks) | Unofficial, daily data only, not related to crypto |
| **Binance Vision** | **Chosen:** official, free, has checksums, history from 2017, legal to download in India |

**Why.** A project you cannot reproduce or legally access is not usable. Binance is also the price source that Polymarket's hourly Bitcoin markets settle on, so the work stays relevant to prediction markets.

**What we got.** A clear data choice with written reasons (`reports/phase1_dataset_selection.md`).

---

## 3. Step 2: Downloading the raw data

**What we did.** We downloaded 110 monthly files of BTCUSDT 1-minute candles (August 2017 to September 2026). We also downloaded 81 files of funding rates (a cost of holding perpetual futures) and 110 files of ETHUSDT for a later check.

**How.**
- Script: `python -m src.data download`.
- Each file comes with a **SHA-256 checksum** from Binance, a fingerprint of the file. We computed the fingerprint of each downloaded file and compared the two. If they match, the file is exactly what Binance published.
- Every download was written to a **manifest** (URL, size, fingerprint, time), so anyone can check it later.

**Why.** It proves the data was not damaged or changed. Binance sometimes replaces old files, so recording the fingerprint and date matters.

**What we got.**

| Item | Result |
|---|---|
| BTCUSDT files | 110 of 110 downloaded, **110 of 110 checksums correct**, 235 MB |
| ETHUSDT files | 110 of 110, all correct, 219 MB |
| Funding-rate files | 81 of 81, all correct |

---

## 4. Step 3: What the raw data looks like

Each row of raw data is **one 1-minute candle**: a summary of all Bitcoin trades on Binance during one minute.

**Size:**
- **Rows (observations):** 4,789,279 one-minute candles.
- **Period:** 2017-08-17 04:00 UTC to 2026-09-30 23:59 UTC (about 9.1 years).
- **Raw columns:** 12 per row, as delivered by Binance. We added 4 helper columns for checking, giving 16 in the loaded table.

**The 12 raw columns:**

| Column | Meaning (simple) | Example (2024-03-05 13:59) |
|---|---|---|
| open_time | When the minute starts (UTC) | 2024-03-05 13:59:00 |
| open | First trade price in that minute (USDT) | 67,814.38 |
| high | Highest price in that minute | 67,814.39 |
| low | Lowest price in that minute | 67,770.01 |
| close | Last trade price in that minute | 67,770.01 |
| volume | Amount of BTC traded | 26.0 BTC |
| close_time | When the minute ends | 13:59:59.999 |
| quote_volume | Amount of USDT traded (≈ price × volume) | |
| n_trades | Number of separate trades | 1,083 |
| taker_buy_base | BTC bought by "aggressive" buyers (people who buy instantly at the asking price) | 11.3 BTC |
| taker_buy_quote | The same in USDT | |
| ignore | Unused by Binance | |

**Typical values across the full data:**

| Column | Average | Smallest | Middle (median) | Largest |
|---|---|---|---|---|
| Price (close) | 39,005 USDT | 2,817 | 29,146 | 126,115 |
| Volume per minute | 41.0 BTC | 0 | 17.4 | 5,878 |
| Trades per minute | 1,404 | 0 | 634 | 213,107 |

The price ranged about **39 times** from its lowest to its highest point over the sample.

**The 4 helper columns we added (for checking only):** timestamp unit (ms or µs), source file name, original row position in the file, and whether the file had a header row.

---

## 5. Step 4: Data-health audit (checking the raw data)

**What we did.** Before using the data, we checked it for every common problem.

**Why.** Every later result depends on the data. Bad timestamps or missing minutes can create fake patterns.

**How (each check and what it found):**

| Check | How we checked | Result |
|---|---|---|
| Columns present | Compared with the expected 12 columns | All present, none extra |
| Timestamp unit | Binance changed from milliseconds to **microseconds** on 2025-01-01. We detected the unit from the size of each number (13 digits = ms, 16 digits = µs) | 89 files in ms, 21 in µs, exactly as documented |
| Timestamps on the minute | Is each start time exactly hh:mm:00? | **21,602 rows were not** (explained below) |
| Missing minutes | Built a full minute-by-minute calendar and compared | 30,163 of 4,797,840 minutes missing (**0.629%**) in 33 gaps; **none from 2022 onward** |
| Duplicates | Same row twice, or same time with different values | 0 and 0 |
| Impossible candles | High below low? High below open or close? Price ≤ 0? Average traded price outside high–low? | 0 |
| Volume problems | Negative volume? Aggressive buying bigger than total volume? | 0 |
| Zero-volume minutes | Minutes with no trades | 24,003 (mostly 2017–18), kept |
| Extreme moves | Moves above 3% in one minute, and moves unusual for that day | 408 large moves, all at known market events (COVID crash 2020, May 2021, FTX 2022, Aug 2024, Oct 2025), kept |

**Missing data by year:** 2017 10.6%, 2018 0.98%, 2019 0.34%, 2020 0.24%, 2021 0.19%, 2023 0.015%, and **0% in 2022, 2024, 2025 and 2026**.

**The most interesting finding: "phase-shifted" candles.**
- From 2017-12-04 to 2017-12-18, Binance's candles start at **20.799 seconds** past each minute instead of exactly on the minute (20,401 rows).
- Around 2018-02-08 to 2018-02-10, the offset is **14.789 seconds** (1,201 rows).
- ETHUSDT has the **same problem at the same times**, so it was an exchange-wide event, not a mistake in our code.

**What we did about it and why.** We treated those minutes as **missing**.
- If we moved 06:00:20.8 back to 06:00, our features would see about 20 seconds of the **future**. That is "look-ahead bias", the most serious backtesting mistake.
- We did not invent a new rule after seeing the data.
- Cost: about 2.3 weeks of 2017–18 training data. The validation and test periods are not affected.

**What we got.** A full audit table (`reports/tables/BTCUSDT_audit_dq_table.csv`) and a list of exactly what was and was not changed.

---

## 6. Step 5: Cleaning and building the 1-minute grid

**What we did.** We put the data on a **regular grid**: one row for every minute from start to end, even if Binance had no candle for that minute.

**How (rules written before seeing the data):**
1. Exact duplicate rows: keep one copy. (We found 0.)
2. Conflicting duplicates: set to missing. (We found 0.)
3. Impossible candles: set to missing. (We found 0.)
4. Missing minutes: **left empty (NaN)**. Never filled with the previous price.
5. Extreme but real moves: **kept** and marked with a flag.

**Why not fill gaps?** Filling with the last price creates fake "no movement" minutes. That lowers measured volatility and can create false patterns around exchange outages.

**Why keep crashes?** Crashes are real risk. Deleting them would make the strategy look safer than it is.

**What we got.** A clean table of **4,797,840 rows** (one per minute) with 12 columns: open, high, low, close, volume, quote volume, number of trades, taker-buy volume (BTC and USDT), and three flags (minute present, zero volume, extreme move).

---

## 7. Step 6: Exploring the data (EDA)

**What we did.** Before any model, we studied returns, volatility, volume and time-of-day patterns.

**Important rule:** anything about the **future** (for example "does this hour predict the next hour?") was studied on data **before 2024 only**. The years 2024–2026 were kept hidden for the final test.

**What we got:**

| Question | How we measured | Answer |
|---|---|---|
| Are returns "normal" (bell-shaped)? | Kurtosis, and the share of hours beyond 4 standard deviations | **No.** Excess kurtosis 38 (a normal distribution has 0); 0.92% of hours are beyond 4σ versus 0.006% for a normal. Worst hour −20.1% (2020-03-12). |
| Does the last hour predict the next hour? | Autocorrelation of hourly returns; Ljung–Box test | Slightly, in the **opposite** direction: −0.047 at lag 1 (statistically significant). This is "reversal". |
| Do big moves follow big moves? | Autocorrelation of absolute returns | Strongly: 0.31 at 1 hour, still 0.22 at 24 hours ("volatility clustering") |
| Does volume predict the next move? | Rank correlation | It predicts the **size** of the next move (0.18), almost not its **direction** (0.012) |
| Time-of-day pattern? | Average absolute return and volume by UTC hour | Busiest and most volatile at 13–16 UTC (US market open); quietest at 03–05 UTC |
| Did the market change over time? | Average hourly volatility by year | From 1.28% (2017) down to about 0.4% (2023, 2025, 2026). The test period is a **calmer** market. |

**Why it matters.** EDA showed that (1) tails are fat, so normal-based risk numbers will be too low; (2) a small reversal effect may exist; (3) the market got calmer, so fixed trading costs will hurt more.

Figures: `figures/eda_*.png`.

---

## 8. Step 7: Writing the plan before looking (pre-registration)

**What we did.** Before downloading any data, we wrote down and saved in git:
- the question;
- the target;
- the 16 features;
- the models;
- how to split time;
- the trading rule options;
- the costs;
- which tests to run;
- how to classify the final result (STRONG / MODERATE / WEAK / NO SIGNAL).

**Why.** The biggest danger in backtesting is trying many versions and only reporting the one that worked ("data snooping"). Fixing the plan first, with a timestamp in git, proves we didn't do that. Any later change had to be written in a **deviation log** with its reason.

**What we got.** `reports/research_design_preregistration.md` and `src/config.py` (commit `b312d58`, before any data).

---

## 9. Step 8: Building the target (what we predict)

**What we did.** We made one decision every hour, on the hour (00:00, 01:00, …).

**Timing for a decision at time *t* (the most important definition):**

| Moment | What happens |
|---|---|
| Up to *t* | We may only use candles that have **already closed** (the last one is the candle from *t*−1 min to *t*) |
| *t* | We make the prediction |
| *t* + 1 minute | We trade at the **open price** of that minute |
| *t* + 61 minutes | We exit at the **open price** of that minute |

**Target:**
- the 60-minute return = ln(exit price / entry price);
- **Y = 1 if the return > 0 (price went up), else Y = 0**.

**Why wait one minute to trade?** The last price we see may be at the buy price or the sell price (the "bid-ask bounce"). Trading at that same price would assume zero delay and could create a fake reversal pattern. Waiting one minute is realistic and safe.

**Why one decision per hour?** Each holding period then ends exactly when the next begins, so labels never overlap. Overlapping labels make statistics look more certain than they are.

**Why up/down (classification) instead of predicting the exact return?** Hourly returns have huge outliers (kurtosis 38). A regression would be driven by a few crash hours. The size of returns still enters later through the profit-and-loss calculation.

**Real example (2024-03-05):**
- Decision at 14:00. Entry at the open of 14:01 = **67,684.40**. Exit at the open of 15:01 = **68,755.26**.
- Return = ln(68,755.26 / 67,684.40) = **+1.57%**, so **Y = 1** (up).

**What we got.** 79,964 hourly decisions from 2017-08 to 2026-09. The share of "up" hours was 50.7%, very close to a coin flip.

---

## 10. Step 9: Feature engineering (what the model sees)

**What we did.** We created **16 features**, in 5 groups. All are computed from candles that closed **before** the decision time.

**Rules for every feature:**
- **Scale-free:** they use returns and ratios, not raw prices, so a $4,000 Bitcoin and a $100,000 Bitcoin are comparable.
- **Past only:** every rolling window ends at the decision time and looks backwards.
- **Coverage rule:** a feature is computed only if **at least 90%** of the minutes in its window exist. Otherwise it is left empty and that hour is dropped. Nothing is filled in.

**The 16 features** (examples are from the real 2024-03-05 14:00 decision):

| # | Feature | Group | How it is calculated | Window | Why it might predict | Example |
|---|---|---|---|---|---|---|
| 1 | ret_1m | Returns | ln(close at *t*−1m / close at *t*−2m) | 1 min | Very short-term reversal | −0.0007 |
| 2 | ret_5m | Returns | Log return over the last 5 minutes | 5 min | Short-term reversal or continuation | +0.0017 |
| 3 | ret_15m | Returns | Log return over the last 15 minutes | 15 min | Same | −0.0008 |
| 4 | ret_30m | Returns | Log return over the last 30 minutes | 30 min | Same | +0.0018 |
| 5 | ret_60m | Returns | Log return over the last 60 minutes | 60 min | Same | +0.0066 |
| 6 | ret_4h | Momentum | Log return over the last 4 hours | 240 min | Intraday trend | +0.0203 |
| 7 | ret_24h | Momentum | Log return over the last 24 hours | 1,440 min | Daily trend | +0.0400 |
| 8 | ma_dist_24h | Momentum | ln(price / average price of the last 24 h) | 1,440 min | Price "stretched" away from its average may snap back | +0.0095 |
| 9 | log_rv_60m | Volatility | ln of realised volatility: √(60 × mean of squared 1-min returns) | 60 min | Volatility level; risk changes behaviour | −4.89 |
| 10 | log_rv_24h | Volatility | The same over 24 h, in hourly units | 1,440 min | Slower volatility level | −4.67 |
| 11 | vol_ratio | Volatility | log_rv_60m − log_rv_24h | 1,440 min | Is the last hour unusually wild? | −0.22 |
| 12 | range_rel_60m | Volatility | ln(highest high / lowest low over 60 min) ÷ 24 h volatility | 1,440 min | Was the last hour's range unusual? | 1.13 |
| 13 | rel_volume_60m | Volume | ln(volume in the last hour / average hourly volume over 7 days) | 10,080 min | Unusual attention or news | +0.27 |
| 14 | volume_chg_60m | Volume | ln(volume in the last hour / volume in the hour before) | 120 min | Activity speeding up | +0.36 |
| 15 | taker_imb_60m | Order flow | (aggressive buys − aggressive sells) ÷ volume over 60 min, from −1 to +1 | 60 min | Buying or selling pressure | +0.19 |
| 16 | clv_60m | Candle | (close − lowest low) ÷ (highest high − lowest low) over 60 min; 0 = closed at the low, 1 = at the high | 60 min | Where the hour finished in its range | 0.64 |

**Why only 16, and why these?**
- Each one has a clear economic reason: momentum, reversal, volatility, attention, order flow.
- Using hundreds of popular technical indicators increases the chance of finding a pattern by luck.

**Typical feature values (all decisions):**

| Feature | Mean | Std. dev. | Min | Median | Max |
|---|---|---|---|---|---|
| ret_60m | 0.0000 | 0.0076 | −0.201 | 0.0001 | 0.160 |
| ret_24h | 0.0007 | 0.0350 | −0.625 | 0.0008 | 0.299 |
| log_rv_60m | −5.35 | 0.71 | −8.69 | −5.38 | −1.63 |
| rel_volume_60m | −0.20 | 0.63 | −4.48 | −0.24 | 3.47 |
| taker_imb_60m | −0.010 | 0.138 | −0.993 | −0.008 | 1.000 |
| clv_60m | 0.516 | 0.285 | 0.000 | 0.522 | 1.000 |

(All 16 are in `reports/tables/feature_summary.csv`.)

**Extra labels (not features; used only for analysis):**
- **Volatility regime:** high / mid / low, based on where today's 24 h volatility sits within the last 90 days.
- **Trend regime:** "trending" if the 24 h move is bigger than one normal daily move, else "range".

---

## 11. Step 10: The final modelling table, before any model is fitted

This is the table the models learn from: **one row per hour**.

| Item | Value |
|---|---|
| Total hourly rows | 79,964 |
| Rows with all 16 features and a target (usable) | **78,522** |
| Rows dropped (a feature window was less than 90% complete) | 1,442 (1.8%) |
| Columns used by the model | 16 features + 1 target (Y) |
| Other columns kept for the backtest | entry time, exit time, entry price, exit price, return, regime labels |
| Development data (before 2024) | 54,427 usable rows, of which 11,012 are before 2019 (training only) |
| **Test data (2024-01 to 2026-09)** | **24,095 rows**, kept hidden until the very end |
| Share of "up" hours | 50.7% overall; 50.5% in the test period |

**The real 2024-03-05 14:00 row** (shortened):

| ret_60m | ret_4h | ret_24h | taker_imb_60m | clv_60m | … (11 more) | entry price | exit price | Y |
|---|---|---|---|---|---|---|---|---|
| +0.0066 | +0.0203 | +0.0400 | +0.19 | 0.64 | … | 67,684.40 | 68,755.26 | 1 |

---

## 12. Step 11: Checking for leakage

**What is leakage?** When a feature accidentally contains information from the future. The model then looks brilliant in the backtest and fails in real life.

**What we did:**
1. **Timing table:** for every feature, we wrote down the last candle it uses (always the one closing at *t*) and when we trade (*t* + 1 min). See `reports/tables/feature_timing_table.csv`.
2. **Perturbation test (the strong check):** at 24 random decision times between 2018 and 2026, we replaced **all data at or after *t*** with random numbers and recomputed the features.
   - If any feature changed, it was using the future.
   - **Result: 0 of 16 features changed at all 24 times.**
   - The target *did* change all 24 times, which proves the random numbers really reached the future data.
3. **Scaling inside training only:** the feature scaling (mean and standard deviation) is learned from training data only, never from the whole dataset.
4. **Purging:** training rows whose 61-minute holding window would run into the next test block were removed.
5. **Unit tests:** 25 automatic tests check timing, leakage, costs and risk formulas.
6. **Synthetic controls:** on 40 fake random-walk price series (where no pattern exists), the pipeline found no average edge (mean AUC 0.498). On 20 fake series with a pattern built in, it found the pattern every time. So the method neither invents signals nor misses real ones.

---

## 13. Step 12: Testing the hypotheses one feature at a time

**What we did.** We tested 5 hypotheses that were written down in advance, using **development data only (before 2024)**.

**How:**
- For each feature, we measured the **Spearman rank correlation** with the next-hour return.
- Hourly data is not independent, so p-values came from a **block bootstrap**: re-sampling whole weeks of data, which keeps volatility clustering intact.
- We tested 16 features at once, so we used the **Holm correction** to avoid false positives.

**What we got:**

| Hypothesis | Result | Verdict |
|---|---|---|
| H1: Past returns predict the next hour | Correlations −0.046 (5 min) to −0.074 (30 min); `clv_60m` −0.076. The 30-min and 4-h ones were negative in **all 13 half-years** 2017–2023 | **Supported: reversal** (after a rise, the next hour is slightly weaker) |
| H2: Volume and order flow predict it | Buying pressure −0.036 (reversal); relative volume +0.012 (tiny); volume change not significant | **Partly supported** |
| H3: Effects differ by market regime | Only 2 of 32 comparisons differed (about 1.6 expected by luck) | **Not supported** |
| H4: A model beats single features | Tested in Step 14 | Supported for prediction |
| H5: It makes money after costs | Tested in Step 16 | **Rejected** |

14 of 16 features were significant after correction. With 55,000 rows, even tiny correlations are "significant", which is why significance alone is not enough: we must test money.

---

## 14. Step 13: Splitting time correctly

**What we did.** We never used random splitting. We used **walk-forward validation**:

```
2017-08 … training (grows each step) … | validate 2019H1 | → retrain → | validate 2019H2 | → … → | validate 2023H2 |
2024-01 … FINAL TEST (hidden until the design was frozen; model refitted every 6 months; no changes) … 2026-09
```

- **Development:** 10 validation blocks of 6 months each (2019H1 to 2023H2). Training size grew from 12,043 to 51,451 rows.
- **Final test:** 24,095 hours, used **once**.

**Why not random splitting?** It trains on the future and tests on the past. Neighbouring hours are very similar because of volatility clustering, so random splits make models look better than they are.

---

## 15. Step 14: Baselines and models

**What we did.** We started simple and only accepted complexity if it clearly helped.

| Model | What it is | Why |
|---|---|---|
| Base rate | Always predicts the historical share of up-hours | The "no skill" benchmark |
| Momentum rule | Up if the last 1 h (or 24 h) return was positive | The simplest trading idea |
| **Logistic regression** | Linear model giving a probability; C = 0.01, 0.1 or 1 (strength of regularisation) | Interpretable, hard to overfit |
| **LightGBM** | Gradient-boosted trees, two small, careful settings (7 or 15 leaves, at least 500–1,000 rows per leaf) | Can capture non-linear patterns |

**Pre-written simplicity rule:** use LightGBM only if its average AUC beats logistic regression by more than 0.005 **and** it has better log-loss in at least 70% of the validation blocks.

**Validation results (2019–2023):**

| Model | AUC | Accuracy | Better log-loss than base rate in |
|---|---|---|---|
| Base rate | 0.496 | — | — |
| Logistic regression | 0.557 | 54.4% | 9 of 10 blocks |
| **LightGBM (7 leaves)** | **0.565** | 54.9% | 9 of 10 blocks |
| Momentum 1 h rule | 0.453 (below 0.5, i.e. reversal) | — | — |

---

## 16. Step 15: Choosing the model and the trading rule

**Model choice.** LightGBM beat logistic regression by +0.0064 AUC and had better log-loss in **exactly 7 of 10 blocks**, which is the limit itself. So it passed the rule, but only just.

**Trading rule.** Let *p* be the model's predicted chance of "up" and *s* the typical spread of its predictions on training data.
- **Long** (buy) if *p* > 0.5 + k·*s*.
- **Short** (sell) if *p* < 0.5 − k·*s* (in long/short mode only).
- Otherwise **no trade** (the no-trade zone).
- We tried k = 0, 0.25, 0.5, 1, 1.5, each with long/short and long-only: 10 options.
- Position size: always 1× (no leverage).

**What we got in validation:** **all 10 options lost money after costs** (net Sharpe −3.9 to −7.2), even though before costs they were positive (0.6 to 1.6). The pre-written rule picked the least-bad option: **k = 1.5, long only**.

This was already a warning sign before the final test. The choice was saved and committed to git (`231634d`) **before** the test was run.

---

## 17. Step 16: The final test (run once)

**What we did.** We ran the frozen model and rule once on 2024-01 to 2026-09. The model was refitted every 6 months on all earlier data, with no changes to settings.

**Prediction results:**

| Measure | Result | Meaning |
|---|---|---|
| AUC | **0.545** (95% CI 0.538–0.552) | Real skill: the whole range is above 0.5 |
| Log-loss vs base rate | Better, p = 0.004 | The probabilities are more informative than a coin |
| Accuracy | 53.5% (base rate 50.5%) | Right a bit more often than chance |
| Label-shuffle test | Real AUC beat all 200 shuffled versions, p = 0.005 | Not luck |
| AUC by half-year | 0.558, 0.567, 0.536, 0.535, 0.534, 0.532 | Always above 0.5, slowly fading |
| Calibration | The top 10% of predictions said 60% up; 56.4% actually went up | Over-confident, but ranked correctly |

**Why did it fade?** The market became calmer. The volatility features **drifted**: PSI (Population Stability Index, the same tool used for loan vintages in the credit project) reached 3.5, where above 0.25 counts as a big shift.

---

## 18. Step 17: The backtest (turning predictions into money)

**What we did.** For every hour we recorded: decision time, trade time, entry price, exit price, position, change in position, gross return, cost and net return.

**Costs (per side, i.e. per buy or per sell):**

| Cost | Value | Source |
|---|---|---|
| Exchange fee | 5 bp (0.05%) | Binance's published fee for futures takers |
| Half the bid-ask spread | 1 bp | Assumption (no historical order book) |
| Slippage | 1 bp | Assumption |
| Funding | Actual history | From Binance funding files |
| **Total** | **7 bp per side = 14 bp per round trip** | |

Costs are charged only when the position changes. Staying long for two hours in a row costs nothing extra.

**What we got:**

| | Our strategy | Just holding Bitcoin |
|---|---|---|
| Net Sharpe | **−5.84** (95% CI −6.94 to −4.77) | 0.76 |
| Gross Sharpe (before costs) | 0.16 | 0.76 |
| Total return | **−92.2%** | +97.6% |
| Max drawdown | −92.3% | −53.7% |
| Time in the market | 9.1% of hours | 100% |
| Round trips per year | 667 | ≈0 |

**Why it fails.**
- **Size:** the average return in each prediction decile was only −1.3 to +1.9 bp, against 14 bp of cost per round trip.
- **Gross vs costs:** gross gains totalled 7.0% of capital, but costs totalled 256.8%.
- **Break-even cost:** the strategy would only break even if trading cost **0.19 bp per side**, about 37 times cheaper than reality.
- **Confidence:** to beat costs, the model would need to be more than 71.6% sure; its most confident predictions averaged 60%.

---

## 19. Step 18: Risk analysis

**What we did.** We measured the tail risk of daily strategy returns.

| Method | What it is | 1-day 99% VaR | 99% ES |
|---|---|---|---|
| Historical | Uses the actual worst days | **3.53%** | 4.54% |
| Normal | Assumes a bell curve | 2.18% | 2.47% |
| Student-t | Fat-tailed curve | Not usable here: the fit broke down because the strategy is out of the market most days | — |

- **VaR** = a loss level exceeded only on about 1 day in 100.
- **ES (Expected Shortfall)** = the average loss on those worst days.
- The normal assumption **understates** risk by about 38%, because returns have fat tails.

**VaR backtest (Kupiec test).** We counted the days when losses beat a rolling VaR forecast. At 99%: 9 days against 7.5 expected (p = 0.60), so the VaR model is fine. This is the same idea as comparing expected and actual defaults in the credit project.

**Monte Carlo (block bootstrap).** We created 5,000 possible one-year futures by re-sampling 10-day blocks of real daily returns.
- Chance of losing money over a year: **100%**.
- Median one-year return: −60%.

We used this instead of the credit project's Gaussian model because here we *observe* the returns directly, so resampling keeps their real fat tails.

---

## 20. Step 19: Stress testing

**What we did.** We asked what happens when conditions get worse.

| Scenario | Net Sharpe | What we learn |
|---|---|---|
| Normal costs | −5.81 | Fails |
| Zero costs | +0.16 | The only positive case, and barely |
| 2× costs / 3× costs | −11.40 / −16.33 | Fails worse |
| +2 / +5 bp extra slippage | −7.46 / −9.86 | Fails |
| Twice the volatility | −2.85 (but 99% VaR doubles to 6.5%) | Bigger moves dilute fixed costs but double the tail risk |
| 10% / 25% / 50% of signals flipped | −6.02 / −6.31 / −6.58 | No safety margin |
| High / mid / low volatility periods | −4.78 / −5.48 / −7.75 | No regime saves it |
| Crash of 2024-08-05 | Strategy −7.2% vs Bitcoin −6.9% | A buy-the-dip rule can catch a falling knife |
| Instant −10% gap while long | −10% loss = 2.8× the daily 99% VaR | VaR understates sudden gaps |

---

## 21. Step 20: Robustness ("trying to kill the signal")

**What we did.** After recording the final result, we changed one thing at a time on the test period (23 variants). These were **only reported**, never used to change the choice.

| What we changed | AUC range | Net Sharpe range |
|---|---|---|
| Thresholds and modes (10 versions) | 0.545 | −5.28 to −10.14 |
| Removing each feature group (5 versions) | 0.537 to 0.545 | −4.84 to −6.02 |
| Logistic regression instead of LightGBM | 0.540 | −4.35 |
| Trading 0 or 5 minutes after the decision | 0.548 / 0.540 | −4.57 / −6.31 |
| Deciding at :15, :30 or :45 past the hour | 0.534 to 0.539 | −4.43 to −4.99 |
| 15- or 30-minute horizon | 0.538 / 0.536 | −17.27 / −10.10 |

**What we got.**
- The **prediction** survived every change (AUC always above 0.53).
- The **money** survived none: 23 of 23 lost after costs.
- The always-in-market version had a gross Sharpe of 2.05, but it trades 3,685 times a year, so it ends at −8.8 net.

---

## 22. Step 21: Second asset (ETHUSDT)

**What we did.** We ran the exact same frozen pipeline on Ethereum, with no re-tuning.

**Why.** To check whether the pattern is real or just a Bitcoin accident.

**What we got.**
- AUC **0.546** (95% CI 0.538–0.554); log-loss improvement p = 0.00001.
- Net Sharpe −3.70. Even before costs it was negative (gross Sharpe −0.52).

The same pattern holds: real prediction, no money.

---

## 23. Step 22: Final answer

| Question | Answer |
|---|---|
| Is there a predictive signal? | **Yes:** short-term reversal |
| Is it statistically real? | **Yes:** test AUC 0.545 (CI above 0.5), shuffle test p = 0.005, holds on ETH |
| Does ML beat a simple rule? | For prediction yes; for money no (buy-and-hold beat everything) |
| Does it survive costs? | **No:** break-even cost 0.19 bp vs 7 bp |
| Is it robust? | The statistics are; the losses are just as robust (23 of 23) |
| **Pre-registered verdict** | **WEAK / INCONCLUSIVE**: "statistical evidence without positive net performance" |

**In plain words:** Bitcoin's next-hour direction can be predicted a little, but the edge is about 1–2 basis points while trading costs are 14. So it cannot be traded profitably on Binance.

**Idea for next time (exploratory, not proven).** Prediction markets like Polymarket pay on **direction only**, not size. The model's most confident "up" calls were right 56.6% of the time, versus 51.75% needed to break even on a 50-cent contract. But we never saw real Polymarket prices, so this needs a new test with real data.

---

## 24. Glossary

| Term | Simple meaning |
|---|---|
| Candle | Summary of one minute of trading: open, high, low, close, volume |
| Basis point (bp) | 0.01%. 14 bp = 0.14% |
| Log return | ln(new price / old price); about the same as the % change for small moves |
| AUC | Chance the model ranks a random "up" hour above a random "down" hour. 0.5 = coin flip |
| Log-loss | Penalty for bad probabilities; lower is better |
| Sharpe ratio | Average return ÷ volatility, per year. Above 1 is good; below 0 means losing |
| Gross / net | Before / after trading costs |
| Drawdown | Fall from a peak to the next low |
| VaR / ES | A rare-loss threshold / the average loss beyond it |
| PSI | How much a variable's distribution has shifted between two periods |
| Walk-forward | Train on the past, test on the next block, move forward, repeat |
| Leakage / look-ahead | Accidentally using future information |
| Block bootstrap | Re-sampling chunks of time to measure uncertainty without breaking time patterns |
| Holm correction | An adjustment for testing many things at once |
| Pre-registration | Writing the plan down before seeing the data |

---

## 25. Where to find everything

| What | File |
|---|---|
| Full research report | `reports/final_report.md` |
| Plan written before the data, and its deviation log | `reports/research_design_preregistration.md` |
| Why Binance | `reports/phase1_dataset_selection.md` |
| Audit tables | `reports/tables/BTCUSDT_audit_*.csv` |
| Feature timing and leakage test | `reports/tables/feature_timing_table.csv`, `leakage_perturbation_real_data.csv` |
| Model, validation and test results | `reports/tables/validation_*.csv`, `test_*.csv` |
| Risk, stress and robustness | `reports/tables/risk_*.csv`, `stress_*.csv`, `robustness_variants.csv` |
| Every experiment run | `reports/tables/experiment_log.csv` (44 rows, exactly 1 final test) |
| Code | `src/` (data, quality, features, models, backtest, risk…) |
| Step-by-step notebooks | `notebooks/02` to `10` |
