# FINAL INTERVIEW-READINESS AUDIT

**Credit Risk Modelling & Quantitative Market Signal Research**

Documentation and interview-readiness audit of the completed project · Kudrati Khoda · 7 October 2026

---

## Scope, method and labels

This is an audit of the project **as it already exists**. Nothing in the market research was re-run, re-trained or re-tuned. The test period, target, features, validation design, costs and results are unchanged. The credit model was not modified or recreated.

**Sources treated as the truth:** the code in `src/` and `scripts/`, `src/config.py`, `reports/selection.json`, `reports/results.json`, the 64 tables in `reports/tables/`, the experiment log, the pre-registration and its deviation log, the git history, the credit notebook and the original credit PDF, the teaching walkthrough, and the files you originally uploaded (credit notebook, credit PDF, CV, Gravia JD).

**Checks done for this audit (read-only):** SHA-256 comparison of the repository's credit files with your uploads; a full-disk search for LendingClub data, model files and predictions; a scan of every notebook cell; the git history of the selection rule; recomputation of the AUC gain from `validation_models.csv`; the experiment-log timestamps against the git commits.

| Label | Meaning in this audit |
|---|---|
| **VERIFIED** | Checked against code **and** a saved output (or a byte-level file check) in the repository |
| **REPORTED** | Stated in your original credit PDF; the code that would produce it exists, but it could not be re-run here (data and outputs absent) |
| **REPORTED BUT NOT CURRENTLY REPRODUCIBLE** | Stated in a report, but no code, model or output exists to reproduce it |
| **NOT VERIFIED** | Nothing in the repository can confirm or reproduce it |

**Documentation fixes made during this audit (documentation only):**
1. `reports/research_design_preregistration.md`: one new deviation-log row clarifying the "≥ / >" wording (Task 2). The original §5 sentence was left exactly as committed.
2. `credit_module/README.md`: an "Evidence status" section and a LightGBM row labelled "not currently reproducible" (Task 1).
3. The teaching walkthrough was updated to match (two notes), and this audit was added to `reports/teaching/`.

---

# TASK 1 — Audit of the Original Credit-Risk Module

## 1.1 What exists in the repository

| Item | Found? | Detail |
|---|---|---|
| Original notebook `notebooks/01_credit_risk.ipynb` | Yes | **Byte-identical** (SHA-256 `63cf49ae…`) to your upload `Credit_Risk_Project3_Validation_Extended.ipynb`. Added in commit `9f5aa93`; never modified since. **VERIFIED unchanged.** |
| Original report `credit_module/Credit_Risk_Project_Report.pdf` | Yes | Byte-identical (SHA-256 `bda85a5b…`) to your upload `Credit_Risk_Project_What_We_Did_Report.pdf`. 19 pages. |
| LendingClub data `accepted_2007_to_2018Q4.csv.gz` | **No** | Not in the repository and not anywhere on this machine (full-disk search). Distributed via Kaggle (account required). |
| Logistic-regression code | Yes | Cell 6 (liblinear, 300,000-loan stratified sample of ≤ 2016 loans) and cell 19 (lbfgs, C = 1, 300,000-loan random sample) |
| LightGBM code | **No** | 0 occurrences of "lightgbm", "LGBM" or "lgb" in any cell |
| Saved notebook outputs | **No** | 13 code cells, 0 with outputs, all execution counts empty |
| Saved predictions, model files, intermediate datasets | **No** | No `.pkl`, `.joblib`, model, prediction or credit dataset file anywhere on disk |
| Code for KS, Gini, Brier, calibration, LGD, ECL, Monte Carlo, stress, vintage AUC, PSI | Yes | Cells 6–27 |

## 1.2 Answers to your ten questions

**1. Is the logistic-regression credit model reproducible from the current repository?**
**Not as it stands.** The code is there, but the data is not, and the notebook has no saved outputs, so nothing can be re-run or checked here. It is *reproducible in principle* (download the Kaggle file, run the notebook). One extra nuance found in the notebook itself: its markdown says the headline results "remain those from the original full-sample run", while the first model cell fits on a 300,000-loan stratified sample. So even with the data, the headline numbers (AUC 0.6999, KS 0.2907) may not come out to the last digit from this exact notebook. The extended cell corresponds to the "extended OOT run" (AUC 0.6992, Brier 0.1639).
Status: **REPORTED** (code present, not re-run).

**2. Is the LightGBM credit model reproducible?** **No.** There is no LightGBM code, no saved model, no predictions and no output in the repository. Status: **REPORTED BUT NOT CURRENTLY REPRODUCIBLE.**

**3. Can AUC 0.708, KS 0.301, Gini 0.416, Brier 0.1615 be verified?** **No.** They appear only in the PDF (§3.2 table: 0.7080, 0.3012, 0.4159, 0.1615). The only check possible is the report's own arithmetic: Gini = 2 × AUC − 1 = 2 × 0.7080 − 1 = 0.4160, consistent with the reported 0.4159 within rounding. That checks the report's internal consistency, **not** the model.

**4. Is the LendingClub data present?** **No.**

**5. Does LightGBM training code exist?** **No.**

**6. Do saved predictions, model files, intermediate datasets or evaluation outputs exist?** **No.** None of any kind for the credit module.

**7. Status of each credit claim:**

| Claim | Value (from your PDF) | Status |
|---|---|---|
| Notebook and PDF are your unchanged originals | — | **VERIFIED** (SHA-256) |
| Binary target: Fully Paid = 0; Charged Off / Default = 1; unresolved loans excluded | — | **VERIFIED as code** (cell 2) |
| Post-origination fields (recoveries, payments, settlement, hardship) excluded from the PD model | — | **VERIFIED as code** (feature lists in cell 4) |
| Out-of-time split: train ≤ 2016, validate 2017, test 2018 | — | **VERIFIED as code** (cell 4) |
| Preprocessing: median / most-frequent imputation, scaling, one-hot | — | **VERIFIED as code** (cell 6) |
| KS, Gini, Brier, decile calibration, LGD, ECL, one-factor Monte Carlo, PD-odds stress, ρ sensitivity, vintage AUC, PSI | — | **VERIFIED as code** (cells 6–27) |
| About 1.345M resolved loans | 1.077M good / 268.6k defaults | **NOT VERIFIED** (reported; data absent) |
| Logistic regression, 2017 OOT | AUC 0.6999, KS 0.2907, Gini 0.3997, Brier 0.1638 | **REPORTED** |
| Extended OOT run | AUC 0.6992, Brier 0.1639; mean PD 19.91% vs observed 23.13% | **REPORTED** |
| Calibration: observed > predicted in all 10 deciles | — | **REPORTED** |
| Aggregate backtest 2017 | expected 33,719 vs actual 39,169 | **REPORTED** |
| Vintage AUC | 0.731 / 0.708 / 0.699 / 0.694 (2015–2018) | **REPORTED** |
| Score PSI vs 2016 | ≤ 0.027 | **REPORTED** |
| Mean realised LGD; ECL | 93.29%; ≈ $494.68M | **REPORTED** (EAD = funded amount, an assumption) |
| Monte Carlo VaR99 / ES99 at ρ = 0.15; stress | $68.54M / $74.61M; $83.01M / $90.33M | **REPORTED** |
| ρ sensitivity of VaR99 | $30.46M (ρ = 0) → $108.36M (ρ = 0.5) | **REPORTED** |
| LightGBM challenger | AUC 0.7080, KS 0.3012, Gini 0.4159, Brier 0.1615 | **REPORTED BUT NOT CURRENTLY REPRODUCIBLE** |

**8. What you can safely say about the credit project.**

> "My first project was a credit-risk workflow on public LendingClub loans. I built a logistic-regression probability-of-default model using only information available at origination, and validated it out of time — trained on loans up to 2016 and tested on 2017 and 2018. In my original write-up the 2017 AUC was about 0.70 and KS about 0.29. The ranking was useful, but the model under-predicted default rates in every risk decile, which taught me that ranking and calibration are different things. I then combined PD with a realised LGD and funded amount as an exposure proxy to get expected credit loss, and ran a one-factor Monte Carlo to get VaR and Expected Shortfall. The main lesson was model risk: 99% VaR went from about \$30 million to over \$100 million just by changing the assumed default correlation from 0 to 0.5. It's an academic framework — the LGD and EAD are simplified — and the notebook isn't saved with outputs, so the numbers come from my original report."

**9. How to present the LightGBM credit result professionally (do not recreate or estimate it).**
* **On the CV:** do not list LightGBM metrics for the credit project, and do not write "using Logistic Regression and LightGBM" for the credit bullet, because a reviewer cannot see that model. (Your current CV says exactly this — see "Remaining actions".)
* **If asked in an interview:** "I also tried a LightGBM challenger. My original report recorded AUC 0.708 versus 0.700 for logistic regression — a small gain. I didn't keep that code in the final notebook, so I treat it as an unreproduced side result and use the logistic model as the reference. The same question came up in my market project, where I made the choice formal with a pre-registered rule."
* **If you still have the original LightGBM code or notebook on your own machine:** add it to the repository **exactly as it was**, unedited, with a note saying when it was added. Then it becomes "REPORTED, code present". Do not rewrite it or tune it to match 0.708.

**10. What needs fixing.**
* **Repository (done in this audit, documentation only):** `credit_module/README.md` now has an evidence-status table and a LightGBM row labelled "not currently reproducible".
* **Your CV (you must do this):** change the credit bullet's "using Logistic Regression and LightGBM" to a logistic-regression wording (suggested text in "Remaining actions").
* **Optional, not required:** re-run the notebook on your own machine with the Kaggle file and commit it with outputs. If you do, report whatever numbers it produces, even if they differ slightly from the PDF, and say why (sampled vs full-sample run).

---

# TASK 2 — The Pre-registration / Model-Selection Wording

## 2.1 What each source says

| Source | Wording | When written (UTC, git) |
|---|---|---|
| Pre-registration §5 (`research_design_preregistration.md`) | "mean validation AUC is **≥ 0.005** higher **and** its log-loss is lower in ≥ 70% of folds" | Commit `b312d58`, 2026-10-05 20:26 |
| `src/config.py` comment, same commit | "improves mean validation AUC **by more than this**" (`LGBM_MIN_AUC_GAIN = 0.005`) | `b312d58`, 20:26 |
| `src/pipeline.py` (the code that decided) | `use_gb = auc_gain > 0.005 and gb_win_share >= 0.70` (strict **>** for AUC; **≥** for folds) | First committed in `ce0af80`, 20:37; unchanged in `9f5aa93` and `231634d` |
| `reports/selection.json` (the frozen decision) | "LightGBM AUC gain +0.0064 (needs **>0.005**), log-loss better in 70% of folds (needs >=70%) -> LightGBM" | Frozen 21:23:10; committed `231634d` 21:23:58 |
| README, notebooks, final report, walkthrough | "more than 0.005" | after the freeze |

**First real data download:** 21:08:08 UTC (`manifest.csv`). So **both** wordings — the "≥" in the document and the strict ">" in the code and config comment — were written **before any data existed.** The code was never changed after the data arrived.

## 2.2 What actually happened (VERIFIED, re-derived from `validation_models.csv`)

* Best logistic regression by pooled log-loss: C = 0.01, mean fold AUC **0.559772**.
* Best LightGBM: config 1 (200 trees, 7 leaves), mean fold AUC **0.566154**.
* Gain = **+0.006383** (your "≈ +0.0064" is correct).
* Fold criterion: LightGBM's log-loss lower than logistic regression's in **7 of 10** folds = 70% (needs ≥ 70%) — **passes exactly at the threshold**.

## 2.3 Does the difference change the decision?

| Rule wording | 0.006383 passes? | 7/10 passes? | Decision |
|---|---|---|---|
| "≥ 0.005" (pre-registration text) | Yes | Yes | LightGBM |
| "> 0.005" (code) | Yes | Yes | LightGBM |

The two wordings only give different answers if the gain is **exactly** 0.005000. It was 0.006383. **The decision is identical.**

## 2.4 Classification

**B — a documentation / wording inconsistency only.** It is not a methodological problem because (a) both versions were written before the data, (b) the implemented rule was never changed, (c) the decision is the same under either version, and (d) nothing downstream (test, verdict) depends on the difference.

A second small wording gap was found and resolved at the same time: §5 says "log-loss is lower in ≥ 70% of folds" without saying *lower than what*. The code compares LightGBM's fold log-loss with **logistic regression's**.

**What *is* worth saying honestly:** the real fragility is not ≥ vs >; it is that LightGBM passed the fold criterion exactly at 7 of 10. One fold the other way and logistic regression would have been chosen. And on the test, logistic regression lost less money (net Sharpe −4.35 vs −5.81). That is a lesson, not an error.

## 2.5 Resolution (wording added to the deviation log, original §5 left untouched)

> **2026-10-07 (after the test; documentation only, no rule change).** Wording clarification of the simplicity rule (§5). §5 says LightGBM's mean validation AUC must be "≥ 0.005 higher". The rule as implemented and applied is strict: `auc_gain > 0.005` (`src/pipeline.py`, committed in `ce0af80` before any data was downloaded; the `src/config.py` comment in this same pre-registration commit `b312d58` already says "by more than this"). "Its log-loss is lower in ≥ 70% of folds" means lower than logistic regression's log-loss in the same fold, which is how the code compares them. The §5 sentence is left exactly as originally committed. **Effect: none.** The observed gain was +0.00638 (mean fold AUC 0.56615 vs 0.55977), which passes under either "≥" or ">"; the wordings differ only if the gain were exactly 0.005. Fold criterion: 7 of 10 = 70%, passing at the threshold. Selection, test results and verdict are unchanged.

**Interview wording:** "My pre-registration document said 'at least 0.005' while the code, written before the data, used 'more than 0.005'. The actual gain was 0.0064, so both versions choose LightGBM. I logged the wording difference in the deviation log rather than editing the original text. The more important point is that LightGBM passed the fold test exactly at 7 out of 10, so I describe the choice as marginal."

---

# TASK 3 — What I Can Claim in an Interview

Column 2 uses **YES**, **YES, WITH QUALIFICATION** or **NO**.

## A. Dataset

| Claim / Statement | Can I say this? | Correct interview wording | Why / Evidence |
|---|---|---|---|
| I used Binance BTCUSDT 1-minute data | YES | "Binance spot BTCUSDT 1-minute klines from Binance's official public archive, data.binance.vision." | `manifest.csv` URLs; `src/data.py` |
| All files were checksum-verified | YES | "All 110 monthly files were verified against Binance's SHA-256 checksums." | `manifest.csv`: 110/110 `checksum_ok` |
| Date range | YES | "17 August 2017 to 30 September 2026, UTC." | `results.json` audit: first/last open time |
| About 4.8 million candles | YES | "4,789,279 one-minute candles on a 4,797,840-minute grid." | `results.json`; audit table |
| 0.63% missing minutes | YES, WITH QUALIFICATION | "30,163 minutes, 0.63%, in 33 gaps. About 78% of them fall in two 2017–18 stretches affected by phase-shifted candles (one also contains a real outage). There are no missing minutes in 2022 or 2024–2026." | `BTCUSDT_audit_gaps_top25.csv` (20,413 + 3,226 minutes); `missing_by_year.csv` |
| Binance changed timestamps from ms to µs | YES | "From 2025 the spot files use microseconds instead of milliseconds; I detect the unit per file from the number of digits and refuse to guess." | 89 ms / 21 µs files; `detect_time_unit`; unit test |
| I found phase-shifted candles in Binance's archive | YES, WITH QUALIFICATION | "21,602 candles in Dec 2017 and Feb 2018 start 20.8 or 14.8 seconds after the minute, in both BTC and ETH files. I don't know Binance's internal cause." | Re-derived from the interim parquet; ETH audit shows the same 21,602 |
| Snapping them would leak future data | YES | "Flooring a candle that starts at 06:00:20.8 to 06:00 would let a 'closed' feature include up to 20.8 seconds of the future, so I treated those minutes as missing, as my pre-registered rule says." | Deviation log; `build_processed` |
| I cleaned the data | YES, WITH QUALIFICATION | "The audit found 0 duplicates, 0 conflicting rows and 0 impossible candles, so no rows had to be removed. Cleaning meant putting candles on a regular grid and leaving gaps empty — never forward-filled or interpolated." | `BTCUSDT_audit_treatments.csv` |
| I removed outliers | NO | "I kept all 408 minutes with moves or ranges above 3%; they match real events like the COVID crash and FTX." | `outlier_scan`; nothing removed |
| The test period has complete data | YES | "2024–2026 has zero missing minutes; 24,095 of 24,096 test hours are usable — the last one has no exit price yet." | Missing-by-year table; dataset counts |

## B. Research design

| Claim / Statement | Can I say this? | Correct interview wording | Why / Evidence |
|---|---|---|---|
| Research question | YES | "Can information available at time t predict the direction of the next 60-minute BTCUSDT return out of sample, well enough to survive realistic trading costs?" | Pre-registration §1 |
| Target definition | YES | "y = 1 if the log return from the open of t+1 minute to the open of t+61 minutes is positive." | `build_dataset`; **not** close(t) → close(t+60) |
| 60-minute horizon | YES, WITH QUALIFICATION | "Chosen before the data for cost, clean statistics, sample size and its link to hourly settlement. Only 15 and 30 minutes were tested as alternatives." | Pre-registration §2; `config.py` |
| Decision time | YES | "On the hour, UTC, using only candles that had closed by then." | `DECISION_OFFSET_MIN = 0`; `shift(1)` |
| Execution timing | YES, WITH QUALIFICATION | "The backtest assumes a fill at the open of the next minute plus costs. It's a simulation, not live execution." | `EXECUTION_LAG_MIN = 1`; `run_backtest` |
| 16 features | YES | "16 scale-free features: returns, momentum, volatility, volume and order flow, and close location in the last hour's range." | `FEATURE_GROUPS` |
| Features use only closed candles | YES | "Features are computed per minute and shifted one minute, so at t they come from the candle that closed at t." | `build_dataset`; leakage unit tests |
| Leakage audit | YES, WITH QUALIFICATION | "At 24 real decision times I randomised everything at or after t; 0 of 16 features changed, while every target did. That tests feature look-ahead; purging and in-fold scaling handle the other leakage routes. It can't prove there is no leakage of any kind." | `leakage_perturbation_real_data.csv` |
| Pre-registration | YES, WITH QUALIFICATION | "I committed the design and every parameter to git before downloading any data. It's a self-registration in my own repository, so the evidence is the commit history, not a public registry." | `b312d58` 20:26 UTC; first download 21:08 UTC |
| Nothing changed after seeing data | YES, WITH QUALIFICATION | "Every later change is in the deviation log; none changed the model, features, selection or test. One pre-registered check, a 10 bp spot-fee scenario, was run late (7 Oct 2026) by re-pricing the frozen positions: net Sharpe −9.86." | Deviation log; `robustness_spot_fee_check.csv` |

## C. Models

| Claim / Statement | Can I say this? | Correct interview wording | Why / Evidence |
|---|---|---|---|
| Logistic regression | YES | "Standardised features inside a pipeline, L2 regularisation, C in {0.01, 0.1, 1}; validation AUC 0.557 for all three." | `make_lr`; `validation_models.csv` |
| LightGBM | YES | "Two small, regularised configurations; the selected one had 200 trees, 7 leaves and at least 500 rows per leaf." | `LGBM_CONFIGS`; `selection.json` |
| LightGBM was clearly better | NO | "LightGBM passed my rule only just — 7 of 10 folds — and on the test it lost more money than logistic regression." | Selection reason; robustness swap −4.35 vs −5.81 |
| Model-selection rule | YES, WITH QUALIFICATION | "Pre-registered: more than 0.005 AUC gain and lower log-loss than LR in at least 70% of folds. It got +0.0064 and exactly 7 of 10. The document said '≥', the code '>'; it makes no difference here and is logged." | Task 2 |
| Walk-forward validation | YES | "Purged expanding walk-forward, ten six-month folds 2019–2023, refitting each time." | `walk_forward_folds`; `validation_folds.csv` |
| Final test was untouched and run once | YES, WITH QUALIFICATION | "The selection was committed before the test ran, and the log has exactly one test row. Data from 2024 on was seen only in the data audit and descriptive charts, never in anything predictive." | `selection.json` 21:23:10; TEST row 21:24:38; commit `231634d` has no test outputs |
| I never tuned on the test | YES | "The 23 robustness variants were run after the result and reported, not adopted — even the ones that looked a bit better." | Pipeline docstring; log notes |
| Validation ran only once | NO | "The validation stage resets the log by design, so the repository only proves the final validation run. Re-running validation can't touch the test data." | `stage_validation` starts a fresh log |

## D. Results

| Claim / Statement | Can I say this? | Correct interview wording | Why / Evidence |
|---|---|---|---|
| Validation AUC | YES | "Pooled validation AUC 0.565 for LightGBM vs 0.557 for logistic regression." | `validation_models.csv` |
| Final test AUC | YES | "0.545 on the untouched test, with a block-bootstrap 95% CI of 0.538 to 0.552." | `results.json` test classification |
| Statistically significant | YES, WITH QUALIFICATION | "The log-loss gain over the base rate has a Newey–West p of 0.004, and a 200-shuffle permutation test with logistic regression gave p = 0.005, the smallest possible with 200 shuffles. That shows the effect isn't zero, not that it's big." | HAC test; `permutation_test` |
| The permutation test validated my LightGBM | NO | "The permutation test used logistic regression for speed; LightGBM's evidence is its bootstrap CI and HAC test." | `permutation_test` code |
| Accuracy | YES, WITH QUALIFICATION | "53.5% versus 50.5% for always predicting up. Accuracy isn't what pays in trading." | Test classification |
| Calibration | YES | "The model is over-confident: the top decile predicts 60.0% up, 56.4% happens." | `test_calibration.csv` |
| Short-term reversal | YES, WITH QUALIFICATION | "The pattern is reversal: past 5-minute to 24-hour returns, closing near the top of the range and taker buying all correlate negatively with the next hour, and momentum rules have AUC below 0.5. Why it exists — for example liquidity provision — is my hypothesis, not tested." | `h1_h2_univariate_dev.csv`; baselines |
| Feature importance shows reversal | NO | "Gain importance shows which features LightGBM used most — 4-hour and 1-hour returns, close location — not the direction; the direction comes from the correlations." | `test_importance.csv` |
| ETH replication | YES, WITH QUALIFICATION | "The frozen BTC design on ETH gave AUC 0.546 without re-tuning. ETH is highly correlated with BTC on the same venue, so it's a replication, not independent proof, and it ran without funding." | `second_asset_ETHUSDT` |
| Regime decay | YES, WITH QUALIFICATION | "AUC fell from 0.567 in 2024H2 to 0.532 in 2026H2 while volatility-feature PSI reached 3.5. The drift coincides with the decay; I can't prove it caused it. The last block is only three months." | `test_per_block.csv`; `test_drift.csv` |
| The model predicts Bitcoin's price | NO | "It ranks hours by the probability of going up, slightly better than chance." | AUC 0.545 |

## E. Trading and backtesting

| Claim / Statement | Can I say this? | Correct interview wording | Why / Evidence |
|---|---|---|---|
| Transaction costs | YES, WITH QUALIFICATION | "7 bp per side: a 5 bp taker fee from Binance's published schedule, plus 1 bp half-spread and 1 bp slippage that are assumptions — I had no order book." | `config.py` labels: SOURCED / ASSUMPTION |
| 14 bp round trip | YES, WITH QUALIFICATION | "Two sides at 7 bp. 10 bp of it is the fee; 4 bp is assumed." | Same |
| Gross return | YES, WITH QUALIFICATION | "Before costs the summed hourly return was +7% over 2.75 years; compounded and excluding funding it was about +3.8%, gross Sharpe 0.16." | `gross_sum`; stress costs ×0 |
| Net return | YES | "After costs and funding, −92.2% compounded, against +97.6% for buy-and-hold." | Test performance |
| Sharpe | YES, WITH QUALIFICATION | "Net Sharpe −5.84, CI −6.94 to −4.77, from hourly returns annualised with a zero risk-free rate, including funding." | `performance`, `sharpe_ci` |
| Maximum drawdown | YES | "−92.3% from January 2024 to September 2026, never recovered." | `results.json` risk drawdown |
| Turnover | YES | "In the market 9% of hours, about 667 round trips a year." | Test performance |
| Break-even cost | YES, WITH QUALIFICATION | "Gross return divided by turnover is 0.19 bp per side, against 7 bp assumed. I computed it from the saved backtest; it isn't a function in the pipeline." | Re-derived: 0.06998 / 3,668 |
| Why it failed | YES | "Each correct call is worth about 1–2 bp while a round trip costs 14 bp, and the strategy trades hundreds of times a year." | Decile returns −1.3 to +1.9 bp |
| It would be profitable with lower fees or as a market maker | NO | "Break-even is below any realistic cost; maker execution was not tested." | No order-book data |
| Realistic execution | YES, WITH QUALIFICATION | "Next-minute-open fills, costs on every position change, real funding. No order book, partial fills or queue position; spot prices proxy the perpetual." | `run_backtest` docstring |

## F. Risk analysis

| Claim / Statement | Can I say this? | Correct interview wording | Why / Evidence |
|---|---|---|---|
| VaR / ES | YES | "Daily 99% historical VaR 3.53% against 2.18% under normality — fat tails. ES99 4.54%." | `risk_var_es_daily.csv` |
| VaR model validated | YES, WITH QUALIFICATION | "A rolling 250-day historical VaR passes Kupiec at 95% and 99%, p = 0.42 and 0.60. Kupiec checks the number of exceptions only, not clustering." | `risk_kupiec.csv` |
| Monte Carlo | YES, WITH QUALIFICATION | "A stationary block bootstrap of the strategy's own daily returns: none of 5,000 one-year paths was profitable; median −60%. It replays the test period; it's not a market forecast." | `risk_bootstrap_mc.csv` |
| Stress testing | YES | "Only the zero-cost case is positive; doubling costs, extra slippage, every regime and signal flips all fail. A −10% gap while long would be 2.8 times daily 99% VaR." | `stress_*.csv` |
| Robustness tests | YES | "23 post-hoc variants: AUC stayed 0.534–0.548, all 23 lost money after costs." | `robustness_variants.csv` |
| VaR shows the strategy is low-risk | NO | "Its daily VaR is low only because it's flat 91% of the time; it still lost 92%." | Exposure 9.1% |

## G. Credit-risk connection

| Claim / Statement | Can I say this? | Correct interview wording | Why / Evidence |
|---|---|---|---|
| I built a PD model on LendingClub | YES, WITH QUALIFICATION | "A logistic-regression PD model, validated out of time; numbers from my original report — the notebook isn't saved with outputs." | Task 1 |
| 2017 OOT AUC ≈ 0.70, KS ≈ 0.29 | YES, WITH QUALIFICATION | "My original report recorded about 0.70 AUC and 0.29 KS on 2017 loans." | REPORTED |
| LightGBM credit AUC 0.708 | NO (as a result) | Only verbally, qualified: "reported in my original write-up; code not retained." | REPORTED BUT NOT CURRENTLY REPRODUCIBLE |
| PD / LGD / EAD / ECL | YES, WITH QUALIFICATION | "ECL = PD × LGD × EAD with realised mean LGD 93% and funded amount as an exposure proxy — about \$495M in my report." | REPORTED; EAD and LGD simplified |
| Backtesting | YES, WITH QUALIFICATION | "Predicted vs observed defaults by decile and in aggregate: it under-predicted in every decile." | REPORTED |
| PSI and stability | YES, WITH QUALIFICATION | "Score PSI stayed under 0.03; vintage AUC drifted from 0.73 to 0.69. 2018 loans are less mature, so that vintage needs care." | REPORTED |
| Stress testing | YES, WITH QUALIFICATION | "PD-odds ×1.5 and LGD +5 pp, plus correlation sensitivity: VaR99 \$30M → \$108M as ρ goes 0 → 0.5." | REPORTED |
| Production / Basel / IFRS 9 model | NO | "An academic framework with simplified LGD and EAD." | Your own PDF scope note |
| Conceptual connection | YES | "Same loop in both: predict a probability, validate out of time, check calibration and drift, then quantify tail risk under stress. In credit the tail depends on correlation; in trading it depends on costs." | `module_connection.md` |

## H. Gravia relevance (honest)

| Skill | Can I claim it? | Correct interview wording | Evidence |
|---|---|---|---|
| Python / pandas | YES | "A 13-module pandas pipeline over 4.8M rows with 26 tests." | `src/`, `tests/` |
| Polars / SQL / ClickHouse | NO | "Not in this project; I used pandas and parquet." | — |
| Statistical intuition | YES | "Block bootstrap, Newey–West, Holm, permutation, Deflated Sharpe — and knowing significance isn't size." | Stats code and outputs |
| Predictive modelling | YES | "LR and LightGBM with calibration, importance and drift analysis." | Test outputs |
| Overfitting scepticism | YES | "Pre-registration, frozen selection, one test run, 23 kill tests, every trial logged." | Git; experiment log |
| Leakage prevention | YES | "Closed-candle features, next-minute execution, purging, in-fold scaling, perturbation test." | Code and tests |
| Walk-forward validation | YES | "Purged expanding folds; six-monthly refits on test." | `validation.py` |
| Backtesting | YES | "Event-level backtest with explicit timing and turnover-based costs." | `backtest.py` |
| Transaction-cost awareness | YES | "Break-even 0.19 bp vs 7 bp; cost stress scenarios; capacity check." | Test, stress |
| Risk management | YES, WITH QUALIFICATION | "Risk measurement and stress testing, in research. Not live risk management." | `risk.py`, `stress_test.py` |
| Prediction vs economic value | YES | "The core finding: statistically real, economically worthless after costs." | Whole project |
| End-to-end research workflow | YES | "From checksum-verified raw files to a pre-registered verdict." | Whole repo |
| Crypto data | YES, WITH QUALIFICATION | "Binance spot klines and perpetual funding only — no on-chain or wallet data." | `data/raw` |
| Prediction markets / Polymarket | NO | "Only a motivation and a future hypothesis; I never used Polymarket data." | No Polymarket data in the repo |

## Things I MUST NOT claim

| Do not say | Why not | Say instead |
|---|---|---|
| "I built a profitable trading strategy" | Net −92.2%; 23/23 variants negative | "I tested a strategy and showed it isn't profitable after costs." |
| "I found profitable alpha" / "I found alpha" | No positive net result anywhere | "I found a statistically real but economically too small signal." |
| "My model beat the market / beat buy-and-hold" | Buy-and-hold +97.6% vs −92.2% | "It beat chance on prediction, not buy-and-hold on money." |
| "AUC 0.545 is high / strong" | It is small | "Small but statistically reliable." |
| "I analysed Polymarket" / "I tested a Polymarket strategy" | No Polymarket data; the 56.6% table's code is not in the repo | "Polymarket is a hypothesis for a future forward test." |
| "I verified Polymarket's settlement rule or fee" | Taken from web research, not verified in the repo | "According to their published rules, as I understand them…" |
| "I used on-chain or wallet data" | None used | — |
| "I built an HFT / high-frequency strategy" | Hourly decisions; 1-minute data only for features and timing | "An hourly strategy built from minute data." |
| "I measured the spread and slippage" | Both assumed (1 bp each) | "Spread and slippage were assumptions; I stress-tested them." |
| "100% probability of loss means BTC will lose money" | It's a bootstrap of the strategy's returns; BTC rose 98% | "Resampling the strategy's own history, no simulated year was profitable." |
| "The permutation test proves my LightGBM works" | It used logistic regression | "A real-label logistic model beat all 200 shuffles." |
| "LightGBM was clearly better than logistic regression" | Passed at the threshold; lost more money | "Marginally better on AUC, worse on money." |
| "My credit LightGBM got 0.708 AUC" (as a result) | No code, model or output | "Reported in my original write-up; not reproducible from my repository." |
| "The credit notebook reproduces all the numbers" | No data, no outputs | "The numbers come from my original report." |
| "This is a Basel / production credit model" | Academic, simplified LGD/EAD | "An academic risk framework." |
| "The model traded live / paper-traded" | Backtest only | — |
| "I proved there is no leakage" | Tests reduce risk; they can't prove absence | "I tested for look-ahead leakage and found none." |
| "The drift caused the decay" | Correlation only | "The decay coincides with strong feature drift." |
| "The pre-registration was public / independently time-stamped" | Own git repository | "Committed to my repository before the data, as the git history shows." |
| "I'd make it profitable by optimising it" | That would be test-set snooping | "I'd test new pre-registered ideas on new data." |

---

# FINAL OUTPUT

## 1. FINAL PROJECT STATUS

**Is it interview-ready?** The market module is. It is complete, reproducible from public data, internally consistent, and every headline number traces to code and a saved output (26/26 tests pass after the completion pass). The credit module is presentable **only with honest wording**: its numbers come from your original report, and its LightGBM result cannot be reproduced.

**What is strong.**
* Pre-registration committed before the data, a frozen selection committed before the test, and exactly one test run, all visible in git and the experiment log.
* Real data problems found and handled with written rules (ms → µs, phase-shifted candles, gaps), with a leakage test on the real data.
* A clear, well-supported negative result: AUC 0.545 [0.538, 0.552], significant, replicated on ETH — but break-even 0.19 bp vs 7 bp and 23/23 variants negative.
* Honest reporting: failures, assumptions, post-hoc status and a deviation log.

**What is weak (and how to frame it).**
* Spread and slippage are assumptions → "break-even is below the fee alone, so the conclusion doesn't depend on them."
* LightGBM was selected at the threshold and lost more money than LR → "a lesson: I'd add an economic criterion to the rule next time."
* The permutation test used LR, not the selected model → say so.
* One venue, two correlated assets; strong feature drift; a pre-registered spot-fee check that was only run late (it made the loss worse: net Sharpe −9.86).
* Credit module: no data, no saved outputs, LightGBM code missing.

**What MUST be fixed before the interview.** Only your CV (see "Remaining actions"). The repository documentation fixes were made in this audit.

## 2. FINAL CLAIMS I CAN MAKE

1. "I built an end-to-end research pipeline on 4.8 million checksum-verified Binance BTCUSDT one-minute candles, 2017 to 2026."
2. "I committed the whole research design to git before downloading any data, and froze the model selection before running the test once."
3. "My audit found Binance's 2025 switch from millisecond to microsecond timestamps, and 21,602 candles in 2017–18 that sit 20.8 or 14.8 seconds off the minute; I treated those as missing to avoid leaking future data."
4. "Features use only candles closed by the decision time, and trades execute one minute later; randomising the future at 24 real decision times changed none of the 16 features."
5. "I used purged expanding walk-forward validation over ten half-years, never random K-fold."
6. "On the untouched 2024–2026 test, the model's AUC was 0.545 with a 95% CI of 0.538 to 0.552."
7. "The improvement over the base rate is significant with Newey–West errors (p = 0.004), and a real-label logistic model beat all 200 label shuffles."
8. "The signal is a short-term reversal, and it replicated on ETH with the frozen design (AUC 0.546)."
9. "After 7 bp per side and real funding, the strategy lost 92% while buy-and-hold made 98%; net Sharpe −5.84."
10. "The break-even cost is 0.19 basis points per side — about 37 times below my cost assumption."
11. "All 23 robustness variants kept AUC between 0.534 and 0.548, and all 23 lost money after costs."
12. "The model is over-confident: its top decile predicts 60% up and 56% happens."
13. "Historical 99% daily VaR is 3.5% versus 2.2% under a normal model, and the rolling VaR passes the Kupiec test."
14. "Predictive accuracy decayed from 0.567 to 0.532 AUC as the market calmed and volatility features drifted (PSI up to 3.5)."
15. "In my credit project, I validated a logistic PD model out of time and showed that tail risk depends heavily on the assumed default correlation."

## 3. CLAIMS I MUST AVOID

1. "I built a profitable trading strategy."
2. "I found alpha."
3. "My model beat buy-and-hold."
4. "AUC 0.545 is a strong result."
5. "I analysed or traded Polymarket."
6. "I used on-chain or wallet data."
7. "I built a high-frequency strategy."
8. "I measured spread and slippage."
9. "There is a 100% probability that Bitcoin will lose money."
10. "The permutation test validated my LightGBM model."
11. "LightGBM was clearly better than logistic regression."
12. "My credit LightGBM model achieved 0.708 AUC" (as a reproducible result).
13. "The credit notebook reproduces all my reported numbers."
14. "I proved there is no leakage at all."
15. "With lower fees or better tuning, it would be profitable."

## 4. INTERVIEW EXPLANATION

### A. 30-second version

> "I extended my credit-risk project to markets. I took 4.8 million one-minute Bitcoin candles from Binance, wrote down my whole research design before downloading the data, and asked whether the next hour's direction can be predicted well enough to trade. It can be predicted a little — out-of-sample AUC 0.545, statistically significant, and the same on Ethereum — but each correct call is worth about 1–2 basis points while a round trip costs 14. So the strategy lost money in every version I tried. The lesson: predictive skill and economic value are different things."

### B. 2-minute version

> "My first project was credit risk on LendingClub loans. I built a logistic-regression default model, validated it on later years — my original report had an AUC around 0.70 — and found it ranked borrowers well but under-predicted default rates. Then I turned PD into expected loss and simulated portfolio losses, and saw that the 99% VaR more than tripled just by changing the assumed default correlation.
>
> For this role I applied the same discipline to markets. I downloaded 4.8 million one-minute BTCUSDT candles from Binance's official archive, checked every file's checksum, and audited the data. I found real problems — a millisecond-to-microsecond timestamp change in 2025, and two weeks of 2017 candles that start 20 seconds after the minute, which I treated as missing so I wouldn't leak future data.
>
> Before downloading anything, I committed the design: decide on the hour using closed candles, trade one minute later, hold an hour; 16 simple features; logistic regression and a small LightGBM; purged walk-forward validation from 2019 to 2023; and one untouched test from 2024 to 2026.
>
> The test AUC was 0.545, with a confidence interval above 0.5, and it replicated on Ethereum. The pattern is a short-term reversal. But after a realistic 7 basis points per side, the strategy lost 92% while buy-and-hold made 98%. The break-even cost is 0.19 basis points, and all 23 robustness variants lost money.
>
> So my conclusion is: the signal is real, but it isn't tradable at these costs — and I'd rather show that honestly than tune until a backtest looks good."

### C. 5-minute technical version

> **Credit context.** "Module A is a PD model on LendingClub: logistic regression on origination-time features, trained on loans up to 2016 and tested out of time on 2017 and 2018. My original report recorded a 2017 AUC of about 0.70 and KS of 0.29, under-prediction in every risk decile, and stable score PSI. ECL is PD times a realised mean LGD of 93% times funded amount as an exposure proxy, and a one-factor Gaussian Monte Carlo shows 99% VaR rising from about \$30M to \$108M as default correlation goes from 0 to 0.5. I'm careful to say those numbers come from my original report, because the notebook isn't saved with outputs, and I don't present the LightGBM challenger as a reproducible result because I didn't keep its code.
>
> **Data.** Module B uses Binance Vision BTCUSDT one-minute klines from August 2017 to September 2026: 110 files, all SHA-256 verified, 4,789,279 rows on a 4,797,840-minute grid. The audit found no duplicates, conflicts or impossible candles; 0.63% missing minutes, none in 2024 to 2026; a millisecond-to-microsecond switch in 2025, handled by detecting the unit per file; and 21,602 candles phase-shifted by 20.8 or 14.8 seconds. Flooring those would leak future data, so under the pre-registered rule they're missing. I never forward-fill, and rolling features need 90% of their window.
>
> **Design.** The design was committed before the data. Decision on the hour using candles closed by t; entry at the open of t plus one minute; exit an hour later; y is one if that log return is positive. Hourly decisions mean labels don't overlap. There are 16 scale-free features: multi-horizon returns, distance from the 24-hour mean, realised volatility and its ratio, relative range, relative volume, volume change, taker-buy imbalance, and close location. Randomising everything at or after t at 24 real times changed none of them.
>
> **Models and validation.** Base rate, logistic regression with three regularisation levels inside a scaling pipeline, and two small LightGBMs. Purged expanding walk-forward over ten half-years. LightGBM had to beat LR by more than 0.005 mean AUC and win on log-loss in at least 70% of folds. It got 0.0064 and exactly seven of ten — so it was chosen, marginally. The signal goes long when the probability exceeds 0.5 plus 1.5 times the standard deviation of training predictions. On validation every threshold lost money after costs; the rule picked the least bad, and I froze it.
>
> **Test.** One run on 2024 to 2026 with six-monthly refits: AUC 0.545, block-bootstrap CI 0.538 to 0.552; Newey–West p of 0.004 for the log-loss gain; a 200-shuffle permutation test with logistic regression at p = 0.005. Calibration is over-confident, and the mean next-hour return by decile is only −1.3 to +1.9 basis points.
>
> **Economics and risk.** Costs are a 5 bp taker fee plus 1 bp assumed spread and 1 bp assumed slippage, plus real funding. In the market 9% of the time, 667 round trips a year: gross plus 7%, costs 257%, net minus 92% versus plus 98% for buy-and-hold, net Sharpe −5.84. Break-even is 0.19 bp per side. Historical 99% VaR is 3.5% versus 2.2% under normality and passes Kupiec. A block bootstrap gives no profitable year out of 5,000.
>
> **Robustness and verdict.** 23 post-hoc kill tests all keep AUC between 0.534 and 0.548 and all lose money; ETH gives AUC 0.546 and also loses. AUC decays from 0.567 to 0.532 as volatility features drift. By my pre-registered rule the verdict is weak or inconclusive: statistical evidence without positive net performance. I wouldn't deploy it. Next, I'd pre-register lower-turnover or volatility-normalised designs on new data — not tune this test."

---

# VERDICT: PROJECT NEEDS FIXES

The repository and documentation are interview-ready after this audit's documentation fixes. What remains is outside the repository, on your CV.

**Remaining actions (only these):**

**Action 1 — Fix the credit bullet on your CV.** It currently says "Developed Probability of Default (PD) models using Logistic Regression and LightGBM … evaluated AUC, KS, Gini, Brier Score". The LightGBM part cannot be shown from your repository. Suggested replacement:

> "Developed a Probability of Default (PD) model with Logistic Regression on historical LendingClub loans; evaluated AUC, KS, Gini, Brier Score and calibration on out-of-time data."

Keep "LightGBM" in your skills list — you used it in the market module. *Exception:* if you still have the original LightGBM credit code, add it to the repository unchanged instead; then you may keep the wording.

**Action 2 — Add the market module to your CV** using the bullets in `reports/gravia_mapping_and_cv.md`, without changing any number.

**Action 3 — Memorise the three explanations above and the "must not claim" list.**
