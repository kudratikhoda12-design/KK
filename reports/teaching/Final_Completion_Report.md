# FINAL COMPLETION REPORT

**Quantitative Market Signal Research & Portfolio Risk Analysis**

Final completion pass · Kudrati Khoda · 7 October 2026

---

## Rules followed in this pass

The core market experiment is frozen and was **not** changed: same research question, target, 60-minute horizon, 16 features, periods, walk-forward design, selection rule, models and results. No model was retrained or tuned, the main experiment was not re-run, and no headline number changed. `reports/results.json`, `reports/selection.json` and `reports/tables/experiment_log.csv` are byte-identical to their state before this pass (MD5 checked). The only new computation is the pre-registered spot-fee check (Part 2), which re-prices the already-saved frozen positions.

---

# PART 1 — Completion Checklist

## 1. Completed

| Item | Evidence |
|---|---|
| Data acquisition and checksum verification (BTCUSDT 110, ETHUSDT 110, funding 81 files) | `manifest.csv` files; all re-hashed in this pass: 0 mismatches |
| Data-health audit, regular 1-minute grid, written treatment rules | `BTCUSDT_audit_*.csv`, `ETHUSDT_audit_*.csv` |
| EDA (distributions, autocorrelation, volume, intraday, regimes) | `eda_*.csv`, 4 figures |
| Hypotheses H1–H3, including the pre-registered sign-by-block and regime-CI criteria | `h1_*`, `h3_*` tables |
| 16 features, timing table, perturbation leakage test on real BTC and ETH data | `feature_timing_table.csv`, `leakage_perturbation_real_data.csv` |
| Walk-forward validation: base rate, 3 LR, 2 LightGBM; signal grid; baselines; frozen selection | `validation_*.csv`, `selection.json` |
| Single untouched test run; per-block, calibration, drift (PSI), importance, capacity | `results.json` → `test`; `test_*.csv` |
| Risk: VaR/ES (historical, normal, Student-t), Kupiec, block-bootstrap Monte Carlo, drawdown | `risk_*.csv` |
| Stress: costs ×0/×1/×2/×3, +2/+5 bp slippage, volatility ×2, signal flips, regimes, worst days, gaps | `stress_*.csv` |
| Robustness: 23 variants, 200-run permutation test, PSR, Deflated Sharpe | `robustness_variants.csv`, `results.json` |
| ETH replication with the frozen design | `second_asset_ETHUSDT_*` |
| Synthetic negative/positive pipeline controls; unit tests | `pipeline_controls.csv`; 26 tests |
| **Pre-registered spot-fee check** (completed in this pass) | `robustness_spot_fee_check.csv` (Part 2) |
| Reports: final report, walkthrough, readiness audit, this report | `reports/`, `reports/teaching/` |

## 2. Completed but only documented

| Item | Status | Necessary to do more? |
|---|---|---|
| "≥ 0.005" vs "> 0.005" wording of the selection rule | Deviation-log entry (7 Oct); no effect (gain +0.00638) | No |
| Credit-module evidence status | `credit_module/README.md` section | No |
| Break-even cost 0.19 bp | Derived from saved outputs (gross sum ÷ turnover); formula documented | No — it is arithmetic on saved values |
| Exploratory Polymarket hit-rate table (`exploratory_polymarket_event_deciles.csv`) | Table exists; **the code that produced it is not committed**; labelled exploratory and not reproducible | No. Regenerating it would require recomputing test-period predictions, i.e. re-running the frozen model on the test. It is not part of any conclusion, so it stays as a labelled exploratory artefact. |
| Earlier aborted validation runs | The validation stage resets the log by design; only the final validation run is recorded | No — validation never touches test data |

## 3. Pre-registered but not executed

| Item | Status now | Necessary? |
|---|---|---|
| Spot-fee robustness check (10 bp fee, long/flat), pre-registration §8 | **Executed in this pass** (Part 2) | It was the only real gap in the pre-registered checklist |
| "Zero-return" baseline (never trade), pre-registration §6 | Not reported as a row | No: its return is exactly 0 and its Sharpe is undefined; it adds no information |

Everything else in the pre-registered "try to kill the signal" checklist (§11) and statistics table (§9) was run.

## 4. Recommended future work (not necessary for this project)

* Christoffersen independence test for VaR exceptions (Kupiec checks the count only).
* Volatility-normalised features to reduce drift.
* Lower-turnover designs (daily horizon, extreme deciles only).
* Maker-order execution — needs historical order-book data.
* An economic criterion in the model-selection rule — for a *new* pre-registered study.
* A prediction-market forward test with real historical prices, fills and settlement data.
* More assets (SOL, XRP were mentioned in the dataset-selection plan but were never pre-registered).

## 5. Cannot be completed because the required data or code is unavailable

| Item | Why |
|---|---|
| Re-running the credit logistic-regression notebook | LendingClub file not present (Kaggle, account required). You can do it locally; not necessary for the interview if described honestly. |
| Reproducing the credit LightGBM challenger | No code exists. Must not be recreated or estimated. |
| Measuring spread and slippage | Binance Vision klines contain no historical order book. Break-even (0.19 bp) is below the fee alone, so this does not affect the conclusion. |
| Historical Polymarket analysis | No legally accessible, reproducible historical dataset in this setting. Positioned as future work. |

---

# PART 2 — The Pre-registered Spot-Fee Check

**What the pre-registration intended (§8).** The cost table lists: "Spot alternative | 0.10% fee per side, long/flat only | Sourced, reported as a robustness check". The instrument paragraph explains why: the main design assumes the perpetual future (which allows shorting); trading spot instead means a higher taker fee and no shorting. So the check asks: *does the result change if the strategy is traded on spot at the spot fee?*

**Can it be run without touching the frozen design? Yes.** The frozen strategy is already long/flat, so the positions do not change at all — only their price. `scripts/run_spot_fee_check.py`:

1. loads the frozen positions saved in `data/processed/BTCUSDT_test_backtest.parquet` (no refit, no new predictions, no re-selection);
2. **consistency check:** re-prices them at the original 7 bp + actual funding and requires the recorded test net Sharpe to be reproduced — it was, exactly (−5.840966);
3. re-prices them at **10 bp fee + 1 bp half-spread + 1 bp slippage = 12 bp per side, with no funding** (a spot position pays no perpetual funding);
4. writes only new files: `reports/tables/robustness_spot_fee_check.csv` and `reports/spot_fee_check.json`.

**Result (VERIFIED):**

| Scenario | Cost/side | Funding | Net Sharpe (95% CI) | Cumulative | Max drawdown | Daily VaR99 |
|---|---|---|---|---|---|---|
| Frozen test (original; re-priced check) | 7 bp | actual | −5.84 (−6.94, −4.77) | −92.2% | −92.3% | 3.53% |
| **Pre-registered spot alternative** | 12 bp | none | **−9.86 (−10.97, −8.79)** | **−98.7%** | −98.7% | 3.77% |

Gross Sharpe (0.16), exposure (9.1%) and turnover (667 round trips a year) are identical, because the positions are identical. The spot result equals the existing "+5 bp extra slippage" stress row to every digit, as it must: both are 12 bp per side without funding.

**Labelling.** Recorded as a "pre-registered check executed later" in the deviation log and in a dated addendum to `final_report.md`. It does **not** alter the model selection, the test result, the robustness results or the verdict. It was kept out of `results.json` and the experiment log so that the original record stays exactly as it was.

**Does it change the final conclusion? No.** It only makes the loss larger. The verdict stays **WEAK / INCONCLUSIVE**.

---

# PART 3 — Verification of the Final Market Pipeline

**TESTS BEFORE:** 25 passed, 0 failed (`python -m pytest -v`, 18 s).
**TESTS AFTER:** 26 passed, 0 failed (one new test, below).

| Component | How it was confirmed in this pass | Result |
|---|---|---|
| Data acquisition | Live download of `BTCUSDT-1m-2026-09-30.zip` from data.binance.vision into a temporary folder with `download_verified` | `ok` |
| Checksums | Same live file verified against Binance's `.CHECKSUM`; all 110 BTC + 110 ETH local zips re-hashed against Binance's published SHA-256; funding manifest 81/81 | 0 mismatches |
| Timestamp handling | Tests `test_time_unit_detection`, `test_microsecond_file_converts_to_correct_date`, `test_close_time_check_is_resolution_safe`; the live 2026 file was detected as µs | Pass |
| Phase-shifted candle treatment | **New test** `test_phase_shifted_candles_become_missing_not_snapped` | Pass |
| Regular-grid construction | `test_processed_grid_applies_rules_without_filling` | Pass |
| Feature construction | `test_feature_alignment_uses_only_closed_candles`, `test_rolling_windows_are_backward_looking`, `test_missing_minutes_are_not_filled` | Pass |
| Leakage tests | `test_perturbing_the_future_never_changes_features`; real-data table (0/16) | Pass |
| Target construction | `test_target_uses_execution_lag` | Pass |
| Walk-forward validation | `test_folds_are_chronological_and_purged` | Pass |
| Model selection, backtest, risk, stress, robustness | Full synthetic end-to-end smoke run of every stage (`scripts/smoke_test_synthetic.py`, 4.8M synthetic minutes, temporary folder only) | Pass: all 8 stages ran on 4.8M synthetic minutes in 431 s. On the pure random walk the selection rule correctly kept logistic regression (LightGBM gain −0.0037, 0/10 folds), test AUC 0.503 (no signal, as expected), net Sharpe −3.72, 23 robustness rows, 0/16 leakage changes |
| Transaction-cost calculation | 4 backtest tests; spot-fee check re-priced the saved positions and reproduced the recorded net Sharpe exactly | Pass |
| Risk analysis | 5 risk tests (drawdown, VaR/ES, Kupiec, DSR, PSI) | Pass |
| ETH replication | `stage_second_asset` run on a synthetic second asset in the same temporary folder | Pass: ran in 337 s and produced all three second-asset tables; synthetic AUC 0.498 (no signal, as expected) |
| Pipeline controls | Random walks show no edge; injected momentum is detected | Pass |

**FILES CHANGED in this pass, and why:**

| File | Change | Why necessary |
|---|---|---|
| `scripts/run_spot_fee_check.py` | New | Execute the one pre-registered check that had not been run, without touching the frozen pipeline |
| `reports/tables/robustness_spot_fee_check.csv`, `reports/spot_fee_check.json` | New outputs | Store that result separately from the original record |
| `tests/test_data_quality.py` | One new test | The phase-shifted-candle rule is a key claim but had no direct test |
| `reports/research_design_preregistration.md` | One deviation-log row | Record the late execution of the §8 check |
| `reports/final_report.md` | Dated addendum at the end | Report the §8 result without editing sections 1–27 |
| `README.md` | Rewritten for recruiters | Requested Gravia README; numbers unchanged |
| `data/README.md` | Measured sizes; funding and ETH commands | Reproducibility: the ETH and funding steps were not documented |
| `reports/gravia_mapping_and_cv.md` | Final title and checked CV bullets; two over-strong phrases softened | CV accuracy |
| `reports/teaching/*` | Walkthrough and audit notes updated for the spot check; this report added | Consistency |

No source module in `src/`, no configuration value and no saved result was modified. Nothing failed, so no engineering fix was needed.

---

# PART 4 — Reproducibility Check

| Requirement | Status | Where |
|---|---|---|
| Public data source documented | Yes | `data/README.md` (URL pattern, columns, units, licence) |
| Checksums documented and enforced | Yes | `download_verified`; `manifest.csv` per dataset |
| Random seeds fixed | Yes | `RANDOM_SEED = 42` for LightGBM, bootstraps, permutations, leakage sampling; LightGBM `deterministic=True` |
| Python dependencies | Yes | `requirements.txt` (minimum versions + tested versions; Python 3.11) |
| Pipeline commands | Yes (completed in this pass) | `README.md` → Reproduce; `data/README.md` |
| Expected outputs | Yes (added in this pass) | `README.md` → Reproduce |
| Important intermediate tables saved | Yes | 65 CSV tables, `results.json`, `selection.json`, test backtest parquet (local) |
| Final results trace back to code | Yes | Teaching walkthrough traces every stage to file and function |
| Raw/processed data in git | No, by design | Licence (CC BY-NC-SA) and size; re-created by script and verified by checksum |

**Can a new person reproduce it?** Yes, for the market module, from public Binance data with the documented commands. Two honest caveats:

1. Re-running the validation and test stages on the real data creates a **new** record (the validation stage starts a new experiment log); it reproduces the method, not the historical "one test run" evidence, which lives in git.
2. Binance could revise archive files in future; the manifest's hashes would detect this.

The credit module is reproducible only by someone with the LendingClub file, and its LightGBM part is not reproducible at all.

---

# PART 5 — Final Project Structure

```
KK/
├── README.md                     recruiter-facing summary (2-3 minutes)
├── credit_module/                1  original credit-risk module (report + evidence status)
├── notebooks/                    01 credit (original) | 02-10 market stages (saved outputs)
├── data/README.md                how to obtain and verify the data (data itself not committed)
├── src/                          2  market-signal research code (13 modules)
├── scripts/                      pipeline runner, spot-fee check, smoke test, downloads
├── tests/                        26 tests incl. synthetic pipeline controls
├── figures/                      all charts used by the reports
└── reports/
    ├── phase1_dataset_selection.md, phase2_data_acquisition.md     3  dataset-selection research
    ├── research_design_preregistration.md                         4  pre-registration  +  5  deviation log (§12)
    ├── selection.json, results.json, spot_fee_check.json
    ├── final_report.md                                            full report + addendum
    ├── tables/                    6 validation_*  7 test_*  8 test_benchmarks / backtest rows
    │                              9 risk_* stress_*  10 robustness_*  11 second_asset_ETHUSDT_* / ETHUSDT_*
    └── teaching/                  12 final interview documentation (Markdown, Word, PDF)
```

Figures stay in `figures/` rather than `reports/figures/`: moving them would break links in the final report, the notebooks, the walkthrough and the plotting code, for no scientific benefit. Nothing was deleted.

---

# PART 6 & 7 — Gravia README and Polymarket Positioning

The main `README.md` was rewritten for a recruiter: title, motivation, dataset and audit, research question, methodology, results, trading economics, robustness and risk, conclusion, future extension, the credit module's evidence status, a repository map and reproduction commands. Every number is an existing verified value.

**Section included in the README — "Why Binance instead of direct Polymarket data?"**

> Polymarket-style hourly "Bitcoin Up or Down" markets were part of the motivation: according to their published market rules (found in my Phase 1 research, not verified in this repository), they settle on the Binance BTC/USDT 1-hour candle. Binance provides long, reproducible, checksum-verified history (2017–2026), while fine-grained Polymarket price history is kept only briefly and the hourly markets are recent; Polymarket access is also blocked in India. **Direct Polymarket order-book, fill or price analysis was not part of this experiment.** So this is quantitative research on the underlying market, not a Polymarket trading strategy.

**Future extension (not done):** use historical prediction-market prices, fills / order book and settlement data to test whether the same signal creates a tradable prediction-market edge — with a new pre-registration.

---

# PART 8 — Final CV Entry (every number checked)

**Quantitative Market Signal Research & Portfolio Risk Analysis** | Python, pandas, scikit-learn, LightGBM, statsmodels

* Built an end-to-end quantitative research pipeline on 4.8M Binance BTCUSDT 1-minute candles (2017–2026), including checksum-verified ingestion, data-quality auditing, timestamp handling, feature engineering and leakage testing.
* Developed Logistic Regression and LightGBM models with purged expanding walk-forward validation to predict the direction of the next 60-minute BTC return; the pre-registered model reached 0.545 out-of-sample AUC (95% CI 0.538–0.552) on the untouched 2024–2026 test set.
* Backtested the signal with explicit execution timing, transaction costs and funding, finding a statistically significant short-term reversal but a 0.19 bp/side break-even cost versus ~7 bp/side assumed cost, making the strategy untradeable at realistic taker costs.
* Performed ETH replication, 23 robustness tests, VaR/ES, Monte Carlo and stress testing, documenting performance decay under feature drift and showing that predictive accuracy did not translate into trading profitability.

**Changes from your draft and why:**

| Your draft | Final | Reason |
|---|---|---|
| "achieved 0.545 out-of-sample AUC" | "the pre-registered model reached 0.545 … (95% CI 0.538–0.552)" | "Achieved" sounds like a strong score; the CI shows it is reliably above 0.5 |
| "economically untradeable" | "untradeable at realistic taker costs" | The conclusion depends on the cost level; this is the precise claim |
| "demonstrating model drift" | "documenting performance decay under feature drift" | Drift coincides with the decay; causation was not proven |
| (bullet 1) | added "checksum-verified ingestion" | Verified, and a strong data-quality signal |

| Number | Repository value | Source |
|---|---|---|
| 4.8M candles | 4,789,279 | `results.json` audit |
| 2017–2026 | 2017-08-17 → 2026-09-30 | `results.json` audit |
| 0.545 [0.538, 0.552] | 0.54475 [0.53826, 0.55158] | `results.json` → test |
| 2024–2026 test | 2024-01-01 → 2026-09-30 | `config.py`, `test_per_block.csv` |
| 0.19 bp/side | 0.1908 bp | gross 0.069985 ÷ turnover 3,668 |
| ~7 bp/side | 7 bp | `config.BASE_COST_PER_SIDE` |
| 23 robustness tests | 23, all net < 0 | `robustness_variants.csv` |

**Title note:** "Portfolio Risk Analysis" is acceptable. If asked, say the market-side risk work (VaR/ES, Monte Carlo, stress) is for a single strategy's P&L, and the portfolio-loss modelling is in the credit project.

**Credit entry (separate, corrected):** "Developed a Probability of Default (PD) model with Logistic Regression on historical LendingClub loans; evaluated AUC, KS, Gini, Brier Score and calibration on out-of-time data." Keep your other credit bullets; drop "and LightGBM" unless you restore that code.

---

# PART 9 — Final Interview Claim Audit (one page)

| Topic | CAN SAY | CAN SAY WITH QUALIFICATION | MUST NOT SAY |
|---|---|---|---|
| Binance | Official Binance Vision archive; 110 files checksum-verified | Fee schedule taken from Binance's published 2026 schedule (cited, not re-checked) | "I had Binance order-book data" |
| Polymarket | It motivated the hourly question | Settlement-rule link per their published rules (my research) | "I analysed / traded Polymarket" |
| BTCUSDT | 4.79M 1-min candles, 2017–2026 | Prices in USDT, not USD | "I traded BTC" |
| LightGBM | Used; selected by a pre-registered rule | Passed exactly at 7/10 folds; lost more than LR on test | "Clearly better than LR"; credit LightGBM 0.708 as a result |
| Logistic regression | Scaled, L2, AUC 0.557 val / 0.540 test | — | — |
| AUC 0.545 | Untouched test, CI 0.538–0.552 | Small, but reliably above chance | "A strong / high AUC" |
| Statistical significance | HAC p = 0.004 | Permutation p = 0.005 used LR; significance ≠ size | "Significant, so profitable" |
| Short-term reversal | The measured pattern | Its cause is a hypothesis | "I explained why it exists" |
| Transaction costs | 7 bp/side, 14 bp round trip, plus real funding | 5 bp sourced, 2 bp assumed | — |
| Spread / slippage | Stress-tested (+2, +5 bp) | **Assumed**, not measured | "I measured slippage" |
| Break-even cost | 0.19 bp/side vs 7 bp | Computed from saved outputs | — |
| Backtest | Event-level, next-minute fills, costs on turnover | No order book, partial fills or latency beyond 1 min | "Realistic live execution" |
| VaR | Hist. 99% 3.53% vs normal 2.18%; Kupiec passes | Validates the risk measure, not the strategy | "VaR shows it is low-risk" |
| Monte Carlo | 5,000 bootstrap paths, none profitable | Replays the test period only | "100% chance BTC loses money" |
| ETH replication | AUC 0.546 with the frozen design | Correlated asset, same venue, no funding | "Independent proof" |
| Robustness | 23 variants, all net negative | Post hoc, never used for selection | "It is robustly profitable" |
| HFT | — | Hourly decisions from minute data | "HFT / high-frequency strategy" |
| Alpha | "A statistically real signal" | Not economically exploitable | "I found alpha" |
| Profitable strategy | — | Profitable only before costs (gross Sharpe 0.16) | "Profitable strategy" |
| On-chain data | — | — | "I used on-chain data" |
| Wallet data | — | — | "I analysed wallets" |
| Live trading | — | Backtest and research only | "Traded live / paper-traded" |

---

# PART 10 — Final Spoken Answers

### 30-second answer

> "I extended my credit-risk project into market research. I took about 4.8 million one-minute Bitcoin candles from Binance, wrote down the full research plan before downloading the data, and asked whether the next hour's direction can be predicted well enough to trade. It can, a little: out-of-sample AUC 0.545, statistically significant, and the same on Ethereum. But each correct call is worth about 1 to 2 basis points, and a round trip costs about 14. So it loses money in every version I tested. The main lesson is that predicting better than chance isn't the same as making money."

### 2-minute answer

> "My first project was credit risk on LendingClub loans. I built a logistic-regression default model, tested it on later years — my original report had an AUC around 0.70 — and found it ranked borrowers well but under-predicted defaults. Then I turned that into expected loss and simulated portfolio losses, and saw the 99% VaR more than triple just by changing the assumed default correlation.
>
> For this role I took the same approach to markets. I downloaded about 4.8 million one-minute BTCUSDT candles from Binance's official archive and checked every file's checksum. The data audit found real issues: Binance switched from milliseconds to microseconds in 2025, and in 2017 there were two weeks of candles starting about 20 seconds after the minute. I treated those as missing, because snapping them to the minute would have leaked future data.
>
> Before touching the data, I fixed the whole design in git: decide on the hour using only closed candles, trade one minute later, hold for an hour; 16 simple features; logistic regression and a small LightGBM; walk-forward validation from 2019 to 2023; and one untouched test from 2024 to 2026.
>
> On the test, the AUC was 0.545, with a confidence interval above 0.5, and it held on Ethereum. The pattern is a short-term reversal. But with about 7 basis points of cost per side, the strategy lost about 92% while buy-and-hold made 98%. The break-even cost is just 0.19 basis points, and all 23 robustness checks lost money.
>
> So the signal is real, but it isn't tradable at realistic costs. I'd rather show that honestly than tune things until the backtest looks good."

### 5-minute technical answer

> **Credit background.** "Module A is a PD model on LendingClub: logistic regression on information known at origination, trained on loans up to 2016 and tested on 2017 and 2018. My original report had a 2017 AUC around 0.70 and KS around 0.29, under-prediction in every risk decile, and a stable score PSI. Expected loss is PD times a realised LGD of about 93% times funded amount as an exposure proxy, and a one-factor Monte Carlo shows the 99% VaR going from about 30 to 108 million dollars as default correlation goes from 0 to 0.5. I'm careful to say those numbers come from my original report, because the notebook isn't saved with outputs.
>
> **Data.** Module B uses Binance BTCUSDT one-minute klines from August 2017 to September 2026: 110 files, every one SHA-256 verified, 4.79 million rows. No duplicates, conflicts or impossible candles; 0.63% missing minutes, none in the 2024–2026 test period. Two real issues: the 2025 switch from milliseconds to microseconds, which I handle by detecting the unit per file, and 21,602 candles in 2017–18 shifted 20.8 or 14.8 seconds off the minute. Flooring them would leak future data, so under my pre-registered rule they're missing. I never forward-fill.
>
> **Design.** The design was committed before the data. Decide on the hour using candles closed by then, enter at the next minute's open, exit an hour later; the label is whether that log return is positive. Hourly decisions mean labels never overlap. Sixteen scale-free features cover recent returns, momentum, volatility, volume and taker-buy imbalance, and where the price closed in the last hour's range. When I randomised everything from the decision time onward, none of the 16 features changed.
>
> **Models.** Base rate, logistic regression with three regularisation levels, and two small LightGBMs, all with purged expanding walk-forward over ten half-years. LightGBM had to beat logistic regression by more than 0.005 AUC and win on log-loss in at least 70% of folds. It got 0.0064 and exactly 7 of 10, so it was chosen — but only just. The trading rule goes long when the probability is above 0.5 plus 1.5 standard deviations of the training predictions. In validation every threshold already lost money after costs; the rule picked the least bad one, and I froze it.
>
> **Test.** One run on 2024 to 2026: AUC 0.545, confidence interval 0.538 to 0.552; Newey–West p of 0.004 for the log-loss gain; and a real-label logistic model beat all 200 label shuffles. The model is over-confident, and the average next-hour return by decile is only about −1 to +2 basis points.
>
> **Economics and risk.** Costs are a 5 bp taker fee plus 1 bp assumed spread and 1 bp assumed slippage, plus real funding. The strategy was in the market 9% of the time with about 667 round trips a year. Gross it made 7%; costs were 257%; net it lost 92% against plus 98% for buy-and-hold, with a net Sharpe of −5.84. Break-even is 0.19 bp per side. The pre-registered spot-fee version, at 12 bp, is worse: Sharpe −9.86. Daily 99% VaR is 3.5% versus 2.2% under a normal model, and it passes the Kupiec test. A block bootstrap gives no profitable year out of 5,000.
>
> **Robustness and verdict.** 23 kill tests keep the AUC between 0.534 and 0.548, and all 23 lose money. ETH gives 0.546 and also loses. The AUC decays from 0.567 to 0.532 as volatility features drift. My pre-registered verdict is weak or inconclusive: real statistical evidence, no positive net performance. I wouldn't deploy it. Next I'd pre-register lower-turnover or volatility-normalised versions on new data — not tune this test."

---

# PART 11 — Final Project Verdict

| | Status | Basis |
|---|---|---|
| **A. Core experiment** | **COMPLETE** | Every pre-registered stage and check has been run, including the late spot-fee check; nothing in the frozen design changed |
| **B. Reproducibility** | **COMPLETE** | Public data with checksums, seeded code, dependencies, commands and expected outputs documented; 26 tests pass; full synthetic end-to-end run passes. (Credit module: needs the LendingClub file; its LightGBM part is not reproducible — documented.) |
| **C. Documentation** | **COMPLETE** | README, final report + addendum, pre-registration + deviation log, walkthrough, audit, this report, CV bullets |
| **D. Interview readiness** | **READY** | Claims table, must-not-say list, and 30-second / 2-minute / 5-minute answers |
| **E. CV** | **NEEDS FIXES** | The CV you uploaded still says the credit PD models used "Logistic Regression and LightGBM", and it does not yet contain this project |

**F. Remaining actions (only these):**

1. **In your CV file**, change the credit bullet to: "Developed a Probability of Default (PD) model with Logistic Regression on historical LendingClub loans; evaluated AUC, KS, Gini, Brier Score and calibration on out-of-time data."
2. **In your CV file**, add the entry "Quantitative Market Signal Research & Portfolio Risk Analysis" with the four bullets in Part 8, exactly as written.

## PROJECT STILL NEEDS FIXES

The project itself — research, code, reproducibility and documentation — is complete. The two remaining fixes are both in your CV file (above). Once they are made: **PROJECT COMPLETE — READY FOR GRAVIA INTERVIEW.**
