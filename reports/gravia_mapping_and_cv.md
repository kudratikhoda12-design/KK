# Gravia mapping, project title and CV bullets

Everything below is based on results that were actually computed (`reports/final_report.md`). Where the project does **not** demonstrate something, this document says so.

## Project title

**Quantitative Market Signal Research & Portfolio Risk Analysis** (final title, 7 Oct 2026)

Why this title: it says *signal research*, not "trading strategy" or "alpha", because the market module found a statistically real signal and showed it is **not** tradable. "Portfolio risk analysis" covers the strategy's VaR/ES, Monte Carlo and stress tests and the credit-portfolio risk work; if asked, say the market-side risk analysis is for a single strategy's P&L, not a multi-asset portfolio.

## Gravia JD → evidence in this project

| Gravia requirement | Evidence in my project | Where |
|---|---|---|
| **Alpha discovery** ("form hypotheses, test them ruthlessly, kill the ones that don't survive") | 5 pre-registered hypotheses. One supported (reversal: test AUC 0.545), one partly supported, one not (regime dependence: 2 of 32 comparisons), one supported only for prediction, and one **killed** (economic value: net Sharpe −5.8). I demonstrated the *process*; I do **not** claim alpha. | final report §8, §23 |
| **Statistical intuition: signal vs overfit** | Block bootstrap, Newey–West (HAC) tests, Holm correction across 16 features, a 200-run label permutation (p = 0.005), Deflated Sharpe, and synthetic negative controls (40 random walks: 5.0% false-positive rate) | §8, §11, §23; `pipeline_validation.md` |
| **Skeptical of own backtests** | Design committed to git before any data; selection frozen before the test; the test run exactly once (the log has 1 TEST row); 23 kill-test variants, all reported | pre-registration, `experiment_log.csv` |
| **ML that trades** ("models that make a decision") | Model → threshold rule → positions → event-level backtest with fees, spread, slippage and actual funding history. The decision it reached was **don't trade**. **Not demonstrated:** live or paper deployment. | §15–18 |
| **Feedback loop: why a model stopped working** | AUC decayed from 0.567 (2024H2) to 0.532 (2026H2). Feature PSI drift explains it: volatility features reached PSI 3.5 as the market calmed. | §18 |
| **Python / pandas** | The whole pipeline: 13 modules, 26 tests, a staged runner, notebooks | `src/`, `tests/` |
| **Polars / SQL / ClickHouse** | **Not demonstrated.** The project uses pandas and parquet. SQL appears only in interview-prep examples. | — |
| **Multi-million-row datasets** | 4.79M-row BTC and 4.79M-row ETH 1-minute tables, plus 1.3M loans in Module A | §4 |
| **Messy data** | Found 21,602 phase-shifted Binance candles (Dec 2017 / Feb 2018, in both BTC and ETH), the ms → µs timestamp switch in 2025, 33 gaps, 18 truncated restart candles and thin-market noise in 2017, each with a written rule | §5 |
| **Feature engineering** | 16 interpretable features, a timing table, and a perturbation leakage test on real data (0 of 16 features changed) | §10–11 |
| **End-to-end ML** | Raw download with checksums → audit → EDA → hypotheses → features → model → validation → signal → backtest → risk → stress → robustness → decision | whole repo |
| **Portfolio / risk analysis** | Module A: PD → ECL → Monte Carlo VaR/ES → stress. Module B: VaR/ES (historical, normal, t), Kupiec test (p = 0.42 / 0.60), block-bootstrap Monte Carlo, drawdowns, cost, volatility and gap stress | §19–21 |
| **Crypto exposure** | Binance BTC/ETH spot and USD-M funding data. **Not demonstrated:** on-chain data, wallet analysis, Polymarket data or trading. Polymarket appears only as a *hypothesis for the next test*. | §26(9) |
| **Model robustness** | Per-half-year AUC, regimes, thresholds, lags, offsets, horizons, feature removal, model swap, second asset | §22 |
| **Data-driven decision making** | Concluded "no tradable edge" from out-of-sample evidence, against the temptation of a gross Sharpe of 2.05 in one post-hoc variant | §22–26 |
| **AI-native, catching a model hallucinating** | I used an AI assistant throughout and verified its work. Real catches included a timestamp check that would have flagged every row; an outlier diagnostic dividing by zero; a figure title stating a "fact" before the data existed; overstated audit claims; a runner bug that would have overwritten BTC results with ETH ones; and notebooks that would have re-run the test. Each was caught by a test, a smoke run or checking claims against tables. | commit history |

## CV entries (final, every number checked against the repository on 7 Oct 2026)

**Quantitative Market Signal Research & Portfolio Risk Analysis** | Python, pandas, scikit-learn, LightGBM, statsmodels

* Built an end-to-end quantitative research pipeline on 4.8M Binance BTCUSDT 1-minute candles (2017–2026), including checksum-verified ingestion, data-quality auditing, timestamp handling, feature engineering and leakage testing.
* Developed Logistic Regression and LightGBM models with purged expanding walk-forward validation to predict the direction of the next 60-minute BTC return; the pre-registered model reached 0.545 out-of-sample AUC (95% CI 0.538–0.552) on the untouched 2024–2026 test set.
* Backtested the signal with explicit execution timing, transaction costs and funding, finding a statistically significant short-term reversal but a 0.19 bp/side break-even cost versus ~7 bp/side assumed cost, making the strategy untradeable at realistic taker costs.
* Performed ETH replication, 23 robustness tests, VaR/ES, Monte Carlo and stress testing, documenting performance decay under feature drift and showing that predictive accuracy did not translate into trading profitability.

**Credit Risk Modeling & Portfolio Stress Testing** (keep as a separate entry; corrected wording)

* Developed a Probability of Default (PD) model with Logistic Regression on historical LendingClub loans; evaluated AUC, KS, Gini, Brier Score and calibration on out-of-time data.
* Keep your existing bullets on backtesting/PSI, ECL and Monte Carlo VaR/ES, and stress testing; they describe code that is in the notebook. Do **not** mention a LightGBM credit model unless you restore its original code.

| Number in the bullets | Repository value | Source |
|---|---|---|
| 4.8M candles | 4,789,279 | `results.json` → `BTCUSDT_audit.schema.n_rows` |
| 2017–2026 | 2017-08-17 → 2026-09-30 | `results.json` → `BTCUSDT_audit.timestamps` |
| 0.545 AUC, CI 0.538–0.552 | 0.54475 [0.53826, 0.55158] | `results.json` → `test.classification` |
| 2024–2026 test | 2024-01-01 → 2026-09-30 | `config.TEST_START`; `test_per_block.csv` |
| 0.19 bp/side | 0.1908 bp (gross sum 0.069985 ÷ turnover 3,668) | `results.json` → `test.performance` |
| ~7 bp/side | 7 bp (5 fee + 1 + 1) | `config.BASE_COST_PER_SIDE` |
| 23 robustness tests | 23 rows, all net Sharpe < 0 | `robustness_variants.csv` |

**Skills shown:** time-series (walk-forward) validation, hypothesis testing (block bootstrap, HAC, Holm, permutation), feature engineering, leakage control, backtesting with transaction costs, VaR/ES, Monte Carlo, stress testing, data-quality auditing, Python (pandas, NumPy, scikit-learn, LightGBM, statsmodels), pytest, git.

## One-line interview explanation

> "I extended my credit-risk project to 4.8 million Binance minutes, found a real short-term reversal in Bitcoin (out-of-sample AUC 0.545), and then showed it can't be traded at realistic costs: its edge is 0.19 basis points per trade side against about 7 basis points of cost."

## Answer to Gravia's prompt: "the most interesting thing you've found in a dataset"

> "In Binance's official BTCUSDT archive, about two weeks of December 2017 candles start 20.799 seconds after every minute instead of on the minute, and the same stretch appears in ETHUSDT. I don't know Binance's internal cause, and I don't guess. Snapping those candles onto the grid the obvious way would have leaked 20 seconds of future data into my features. The bigger finding: Bitcoin's hourly direction *is* predictable out of sample, but the conditional returns are about 1–2 bp, so the signal dies against costs. In a binary market that pays on direction rather than size, the same signal might matter. That's the experiment I'd want to run next with real prediction-market prices."
