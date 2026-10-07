# Ethereum Order-Flow Alpha
### Short-horizon return prediction and a cost-aware trading strategy on the Binance ETHUSDT perpetual

> **Headline result.** One-minute ETH price paths contain small, stable, out-of-sample predictability: holdout AUC
> **0.5436** (95 % CI 0.5394–0.5480), above 0.5 in 12 of 12 holdout months. Taker order flow predicts a slight *reversal*.
> The predictable edge is about **0.5 bps per side** per trade, while realistic taker costs are **5.5 bps per side**. The strategy
> earns a gross Sharpe of 2.03 but loses **99.6 %** net of costs. **The signal is statistically significant and economically untradable.**
>
> Every number below comes from a pipeline output. The full analysis is in [`docs/research_report.md`](docs/research_report.md).

---

## Overview
This is an end-to-end quantitative research project. It goes from raw exchange data to a frozen, out-of-sample-tested trading strategy:

1. raw Binance trades and order-book depth;
2. data engineering and quality control;
3. 47 causal microstructure features;
4. statistical tests built for overlapping, heavy-tailed data;
5. leakage-controlled walk-forward ML;
6. a frozen methodology evaluated once on an untouched year;
7. a cost-aware backtest, robustness checks and a replication on SOL.

## Research question
Does order-flow and market-microstructure information carry statistically significant information about 5- to 60-minute
ETH returns, and does it survive transaction costs, slippage and funding as tradable alpha?

## Data
| | |
|---|---|
| Source | Binance public archive, https://data.binance.vision (USDⓈ-M futures, ETHUSDT perpetual), SHA-256 verified |
| Period | 2023-03-01 → 2026-09-30 UTC. Development to 2025-09-30; **holdout 2025-10-01 → 2026-09-30** |
| Trades | **1,877,097,488** aggregated trades with the taker side given explicitly (`is_buyer_maker`) |
| Bars | **1,886,400** one-minute bars (every minute; 53 have no trade) |
| Order book | **3,710,848** snapshots of *cumulative resting depth within ±1…5 % of price* (percentage bands, **not** a level-by-level L2 book; no best bid/ask) |
| Other | Funding (3,930 settlements), open interest (5 min), quoted spread measured from `bookTicker` sample days |

Data-quality report: [`docs/data_quality_report.md`](docs/data_quality_report.md). One notable finding is that the archive's order-book
feed is **frozen for a month** (2025-04-16 → 2025-05-19, 94,730 identical snapshots). It is detected and treated as missing.

## Methodology
- **Timeline.** Features use data stamped before the close of bar t. The order executes at the **VWAP of bar t+1**. Each signal holds a
  1/15 position for 15 bars. The target is (P_{t+15} − P_t) / P_t.
- **Feature engineering.** Five groups: price (16 features), volume (6), taker trade flow (13: order-flow imbalance over 1/5/15/60 min, trade
  imbalance, signed flow, large-trade share), book bands (7: depth imbalance at 1/2/3/5 %), and funding/open interest (5).
  All windows are trailing. Undefined ratios and stale books become missing; nothing is interpolated.
- **Leakage audit** ([`docs/leakage_audit.md`](docs/leakage_audit.md)): 8 automated checks, all passing. They include truncation invariance over 34.8 M feature values, as-of join checks, and purge/embargo checks.
- **Statistical tests.** Newey–West HAC regressions, non-overlapping re-checks, controls, decile spreads with a day-block bootstrap,
  an event study, and Holm/BH multiple-testing adjustment.
- **ML models.** Logistic regression (L2), random forest, LightGBM (8-point grid), plus rule baselines (prior, momentum, order-flow rule).
  Feature ablation runs on sets A = price → B = +volume → C = +trade flow → D = +book → E = +funding/OI.
- **Validation.** Expanding walk-forward with 7 blocks of 3 months, purge plus a 1-day embargo, and all preprocessing fitted inside each training fold.
  Model choice (lowest validation log loss) and signal threshold (best out-of-fold base-cost Sharpe) follow pre-declared rules.
- **Frozen methodology.** [`docs/final_methodology.md`](docs/final_methodology.md) was committed before the holdout was opened. Every holdout access
  is logged with that file's SHA-256 ([`reports/holdout_access_log.jsonl`](reports/holdout_access_log.jsonl)).
- **Backtesting.** Fees + slippage + half spread at low / base / high = 2.52 / **5.52** / 7.02 bps per side (Binance VIP0 fees). Realised
  funding is charged at settlement. The break-even cost is reported for every strategy.

## Results

**Prediction (frozen random forest, all features, 15-minute horizon):**

| | Development walk-forward (19 months) | Untouched holdout (12 months) |
|---|---:|---:|
| AUC | 0.5381 ± 0.0087 (7/7 folds > 0.5) | **0.5436** [0.5394, 0.5480] |
| Log loss vs prior | 0.6911 vs ≈ 0.6931 | 0.6902 vs 0.6931 |
| Calibration error (ECE) | n/a | 0.0028 |

**Trading (holdout, frozen rule):**

| Strategy | Gross return | Gross Sharpe | Net return (base) | Net Sharpe | Max DD (net) | Break-even (bps/side) |
|---|---:|---:|---:|---:|---:|---:|
| **ML strategy** | +67.5 % | 2.03 | **−99.6 %** | −20.37 | −99.6 % | **0.49** |
| Momentum | −78.1 % | −3.18 | −100.0 % | −44.11 | −100.0 % | (gross loss) |
| Order-flow rule | −51.9 % | −2.81 | −100.0 % | −41.43 | −100.0 % | (gross loss) |
| Buy & hold | −36.9 % | −0.48 | −36.9 % | −0.48 | −68.7 % | – |

The full model × feature-set table is in [`docs/results_table.md`](docs/results_table.md).

## Robustness
Details: [`docs/robustness_report.md`](docs/robustness_report.md).

- **Horizons:** every horizon from 5 to 60 minutes shows a similar AUC (0.526–0.544 on the holdout).
- **Model classes:** all three are comparable (holdout AUC 0.534–0.544).
- **Volatility regimes:** AUC is 0.541–0.550; net P&L is negative in all of them.
- **Cost scenarios:** even the low-cost case loses 89.8 %.
- **Replication on SOLUSDT:** holdout AUC 0.534; break-even 0.89 bps per side; net −87.7 %.
- **Selection bias:** the holdout AUC survives a Bonferroni correction across 43 configurations.

## Key findings
1. **Order flow is informative but predicts reversal.** 1-minute OFI predicts slight reversal: −0.15 bps per sd at 5 min, HAC t −9.7, Holm-adjusted in development. On the holdout the effect replicates at about half the size (t −2.74).
2. **Order flow and book bands add almost nothing beyond price.** Price-only models reach almost the full AUC (holdout RF: 0.543 vs 0.544). Price features drive the importance (permutation AUC drop 0.0164 vs 0.0020 for book and 0.0010 for flow).
3. **ML beats simple rules as a forecaster.** Momentum and order-flow continuation rules are *anti*-predictive (AUC 0.48–0.49).
4. **Predictability is stable.** AUC is above 0.5 in every month across 31 months of out-of-sample evaluation.
5. **It is not economically viable.** The break-even cost is about one tenth of a realistic taker cost. **Predictive accuracy ≠ trading profitability.**

## Limitations
- The cost model assumes taker execution only. A maker or queue-aware strategy needs L2/L3 data that this archive does not provide.
- Order-book information is percentage-band depth only: no best bid/ask, touch-level imbalance or queue data.
- The analysis uses a single venue at 1-minute resolution.
- Slippage is an assumed parameter.
- SOL costs use ETH's half-spread, which understates SOL's costs.
- The gross Sharpe (CI 0.32–3.75) is below a deflated-Sharpe hurdle for the 215 strategy variants tried.

## Reproducibility
```bash
pip install -r requirements.txt            # exact versions; Python 3.13
pytest -q                                  # 30 unit tests (uses the 1-hour official-data sample in data_sample/)

python scripts/run_stage2.py               # download + SHA-256 verify + QC + 1-min bars (~25 GB streamed; ~1 GB kept)
python scripts/measure_spread.py           # quoted spread from bookTicker sample days
python scripts/build_dataset.py            # features + targets -> data/processed/research/
python scripts/leakage_audit.py            # -> docs/leakage_audit.md
python scripts/run_eda_stats.py            # EDA + statistical tests (development only)
python scripts/run_models.py               # walk-forward grid (development only)
python scripts/run_dev_backtest.py         # selections, out-of-fold backtest, regimes
python scripts/run_interpret.py            # permutation importance / SHAP
python scripts/freeze_methodology.py       # writes docs/final_methodology.md (refuses after the holdout is opened)
python scripts/run_holdout.py              # FINAL holdout (guarded, logged)
python scripts/run_stage2.py --symbol SOLUSDT --data-dir data/sol && \
python scripts/build_dataset.py --symbol SOLUSDT --data-dir data/sol && \
python scripts/run_replication.py && python scripts/run_holdout.py --symbol SOLUSDT --data-dir data/sol --tag replication --skip-secondary
python scripts/run_significance.py && python scripts/make_results_table.py && python scripts/make_notebooks.py
python scripts/make_pdf_report.py          # end-to-end PDF: reports/ETH_OrderFlow_Alpha_Report.pdf
python scripts/make_full_pdf.py            # complete documentation PDF: reports/ETH_OrderFlow_Alpha_Full_Documentation.pdf
```

- **Data URLs:** `https://data.binance.vision/data/futures/um/{daily|monthly}/{aggTrades|bookDepth|metrics|fundingRate|bookTicker}/{SYMBOL}/{SYMBOL}-{dataset}-{period}.zip` (+ `.CHECKSUM`).
  Downloaded 2026-10-07. Each file's URL, size and SHA-256 are written to `data/manifest.csv`.
- **Repository contents:** raw and processed data are not committed. The repository holds code, configuration, schemas, QC tables, result tables, figures and a small sample.
- **Seeds:** fixed (model seed 20231; bootstrap seeds in code).
- **Configuration:** [`config/config.yaml`](config/config.yaml).
- **Schema:** [`docs/data_schema.md`](docs/data_schema.md).

## Repository layout
```
config/config.yaml           parameters: periods, split, QC thresholds, costs, grids (pre-registered)
src/ethof/                   data.py (download/verify) · preprocessing.py (QC, 1-min bars) · pipeline.py · quality.py
                             features.py · targets.py · statistics.py · validation.py (folds, holdout guard)
                             models.py · backtest.py (signals, P&L, costs, funding) · strategy.py · evaluation.py
scripts/                     one script per stage (see Reproducibility)
notebooks/01..08             executed narrative notebooks that read pipeline outputs
docs/                        stage1_data_selection · data_quality_report · data_schema · leakage_audit · final_methodology
                             research_report · robustness_report · results_table · interview_questions · resume_bullets
reports/tables, figures      every result table and chart
tests/                       unit tests: timestamps, order flow, imbalance, rolling features, targets, signals, costs, P&L, funding
data_sample/                 1-hour subset of official files for tests
```
