# Ethereum Order-Flow Alpha: Short-Horizon Return Prediction and a Cost-Aware Trading Strategy

**Instrument:** Binance USDⓈ-M ETHUSDT perpetual · **Period:** 2023-03-01 → 2026-09-30 (UTC) ·
**Untouched holdout:** 2025-10-01 → 2026-09-30 · Replication on SOLUSDT

Every number in this report is copied from a file produced by the pipeline. The source file is named next to each table, or in
§16. The results table is generated automatically in [`results_table.md`](results_table.md).

---

## 1. Abstract

We test whether taker order flow and percentage-band order-book imbalance predict 5- to 60-minute ETHUSDT returns, and
whether that prediction survives realistic execution costs. We built the data from **1,877,097,488** Binance aggregated trades and
**3,710,848** order-book depth snapshots, aggregated to a complete grid of **1,886,400** one-minute bars.

- **Order flow predicts reversal, not continuation, and the effect is tiny.** One-minute order-flow imbalance (OFI) predicts a statistically significant *reversal*: −0.15 to −0.24 bps per standard deviation, HAC t from −9.7 to −4.0, Holm-adjusted. The effect is two orders of magnitude smaller than trading costs.
- **Machine learning predicts out of sample.** Learning on 47 causal features with walk-forward validation, the frozen model reaches a holdout AUC of **0.5436** (95% day-block bootstrap CI 0.5394–0.5480). The probabilities are well calibrated (ECE 0.0028), and AUC is above 0.5 in every holdout month.
- **The signal is almost entirely price-based mean reversion.** Order-flow and order-book features add at most ~0.001–0.002 AUC in development and no robust increment on the holdout.
- **The strategy is not tradable.** Executed one bar later at VWAP, the strategy earns a gross Sharpe of **2.03** (+67.5%). Its break-even cost is only **0.49 bps per side**, against a realistic base cost of **5.52 bps per side**. Net of costs it loses **99.6%**.
- **SOLUSDT shows the same pattern.** The predictability is statistically real, but economically worthless at taker costs.

## 2. Motivation

Order-flow imbalance is the most direct observable trace of informed or liquidity-demanding trading, and it explains a large share of
contemporaneous price changes. Whether it carries forward-looking information that an outside participant can monetise is a different
question. Crypto perpetuals are a natural test bed: the trade side is recorded, the data are public and granular, and the market trades 24/7.
They are also a hard test, because fees are an order of magnitude larger than the 1-tick spread. The project is built to separate three claims that are
often conflated:

1. *statistical* predictability,
2. *out-of-sample* predictability, and
3. *economic* predictability after execution.

## 3. Research question

> Does order-flow and market-microstructure information contain statistically significant predictive information about
> short-horizon ETH returns, and does it translate into economically meaningful alpha after transaction costs and slippage?

Six sub-questions are answered in §15: (1) predictive content, (2) incremental value over price/volume, (3) ML vs simple
rules, (4) out-of-sample robustness, (5) cost survival, and (6) stability across time, regimes and assets.

## 4. Data

| Item | Value | Source |
|---|---|---|
| Venue / instrument | Binance USDⓈ-M futures, ETHUSDT perpetual | data.binance.vision |
| Files verified (SHA-256 vs published checksum) | 3,976 (1,310 aggTrades, 1,310 bookDepth, 1,310 metrics, 43 funding, 3 bookTicker) | `docs/data_quality_report.md` |
| Raw aggregated trades | 1,877,097,488 (0 removed as errors, 0 duplicates) | same |
| 1-minute bars | 1,886,400 = every minute of the period; 53 minutes have no trade | same |
| Book snapshots (±1…5 % bands, ~30 s) | 3,710,848 kept | same |
| Funding settlements | 3,930 (8-hour interval throughout) | same |
| Open interest (5 min) | 377,147 rows | same |
| Development / holdout rows | 1,360,740 / 525,600 minutes | `reports/tables/stats_summary.json`, `holdout/summary.json` |

The trade side comes from the data, not from inference: `is_buyer_maker = false` means a buyer-initiated taker trade. The
order-book data are **percentage-band aggregates** (cumulative resting quantity within k % of the price), not a level-by-level book. No best bid/ask
exists after 2024-03-30. The quoted spread was therefore measured from `bookTicker` on three development days: one tick in
99.5–99.6% of seconds, a mean of 0.046 bps (`reports/tables/spread_measurement.csv`).

## 5. Data quality

Full report: [`data_quality_report.md`](data_quality_report.md). Findings that matter for the research:

- **Frozen order-book feed.** From 2025-04-16 10:31 to 2025-05-19 10:26 the archive repeats one snapshot 94,730 times. There are 97,477
  identical consecutive snapshots in total. The daily files *look* complete; Stage 1 had wrongly read their small size as better
  compression. Book age is now measured from the last snapshot whose values changed, so 58,081 minutes have missing book features
  instead of month-old values.
- **Schema change.** Binance added ±0.2 % bands on 2026-01-15, inside the holdout. They are not used.
- **Extreme moves are kept.** 83 one-minute bars move more than 3 %. None meets the error-candidate rule; they are liquidation cascades, so
  they are kept and not winsorised.
- **Zero open-interest readings** (91) and ±inf ratios are set to missing. 4,047,288 underlying fill ids (0.216 %) are absent
  from `aggTrades`. Whether those omissions are side-neutral cannot be verified.
- **Time zone and sign checks.** At zero clock shift, 94.06 % of snapshots have implied bid < trade VWAP < implied ask. A ±1 h shift reduces
  that to 64 %. This supports UTC timestamps and negative percentage = bids.

## 6. Feature engineering

There are 47 causal features. The decision time is the bar close; all windows are trailing (`src/ethof/features.py`).

| Group | Examples |
|---|---|
| price (16) | log returns over 1/5/15/30/60 min, realised vol over 15/60/240/1440 min, risk-adjusted 240-min momentum, distance from the 60/240-min mean and the 15-min VWAP, 15-min range, hour-of-day |
| volume (6) | log volume, 1-day volume z-score, 15/60-min volume ratio, trade intensity, trade-count z-score |
| trade flow (13) | OFI over 1/5/15/60 min, trade-count imbalance, signed flow / typical volume, OFI z-scores, average trade size, large-trade (≥ 10 ETH) share and imbalance |
| book (7) | band imbalance at 1/2/3/5 %, 15-min change and mean of the 1 % imbalance, total 1 % depth z-score |
| funding/OI (5) | last settled funding rate, 3-settlement mean, minutes to next funding, 60-min and 1-day open-interest change (lagged ≥ 5 min) |

Buy and sell ratios are affine transforms of OFI (BuyRatio = (1 + OFI)/2), so they are not duplicated.

**Leakage audit** ([`leakage_audit.md`](leakage_audit.md)): all eight automated checks pass.

- **Truncation invariance:** features recomputed on data cut at 12 random times; 34.8 M values compared; max difference 0.
- **As-of joins:** book, funding and OI joins checked.
- **Targets:** exact.
- **Folds:** purge and embargo respected.
- **Preprocessing:** fitted inside each training fold.

## 7. Statistical methodology and results (development data only)

**Tests:**

- Forward returns over h minutes overlap, so OLS uses Newey–West HAC errors (2h lags). Non-overlapping re-estimates are reported as well.
- Decile spreads use a 5-day moving-block bootstrap.
- All 36 signal × horizon tests are adjusted with Holm (FWER) and Benjamini–Hochberg (FDR).

**Contemporaneous vs predictive.** corr(OFI₁, r₁) = **0.467**: order flow explains same-minute price change very well. The
predictive relationships are much weaker:

| Signal → horizon | β (bps per sd) | HAC t | Holm p | Non-overlap t |
|---|---:|---:|---:|---:|
| OFI 1-min → 5 min | −0.146 | −9.74 | 7.4e-21 | −1.99 |
| OFI 1-min → 15 min | −0.170 | −5.99 | 7.2e-8 | −1.09 |
| OFI 1-min → 30 min | −0.194 | −4.77 | 6.2e-5 | −2.54 |
| OFI 1-min → 60 min | −0.237 | −4.02 | 0.0018 | −1.07 |
| OFI 15-min → 15 min | −0.148 | −2.02 | 0.96 | −3.54 |
| Book imbalance 1 % → 5 min | +0.146 | +4.03 | 0.0018 | +3.29 |
| Book imbalance 1 % → 15 min | +0.237 | +2.41 | 0.40 | +1.22 |

(`reports/tables/stats_univariate_predictive.csv`)

**Interpretation:**

- **1-min OFI reversal is robust.** It stays significant after controlling for 15/60-min returns, volatility and volume (t = −5.87 at 5 min,
  −4.37 at 15 min; `stats_with_controls.csv`).
- **Book imbalance does not survive the controls** (t = 0.90 and 1.37). Its information overlaps with recent returns.
- **The decile spreads are tiny.** For 1-min OFI, top minus bottom decile is −0.29 bps over 15 minutes (95 % CI −0.47 to −0.09).
- **Event study.** Extreme selling (OFI₁ < 5th percentile) is followed by +0.18 bps over 5 minutes (CI 0.07–0.28), and extreme buying by −0.10 bps (CI −0.21 to −0.01). That is
  partial reversal of a 5.3–5.4 bps event-bar move (`stats_event_study.csv`).
- **The 1-min return autocorrelation is −0.030** (`eda_acf.csv`).
- **Statistical vs economic significance.** These effects are highly significant statistically, yet they are 1–2 orders of magnitude below a round-trip cost of about 11 bps.

## 8. ML methodology

**Target.** The primary target is `1{fwd_ret_15 > 0}` (base rate ≈ 0.50). Secondary targets:

- a 3-class target with a ±11 bps neutral band (= base round-trip cost);
- a regression on fwd_ret_15;
- the 5, 30 and 60-minute horizons.

**Models:**

- **Logistic regression:** L2, C = 0.1, median imputation with missing-value indicators, standardisation.
- **Random forest:** 200 trees, depth 10, at least 500 rows per leaf.
- **LightGBM:** an 8-point grid (`lgbm_tuning.csv`); the selected configuration is the smallest, 15 leaves / 500 minimum rows per leaf / 200 trees.

All preprocessing is inside per-fold pipelines.

**Rule baselines** (prior, momentum, flow rule) were scored on the same folds. Momentum and the flow-continuation rule have AUC 0.48–0.49 **in every fold and
at every horizon**: continuation is anti-predictive here (`baseline_rules_folds.csv`).

## 9. Validation

- **Scheme:** expanding walk-forward; first 12 months for initial training; seven validation blocks of 3 months (the last is 1 month) covering
  2024-03 → 2025-09.
- **Purge:** training labels must end before (validation start − 1 day embargo).
- **Stride:** training uses every 5th minute (every 15th for RF), because overlapping 15-minute labels make neighbouring rows near-duplicates.
- **Why not random cross-validation:** random CV would put minutes that share future price paths into both train and test.
- **Pre-declared selection rules:** lowest mean validation log loss (model); highest out-of-fold base-cost Sharpe (δ).

| Model (set E, h = 15) | Validation AUC mean ± sd | min fold | log loss | IC |
|---|---:|---:|---:|---:|
| Logistic regression | 0.5367 ± 0.0081 | 0.5244 | 0.69167 | 0.055 |
| **Random forest (selected)** | 0.5381 ± 0.0087 | 0.5265 | **0.69108** | 0.057 |
| LightGBM | 0.5388 ± 0.0084 | 0.5284 | 0.69113 | 0.059 |

(`model_summary.csv`; prior log loss ≈ 0.6931)

All 41 binary classifier configurations have validation AUC > 0.5 in all 7 folds. The frozen methodology
([`final_methodology.md`](final_methodology.md), SHA-256 `f890b16b…`) was committed in `71ca5ad` before the first holdout
access. The access log records that hash on both holdout accesses (ETH final, SOL replication).

## 10. Backtesting

**Execution assumptions:**

- **Timing:** the signal is formed at the bar close and executed at the **next bar's VWAP**.
- **Holding:** each signal holds a 1/15 sub-position for 15 bars (staggered), so the net |position| is at most 1.
- **Leverage and shorting:** no leverage; shorting allowed.
- **Funding:** the realised rate is charged to the position held over each settlement.

**Trading rule:** LONG if P(up) > 0.5 + δ, SHORT if P(up) < 0.5 − δ, otherwise FLAT. The pre-declared rule selected **δ = 0.05**, the most selective
value in the grid. Every δ loses money net of costs in development (`dev_delta_selection.csv`).

## 11. Transaction costs

| Scenario | fee | slippage | half spread | total per side |
|---|---:|---:|---:|---:|
| low | 2.0 | 0.5 | 0.023 | 2.523 bps |
| **base** | 4.5 | 1.0 | 0.023 | **5.523 bps** |
| high | 5.0 | 2.0 | 0.023 | 7.023 bps |

Fees are Binance USD-M VIP0 (2 bps maker, 5 bps taker, 10 % discount when paid in BNB). The half spread is measured. Slippage is an assumption,
bracketed by the low and high scenarios. Break-even cost = (gross P&L + funding) / turnover.

## 12. Results

### 12.1 Out-of-sample prediction (holdout, frozen)

| | Value | Source |
|---|---:|---|
| Pooled AUC | 0.5436, block-bootstrap 95 % CI [0.5394, 0.5480] | `holdout/significance.json` |
| AUC by 3-month block | 0.536 / 0.549 / 0.549 / 0.541 | `holdout/primary_blocks.csv` |
| Months with AUC > 0.5 | 12 / 12 (range 0.521–0.559) | `holdout/monthly_stability.csv` |
| Log loss vs prior | 0.6902 vs 0.6931 | `holdout/summary.json` |
| Accuracy / precision / recall / F1 | 0.531 / 0.529 / 0.538 / 0.533 | same |
| Calibration (ECE) / Spearman IC | 0.0028 / 0.062 | same |
| AUC against the *tradable* VWAP→VWAP return | 0.548 | same |

Holdout AUC is slightly *higher* than in development. There is no sign of decay over the 31 months of out-of-sample
evaluation (19 development months plus 12 holdout months).

### 12.2 Trading (holdout, base cost unless stated)

| Strategy | Gross return | Gross Sharpe | Net return | Net Sharpe | Max DD (net) | Trades | Break-even bps/side |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ML (frozen RF, δ = 0.05)** | **+67.5 %** | **2.03** | **−99.6 %** | **−20.37** | −99.6 % | 117,461 | **0.49** |
| Momentum sign(ret₁₅) | −78.1 % | −3.18 | −100.0 % | −44.11 | −100.0 % | 525,022 | n/a (gross loss) |
| Order-flow rule (OFI z > 1) | −51.9 % | −2.81 | −100.0 % | −41.43 | −100.0 % | 175,617 | n/a (gross loss) |
| Random (same activity) | +7.2 % | 0.91 | −99.9 % | −94.16 | −99.9 % | 117,194 | 0.05 |
| Buy & hold | −36.9 % | −0.48 | −36.9 % | −0.48 | −68.7 % | 1 | n/a |

(`holdout/backtest_summary.csv`)

**ML strategy details:**

- Gross: win rate 57.1 %, average trade 0.70 bps gross, turnover 30.4× notional per day, mean |position| ≈ 0.2.
- Net at base cost: win rate 36.0 %, average trade −10.35 bps.
- Funding is negligible: +0.08 % of capital over the year.
- Cost sweep (`holdout/cost_sensitivity_ml.csv`): the total return is +67.5 % at 0 bps, −3.9 % at 0.5 bps, −44.8 % at 1 bp and −99.6 % at 5.52 bps.
- The gross Sharpe of 2.03 has a bootstrap CI of [0.32, 3.75] (`holdout/significance.json`).

## 13. Robustness

Details in [`robustness_report.md`](robustness_report.md). In summary:

- **Model class:** logistic, RF and LightGBM all reach holdout AUC 0.534–0.544. None has a break-even cost above 1.07 bps per side.
- **Feature ablation:** price-only models already reach holdout AUC 0.543 (RF) and 0.541 (LightGBM), against 0.544 and 0.542 with all features.
  For logistic regression, adding book and funding features *lowers* holdout AUC (0.536 → 0.534) and worsens log loss beyond the prior (0.6939).
- **Horizons:** holdout AUC is 0.531–0.537 at 5 min, 0.534–0.544 at 15 min (all ablation models), 0.530–0.540 at 30 min, and 0.526–0.537 at 60 min. No horizon is economically different.
- **Costs:** the low scenario (2.52 bps) still loses 89.8 % net.
- **Volatility regimes:** AUC is 0.550 / 0.547 / 0.541 in low / mid / high volatility (holdout). Gross P&L is positive only in mid and high volatility, and net P&L is negative in every regime.
- **Confirmatory test:** on the holdout, 1-min OFI reversal at 5 min replicates with about half the development effect (β −0.069 bps per sd, t −2.74, Holm p 0.037). At 15 min it is not significant.
- **SOLUSDT, frozen settings:** holdout AUC 0.534, gross Sharpe 1.74, break-even 0.89 bps per side, net −87.7 %.
- **Model-agnostic interpretability** (out of sample, permutation importance): price features AUC drop 0.0164; book 0.0020; flow 0.0010; volume and funding/OI ≈ 0. The largest single features are the distance from the 60/240-min moving averages and recent returns, i.e. short-horizon mean reversion.

## 14. Limitations

1. **Taker-only execution.** The edge per trade is about 0.7 bps, so viability would require maker execution (2 bps fee or rebates) with queue
   priority. Modelling passive fills requires L2/L3 queue data, which this dataset does not contain. A maker strategy also suffers adverse selection
   precisely when the reversal signal fires, so it cannot be assessed here.
2. **Coarse order-book data.** ±1–5 % bands are far deeper than the touch (±1 % ≈ 300 ticks). Touch-level imbalance, which the literature finds most
   predictive at second horizons, is unobservable. The negative result for "book features" applies to band depth only.
3. **Single venue, 1-minute resolution.** Sub-minute dynamics and cross-venue flow are not used. Faster horizons may hold more edge, but would demand
   lower costs still.
4. **The slippage assumption and the use of VWAP as execution price are approximations.** They are generous rather than conservative for a 1/15-notional order in a deep market, so
   the true outcome is unlikely to be better.
5. **Researcher degrees of freedom.** 43 development model configurations, 5 thresholds and 36 univariate tests were run (§13 of the robustness
   report). The holdout AUC (z = 19.6) survives a Bonferroni correction over 43 configurations (z needed 3.05). The gross Sharpe of 2.03 is
   *below* the ~3.3 expected maximum of 215 zero-skill strategy variants, so even the gross profitability claim is weak.
6. **SOL costs are understated.** SOL uses ETH's measured half-spread; SOL's real tick-relative spread is larger.

## 15. Conclusion

| Question | Answer (evidence) |
|---|---|
| 1. Is there order-flow predictability? | **Yes, statistically; no, economically.** 1-min OFI predicts a reversal with Holm-adjusted p down to 7e-21 in development and p = 0.037 on the holdout (5 min), but the effect is ≈ 0.07–0.24 bps per standard deviation. |
| 2. Does order flow add information beyond price and volume? | **Marginally in development, not robustly on the holdout.** It adds +0.0001 to +0.002 AUC in development; holdout increments are within ±0.002 and negative for logistic regression with book features. |
| 3. Does ML improve the signal? | **Yes, as a forecaster.** ML is far better than the rule baselines (rules AUC < 0.5; ML 0.54). Trees beat logistic regression by ≈ 0.002–0.010 AUC on the holdout. It does not create an economically viable signal. |
| 4. Does the signal survive out of sample? | **Yes.** Holdout AUC is 0.5436 [0.5394, 0.5480], 12/12 months > 0.5, calibration ECE 0.003, tradable AUC 0.548. |
| 5. Does it survive realistic costs? | **No.** The break-even cost is 0.49 bps per side against 5.52 bps base (2.52 bps even at the low scenario). Net −99.6 %. |
| 6a. Is it stable across time? | **Yes for prediction.** Monthly AUC ranges from 0.520 to 0.558 in development and 0.521 to 0.559 on the holdout, with no decay. Development quarterly IC for OFI-15 is negative in all 11 quarters. |
| 6b. Is it stable across volatility regimes? | **Prediction yes.** AUC is 0.541–0.550 on the holdout across regimes. Profit is negative net of costs in every regime. |
| 6c. Does it replicate on another asset? | **Yes, qualitatively.** SOL holdout AUC is 0.534, with a stronger OFI reversal in development (t −11.4). It is also not tradable (break-even 0.89 bps per side). |
| Strongest limitation | Taker-only cost model and band-level book data. A maker or sub-second strategy cannot be evaluated with these data. |

**Bottom line.** There is strong, stable, out-of-sample statistical evidence that one-minute ETH perpetual price paths mean-revert, and that
taker order flow is mildly informative about that reversal. The predictable component is worth about 0.5 bps per side per trade, roughly one tenth
of the cost of taking liquidity. **The signal is statistically significant and economically untradable at realistic costs.** Higher accuracy does not mean higher profit.

## 16. Reproduction and file map

See the README. Primary sources:

- tables: `reports/tables/{stats_*, model_summary, dev_*, holdout/*, replication_SOLUSDT/*}`
- figures: `reports/figures/`
- narrative notebooks: `notebooks/01–08`
