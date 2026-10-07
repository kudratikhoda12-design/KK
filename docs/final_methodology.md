# Final Methodology (FROZEN)

Frozen at **2026-10-07T14:21:08+00:00**, before any holdout row was evaluated.
`scripts/freeze_methodology.py` generated this file from `config/config.yaml` and the
development-period selection outputs. `reports/tables/frozen_parameters.json` is the machine-readable copy
that `scripts/run_holdout.py` reads. Every holdout access is logged in `reports/holdout_access_log.jsonl` together with the
SHA-256 of this file. If the file changes after the holdout is opened, the hashes in the log will no longer match.

Environment: Python 3.13.16, packages pinned in `requirements.txt`. Seeds: model seed 20231
(`ethof.models.SEED`), bootstrap seeds fixed in code.

## 1. Data and periods
- Instrument: Binance USDⓈ-M **ETHUSDT perpetual** (public archive, SHA-256-verified files).
- Full period 2023-03-01 → 2026-09-30 UTC. **Development** 2023-03-01 → 2025-09-30. **Holdout**
  2025-10-01 → 2026-10-01 (exclusive).
- Research frequency: complete 1-minute grid. Bar t covers [ts, ts+1 min).
- Data treatment: as in `docs/data_quality_report.md`. Nothing is deleted for being extreme. Book features are missing when
  the snapshot is older than 120 s, where age is measured from the last snapshot whose values *changed*
  (this covers the frozen feed). Open interest ≤ 0 is set to missing, and ±inf ratios are set to missing.

## 2. Timeline
| Step | Time |
|---|---|
| features of row t | data stamped < τ_t = ts_t + 1 min (trades of bars ≤ t; last book snapshot < τ_t; settled funding ≤ τ_t; OI created ≤ τ_t − 5 min) |
| signal | at τ_t |
| execution | during bar t+1 at its VWAP (taker); never at the signal price |
| holding | 15 bars per signal (staggered: net position = mean of the last 15 signals) |
| exit | VWAP of bar t+1+15 (implicitly, through the staggered position) |
| target | (P[t+15] − P[t]) / P[t], P = last trade price at or before the bar close |

## 3. Features (47 in the primary set E)
| Group | n | Features |
|---|---|---|
| price | 16 | `ret_1`, `ret_5`, `ret_15`, `ret_30`, `ret_60`, `vol_15`, `vol_60`, `vol_240`, `vol_1440`, `mom_240_riskadj`, `dist_ma_60`, `dist_ma_240`, `range_15`, `dist_vwap_15`, `hour_sin`, `hour_cos` |
| volume | 6 | `log_volume_1`, `volume_z_1440`, `volume_ratio_15`, `volume_ratio_60`, `trade_intensity_15`, `trade_count_z_1440` |
| flow | 13 | `ofi_1`, `ofi_5`, `ofi_15`, `ofi_60`, `trade_imb_1`, `trade_imb_15`, `signed_flow_15`, `ofi_15_z_1440`, `ofi_1_z_1440`, `avg_trade_size_15`, `avg_trade_size_z_1440`, `large_share_15`, `large_ofi_15` |
| book | 7 | `obi_1pct`, `obi_2pct`, `obi_3pct`, `obi_5pct`, `obi_1pct_chg_15`, `obi_1pct_mean_15`, `depth_1pct_z_1440` |
| funding_oi | 5 | `funding_rate_last`, `funding_rate_mean_3`, `minutes_to_funding`, `oi_chg_60`, `oi_chg_1440` |

The order-book features are **percentage-band** imbalances (cumulative resting quantity within ±1/2/3/5 % of the price).
They are **not** top-N level features, and no best bid/ask or spread information is used. Trailing windows only. Large-trade threshold
is ≥ 10 ETH (rule: the size bucket closest to the median per-minute p95 trade size in 2023-03..2024-02).

Ablation sets: A = price; B = A + volume; C = B + flow; D = C + book; E = D + funding/OI.

## 4. Target
- Primary: binary `up_15 = 1{fwd_ret_15 > 0}`. Secondary (reported, not used for trading): 3-class with neutral band
  ±11 bps (= base round-trip cost); regression on fwd_ret_15; horizons 5/30/60.

## 5. Validation and preprocessing
- Expanding walk-forward, 3-month blocks, first validation block 2024-03-01, purge of
  training rows whose label ends after (block start − 1 day embargo), training stride 5 minutes.
- Imputation and scaling happen inside sklearn pipelines fitted per training fold. LightGBM uses native missing-value handling.

## 6. Model (frozen)
- Selection rule (pre-declared): the lowest mean validation log loss on set E, h = 15, among logit / RF / LightGBM.
- **Selected: `rf_E_all_h15`** (model class `rf`).
- LightGBM hyper-parameters (8-point grid, chosen by mean validation log loss): `{"num_leaves": 15, "min_child_samples": 500, "n_estimators": 200}`;
  other parameters are fixed in `ethof.models.LGBM_DEFAULT`. Logit C = 0.1. RF: `{"n_estimators": 200, "min_samples_leaf": 500, "max_depth": 10}`, stride 15.
- Holdout procedure: four 3-month blocks. Before each block the frozen model is re-fitted on all data from 2023-03-01 to
  (block start − 1 day) (the same expanding scheme as development). No parameter, feature or threshold is revisited.

## 7. Trading rule (frozen)
- LONG if P(up) > 0.5 + δ, SHORT if P(up) < 0.5 − δ, FLAT otherwise. **δ = 0.05**, chosen as the highest
  out-of-fold net Sharpe at base cost among [0.005, 0.01, 0.02, 0.03, 0.05] (development out-of-fold, 2024-03 → 2025-09).
- Position sizing: each signal is a sub-position of 1/15 of capital held 15 bars; net |position| ≤ 1; no leverage;
  shorting allowed (perpetual).
- Benchmarks: buy-and-hold (long 1×), momentum sign(ret_15), order-flow rule (long if ofi_15_z_1440 > 1.0, short if < −1.0),
  random signals with the ML strategy's activity rate (seeded).

## 8. Costs, slippage, funding (per side, on traded notional)
| Scenario | fee (bps) | slippage (bps) | half spread (bps) | total per side (bps) |
|---|---|---|---|---|
| low | 2.0 | 0.5 | 0.023 | 2.523 |
| base | 4.5 | 1.0 | 0.023 | 5.523 |
| high | 5.0 | 2.0 | 0.023 | 7.023 |

Sources: Binance USD-M VIP0 fees are 2 bps maker and 5 bps taker, with a 10% discount when paid in BNB. The half spread was measured from `bookTicker` on 3 development
days (`reports/tables/spread_measurement.csv`). Slippage is an assumption, bracketed by the low and high scenarios.
**Base** is the headline scenario. Funding: the realised rate at each settlement is charged to the position held over the settlement
(longs pay a positive rate, shorts receive it).
The break-even cost is (gross P&L + funding) / total turnover, expressed per side.

## 9. Metrics and pre-declared secondary holdout evaluations
- Classification: AUC, log loss vs prior, Brier, accuracy, precision/recall/F1, ECE, Spearman IC, and *tradable AUC* (against the
  VWAP-to-VWAP return actually capturable).
- Trading: total and annualised return, volatility, Sharpe and Sortino (daily, 365-day year), max drawdown, Calmar, win rate,
  profit factor, average trade (bps, net of a full round trip), turnover, exposure, funding, break-even cost.
- Also evaluated on the holdout, with the same frozen procedure: ablation sets A–E × {logit, LightGBM, RF} at h = 15; horizons 5/30/60
  (sets A, C, E × logit and LightGBM); a confirmatory HAC test of 1-minute OFI reversal; volatility regimes (tercile cut-offs
  4.838e-04 / 6.694e-04 on vol_1440, estimated on 2023-03..2024-02); monthly stability; SOLUSDT replication
  with identical settings.
