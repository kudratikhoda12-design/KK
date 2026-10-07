# Robustness Report

Each row is a robustness test. Values are copied from the named file.

- **Dev** = development walk-forward out-of-fold results (2024-03 → 2025-09).
- **Holdout** = the untouched holdout under the frozen procedure (2025-10 → 2026-09).
- **BE** = break-even cost, in bps per side.

All holdout experiments listed here were declared in `docs/final_methodology.md` §9 before the holdout was opened.

| # | Test | Result | Source |
|---|---|---|---|
| 1 | Horizon 5 / 15 / 30 / 60 min (holdout AUC, logit + LightGBM; 15 min includes RF) | 0.531–0.537 / 0.534–0.544 / 0.530–0.540 / 0.526–0.537. Predictability at every horizon, no horizon stands out | `holdout/horizons.csv`, `holdout/ablation.csv` |
| 2 | Horizon (dev, same models) | best mean AUC 0.531 / 0.539 / 0.539 / 0.536 | `model_summary.csv` |
| 3 | Model class (holdout, set E) | logit 0.534, RF 0.544, LightGBM 0.542. BE 1.07 / 0.49 / 0.55 | `holdout/ablation.csv` |
| 4 | Feature ablation A → E (dev, LightGBM) | 0.5375 → 0.5378 → 0.5380 → 0.5391 → 0.5388 | `model_summary.csv` |
| 5 | Feature ablation A → E (holdout, RF) | 0.5428 → 0.5424 → 0.5427 → 0.5434 → 0.5436 | `holdout/ablation.csv` |
| 6 | Feature ablation A → E (holdout, logit) | 0.5346 → 0.5336 → 0.5363 → 0.5338 → 0.5337; log loss for D and E (0.6939) is worse than the prior (0.6931) | `holdout/ablation.csv` |
| 7 | Cost scenarios (holdout ML, frozen δ) | gross +67.5 %; low (2.52 bps) −89.8 %; base (5.52) −99.6 %; high (7.02) −99.9 % | `holdout/backtest_summary.csv` |
| 8 | Cost sweep (holdout ML) | +67.5 % at 0 bps; −3.9 % at 0.5; −44.8 % at 1.0. BE 0.49 bps | `holdout/cost_sensitivity_ml.csv` |
| 9 | Signal selectivity (dev, EXPLORATORY, non-binding) | mean gross trade +0.19 bps (all minutes) → +0.76 (conf > 0.05) → +2.96 (conf > 0.10, 1.1 % of minutes). Never above the 11.05 bps round trip | `dev_edge_by_confidence_EXPLORATORY.csv` |
| 10 | Volatility regimes (holdout; tercile cut-offs from 2023-03..2024-02) | AUC low 0.550 / mid 0.547 / high 0.541. Gross sum of bar returns −0.117 / +0.168 / +0.496. Net negative in all three | `holdout/regimes.csv` |
| 11 | Volatility regimes (dev) | AUC 0.546 / 0.535 / 0.536. Gross positive only in high volatility. Net negative in all three | `dev_regimes.csv` |
| 12 | Time stability | monthly AUC > 0.5 in 19/19 dev months (0.520–0.558) and 12/12 holdout months (0.521–0.559). Quarterly IC of 15-min OFI is negative in 11/11 dev quarters | `dev_monthly_stability.csv`, `holdout/monthly_stability.csv`, `stats_quarterly_rank_ic.csv` |
| 13 | Confirmatory test of the main dev finding on the holdout | 1-min OFI → 5 min: β −0.069 bps per sd, t −2.74, Holm p 0.037 (dev: −0.146, t −9.74). → 15 min: not significant (p 0.23) | `holdout/confirmatory_tests.csv` |
| 14 | Replication on SOLUSDT (frozen settings, RF on set E) | dev AUC 0.529, BE 2.08 bps. Holdout AUC 0.534, gross Sharpe 1.74, BE 0.89 bps, net −87.7 % at base | `replication_SOLUSDT/{dev_summary,backtest_summary,summary}` |
| 15 | Alternative targets (dev, LightGBM set E) | 3-class: macro AUC 0.614, but UP-vs-DOWN AUC excluding neutral is 0.546, so most of the macro AUC is volatility, not direction. Regression: RMSE 35.66 vs 35.61 for predicting the mean, IC 0.013, so it fails | `model_folds_all.csv` |
| 16 | Execution-price realism | AUC against the VWAP_{t+1} → VWAP_{t+16} return is 0.548 vs 0.544 close-to-close (holdout). The edge survives the one-bar delay statistically | `holdout/summary.json` |

## Multiple testing and data snooping

**What was tried.** Development used 43 model configurations (`experiment_registry.csv`: 8 tuning points, 15 ablation
configurations, 18 horizon configurations, 2 alternative targets), 5 signal thresholds, 36 univariate signal tests, 16 controlled
regressions, 12 decile spreads and 4 event definitions.

**How the noise was controlled:**

- **Univariate tests.** Holm (family-wise error) and Benjamini–Hochberg (false discovery rate) adjustments are applied within each table. Holm is used for the headline claims
  because the tests are few and strongly dependent. 1-min OFI reversal survives Holm at every horizon in development. 15-min OFI and
  most other signals do not.
- **Selection.** Model class, LightGBM hyper-parameters and δ were each chosen by a single pre-declared criterion on development data, then
  frozen (commit `71ca5ad`, SHA-256 `f890b16b…`). The holdout was evaluated once for ETH and once for the SOL replication. Both
  accesses are logged with the unchanged methodology hash (`reports/holdout_access_log.jsonl`).
- **Bonferroni on the holdout AUC.** With 43 configurations, a one-sided z of 3.05 is required. The holdout AUC has z = 19.6 against a day-block
  bootstrap SE of 0.0022 (`holdout/significance.json`). **Predictability is not a selection artefact.**
- **Deflated-Sharpe hurdle.** With 43 × 5 = 215 strategy variants and one year of data, the expected maximum Sharpe of zero-skill variants is
  ≈ 3.28. The holdout gross Sharpe of 2.03 (bootstrap CI 0.32–3.75) is below this hurdle. The gross profit would not be convincing
  even if trading were free, and it is irrelevant at realistic costs.
- **Not used.** A full White reality check / Hansen SPA over the strategy universe was not run. It would only matter if some strategy were
  net-profitable, and none is: net Sharpe is negative for every model, feature set and cost scenario tested.

## Where we deviated from the original plan, and why

1. **Model results produced before the frozen-feed fix were discarded and re-run.** The fix was triggered by a crash (±inf in features), not by results.
2. **Trades are processed from daily files instead of monthly files.** A single month needed more than 13 GB of RAM. This is a computational change with identical content.
3. **Order-book features use percentage bands, not top-N levels**, because the data provide only bands (Stage 1 decision).
4. **A post-holdout bootstrap and snooping accounting was added.** It is inference on frozen predictions, labelled as such.
