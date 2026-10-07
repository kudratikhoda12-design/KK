# Methodology summary

**Question.** Do accurate, explainable demand forecasts lead to better inventory decisions (service vs holding cost)?

| step | what was done |
|---|---|
| Data | M5 Walmart (30,490 series x 1,941 days) from two SHA-256-verified public mirrors; sample of 18 item-store series chosen with TRAIN information only (store per state, listing eligibility, percentile demand tiers, seeded draws) |
| Split | chronological 2011-01-29..2014-10-17 / 2014-10-18..2015-08-04 / 2015-08-05..2016-05-22; daily rolling origins |
| Target | Target_H(t) = sum of demand over t+1..t+H, H = 7, 14, 28 (H = 3 supplementary for the 3-day lead time) |
| Baselines | naive, seasonal naive, moving average (window by validation), Croston-SBA |
| Statistical | SES, Holt (damped), Holt-Winters additive s = 7 (AICc), SARIMA 5 candidates s = 7 (validation) with ADF/KPSS/ACF/PACF |
| ML | direct XGBoost per horizon, global model, 25 leakage-safe features (history / calendar-of-origin / known-ahead calendar counts), 24-config validation search |
| Leakage control | features at origin; training rows cut before the next period; perturbation tests (40 checks) incl. negative control |
| Explainability | TreeSHAP on the final models, additivity verified, 3 rule-selected individual explanations |
| Uncertainty | sigma_L = validation RMSE of the H-day sum; Gaussian vs empirical intervals chosen by split-half calibration; test coverage reported |
| Inventory | (s, S) policy: ROP = mu_L + z sigma_L, Q = EOQ; order arrives at start of day t+L+1; lost sales; hypothetical costs |
| Policies | A: trailing 28-day mean and sqrt(L) x trailing std; B: forecast model + validation sigma; identical lead times, costs, period and initial inventory |
| Evaluation | MAE, RMSE, sMAPE, WAPE; Diebold-Mariano (HAC, HLN, Holm); paired bootstrap over series for costs; fill rate, cycle service level, in-stock days, turnover |
| Sensitivity | 3 penalty scenarios x 3 service levels x 3 lead times; accuracy-vs-cost rank correlation; tier and regularity subgroups |

