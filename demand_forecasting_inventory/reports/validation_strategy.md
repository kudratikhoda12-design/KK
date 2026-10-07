# Validation strategy (Phase 9)

## 1. Chronological split - never random

| period     |   first_day_index |   last_day_index |   n_days | first_date   | last_date   |   share_of_days |
|:-----------|------------------:|-----------------:|---------:|:-------------|:------------|----------------:|
| train      |                 0 |             1357 |     1358 | 2011-01-29   | 2014-10-17  |            0.70 |
| validation |              1358 |             1648 |      291 | 2014-10-18   | 2015-08-04  |            0.15 |
| test       |              1649 |             1940 |      292 | 2015-08-05   | 2016-05-22  |            0.15 |

The 1,941 observed days are split by *date*: 70% train / 15% validation / remainder test (`floor(0.70 x N) = 1358` and `floor(0.15 x N) = 291` days; the test period gets the remaining 292). No observation is ever shuffled; a static check (`leakage_audit.check_no_random_splitting`) confirms that no random-split API is used.

## 2. Rolling-origin (daily) evaluation of H-day demand sums

* A **forecast origin** `t` is the last day whose sales are known. The forecast is for `sum(y[t+1..t+H])`.
* Every day of the validation (test) period that leaves the *complete* H-day window inside the period is an origin, so each model is scored on hundreds of overlapping windows (consecutive windows share H-1 days; this is why the Diebold-Mariano test uses a HAC variance with lag H-1).
* Origin `t = train_end - 1` (last training day) is the first validation origin; `t = val_end - 1` is the first test origin.

|   horizon H | role                          |   validation origins |   test origins |   XGBoost tuning rows (train) |   XGBoost final-fit rows (train+val) |
|------------:|:------------------------------|---------------------:|---------------:|------------------------------:|-------------------------------------:|
|           3 | supplementary (lead time L=3) |                  289 |            290 |                         23904 |                                29142 |
|           7 | primary                       |                  285 |            286 |                         23832 |                                29070 |
|          14 | primary                       |                  278 |            279 |                         23706 |                                28944 |
|          28 | primary                       |                  264 |            265 |                         23454 |                                28692 |

## 3. What is chosen on which data

| decision | made on | never uses |
|---|---|---|
| moving-average window (7/14/28) | validation WAPE (per horizon) | test |
| ETS variant (Holt-Winters with / without damped trend) | AICc on the training window | validation, test |
| SARIMA structure (5 candidates, per series) | validation error of the H = 7, 14, 28 sums (AIC/BIC reported; AIC is not comparable across differencing orders) | test |
| XGBoost hyper-parameters, number of trees (early stopping) | validation rows only (24 configurations x 4 horizons) | test |
| sigma_L of the safety stock; prediction-interval method | VALIDATION residuals of the model trained on TRAIN | test |
| headline Policy-B forecasting model per lead time | lowest validation WAPE at H = L | test cost |
| cost parameters, scenarios, service levels, lead times | fixed a priori (config.py; EOQ cycle lengths sanity-checked on TRAIN only) | any result |

## 4. Two-phase protocol (identical for every model family)

1. **Validation phase:** fit on `y[:train_end]`; produce forecasts for validation origins (parameters frozen, state updated day by day); select / calibrate.
2. **Test phase:** with the structure chosen in phase 1, **refit on train + validation** (`y[:val_end]`); produce forecasts for test origins; evaluate **once**.
3. XGBoost training rows are cut so that their target windows end before the period being predicted (`origin + H <= fit_end - 1`; asserted in code and in the audit).

## 5. Why not k-fold or a longer rolling re-fit?
k-fold would shuffle time. A fully rolling re-fit (re-estimating every day) would multiply the cost by ~290 for SARIMA/XGBoost without changing the protocol's logic; instead parameters are re-estimated once per phase and the *states* (ETS level/seasonals, SARIMA Kalman state) and *features* (lags, rolling means) are updated daily - the way such models run in production between periodic re-fits.

## 6. Known limitations of the design
* One validation and one test period (292 days, one calendar year of seasonality): results describe this period, not all futures.
* Overlapping windows make errors strongly autocorrelated; the effective number of independent windows is roughly n / H.
* The test period turned out to be harder than validation for *every* model (pooled RMSE ratio test/validation 1.17 to 2.23), which matters for validation-based safety stocks (section 15 of the final report).

