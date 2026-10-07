# Results summary

*Test-period results; costs hypothetical.*

## Forecasting (pooled test WAPE, lower is better)

| Model         |   7-day WAPE % |   14-day WAPE % |   28-day WAPE % |
|:--------------|---------------:|----------------:|----------------:|
| XGBoost       |          38.46 |           32.79 |           29.85 |
| SARIMA        |          38.01 |           33.71 |           31.56 |
| MovingAverage |          39.42 |           34.48 |           32.15 |
| Holt          |          37.57 |           34.22 |           32.73 |
| HoltWinters   |          37.70 |           34.35 |           32.85 |
| SES           |          37.60 |           34.31 |           32.86 |
| Croston       |          40.66 |           36.04 |           33.63 |
| SeasonalNaive |          40.91 |           39.90 |           40.33 |
| Naive         |          70.42 |           70.88 |           72.82 |

Best by horizon: 7-day: Holt (37.6%), 14-day: XGBoost (32.8%), 28-day: XGBoost (29.8%).

## Inventory (MEDIUM penalty, 95% target)

| Policy            | Forecast_Model                  |   Lead_Time_days |   Stockouts |   Fill_Rate |   Cycle_Service_Level |   Average_Inventory |   Holding_Cost |   Ordering_Cost |   Stockout_Cost |   Total_Cost |
|:------------------|:--------------------------------|-----------------:|------------:|------------:|----------------------:|--------------------:|---------------:|----------------:|----------------:|-------------:|
| A_historical      | none (28-day trailing mean/std) |                3 |          43 |       0.981 |                 0.794 |             594.387 |        195.243 |         145.000 |         170.718 |      510.961 |
| B_forecast_driven | XGBoost                         |                3 |          29 |       0.975 |                 0.879 |             607.904 |        212.396 |         141.000 |         145.035 |      498.431 |
| A_historical      | none (28-day trailing mean/std) |                7 |          65 |       0.960 |                 0.746 |             675.990 |        223.417 |         141.000 |         283.443 |      647.860 |
| B_forecast_driven | SARIMA                          |                7 |          45 |       0.963 |                 0.848 |             661.123 |        237.717 |         140.000 |         215.067 |      592.784 |
| A_historical      | none (28-day trailing mean/std) |               14 |          96 |       0.936 |                 0.752 |             750.175 |        252.691 |         141.000 |         447.021 |      840.712 |
| B_forecast_driven | SARIMA                          |               14 |          55 |       0.945 |                 0.887 |             787.284 |        287.777 |         143.000 |         293.415 |      724.192 |

B vs A total cost: -2.5% (L=3), -8.5% (L=7), -13.9% (L=14); cheaper in 21/27 grid cells; bootstrap intervals include zero in 27/27.

Accuracy vs cost: lowest-WAPE model is the cheapest in 2/27 cells; mean Spearman rho (excl. Naive) LOW +0.47, MEDIUM -0.08, HIGH -0.26.

