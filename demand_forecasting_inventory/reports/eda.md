# Exploratory data analysis (Phase 7)

All statistics use the **TRAIN period only** (2011-01-29 to 2014-10-17, 1358 days) so that EDA does not leak validation/test information into modelling decisions. Figures are in `outputs/figures/eda_*.png`; the full numbers are in `outputs/tables/eda_summary_stats.csv`. Statements below describe patterns; **no causal claims** are made from correlations.

## Representative series (rule: per tier, the series closest to the tier's median TRAIN mean)

| tier   |   series_idx | series_id            |   train_mean | sb_class     |
|:-------|-------------:|:---------------------|-------------:|:-------------|
| high   |            6 | FOODS_2_347_TX_2     |        3.999 | smooth       |
| medium |            2 | FOODS_3_784_CA_3     |        1.345 | intermittent |
| low    |            5 | HOUSEHOLD_2_228_CA_3 |        0.513 | intermittent |

## Summary statistics (TRAIN, all 18 series)

| series_id            | tier   |   mean |   median |   std |   cv |   min |   max |   zero_pct |   adi |   cv2 | sb_class     |
|:---------------------|:-------|-------:|---------:|------:|-----:|------:|------:|-----------:|------:|------:|:-------------|
| FOODS_3_449_CA_3     | high   |   8.3  |        8 |  5.55 | 0.67 |     0 |    42 |       8.17 |  1.09 |  0.33 | smooth       |
| HOUSEHOLD_1_114_CA_3 | high   |   9    |        9 |  5.62 | 0.62 |     0 |    38 |       8.25 |  1.09 |  0.28 | smooth       |
| FOODS_3_784_CA_3     | medium |   1.35 |        0 |  1.87 | 1.39 |     0 |    10 |      50.22 |  2.01 |  0.46 | intermittent |
| HOUSEHOLD_2_270_CA_3 | medium |   0.87 |        0 |  1.26 | 1.45 |     0 |     9 |      55.23 |  2.23 |  0.39 | intermittent |
| HOBBIES_1_377_CA_3   | low    |   0.3  |        0 |  0.56 | 1.84 |     0 |     3 |      74.15 |  3.87 |  0.13 | intermittent |
| HOUSEHOLD_2_228_CA_3 | low    |   0.51 |        0 |  0.84 | 1.64 |     0 |     7 |      64.58 |  2.82 |  0.31 | intermittent |
| FOODS_2_347_TX_2     | high   |   4    |        4 |  2.89 | 0.72 |     0 |    19 |      12    |  1.14 |  0.34 | smooth       |
| HOUSEHOLD_1_247_TX_2 | high   |   3.45 |        3 |  2.45 | 0.71 |     0 |    13 |      10.31 |  1.11 |  0.35 | smooth       |
| FOODS_3_729_TX_2     | medium |   1.38 |        1 |  1.99 | 1.44 |     0 |    16 |      47.57 |  1.91 |  0.61 | lumpy        |
| HOBBIES_1_034_TX_2   | medium |   0.87 |        1 |  1.13 | 1.3  |     0 |     9 |      48.9  |  1.96 |  0.37 | intermittent |
| FOODS_2_196_TX_2     | low    |   0.53 |        0 |  0.87 | 1.64 |     0 |    12 |      62.22 |  2.65 |  0.39 | intermittent |
| HOUSEHOLD_2_315_TX_2 | low    |   0.44 |        0 |  0.79 | 1.81 |     0 |     9 |      67.97 |  3.12 |  0.37 | intermittent |
| FOODS_2_056_WI_3     | high   |   4.14 |        2 |  5.32 | 1.28 |     0 |    30 |      45.73 |  1.84 |  0.44 | intermittent |
| HOBBIES_1_312_WI_3   | high   |   3.71 |        2 |  6.15 | 1.66 |     0 |    64 |      35.64 |  1.55 |  1.42 | lumpy        |
| FOODS_3_128_WI_3     | medium |   1.38 |        1 |  1.54 | 1.11 |     0 |    14 |      37.19 |  1.59 |  0.41 | intermittent |
| HOBBIES_1_036_WI_3   | medium |   0.89 |        0 |  1.38 | 1.56 |     0 |     8 |      56.48 |  2.3  |  0.49 | lumpy        |
| FOODS_3_520_WI_3     | low    |   0.58 |        0 |  0.92 | 1.6  |     0 |     6 |      61.63 |  2.61 |  0.36 | intermittent |
| HOUSEHOLD_1_464_WI_3 | low    |   0.29 |        0 |  0.6  | 2.08 |     0 |     4 |      77.39 |  4.42 |  0.2  | intermittent |

## Findings

* **Volume and zeros:** daily means range from 0.29 to 9.00 units; the share of zero-sales days ranges from 8% to 77% (median 50%). 14 of 18 series have ADI >= 1.32 (intermittent or lumpy).
* **Variability:** the coefficient of variation ranges from 0.62 to 2.08; median CV by tier: high 0.72, medium 1.42, low 1.73.
* **Weekly pattern: weak.** Autocorrelation at lag 7 is above the 95% noise bound for 15 of 18 series (mean 0.16, max 0.49). Robust-STL seasonal strength F_s (period 7): median 0.09, max 0.24; 0 of 18 exceed 0.64 (the usual threshold for suggesting seasonal differencing). Day-of-week indices (min-max, series mean = 1) of the representative series: high: 0.81-1.26; medium: 0.74-1.43; low: 0.82-1.31; busiest weekday over all series: {'Sat': 9, 'Sun': 6, 'Fri': 2, 'Wed': 1}.
* **Yearly pattern:** monthly indices of the representative series range high: 0.85-1.16; medium: 0.66-1.97; low: 0.65-1.38 (TRAIN has only ~3.7 years, so each month effect rests on 3-4 observations - indicative only).
* **Trend:** an OLS slope of weekly sales on time is significant (p < 0.05) for 12 of 18 series, and *negative* for 12 of them (median trend over all series -14.3% of the mean per year). Because eligibility required the item to be listed since d_1, the sample consists of long-established items; their decline is consistent with product life-cycle effects, but the data cannot establish the cause. Long zero spells (up to ~1 year) behave like regime changes (see `zero_sales_analysis.md`) rather than a smooth trend.
* **Stationarity (ADF/KPSS on levels, TRAIN):** conflict: ADF stationary / KPSS non-stationary: 14, stationary (both tests agree): 3, unit root (both tests agree): 1. 'Conflict' (ADF rejects a unit root but KPSS rejects level-stationarity) is the classic signature of a slowly moving mean or structural breaks; it motivates SARIMA candidates both with and without differencing.
* **Prices:** the median number of weekly price changes over TRAIN is 2 (range 0-15); prices are piecewise constant, so variation is sparse.
* **Outliers:** Tukey far-out fence (Q3 + 3 IQR of non-zero demand) flags 204 series-days (0.83% of TRAIN series-days); they are retained (see `data_cleaning.md`).

## Implications for modelling
* Weekly seasonality is weak: a seasonal period of 7 is still the natural choice for seasonal baselines, Holt-Winters and SARIMA, but gains from modelling it should be modest.
* Intermittency and skewness -> Gaussian assumptions are approximate; Croston-SBA is included as a baseline and prediction intervals are checked empirically.
* Level shifts, declines and dead spells -> slow-adapting models (small smoothing parameters) can lag; long moving averages are a strong baseline.
* Sparse price variation -> price features are expected to matter little; SHAP will show whether that is the case.

