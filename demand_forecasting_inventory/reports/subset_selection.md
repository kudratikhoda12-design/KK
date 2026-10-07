# Subset selection (Phase 6)

## Algorithm (uses TRAIN-period information only; no validation/test sales or model performance)

1. **Chronological split** of the 1,941 observed days: train 2011-01-29 to 2014-10-17; validation 2014-10-18 to 2015-08-04; test 2015-08-05 to 2016-05-22.
2. **Stores:** in each state the store with the largest TRAIN-period unit sales -> CA_3, TX_2, WI_3 (one store per state, covering the three SNAP regimes).
3. **Eligibility:** item-store listed (price row present) on *every* TRAIN day from d_1 -> 3,590 eligible series (CA_3: 1223, TX_2: 1208, WI_3: 1159). Post-hoc diagnostic (not a selection input): 100.0% of the eligible pool is also listed on every validation/test day, so no delisting handling is required.
4. **Demand tiers:** eligible series of the three stores are pooled and ranked by TRAIN mean daily sales. Percentile bands (config `TIER_BANDS`): low = P10-P35 (0.22-0.60 units/day), medium = P45-P65 (0.81-1.47), high = P85-P97 (3.35-12.64). The extreme tails (below P10, above P97) are excluded as unrepresentative.
5. **Sampling:** for every store x tier, 2 series are drawn at random with `numpy.random.default_rng(42)`, preferring different categories and never re-using an item -> 3 stores x 3 tiers x 2 = **18 series / 18 distinct products**.
6. **Profile** (mean, median, std, CV, zero share, ADI, CV^2, Syntetos-Boylan class) is computed on TRAIN data only.

Candidates available per store x tier cell: CA_3|high: 200, CA_3|medium: 263, CA_3|low: 204, TX_2|high: 125, TX_2|medium: 243, TX_2|low: 357, WI_3|high: 102, WI_3|medium: 209, WI_3|low: 336.

## Selected series

|   series_idx | series_id            | tier   | cat_id    | state_id   |   train_mean |   train_median |   train_zero_share |   train_adi |   train_cv2 | train_sb_class   |   last_train_price |
|-------------:|:---------------------|:-------|:----------|:-----------|-------------:|---------------:|-------------------:|------------:|------------:|:-----------------|-------------------:|
|            0 | FOODS_3_449_CA_3     | high   | FOODS     | CA         |        8.301 |              8 |              0.082 |       1.089 |       0.328 | smooth           |               1.98 |
|            1 | HOUSEHOLD_1_114_CA_3 | high   | HOUSEHOLD | CA         |        8.996 |              9 |              0.082 |       1.09  |       0.276 | smooth           |               0.97 |
|            2 | FOODS_3_784_CA_3     | medium | FOODS     | CA         |        1.345 |              0 |              0.502 |       2.009 |       0.462 | intermittent     |               1.48 |
|            3 | HOUSEHOLD_2_270_CA_3 | medium | HOUSEHOLD | CA         |        0.869 |              0 |              0.552 |       2.234 |       0.386 | intermittent     |               6.48 |
|            4 | HOBBIES_1_377_CA_3   | low    | HOBBIES   | CA         |        0.303 |              0 |              0.742 |       3.869 |       0.129 | intermittent     |               9.98 |
|            5 | HOUSEHOLD_2_228_CA_3 | low    | HOUSEHOLD | CA         |        0.513 |              0 |              0.646 |       2.823 |       0.312 | intermittent     |               5.47 |
|            6 | FOODS_2_347_TX_2     | high   | FOODS     | TX         |        3.999 |              4 |              0.12  |       1.136 |       0.339 | smooth           |               1.78 |
|            7 | HOUSEHOLD_1_247_TX_2 | high   | HOUSEHOLD | TX         |        3.452 |              3 |              0.103 |       1.115 |       0.347 | smooth           |               0.97 |
|            8 | FOODS_3_729_TX_2     | medium | FOODS     | TX         |        1.38  |              1 |              0.476 |       1.907 |       0.615 | lumpy            |               3.98 |
|            9 | HOBBIES_1_034_TX_2   | medium | HOBBIES   | TX         |        0.875 |              1 |              0.489 |       1.957 |       0.37  | intermittent     |               4.66 |
|           10 | FOODS_2_196_TX_2     | low    | FOODS     | TX         |        0.532 |              0 |              0.622 |       2.647 |       0.393 | intermittent     |               3.87 |
|           11 | HOUSEHOLD_2_315_TX_2 | low    | HOUSEHOLD | TX         |        0.437 |              0 |              0.68  |       3.122 |       0.367 | intermittent     |               7.67 |
|           12 | FOODS_2_056_WI_3     | high   | FOODS     | WI         |        4.138 |              2 |              0.457 |       1.843 |       0.437 | intermittent     |               6.98 |
|           13 | HOBBIES_1_312_WI_3   | high   | HOBBIES   | WI         |        3.707 |              2 |              0.356 |       1.554 |       1.416 | lumpy            |               0.58 |
|           14 | FOODS_3_128_WI_3     | medium | FOODS     | WI         |        1.381 |              1 |              0.372 |       1.592 |       0.407 | intermittent     |               2.98 |
|           15 | HOBBIES_1_036_WI_3   | medium | HOBBIES   | WI         |        0.885 |              0 |              0.565 |       2.298 |       0.492 | lumpy            |               1.12 |
|           16 | FOODS_3_520_WI_3     | low    | FOODS     | WI         |        0.579 |              0 |              0.616 |       2.607 |       0.359 | intermittent     |               3.98 |
|           17 | HOUSEHOLD_1_464_WI_3 | low    | HOUSEHOLD | WI         |        0.289 |              0 |              0.774 |       4.423 |       0.199 | intermittent     |               9.47 |

## What the selection means
* Demand classes (TRAIN): {'intermittent': 11, 'smooth': 4, 'lumpy': 3}. Intermittent/lumpy demand dominates, which is why a **Croston-SBA baseline** is added.
* Categories: {'FOODS': 8, 'HOUSEHOLD': 6, 'HOBBIES': 4}.
* Selection was run once with the fixed seed; it was not repeated or adjusted after seeing any forecasting result.
* **Limitation:** only continuously listed items are eligible (survivorship of the assortment); high-volume extremes (> P97) are not represented.
