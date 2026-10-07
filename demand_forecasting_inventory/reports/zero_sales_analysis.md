# Zero sales and the stockout problem (Phase 5)

## Question
Can the dataset distinguish genuine zero demand, product unavailability, store closure, stockout and missing observation?

## What the data *can* identify
| zero type | identifiable? | how |
|---|---|---|
| Product not available / not yet launched / delisted | **Yes** | no price row in `sell_prices` for the week (sales are never positive in such weeks) |
| Store closure / outage | **Yes (day level)** | store-wide total < 5 % of its rolling median; Christmas is also an explicit calendar event |
| Missing observation | **Yes** | none exist (no NaN in sales, calendar contiguous) |
| Stockout (item sold out) | **No** | the file has no inventory, shelf-availability or lost-sales information |
| Genuine zero demand | **No** | indistinguishable from a stockout when the item is listed and the store is open |

## Dataset-wide decomposition of zero cells (30,490 series x 1,941 days)

| category                                                                         |    cells | share_of_zero_cells   |
|:---------------------------------------------------------------------------------|---------:|:----------------------|
| zero cells (all series-days)                                                     | 40241819 | 100.0 %               |
| A. structural: item not listed that week (no price row)                          | 12299413 | 30.56 %               |
| B. store-wide closure / outage day (listed item)                                 |   128154 | 0.32 %                |
| C. ambiguous: store open, item listed, zero sales (true zero demand OR stockout) | 27814252 | 69.12 %               |

## Selected series (TRAIN period; store open, item listed)

| series_id            | tier   | sb_class     |   zero_share |   zero_on_closure_days |   longest_zero_run_open_days |   poisson_lambda_open_days |   suspicious_zero_runs |   suspicious_zero_days |
|:---------------------|:-------|:-------------|-------------:|-----------------------:|-----------------------------:|---------------------------:|-----------------------:|-----------------------:|
| FOODS_3_449_CA_3     | high   | smooth       |        0.082 |                      3 |                            7 |                      8.32  |                     21 |                     63 |
| HOUSEHOLD_1_114_CA_3 | high   | smooth       |        0.082 |                      3 |                            9 |                      9.016 |                     24 |                     69 |
| FOODS_3_784_CA_3     | medium | intermittent |        0.502 |                      3 |                          241 |                      1.348 |                      8 |                    333 |
| HOUSEHOLD_2_270_CA_3 | medium | intermittent |        0.552 |                      3 |                          162 |                      0.871 |                      8 |                    465 |
| HOBBIES_1_377_CA_3   | low    | intermittent |        0.742 |                      3 |                           20 |                      0.304 |                    nan |                    nan |
| HOUSEHOLD_2_228_CA_3 | low    | intermittent |        0.646 |                      3 |                           28 |                      0.514 |                      2 |                     53 |
| FOODS_2_347_TX_2     | high   | smooth       |        0.12  |                      3 |                           12 |                      4.007 |                     16 |                     80 |
| HOUSEHOLD_1_247_TX_2 | high   | smooth       |        0.103 |                      3 |                            4 |                      3.46  |                      3 |                     12 |
| FOODS_3_729_TX_2     | medium | lumpy        |        0.476 |                      3 |                          358 |                      1.383 |                      1 |                    358 |
| HOBBIES_1_034_TX_2   | medium | intermittent |        0.489 |                      3 |                           11 |                      0.877 |                      0 |                      0 |
| FOODS_2_196_TX_2     | low    | intermittent |        0.622 |                      3 |                           48 |                      0.533 |                      4 |                    145 |
| HOUSEHOLD_2_315_TX_2 | low    | intermittent |        0.68  |                      3 |                           48 |                      0.438 |                    nan |                    nan |
| FOODS_2_056_WI_3     | high   | intermittent |        0.457 |                      3 |                          349 |                      4.148 |                     19 |                    591 |
| HOBBIES_1_312_WI_3   | high   | lumpy        |        0.356 |                      3 |                           93 |                      3.715 |                     17 |                    219 |
| FOODS_3_128_WI_3     | medium | intermittent |        0.372 |                      3 |                          100 |                      1.385 |                      4 |                    130 |
| HOBBIES_1_036_WI_3   | medium | lumpy        |        0.565 |                      3 |                           41 |                      0.887 |                      2 |                     54 |
| FOODS_3_520_WI_3     | low    | intermittent |        0.616 |                      3 |                           48 |                      0.58  |                      2 |                     67 |
| HOUSEHOLD_1_464_WI_3 | low    | intermittent |        0.774 |                      3 |                          107 |                      0.289 |                    nan |                    nan |

### Indicative stockout detector (Poisson zero-run test)
For series with mean demand >= 0.5/day a zero run of length *r* is called *suspicious* when its expected number of occurrences under an i.i.d. Poisson null is < 0.01. Real demand is over-dispersed, so this over-flags; it is an **indication**, not proof, of stockout-like spells. Informative for 15 of 18 selected series; total suspicious zero days = 2639. For low-volume series long zero spells are normal and carry no information.

**Observation:** 7 selected series have a zero spell of >= 90 consecutive store-open days in TRAIN while the item remains listed (see `longest_zero_run_open_days`). Such spells most likely reflect extended unavailability (out-of-stock or out-of-assortment) that the price table does not reveal, or a regime change in demand; the data cannot say which. Selection was **not** altered because of this: the series were drawn by the pre-defined algorithm and the same zero-spell behaviour is part of the real-world difficulty (regime changes in demand). It is carried into the robustness analysis (intermittent / lumpy series).

## Decision
All zero observations of listed items are **retained** as demand observations. The following limitation must be (and is) carried into the final report:

> **Observed sales may underestimate true demand during stockout periods.** Forecast accuracy is therefore measured against *observed sales*, and the inventory simulation treats observed sales as demand.

Consequences: (i) forecasts trained on observed sales can be biased low in exactly the situations that matter most for service levels; (ii) simulated stockout and fill-rate figures are conditional on observed (possibly censored) demand; (iii) a real deployment would need inventory records and censored-demand estimation (e.g. Tobit/Kaplan-Meier).
