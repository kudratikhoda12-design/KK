# Data cleaning log (Phase 4)

Every decision is documented as **PROBLEM -> DIAGNOSIS -> ACTION -> REASON**. Numbers are computed by `src/data_prep.py`.
Principle: *do not blindly remove observations* - nothing is deleted or winsorised in this project.

## 1. Wide-to-long conversion
* **PROBLEM:** Sales are stored wide (one column per day, d_1...d_1941) and prices weekly in a separate table.
* **DIAGNOSIS:** 30,490 series x 1941 days = 59,181,090 cells; calendar links `d` and `wm_yr_wk`.
* **ACTION:** Melted the 18 selected series to long format (date, item, store, category/department, sales) and merged calendar (weekday, events, SNAP of the series' state) and the weekly price (via `wm_yr_wk`). Sorted by item, store, date.
* **REASON:** Modelling and feature engineering need a (series, date) panel; sorting guarantees that lags/rolling windows use past rows only.

## 2. Duplicates
* **PROBLEM:** Duplicated series, price keys or dates would double-count demand.
* **DIAGNOSIS:** duplicate series ids = 0, duplicate (store, item, week) price keys = 0, duplicate calendar dates = 0 (see data_inspection_checks.csv); duplicate (series, date) rows in the panel = 0.
* **ACTION:** None needed.
* **REASON:** No duplicates exist; asserted in code so a different data version would fail loudly.

## 3. Missing dates
* **PROBLEM:** Gaps in the daily time axis break lags and seasonality.
* **DIAGNOSIS:** Calendar is contiguous (1,969 consecutive days); every selected series has 1941-1941 rows (expected 1941); contiguous in every series = True.
* **ACTION:** None needed.
* **REASON:** The panel is complete, so lag k always means exactly k days.

## 4. Missing / negative sales
* **PROBLEM:** Missing or negative unit sales would indicate corrupted records or returns.
* **DIAGNOSIS:** missing sales = 0; negative sales = 0; min/max sales in the whole file = 0 / 763.
* **ACTION:** None needed.
* **REASON:** Data are clean on these dimensions.

## 5. Missing prices
* **PROBLEM:** Price rows are absent for weeks in which an item is not on sale (20.8 % of all series-days).
* **DIAGNOSIS:** Across the whole file prices are missing for 20.8 % of series-days and `sales > 0` never occurs on such days (0 cases) - the gap means *not sold*, not *unknown price*. For the selected series: missing price rows = 0, unlisted days = 0 (selection requires continuous listing).
* **ACTION:** No imputation. Eligibility rule requires continuous listing; a code assertion confirms no NaN price in the panel. (Fallback implemented for other data: past-only forward fill of prices.)
* **REASON:** Imputing prices of unlisted weeks would invent information. Restricting to continuously listed items removes the structural zeros that the price table can identify; the zeros that remain are ambiguous by nature (see entry 6).

## 6. Zero sales (68 % of all cells)
* **PROBLEM:** Most cells are zero. They can be (A) structural - item not listed, (B) store closed, (C) true zero demand or a stockout.
* **DIAGNOSIS:** Dataset-wide share of zero cells: (A) structural/unlisted = 30.6 %; (B) closure/outage = 0.32 %; (C) ambiguous = 69.1 %. Overall zero share = 68.0 %. Among the selected series, 7 of 18 contain zero spells of >= 90 consecutive store-open days in TRAIN although the item stays *listed* - the price table does not capture every unavailability. Details: `zero_sales_analysis.md`.
* **ACTION:** All zero observations of the selected (listed) series are **kept** as demand observations; closure days are flagged (`store_outage`) and handled through the calendar feature `christmas` for the models that can use it.
* **REASON:** Category C cannot be separated into 'no customer' and 'no stock' with this data; removing zeros would bias demand upward. Limitation stated in the final report: *observed sales may underestimate true demand during stockout periods*.

## 7. Store closures and outages (Christmas, unexplained)
* **PROBLEM:** On certain days every item of a store sells (almost) nothing.
* **DIAGNOSIS:** A store-day is flagged when store total sales < 5 % of its centred 29-day rolling median. Result: Dec 25 of 2011-2015 in all 10 stores (M5 calendar event `Christmas`), plus two unexplained store-wide outages: WI_1 on 2011-02-02 and TX_2 on 2015-03-24. Selected series' zero-days that fall on flagged days (train period) = 54; unexplained-outage series-days in the panel = 6.
* **ACTION:** Retained. Christmas is encoded as calendar information (`christmas_in_window`, `event_days_window`) for XGBoost; the univariate statistical models cannot use it (documented as a limitation of those models). Unexplained outages are retained and flagged.
* **REASON:** A closure is genuine zero sales that a model should anticipate (Christmas is known in advance); deleting it would hide a real pattern. Unexplained outages cannot be predicted, so they add the same noise for every model.

## 8. Outliers / demand spikes
* **PROBLEM:** Some days have unusually large sales.
* **DIAGNOSIS:** Tukey far-out fence on non-zero TRAIN demand (Q3 + 3 IQR): 204 flagged series-days of 24,444 (0.83 %). Median share of flagged days that fall on event/SNAP days = 53.6 % vs 38.3 % of all days.
* **ACTION:** No removal, no winsorising.
* **REASON:** Spikes are plausible demand (promotions, events, SNAP, bulk purchases) and are exactly the peaks inventory must cover; inventory decisions fail if they are clipped. Tree models are robust to them; RMSE is reported next to MAE to expose their influence.

## 9. Prices
* **PROBLEM:** Price outliers or unit errors would distort price features and unit-cost assumptions.
* **DIAGNOSIS:** Panel price range: 0.47 - 9.98 USD; prices are weekly and constant within a week.
* **ACTION:** None; price features use the price at the forecast origin only.
* **REASON:** No evidence of data errors for the selected items.

## 10. Raw data integrity
* **PROBLEM:** The official Kaggle download needs credentials.
* **DIAGNOSIS:** Two independent public mirrors provide byte-identical files (SHA-256 verified).
* **ACTION:** Used the mirror; raw files are never modified (all processing writes to `data/processed/`).
* **REASON:** See `data_source.md`.

