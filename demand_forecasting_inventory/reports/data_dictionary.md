# Data dictionary

For every important variable: meaning, data type, intended use and **leakage risk**.
Data types are read from the loaded tables; semantic columns are curated.

| table | variable | meaning | dtype | intended use | leakage risk |
|---|---|---|---|---|---|
| calendar | `date` | Calendar date of the day | datetime64[us] | Index of the time axis; derive calendar features | None - known in advance |
| calendar | `wm_yr_wk` | Walmart week id (links days to weekly prices) | int64 | Join key to sell_prices | None - known in advance |
| calendar | `weekday / wday` | Day name / numeric day of week (1 = Saturday) | str | Weekly seasonality features | None - known in advance |
| calendar | `month, year` | Month and year of the date | int64 | Seasonality / trend-regime features | None - known in advance. `year` cannot extrapolate beyond the training range |
| calendar | `d` | Day index label `d_1`...`d_1969` | str | Join key to the wide sales table | None |
| calendar | `event_name_1/2, event_type_1/2` | Holiday / sporting / cultural / religious event of the day (second slot rarely used) | mixed | Event indicators (origin day and 'in forecast window' counts) | None for *future* days: the event calendar is published in advance (documented as known-ahead feature) |
| calendar | `snap_CA / snap_TX / snap_WI` | 1 if the state allows SNAP (food-stamp) purchases that day | int64 | State-specific SNAP indicator (FOODS demand uplift) | None - SNAP schedule is fixed by law in advance (known-ahead feature) |
| sell_prices | `store_id, item_id` | Store and item identifiers | category | Join keys | None |
| sell_prices | `wm_yr_wk` | Walmart week in which the price applied | int64 | Join key | None |
| sell_prices | `sell_price` | Weekly selling price in USD. **A row exists only for weeks in which the item was sold in that store** (absence = not on sale) | float32 | Price features; availability (listing) indicator; unit-cost reference for inventory costs | Moderate: prices of FUTURE weeks may reflect planned promotions. Only price at/before the forecast origin is used as a feature; the full-window price is used only for the listing/eligibility check |
| sales (wide) | `id` | `<item>_<store>_evaluation` unique series id | str / category | Series key | None |
| sales (wide) | `item_id, dept_id, cat_id` | Product, department (7) and category (HOBBIES/HOUSEHOLD/FOODS) | str / category | Stratification and grouping | None |
| sales (wide) | `store_id, state_id` | Store (10) and state (CA, TX, WI) | str / category | Stratification; selects the right SNAP flag | None |
| sales (wide) | `d_1 ... d_1941` | Unit sales of the item in the store on that day (non-negative integers) | int16 (non-negative count) | **TARGET** y_t = daily observed unit sales; basis for all lag/rolling features | HIGH if future values enter features/targets of earlier rows - guarded by tests/perturbation audit. Observed sales = censored demand (stockouts not recorded) |
| derived | `listed` | True when a price exists for the item/store in the day's week | float / bool (derived) | Eligibility, structural-zero detection, evaluation masks | None |
| derived | `store_outage` | True when the store's total sales that day are < 5 % of its rolling median (closure/outage) | float / bool (derived) | Zero-sales classification only (not a model feature) | Uses same-day sales -> never used as a feature |
| derived | `lag_k (k = 1,7,14,21,28)` | Sales k days before the first forecast day (lag_1 = last observed day) | float / bool (derived) | XGBoost feature | Safe if built with past values only (tested) |
| derived | `rolling_mean_w / rolling_std_w` | Mean / std of the last w observed days including the origin day | float / bool (derived) | XGBoost feature | Safe: window ends at the origin (tested) |
| derived | `price, price_change, relative_price` | Price at origin; week-over-week % change; price / mean price of the last 91 days | float / bool (derived) | XGBoost features | Safe: only price at or before the origin |
| derived | `snap_days_window, event_days_window, christmas_in_window` | Calendar counts over the forecast window t+1..t+H | float / bool (derived) | XGBoost known-ahead features | Deterministic functions of the calendar only (tested to be invariant to sales) |
| derived | `target_H` | Sum of daily sales over t+1...t+H (H = 3, 7, 14, 28) | float / bool (derived) | Direct multi-horizon target and the quantity every model is evaluated on | Is the label - must never enter features; training rows are cut so the window ends before the validation/test period |

## Key semantic facts
* **Target** `y_t`: observed daily unit sales of one item in one store (the data cannot distinguish zero demand from a stockout, see `zero_sales_analysis.md`).
* **Series id**: `<item_id>_<store_id>`; 30,490 series in total, 18 are modelled (see `subset_selection.md`).
* **Leakage rule of thumb**: a feature is admissible only if its value is determined by information available at the end of the forecast-origin day, *or* it is a deterministic function of the calendar (published in advance).
