"""Phase 3 - automatic inspection of every raw table + data dictionary.

All statistics in the generated markdown are computed from the files; nothing is typed by hand
except the *semantic* descriptions (meaning / intended use / leakage risk) in ``DICTIONARY``.
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from . import data_loading as dl


# --------------------------------------------------------------------------------------
# Semantic dictionary (meaning, intended use, leakage risk)
# --------------------------------------------------------------------------------------
DICTIONARY = [
    # table, variable, meaning, intended use, leakage risk
    ("calendar", "date", "Calendar date of the day", "Index of the time axis; derive calendar features", "None - known in advance"),
    ("calendar", "wm_yr_wk", "Walmart week id (links days to weekly prices)", "Join key to sell_prices", "None - known in advance"),
    ("calendar", "weekday / wday", "Day name / numeric day of week (1 = Saturday)", "Weekly seasonality features", "None - known in advance"),
    ("calendar", "month, year", "Month and year of the date", "Seasonality / trend-regime features", "None - known in advance. `year` cannot extrapolate beyond the training range"),
    ("calendar", "d", "Day index label `d_1`...`d_1969`", "Join key to the wide sales table", "None"),
    ("calendar", "event_name_1/2, event_type_1/2", "Holiday / sporting / cultural / religious event of the day (second slot rarely used)", "Event indicators (origin day and 'in forecast window' counts)", "None for *future* days: the event calendar is published in advance (documented as known-ahead feature)"),
    ("calendar", "snap_CA / snap_TX / snap_WI", "1 if the state allows SNAP (food-stamp) purchases that day", "State-specific SNAP indicator (FOODS demand uplift)", "None - SNAP schedule is fixed by law in advance (known-ahead feature)"),
    ("sell_prices", "store_id, item_id", "Store and item identifiers", "Join keys", "None"),
    ("sell_prices", "wm_yr_wk", "Walmart week in which the price applied", "Join key", "None"),
    ("sell_prices", "sell_price", "Weekly selling price in USD. **A row exists only for weeks in which the item was sold in that store** (absence = not on sale)", "Price features; availability (listing) indicator; unit-cost reference for inventory costs", "Moderate: prices of FUTURE weeks may reflect planned promotions. Only price at/before the forecast origin is used as a feature; the full-window price is used only for the listing/eligibility check"),
    ("sales (wide)", "id", "`<item>_<store>_evaluation` unique series id", "Series key", "None"),
    ("sales (wide)", "item_id, dept_id, cat_id", "Product, department (7) and category (HOBBIES/HOUSEHOLD/FOODS)", "Stratification and grouping", "None"),
    ("sales (wide)", "store_id, state_id", "Store (10) and state (CA, TX, WI)", "Stratification; selects the right SNAP flag", "None"),
    ("sales (wide)", "d_1 ... d_1941", "Unit sales of the item in the store on that day (non-negative integers)", "**TARGET** y_t = daily observed unit sales; basis for all lag/rolling features", "HIGH if future values enter features/targets of earlier rows - guarded by tests/perturbation audit. Observed sales = censored demand (stockouts not recorded)"),
    ("derived", "listed", "True when a price exists for the item/store in the day's week", "Eligibility, structural-zero detection, evaluation masks", "None"),
    ("derived", "store_outage", "True when the store's total sales that day are < 5 % of its rolling median (closure/outage)", "Zero-sales classification only (not a model feature)", "Uses same-day sales -> never used as a feature"),
    ("derived", "lag_k (k = 1,7,14,21,28)", "Sales k days before the first forecast day (lag_1 = last observed day)", "XGBoost feature", "Safe if built with past values only (tested)"),
    ("derived", "rolling_mean_w / rolling_std_w", "Mean / std of the last w observed days including the origin day", "XGBoost feature", "Safe: window ends at the origin (tested)"),
    ("derived", "price, price_change, relative_price", "Price at origin; week-over-week % change; price / mean price of the last 91 days", "XGBoost features", "Safe: only price at or before the origin"),
    ("derived", "snap_days_window, event_days_window, christmas_in_window", "Calendar counts over the forecast window t+1..t+H", "XGBoost known-ahead features", "Deterministic functions of the calendar only (tested to be invariant to sales)"),
    ("derived", "target_H", "Sum of daily sales over t+1...t+H (H = 3, 7, 14, 28)", "Direct multi-horizon target and the quantity every model is evaluated on", "Is the label - must never enter features; training rows are cut so the window ends before the validation/test period"),
]


def _describe_frame(name: str, df: pd.DataFrame, extra: dict | None = None) -> dict:
    dup = int(df.duplicated().sum())
    out = {
        "table": name,
        "rows": len(df),
        "columns": df.shape[1],
        "memory_MB": round(df.memory_usage(deep=True).sum() / 1e6, 1),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": dup,
    }
    if extra:
        out.update(extra)
    return out


def column_profile(name: str, df: pd.DataFrame, max_cols: int | None = None) -> pd.DataFrame:
    cols = list(df.columns) if max_cols is None else list(df.columns)[:max_cols]
    rows = []
    for c in cols:
        s = df[c]
        rows.append({"table": name, "column": c, "dtype": str(s.dtype), "missing": int(s.isna().sum()),
                     "unique": int(s.nunique(dropna=True))})
    return pd.DataFrame(rows)


def inspect_all(raw: dl.RawData | None = None) -> dict:
    """Run the full Phase 3 inspection; save tables and return a dict of results."""
    C.ensure_dirs()
    raw = raw or dl.load_raw()
    cal = raw.calendar
    sales_wide = dl.load_sales_wide("sales_evaluation")
    sales_val = dl.load_sales_wide("sales_validation")
    prices = dl.load_prices()

    d_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    n_days = len(d_cols)
    dates = raw.dates

    # ---- table-level summary ------------------------------------------------------
    summary = pd.DataFrame(
        [
            _describe_frame("calendar", cal.drop(columns=["d_num"]), {
                "date_min": str(cal["date"].min().date()), "date_max": str(cal["date"].max().date()),
                "unique_days": int(cal["date"].nunique())}),
            _describe_frame("sell_prices", prices, {
                "n_items": int(prices["item_id"].nunique()), "n_stores": int(prices["store_id"].nunique()),
                "n_weeks": int(prices["wm_yr_wk"].nunique())}),
            _describe_frame("sales_train_evaluation", sales_wide, {
                "date_min": str(dates.min().date()), "date_max": str(dates.max().date()),
                "n_items": int(sales_wide["item_id"].nunique()), "n_stores": int(sales_wide["store_id"].nunique()),
                "n_departments": int(sales_wide["dept_id"].nunique()), "n_categories": int(sales_wide["cat_id"].nunique()),
                "n_states": int(sales_wide["state_id"].nunique()), "n_day_columns": n_days}),
        ]
    )
    summary.to_csv(C.OUT_TAB / "data_inspection_summary.csv", index=False)

    profile = pd.concat(
        [column_profile("calendar", cal.drop(columns=["d_num"])),
         column_profile("sell_prices", prices),
         column_profile("sales_train_evaluation (id columns)", sales_wide[dl.ID_COLS])]
    )
    profile.to_csv(C.OUT_TAB / "data_inspection_columns.csv", index=False)

    # ---- sales-matrix level checks ------------------------------------------------
    Y = raw.Y
    ids_val = sales_val["id"].astype(str).str.replace("_validation", "", regex=False).to_numpy()
    ids_eval = sales_wide["id"].astype(str).str.replace("_evaluation", "", regex=False).to_numpy()
    val_cols = [c for c in sales_val.columns if c.startswith("d_")]
    Yv = sales_val[val_cols].to_numpy()
    checks = {
        "n_series": Y.shape[0],
        "n_days": n_days,
        "sales_min": int(Y.min()),
        "sales_max": int(Y.max()),
        "negative_sales_cells": int((Y < 0).sum()),
        "missing_sales_cells": int(sales_wide[d_cols].isna().sum().sum()),
        "zero_share_overall": float((Y == 0).mean()),
        "duplicate_series_ids": int(sales_wide["id"].duplicated().sum()),
        "duplicate_price_keys": int(prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).sum()),
        "calendar_days_contiguous": bool((cal["date"].diff().dropna() == pd.Timedelta(days=1)).all()),
        "calendar_duplicate_dates": int(cal["date"].duplicated().sum()),
        "validation_file_is_prefix_of_evaluation": bool((Yv == Y[:, : Yv.shape[1]]).all() and (ids_val == ids_eval).all()),
        "validation_file_days": int(Yv.shape[1]),
        "price_min": float(prices["sell_price"].min()),
        "price_max": float(prices["sell_price"].max()),
        "price_rows": len(prices),
    }
    listed = raw.listed_matrix()
    checks["sales_positive_while_unlisted"] = int(((Y > 0) & ~listed).sum())
    checks["share_series_days_listed"] = float(listed.mean())
    checks["zero_cells_listed"] = int(((Y == 0) & listed).sum())
    checks["zero_cells_unlisted"] = int(((Y == 0) & ~listed).sum())
    chk = pd.DataFrame([checks]).T.reset_index()
    chk.columns = ["check", "value"]
    chk.to_csv(C.OUT_TAB / "data_inspection_checks.csv", index=False)

    # ---- hierarchy unique values ----------------------------------------------------
    hier = {
        "stores": sorted(sales_wide["store_id"].astype(str).unique()),
        "states": sorted(sales_wide["state_id"].astype(str).unique()),
        "categories": sorted(sales_wide["cat_id"].astype(str).unique()),
        "departments": sorted(sales_wide["dept_id"].astype(str).unique()),
        "event_types": sorted(cal["event_type_1"].dropna().unique()),
        "n_event_names": int(cal["event_name_1"].nunique()),
    }
    by_cat = (pd.DataFrame({"cat": sales_wide["cat_id"].astype(str), "total": Y.sum(axis=1)})
              .groupby("cat")["total"].agg(["count", "sum"]).rename(columns={"count": "series", "sum": "units"}))
    by_store = (pd.DataFrame({"store": sales_wide["store_id"].astype(str), "total": Y.sum(axis=1)})
                .groupby("store")["total"].sum().rename("units").to_frame())
    by_cat.to_csv(C.OUT_TAB / "data_inspection_by_category.csv")
    by_store.to_csv(C.OUT_TAB / "data_inspection_by_store.csv")

    write_inspection_report(summary, profile, chk, hier, by_cat, by_store)
    write_data_dictionary(cal, prices, sales_wide)
    return {"summary": summary, "checks": checks, "hierarchy": hier}


def write_inspection_report(summary, profile, chk, hier, by_cat, by_store) -> None:
    lines = [
        "# Data inspection (Phase 3)",
        "",
        "Generated by `src/data_inspection.py` from the raw files in `data/raw/` (unchanged).",
        "",
        "## Table-level summary",
        "",
        summary.fillna("").to_markdown(index=False),
        "",
        "## Integrity and quality checks",
        "",
        chk.to_markdown(index=False),
        "",
        "## Hierarchy (unique values)",
        "",
        f"* Stores ({len(hier['stores'])}): {', '.join(hier['stores'])}",
        f"* States ({len(hier['states'])}): {', '.join(hier['states'])}",
        f"* Categories ({len(hier['categories'])}): {', '.join(hier['categories'])}",
        f"* Departments ({len(hier['departments'])}): {', '.join(hier['departments'])}",
        f"* Event types: {', '.join(hier['event_types'])}; distinct event names: {hier['n_event_names']}",
        "",
        "Units sold by category:",
        "",
        by_cat.to_markdown(),
        "",
        "Units sold by store:",
        "",
        by_store.to_markdown(),
        "",
        "## Column profile",
        "",
        profile.to_markdown(index=False),
        "",
        "## Reading guide",
        "* `sales_positive_while_unlisted = 0` means an item-store never sells in a week without a listed price, so "
        "the *absence of a price row is a clean signal for 'product not on sale'* (used in the zero-sales analysis).",
        "* `validation_file_is_prefix_of_evaluation = True` confirms that `sales_train_validation.csv` is exactly the first "
        "1,913 days of `sales_train_evaluation.csv`; only the latter (1,941 days) is used for modelling.",
    ]
    (C.REPORTS / "data_inspection.md").write_text("\n".join(lines) + "\n")


def write_data_dictionary(cal: pd.DataFrame, prices: pd.DataFrame, sales: pd.DataFrame) -> None:
    dtypes = {
        "calendar": cal.dtypes.astype(str).to_dict(),
        "sell_prices": prices.dtypes.astype(str).to_dict(),
    }
    lines = [
        "# Data dictionary",
        "",
        "For every important variable: meaning, data type, intended use and **leakage risk**.",
        "Data types are read from the loaded tables; semantic columns are curated.",
        "",
        "| table | variable | meaning | dtype | intended use | leakage risk |",
        "|---|---|---|---|---|---|",
    ]
    for table, var, meaning, use, risk in DICTIONARY:
        base = var.split(" ")[0].strip(",/")
        if table in dtypes and base in dtypes[table]:
            dt = dtypes[table][base]
        elif table == "sales (wide)" and base.startswith("d_"):
            dt = "int16 (non-negative count)"
        elif table == "derived":
            dt = "float / bool (derived)"
        else:
            dt = "str / category" if table != "calendar" else "mixed"
        lines.append(f"| {table} | `{var}` | {meaning} | {dt} | {use} | {risk} |")
    lines += [
        "",
        "## Key semantic facts",
        "* **Target** `y_t`: observed daily unit sales of one item in one store (the data cannot distinguish zero demand from "
        "a stockout, see `zero_sales_analysis.md`).",
        "* **Series id**: `<item_id>_<store_id>`; 30,490 series in total, 18 are modelled (see `subset_selection.md`).",
        "* **Leakage rule of thumb**: a feature is admissible only if its value is determined by information available at the "
        "end of the forecast-origin day, *or* it is a deterministic function of the calendar (published in advance).",
    ]
    (C.REPORTS / "data_dictionary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    res = inspect_all()
    print(res["summary"].to_string())
    print(pd.Series(res["checks"]).to_string())
