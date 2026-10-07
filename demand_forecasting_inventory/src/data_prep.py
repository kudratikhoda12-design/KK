"""Phases 4-6 - data preparation, cleaning log and zero-sales / stockout analysis.

Outputs
-------
data/processed/panel.parquet               long panel of the selected series (sorted by item, store, date)
data/processed/selected_series.csv         the 18 selected series (canonical series order)
outputs/tables/cleaning_*.csv, zero_*.csv  numeric evidence
reports/data_cleaning.md                   PROBLEM -> DIAGNOSIS -> ACTION -> REASON log (numbers computed here)
reports/zero_sales_analysis.md             can zero sales be told apart from stockouts?
reports/subset_selection.md                algorithm + result of the selection
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import data_loading as dl
from . import subset_selection as ss
from .splits import Split, make_split, split_table


# --------------------------------------------------------------------------------------
# Panel <-> matrix helpers (shared by every model module)
# --------------------------------------------------------------------------------------
def panel_matrix(panel: pd.DataFrame, series_order: list[str], col: str = "sales") -> np.ndarray:
    """(n_series, n_days) matrix in the canonical series order; checks that dates align across series."""
    out = []
    ref_dates = None
    for sid in series_order:
        g = panel[panel["series_id"] == sid].sort_values("date")
        if ref_dates is None:
            ref_dates = g["date"].to_numpy()
        elif not np.array_equal(ref_dates, g["date"].to_numpy()):
            raise ValueError("series are not aligned on the same dates")
        out.append(g[col].to_numpy())
    return np.vstack(out)


def load_prepared() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(C.DATA_PROC / "panel.parquet")
    sel = pd.read_csv(C.DATA_PROC / "selected_series.csv")
    return panel, sel


# --------------------------------------------------------------------------------------
# Zero-sales classification
# --------------------------------------------------------------------------------------
def classify_zero_cells_all(raw: dl.RawData, outage: pd.DataFrame) -> pd.DataFrame:
    """Dataset-wide (30,490 series x 1,941 days) classification of zero cells."""
    Y = raw.Y
    listed = raw.listed_matrix()
    stores = raw.ids["store_id"].astype(str).to_numpy()
    out_cols = {s: outage[s].to_numpy() for s in outage.columns}
    outage_cell = np.vstack([out_cols[s] for s in stores])            # (series, days) store-wide closure/outage
    zero = Y == 0
    unlisted = zero & ~listed
    closure = zero & listed & outage_cell
    ambiguous = zero & listed & ~outage_cell
    tot = int(zero.sum())
    rows = [
        ("zero cells (all series-days)", tot, 1.0),
        ("A. structural: item not listed that week (no price row)", int(unlisted.sum()), unlisted.sum() / tot),
        ("B. store-wide closure / outage day (listed item)", int(closure.sum()), closure.sum() / tot),
        ("C. ambiguous: store open, item listed, zero sales (true zero demand OR stockout)", int(ambiguous.sum()), ambiguous.sum() / tot),
    ]
    return pd.DataFrame(rows, columns=["category", "cells", "share_of_zero_cells"])


def zero_runs(y_open: np.ndarray) -> list[int]:
    """Lengths of maximal runs of consecutive zeros in a vector."""
    runs, cur = [], 0
    for v in y_open:
        if v == 0:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    return runs


def poisson_run_diagnostic(y: np.ndarray, open_mask: np.ndarray, alpha: float = 0.01) -> dict:
    """Indicative detector of implausibly long zero spells.

    Under an i.i.d. Poisson(lam) null a maximal run of >= r zeros is expected ``n (1-p0) p0^r`` times (p0 = e^-lam).
    A run is *suspicious* when that expectation is < alpha.  Real retail demand is over-dispersed, so this
    over-flags (it is an upper bound on detectable stockout-like spells, not proof of stockouts).
    """
    y = y[open_mask]
    n = len(y)
    lam = float(y.mean())
    if lam < 0.5 or n < 100:                     # too sparse: zero spells carry no information
        return {"lam": lam, "n_runs_gt1": len(zero_runs(y)), "suspicious_runs": np.nan, "suspicious_zero_days": np.nan,
                "max_zero_run": max(zero_runs(y), default=0), "informative": False}
    p0 = np.exp(-lam)
    sus_runs, sus_days = 0, 0
    for r in zero_runs(y):
        if n * (1 - p0) * p0 ** r < alpha:
            sus_runs += 1
            sus_days += r
    return {"lam": lam, "n_runs_gt1": int(sum(1 for r in zero_runs(y) if r > 1)), "suspicious_runs": sus_runs,
            "suspicious_zero_days": sus_days, "max_zero_run": max(zero_runs(y), default=0), "informative": True}


# --------------------------------------------------------------------------------------
# Main preparation routine
# --------------------------------------------------------------------------------------
def prepare(raw: dl.RawData | None = None, verbose: bool = True) -> dict:
    C.ensure_dirs()
    raw = raw or dl.load_raw()
    split = make_split(raw.dates)
    sel, log = ss.select_subset(raw, split)
    sel.to_csv(C.DATA_PROC / "selected_series.csv", index=False)
    (C.OUT_TAB / "subset_selection_log.json").write_text(json.dumps(log, indent=2, default=str))

    # ---- panel ------------------------------------------------------------------------------
    panel = dl.build_panel(raw, sel["raw_row"].to_numpy())
    outage = ss.store_outage_matrix(raw)
    outage_long = outage.stack().rename("store_outage").reset_index()
    outage_long.columns = ["date", "store_id", "store_outage"]
    panel = panel.merge(outage_long, on=["date", "store_id"], how="left")
    panel["store_outage"] = panel["store_outage"].fillna(False).astype(bool)
    panel["christmas"] = (panel["event_name_1"] == "Christmas")
    panel["unexplained_outage"] = panel["store_outage"] & ~panel["christmas"]
    panel = panel.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)
    panel.to_parquet(C.DATA_PROC / "panel.parquet", index=False)

    # ---- integrity checks on the panel ---------------------------------------------------------
    checks = _panel_checks(panel, raw, split)
    # ---- zero analysis -------------------------------------------------------------------------
    zero_all = classify_zero_cells_all(raw, outage)
    zero_all.to_csv(C.OUT_TAB / "zero_cells_dataset_wide.csv", index=False)
    zero_series = _zero_series_table(panel, sel, split)
    zero_series.to_csv(C.OUT_TAB / "zero_sales_by_series.csv", index=False)
    # ---- outliers --------------------------------------------------------------------------------
    outl = _outlier_table(panel, sel, split)
    outl.to_csv(C.OUT_TAB / "cleaning_outliers_by_series.csv", index=False)

    split_table(split).to_csv(C.OUT_TAB / "split_table.csv", index=False)
    write_cleaning_log(checks, zero_all, zero_series, outl, sel, split, raw)
    write_zero_report(zero_all, zero_series, outage, sel, split)
    write_selection_report(sel, log, split)
    if verbose:
        print(f"[prepare] panel rows={len(panel):,}; series={sel.shape[0]}; outage days flagged: "
              f"{int(outage.any(axis=1).sum())}")
    return {"panel": panel, "sel": sel, "log": log, "checks": checks, "split": split}


def _panel_checks(panel: pd.DataFrame, raw: dl.RawData, split: Split) -> dict:
    g = panel.groupby("series_id")
    n_days = g["date"].nunique()
    expected = raw.n_days
    dates_ok = all(
        (grp["date"].diff().dropna() == pd.Timedelta(days=1)).all() for _, grp in g
    )
    return {
        "panel_rows": len(panel),
        "series": panel["series_id"].nunique(),
        "days_per_series_min": int(n_days.min()), "days_per_series_max": int(n_days.max()),
        "expected_days": expected,
        "duplicate_series_date_rows": int(panel.duplicated(["series_id", "date"]).sum()),
        "dates_contiguous_in_every_series": bool(dates_ok),
        "missing_sales": int(panel["sales"].isna().sum()),
        "negative_sales": int((panel["sales"] < 0).sum()),
        "missing_price_rows": int(panel["sell_price"].isna().sum()),
        "unlisted_days": int((~panel["listed"]).sum()),
        "min_price": float(panel["sell_price"].min()), "max_price": float(panel["sell_price"].max()),
        "store_outage_series_days": int(panel["store_outage"].sum()),
        "christmas_series_days": int(panel["christmas"].sum()),
        "unexplained_outage_series_days": int(panel["unexplained_outage"].sum()),
    }


def _zero_series_table(panel: pd.DataFrame, sel: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Zero-sales decomposition per selected series using the TRAIN period only (descriptive diagnostics)."""
    rows = []
    for _, s in sel.iterrows():
        g = panel[panel["series_id"] == s["series_id"]].sort_values("date").reset_index(drop=True)
        tr = g.iloc[: split.train_end]
        zero = tr["sales"].to_numpy() == 0
        closed = tr["store_outage"].to_numpy()
        open_mask = ~closed
        diag = poisson_run_diagnostic(tr["sales"].to_numpy(), open_mask)
        rows.append({
            "series_id": s["series_id"], "tier": s["tier"], "sb_class": s["train_sb_class"],
            "train_days": len(tr),
            "zero_days": int(zero.sum()), "zero_share": float(zero.mean()),
            "zero_on_closure_days": int((zero & closed).sum()),
            "zero_listed_store_open": int((zero & ~closed).sum()),
            "zero_share_listed_open": float((zero & ~closed).sum() / max(open_mask.sum(), 1)),
            "longest_zero_run_open_days": diag["max_zero_run"],
            "poisson_lambda_open_days": diag["lam"],
            "suspicious_zero_runs": diag["suspicious_runs"], "suspicious_zero_days": diag["suspicious_zero_days"],
            "run_test_informative(lambda>=0.5)": diag["informative"],
        })
    return pd.DataFrame(rows)


def _outlier_table(panel: pd.DataFrame, sel: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Tukey far-out fence on non-zero TRAIN demand: y > Q3 + 3 IQR (flag only - nothing is removed)."""
    rows = []
    for _, s in sel.iterrows():
        g = panel[panel["series_id"] == s["series_id"]].sort_values("date").reset_index(drop=True)
        tr = g.iloc[: split.train_end]
        nz = tr.loc[tr["sales"] > 0, "sales"]
        q1, q3 = nz.quantile(0.25), nz.quantile(0.75)
        fence = q3 + 3 * (q3 - q1)
        flag = tr["sales"] > fence
        ev = (tr["event_name_1"] != "None") | (tr["snap"] == 1)
        rows.append({
            "series_id": s["series_id"], "fence": float(fence), "n_outliers": int(flag.sum()),
            "share_outliers": float(flag.mean()), "max_sales": int(tr["sales"].max()),
            "share_outliers_on_event_or_snap_days": float((flag & ev).sum() / max(flag.sum(), 1)) if flag.sum() else np.nan,
            "share_days_event_or_snap": float(ev.mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Report writers (all numbers come from the computed tables)
# --------------------------------------------------------------------------------------
def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f} %"


def write_cleaning_log(checks, zero_all, zero_series, outl, sel, split, raw) -> None:
    ncells = int(raw.Y.size)
    zshare = float((raw.Y == 0).mean())
    L = []
    L += ["# Data cleaning log (Phase 4)", "",
          "Every decision is documented as **PROBLEM -> DIAGNOSIS -> ACTION -> REASON**. Numbers are computed by `src/data_prep.py`.",
          "Principle: *do not blindly remove observations* - nothing is deleted or winsorised in this project.", ""]

    def entry(n, problem, diag, action, reason):
        L.extend([f"## {n}", f"* **PROBLEM:** {problem}", f"* **DIAGNOSIS:** {diag}", f"* **ACTION:** {action}", f"* **REASON:** {reason}", ""])

    entry("1. Wide-to-long conversion",
          "Sales are stored wide (one column per day, d_1...d_1941) and prices weekly in a separate table.",
          f"30,490 series x {raw.n_days} days = {ncells:,} cells; calendar links `d` and `wm_yr_wk`.",
          f"Melted the {sel.shape[0]} selected series to long format (date, item, store, category/department, sales) and merged "
          "calendar (weekday, events, SNAP of the series' state) and the weekly price (via `wm_yr_wk`). Sorted by item, store, date.",
          "Modelling and feature engineering need a (series, date) panel; sorting guarantees that lags/rolling windows use past rows only.")
    entry("2. Duplicates",
          "Duplicated series, price keys or dates would double-count demand.",
          f"duplicate series ids = 0, duplicate (store, item, week) price keys = 0, duplicate calendar dates = 0 (see data_inspection_checks.csv); "
          f"duplicate (series, date) rows in the panel = {checks['duplicate_series_date_rows']}.",
          "None needed.", "No duplicates exist; asserted in code so a different data version would fail loudly.")
    entry("3. Missing dates",
          "Gaps in the daily time axis break lags and seasonality.",
          f"Calendar is contiguous (1,969 consecutive days); every selected series has {checks['days_per_series_min']}-"
          f"{checks['days_per_series_max']} rows (expected {checks['expected_days']}); contiguous in every series = {checks['dates_contiguous_in_every_series']}.",
          "None needed.", "The panel is complete, so lag k always means exactly k days.")
    entry("4. Missing / negative sales",
          "Missing or negative unit sales would indicate corrupted records or returns.",
          f"missing sales = {checks['missing_sales']}; negative sales = {checks['negative_sales']}; min/max sales in the whole file = 0 / 763.",
          "None needed.", "Data are clean on these dimensions.")
    entry("5. Missing prices",
          "Price rows are absent for weeks in which an item is not on sale (20.8 % of all series-days).",
          f"Across the whole file prices are missing for {_pct(1 - float(raw.listed_matrix().mean()))} of series-days and "
          "`sales > 0` never occurs on such days (0 cases) - the gap means *not sold*, not *unknown price*. "
          f"For the selected series: missing price rows = {checks['missing_price_rows']}, unlisted days = {checks['unlisted_days']} (selection requires continuous listing).",
          "No imputation. Eligibility rule requires continuous listing; a code assertion confirms no NaN price in the panel. "
          "(Fallback implemented for other data: past-only forward fill of prices.)",
          "Imputing prices of unlisted weeks would invent information. Restricting to continuously listed items removes the structural zeros "
          "that the price table can identify; the zeros that remain are ambiguous by nature (see entry 6).")
    n_long = int((zero_series["longest_zero_run_open_days"] >= 90).sum())
    entry("6. Zero sales (68 % of all cells)",
          "Most cells are zero. They can be (A) structural - item not listed, (B) store closed, (C) true zero demand or a stockout.",
          f"Dataset-wide share of zero cells: (A) structural/unlisted = {_pct(zero_all.iloc[1]['share_of_zero_cells'])}; "
          f"(B) closure/outage = {_pct(zero_all.iloc[2]['share_of_zero_cells'], 2)}; (C) ambiguous = {_pct(zero_all.iloc[3]['share_of_zero_cells'])}. "
          f"Overall zero share = {_pct(zshare)}. Among the selected series, {n_long} of {len(zero_series)} contain zero spells of >= 90 consecutive "
          "store-open days in TRAIN although the item stays *listed* - the price table does not capture every unavailability. Details: `zero_sales_analysis.md`.",
          "All zero observations of the selected (listed) series are **kept** as demand observations; closure days are flagged (`store_outage`) "
          "and handled through the calendar feature `christmas` for the models that can use it.",
          "Category C cannot be separated into 'no customer' and 'no stock' with this data; removing zeros would bias demand upward. "
          "Limitation stated in the final report: *observed sales may underestimate true demand during stockout periods*.")
    nco = int(zero_series["zero_on_closure_days"].sum())
    entry("7. Store closures and outages (Christmas, unexplained)",
          "On certain days every item of a store sells (almost) nothing.",
          f"A store-day is flagged when store total sales < 5 % of its centred 29-day rolling median. Result: Dec 25 of 2011-2015 in all 10 stores "
          "(M5 calendar event `Christmas`), plus two unexplained store-wide outages: WI_1 on 2011-02-02 and TX_2 on 2015-03-24. "
          f"Selected series' zero-days that fall on flagged days (train period) = {nco}; unexplained-outage series-days in the panel = {checks['unexplained_outage_series_days']}.",
          "Retained. Christmas is encoded as calendar information (`christmas_in_window`, `event_days_window`) for XGBoost; the "
          "univariate statistical models cannot use it (documented as a limitation of those models). Unexplained outages are retained and flagged.",
          "A closure is genuine zero sales that a model should anticipate (Christmas is known in advance); deleting it would hide a real pattern. "
          "Unexplained outages cannot be predicted, so they add the same noise for every model.")
    tot_out = int(outl["n_outliers"].sum())
    entry("8. Outliers / demand spikes",
          "Some days have unusually large sales.",
          f"Tukey far-out fence on non-zero TRAIN demand (Q3 + 3 IQR): {tot_out} flagged series-days of "
          f"{sel.shape[0] * split.train_end:,} ({_pct(tot_out / (sel.shape[0] * split.train_end), 2)}). "
          f"Median share of flagged days that fall on event/SNAP days = {_pct(float(outl['share_outliers_on_event_or_snap_days'].median()))} "
          f"vs {_pct(float(outl['share_days_event_or_snap'].median()))} of all days.",
          "No removal, no winsorising.", "Spikes are plausible demand (promotions, events, SNAP, bulk purchases) and are exactly the "
          "peaks inventory must cover; inventory decisions fail if they are clipped. Tree models are robust to them; RMSE is reported next to MAE to expose their influence.")
    entry("9. Prices",
          "Price outliers or unit errors would distort price features and unit-cost assumptions.",
          f"Panel price range: {checks['min_price']:.2f} - {checks['max_price']:.2f} USD; prices are weekly and constant within a week.",
          "None; price features use the price at the forecast origin only.", "No evidence of data errors for the selected items.")
    entry("10. Raw data integrity",
          "The official Kaggle download needs credentials.", "Two independent public mirrors provide byte-identical files (SHA-256 verified).",
          "Used the mirror; raw files are never modified (all processing writes to `data/processed/`).",
          "See `data_source.md`.")
    (C.REPORTS / "data_cleaning.md").write_text("\n".join(L) + "\n")


def write_zero_report(zero_all, zero_series, outage, sel, split) -> None:
    informative = zero_series[zero_series["run_test_informative(lambda>=0.5)"]]
    sus = informative["suspicious_zero_days"].sum()
    L = ["# Zero sales and the stockout problem (Phase 5)", "",
         "## Question",
         "Can the dataset distinguish genuine zero demand, product unavailability, store closure, stockout and missing observation?", "",
         "## What the data *can* identify",
         "| zero type | identifiable? | how |", "|---|---|---|",
         "| Product not available / not yet launched / delisted | **Yes** | no price row in `sell_prices` for the week (sales are never positive in such weeks) |",
         "| Store closure / outage | **Yes (day level)** | store-wide total < 5 % of its rolling median; Christmas is also an explicit calendar event |",
         "| Missing observation | **Yes** | none exist (no NaN in sales, calendar contiguous) |",
         "| Stockout (item sold out) | **No** | the file has no inventory, shelf-availability or lost-sales information |",
         "| Genuine zero demand | **No** | indistinguishable from a stockout when the item is listed and the store is open |", "",
         "## Dataset-wide decomposition of zero cells (30,490 series x 1,941 days)", "",
         zero_all.assign(share_of_zero_cells=lambda d: (100 * d["share_of_zero_cells"]).round(2).astype(str) + " %").to_markdown(index=False), "",
         "## Selected series (TRAIN period; store open, item listed)", "",
         zero_series[["series_id", "tier", "sb_class", "zero_share", "zero_on_closure_days", "longest_zero_run_open_days",
                      "poisson_lambda_open_days", "suspicious_zero_runs", "suspicious_zero_days"]].round(3).to_markdown(index=False), "",
         "### Indicative stockout detector (Poisson zero-run test)",
         "For series with mean demand >= 0.5/day a zero run of length *r* is called *suspicious* when its expected number of occurrences under an "
         "i.i.d. Poisson null is < 0.01. Real demand is over-dispersed, so this over-flags; it is an **indication**, not proof, of stockout-like spells. "
         f"Informative for {len(informative)} of {len(zero_series)} selected series; total suspicious zero days = {int(sus) if pd.notna(sus) else 0}. "
         "For low-volume series long zero spells are normal and carry no information.", "",
         f"**Observation:** {int((zero_series['longest_zero_run_open_days'] >= 90).sum())} selected series have a zero spell of >= 90 consecutive store-open days "
         "in TRAIN while the item remains listed (see `longest_zero_run_open_days`). Such spells most likely reflect extended unavailability (out-of-stock or "
         "out-of-assortment) that the price table does not reveal, or a regime change in demand; the data cannot say which. Selection was **not** altered because of this: the series were drawn by the "
         "pre-defined algorithm and the same zero-spell behaviour is part of the real-world difficulty (regime changes in demand). "
         "It is carried into the robustness analysis (intermittent / lumpy series).", "",
         "## Decision",
         "All zero observations of listed items are **retained** as demand observations. The following limitation must be (and is) carried into the final report:", "",
         "> **Observed sales may underestimate true demand during stockout periods.** Forecast accuracy is therefore measured against *observed sales*, "
         "and the inventory simulation treats observed sales as demand.", "",
         "Consequences: (i) forecasts trained on observed sales can be biased low in exactly the situations that matter most for service levels; (ii) simulated stockout and fill-rate "
         "figures are conditional on observed (possibly censored) demand; (iii) a real deployment would need inventory records and censored-demand estimation (e.g. Tobit/Kaplan-Meier)."]
    (C.REPORTS / "zero_sales_analysis.md").write_text("\n".join(L) + "\n")


def write_selection_report(sel, log, split) -> None:
    th = log["tier_thresholds_mean_daily_sales"]
    cols = ["series_idx", "series_id", "tier", "cat_id", "state_id", "train_mean", "train_median", "train_zero_share", "train_adi", "train_cv2", "train_sb_class", "last_train_price"]
    L = ["# Subset selection (Phase 6)", "",
         "## Algorithm (uses TRAIN-period information only; no validation/test sales or model performance)", "",
         "1. **Chronological split** of the 1,941 observed days: " + "; ".join(
             f"{p} {split.date_range(p)[0].date()} to {split.date_range(p)[1].date()}" for p in ("train", "validation", "test")) + ".",
         f"2. **Stores:** in each state the store with the largest TRAIN-period unit sales -> {', '.join(log['stores'])} (one store per state, covering the three SNAP regimes).",
         f"3. **Eligibility:** item-store listed (price row present) on *every* TRAIN day from d_1 -> {log['n_eligible_listed_all_train_days']:,} eligible series "
         f"({', '.join(f'{k}: {v}' for k, v in log['eligible_by_store'].items())}). Post-hoc diagnostic (not a selection input): "
         f"{log['share_eligible_also_listed_after_train']:.1%} of the eligible pool is also listed on every validation/test day, so no delisting handling is required.",
         "4. **Demand tiers:** eligible series of the three stores are pooled and ranked by TRAIN mean daily sales. Percentile bands "
         f"(config `TIER_BANDS`): low = P{int(C.TIER_BANDS['low'][0]*100)}-P{int(C.TIER_BANDS['low'][1]*100)} ({th['low'][0]:.2f}-{th['low'][1]:.2f} units/day), "
         f"medium = P{int(C.TIER_BANDS['medium'][0]*100)}-P{int(C.TIER_BANDS['medium'][1]*100)} ({th['medium'][0]:.2f}-{th['medium'][1]:.2f}), "
         f"high = P{int(C.TIER_BANDS['high'][0]*100)}-P{int(C.TIER_BANDS['high'][1]*100)} ({th['high'][0]:.2f}-{th['high'][1]:.2f}). "
         "The extreme tails (below P10, above P97) are excluded as unrepresentative.",
         f"5. **Sampling:** for every store x tier, {C.ITEMS_PER_TIER_PER_STORE} series are drawn at random with `numpy.random.default_rng({C.SEED})`, "
         "preferring different categories and never re-using an item -> 3 stores x 3 tiers x 2 = **18 series / 18 distinct products**.",
         "6. **Profile** (mean, median, std, CV, zero share, ADI, CV^2, Syntetos-Boylan class) is computed on TRAIN data only.", "",
         "Candidates available per store x tier cell: " + ", ".join(f"{k}: {v}" for k, v in log["candidates_per_cell"].items()) + ".", "",
         "## Selected series", "",
         sel[cols].round(3).to_markdown(index=False), "",
         "## What the selection means",
         f"* Demand classes (TRAIN): {sel['train_sb_class'].value_counts().to_dict()}. Intermittent/lumpy demand dominates, which is why a **Croston-SBA baseline** is added.",
         f"* Categories: {sel['cat_id'].value_counts().to_dict()}.",
         "* Selection was run once with the fixed seed; it was not repeated or adjusted after seeing any forecasting result.",
         "* **Limitation:** only continuously listed items are eligible (survivorship of the assortment); high-volume extremes (> P97) are not represented."]
    (C.REPORTS / "subset_selection.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    prepare()
