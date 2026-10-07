"""Loading of the raw M5 files and construction of the (series x day) arrays used everywhere else.

Conventions
-----------
* Day axis: 0-based position ``j`` corresponds to M5 day ``d_{j+1}`` (``d_1`` = 2011-01-29).
* Series axis: row order of ``sales_train_evaluation.csv`` (30,490 item-store series).
* ``listed[s, j]`` is True when a selling price exists for the series' store/item in the Walmart week of day j.
  In M5 an item-store has *no* price row for weeks in which it is not sold, so this is the (only) availability signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


# --------------------------------------------------------------------------------------
# Individual loaders
# --------------------------------------------------------------------------------------
def load_calendar() -> pd.DataFrame:
    cal = pd.read_csv(C.DATA_RAW / C.RAW_FILES["calendar"])
    cal["date"] = pd.to_datetime(cal["date"])
    cal["d_num"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    return cal


def load_sales_wide(kind: str = "sales_evaluation") -> pd.DataFrame:
    """Wide sales table (one row per series, one column per day).  Counts are downcast to int16."""
    df = pd.read_csv(C.DATA_RAW / C.RAW_FILES[kind], engine="pyarrow")
    dcols = [c for c in df.columns if c.startswith("d_")]
    df[dcols] = df[dcols].astype("int16")
    for c in ID_COLS:
        df[c] = df[c].astype("category")
    return df


def load_prices() -> pd.DataFrame:
    pr = pd.read_csv(C.DATA_RAW / C.RAW_FILES["prices"], engine="pyarrow")
    pr["sell_price"] = pr["sell_price"].astype("float32")
    pr["store_id"] = pr["store_id"].astype("category")
    pr["item_id"] = pr["item_id"].astype("category")
    return pr


# --------------------------------------------------------------------------------------
# Container with aligned arrays
# --------------------------------------------------------------------------------------
@dataclass
class RawData:
    calendar: pd.DataFrame           # 1,969 rows
    ids: pd.DataFrame                # 30,490 x 6 identifier columns (category dtype)
    Y: np.ndarray                    # (30490, n_days) int16 observed unit sales
    price_week: np.ndarray           # (30490, n_weeks) float32, NaN when not listed
    week_ids: np.ndarray             # sorted Walmart week ids covering the calendar
    day_week_idx: np.ndarray         # (n_cal_days,) index of each calendar day's week in week_ids
    n_days: int                      # observed sales days (1,941)

    # ---- derived views -----------------------------------------------------------
    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.calendar["date"].iloc[: self.n_days])

    def listed_matrix(self) -> np.ndarray:
        """(30490, n_days) boolean: price listed in that day's week."""
        return ~np.isnan(self.price_week[:, self.day_week_idx[: self.n_days]])

    def price_matrix(self) -> np.ndarray:
        """(30490, n_cal_days) float32 daily price (NaN when not listed); covers the full calendar."""
        return self.price_week[:, self.day_week_idx]

    def series_index(self, store_id: str, item_id: str) -> int:
        m = (self.ids["store_id"] == store_id) & (self.ids["item_id"] == item_id)
        idx = np.flatnonzero(m.to_numpy())
        if len(idx) != 1:
            raise KeyError((store_id, item_id))
        return int(idx[0])


def load_raw(kind: str = "sales_evaluation") -> RawData:
    cal = load_calendar()
    sales = load_sales_wide(kind)
    dcols = [c for c in sales.columns if c.startswith("d_")]
    Y = sales[dcols].to_numpy(dtype=np.int16)
    ids = sales[ID_COLS].copy()
    pr = load_prices()

    week_ids = np.sort(cal["wm_yr_wk"].unique())
    day_week_idx = np.searchsorted(week_ids, cal["wm_yr_wk"].to_numpy())

    stores = list(ids["store_id"].cat.categories)
    items = list(ids["item_id"].cat.categories)
    lookup = np.full((len(stores), len(items)), -1, dtype=np.int64)
    s_code = ids["store_id"].cat.codes.to_numpy()
    i_code = ids["item_id"].cat.codes.to_numpy()
    lookup[s_code, i_code] = np.arange(len(ids))

    p_store = pd.Categorical(pr["store_id"].astype(str), categories=stores).codes
    p_item = pd.Categorical(pr["item_id"].astype(str), categories=items).codes
    if (p_store < 0).any() or (p_item < 0).any():
        raise ValueError("price table contains store/item ids that are absent from the sales table")
    p_series = lookup[p_store, p_item]
    if (p_series < 0).any():
        raise ValueError("price rows for series that do not exist in the sales table")
    p_week = np.searchsorted(week_ids, pr["wm_yr_wk"].to_numpy())
    price_week = np.full((len(ids), len(week_ids)), np.nan, dtype=np.float32)
    price_week[p_series, p_week] = pr["sell_price"].to_numpy()
    return RawData(cal, ids, Y, price_week, week_ids, day_week_idx, Y.shape[1])


# --------------------------------------------------------------------------------------
# Long-format panel for a chosen set of series
# --------------------------------------------------------------------------------------
STATE_SNAP = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}


def build_panel(raw: RawData, series_rows: np.ndarray) -> pd.DataFrame:
    """Long format: one row per (series, day) for the selected series, sorted by item, store, date.

    Columns: series_id, item_id, store_id, dept_id, cat_id, state_id, date, d_num, sales, sell_price, listed,
    weekday, wday, month, year, event_name_1/2, event_type_1/2, snap.
    """
    n = raw.n_days
    cal = raw.calendar.iloc[:n].reset_index(drop=True)
    price_day = raw.price_matrix()[:, :n]
    frames = []
    for r in series_rows:
        meta = raw.ids.iloc[r]
        state = str(meta["state_id"])
        df = pd.DataFrame(
            {
                "series_id": f"{meta['item_id']}_{meta['store_id']}",
                "item_id": str(meta["item_id"]),
                "store_id": str(meta["store_id"]),
                "dept_id": str(meta["dept_id"]),
                "cat_id": str(meta["cat_id"]),
                "state_id": state,
                "date": cal["date"].to_numpy(),
                "d_num": cal["d_num"].to_numpy(),
                "sales": raw.Y[r].astype(np.int32),
                "sell_price": price_day[r].astype(np.float64),
                "weekday": cal["weekday"].to_numpy(),
                "wday": cal["wday"].to_numpy(),
                "month": cal["month"].to_numpy(),
                "year": cal["year"].to_numpy(),
                "event_name_1": cal["event_name_1"].fillna("None").to_numpy(),
                "event_type_1": cal["event_type_1"].fillna("None").to_numpy(),
                "event_name_2": cal["event_name_2"].fillna("None").to_numpy(),
                "event_type_2": cal["event_type_2"].fillna("None").to_numpy(),
                "snap": cal[STATE_SNAP[state]].to_numpy(),
            }
        )
        df["listed"] = ~np.isnan(df["sell_price"].to_numpy())
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)


def full_calendar_features(raw: RawData) -> pd.DataFrame:
    """Calendar for ALL 1,969 days (includes the 28 days after the last observed sale)."""
    cal = raw.calendar.copy()
    for c in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        cal[c] = cal[c].fillna("None")
    return cal
