"""Data loading, wide-to-long conversion, subset selection and panel integrity."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import data_loading as dl
from src import subset_selection as ss
from tests.conftest import needs_prepared


def _tiny_raw():
    """Two series, 14 days, three weekly price rows, SNAP flags for CA."""
    n_days = 14
    dates = pd.date_range("2011-01-29", periods=n_days + 28)
    cal = pd.DataFrame({
        "date": dates, "wm_yr_wk": np.repeat(np.arange(11101, 11101 + (n_days + 28) // 7 + 1), 7)[: n_days + 28],
        "weekday": dates.day_name(), "wday": (np.arange(n_days + 28) % 7) + 1, "month": dates.month, "year": dates.year,
        "d": [f"d_{i + 1}" for i in range(n_days + 28)], "d_num": np.arange(1, n_days + 29),
        "event_name_1": [np.nan] * 5 + ["Christmas"] + [np.nan] * (n_days + 22), "event_type_1": [np.nan] * 5 + ["National"] + [np.nan] * (n_days + 22),
        "event_name_2": np.nan, "event_type_2": np.nan, "snap_CA": 1, "snap_TX": 0, "snap_WI": 0})
    ids = pd.DataFrame({"id": ["A_CA_1_evaluation", "B_CA_1_evaluation"], "item_id": ["A", "B"], "dept_id": ["FOODS_1"] * 2, "cat_id": ["FOODS"] * 2,
                        "store_id": ["CA_1"] * 2, "state_id": ["CA"] * 2}).astype("category")
    Y = np.array([np.arange(n_days), np.arange(n_days)[::-1]], dtype=np.int16)
    week_ids = np.sort(cal["wm_yr_wk"].unique())
    price_week = np.full((2, len(week_ids)), np.nan, dtype=np.float32)
    price_week[0, :] = np.linspace(1.0, 3.0, len(week_ids))                                        # series A listed every week
    price_week[1, 0] = 5.0                                                                       # series B is listed in week 0 only
    day_week_idx = np.searchsorted(week_ids, cal["wm_yr_wk"].to_numpy())
    return dl.RawData(cal, ids, Y, price_week, week_ids, day_week_idx, n_days)


def test_wide_to_long_conversion_preserves_every_value():
    raw = _tiny_raw()
    panel = dl.build_panel(raw, np.array([0, 1]))
    assert len(panel) == 2 * raw.n_days
    assert list(panel.columns[:6]) == ["series_id", "item_id", "store_id", "dept_id", "cat_id", "state_id"]
    for i, item in enumerate(["A", "B"]):
        g = panel[panel["item_id"] == item].sort_values("date")
        assert np.array_equal(g["sales"].to_numpy(), raw.Y[i].astype(int))                      # long == wide
        assert g["date"].is_monotonic_increasing and (g["date"].diff().dropna() == pd.Timedelta(days=1)).all()
    assert panel["snap"].eq(1).all()                                                            # CA flag chosen via state
    assert (panel["event_name_1"] == "Christmas").sum() == 2                                    # one Christmas day per series
    assert (panel.loc[panel["item_id"] == "A", "sell_price"].notna()).all()
    b = panel[panel["item_id"] == "B"].sort_values("date")
    assert b["listed"].iloc[:7].all() and (~b["listed"].iloc[7:]).all()                          # listing derived from the price table


def test_panel_sorted_by_item_store_date():
    raw = _tiny_raw()
    panel = dl.build_panel(raw, np.array([1, 0]))
    keys = list(zip(panel["item_id"], panel["store_id"], panel["date"]))
    assert keys == sorted(keys)


def test_listed_matrix_matches_price_presence():
    raw = _tiny_raw()
    L = raw.listed_matrix()
    assert L.shape == (2, raw.n_days) and L[0].all() and L[1, :7].all() and not L[1, 7:].any()


def test_syntetos_boylan_classes_and_profile():
    assert ss.sb_class(1.0, 0.2) == "smooth" and ss.sb_class(1.0, 0.6) == "erratic"
    assert ss.sb_class(2.0, 0.2) == "intermittent" and ss.sb_class(2.0, 0.6) == "lumpy"
    y = np.array([0, 0, 4, 0, 0, 4, 0, 0, 0, 4], float)
    p = ss.demand_profile(y)
    assert p["adi"] == pytest.approx(10 / 3) and p["cv2"] == pytest.approx(0.0) and p["zero_share"] == pytest.approx(0.7)
    assert p["sb_class"] == "intermittent"


# ------------------------------------------------------------------ real prepared data
@needs_prepared
def test_prepared_panel_integrity():
    panel = pd.read_parquet(C.DATA_PROC / "panel.parquet")
    sel = pd.read_csv(C.DATA_PROC / "selected_series.csv")
    assert panel["series_id"].nunique() == len(sel) == 18
    assert not panel.duplicated(["series_id", "date"]).any()
    assert panel["sales"].notna().all() and (panel["sales"] >= 0).all()
    assert panel["sell_price"].notna().all() and panel["listed"].all()                           # eligibility: continuously listed
    for sid, g in panel.groupby("series_id"):
        assert len(g) == 1941 and (g.sort_values("date")["date"].diff().dropna() == pd.Timedelta(days=1)).all()


@needs_prepared
def test_selection_design_properties():
    sel = pd.read_csv(C.DATA_PROC / "selected_series.csv")
    assert sel["store_id"].nunique() == 3 and sel.groupby("state_id")["store_id"].nunique().eq(1).all()      # one store per state
    assert sel["item_id"].is_unique                                                                          # 18 distinct products
    assert (sel["tier"].value_counts() == 6).all() and sel["cat_id"].nunique() == 3
    assert (sel["train_zero_share"] < 1).all()


@needs_prepared
def test_raw_files_unchanged_hashes_documented():
    ver = pd.read_csv(C.OUT_TAB / "raw_data_verification.csv")
    for col in ("size_matches_primary", "size_matches_crosscheck", "hash_matches_primary", "hash_matches_crosscheck"):
        assert ver[col].all()


@needs_prepared
def test_calendar_loader_shape_and_dates():
    cal = dl.load_calendar()
    assert len(cal) == 1969 and cal["date"].iloc[0] == pd.Timestamp("2011-01-29") and cal["date"].iloc[-1] == pd.Timestamp("2016-06-19")
    assert (cal["date"].diff().dropna() == pd.Timedelta(days=1)).all()
    assert (cal["event_name_1"] == "Christmas").sum() == 5
