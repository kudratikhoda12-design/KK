"""Unit tests for Stage 2 (acquisition, QC, aggregation). Run: pytest -q"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ethof import data as dl  # noqa: E402
from ethof import preprocessing as pp  # noqa: E402

SAMPLE = ROOT / "data_sample"
T0 = datetime(2024, 6, 11, tzinfo=timezone.utc)
MS0 = int(T0.timestamp() * 1000)


def _trades(rows):
    """rows: (agg_id, price, qty, first_id, last_id, ms_offset, is_buyer_maker)"""
    return pl.DataFrame(
        [dict(zip(pp.AGG_COLUMNS, (a, p, q, f, l, MS0 + t, m))) for a, p, q, f, l, t, m in rows],
        schema=pp.AGG_SCHEMA)


# ----------------------------------------------------------------------------- acquisition

def test_url_convention():
    rf = dl.RemoteFile("aggTrades", "daily", "ETHUSDT", "2024-06-11",
                       "https://data.binance.vision/data/futures/um")
    assert rf.url == ("https://data.binance.vision/data/futures/um/daily/aggTrades/ETHUSDT/"
                      "ETHUSDT-aggTrades-2024-06-11.zip")
    assert rf.checksum_url == rf.url + ".CHECKSUM"


def test_periods():
    assert dl.periods("2023-11-15", "2024-02-01", "monthly") == ["2023-11", "2023-12", "2024-01", "2024-02"]
    d = dl.periods("2024-02-27", "2024-03-01", "daily")
    assert d == ["2024-02-27", "2024-02-28", "2024-02-29", "2024-03-01"]


def test_sample_checksums():
    for line in (SAMPLE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert dl.sha256_file(SAMPLE / name) == digest


# ----------------------------------------------------------------------------- timestamps

@pytest.mark.parametrize("scale,unit", [(1, "ms"), (1_000, "us"), (1_000_000, "ns")])
def test_detect_time_unit(scale, unit):
    assert pp.detect_time_unit(pl.Series([MS0 * scale, (MS0 + 5) * scale])) == unit


def test_detect_time_unit_mixed_raises():
    with pytest.raises(ValueError):
        pp.detect_time_unit(pl.Series([MS0, MS0 * 1000]))


# ----------------------------------------------------------------------------- cleaning

def test_clean_aggtrades_counts_and_policy():
    raw = _trades([
        (1, 100.0, 1.0, 10, 10, 1_000, "false"),        # buyer-initiated
        (2, 100.5, 2.0, 11, 12, 2_000, "true"),         # seller-initiated
        (2, 100.5, 2.0, 11, 12, 2_000, "true"),         # exact duplicate -> removed
        (3, -1.0, 1.0, 13, 13, 3_000, "false"),         # price <= 0 -> removed
        (4, 100.0, 0.0, 14, 14, 4_000, "false"),        # qty <= 0 -> removed
        (5, 100.0, 1.0, 15, 15, -5_000, "false"),       # before the day -> removed
        (6, 100.2, 1.0, 20, 20, 5_000, "false"),        # trade-id gap 16..19 -> flagged only
        (7, 150.0, 0.1, 21, 21, 6_000, "true"),         # 50% off minute median -> flagged, kept
        (8, 100.1, 1.0, 22, 22, 7_000, "true"),
        (8, 100.9, 1.0, 22, 22, 7_500, "true"),         # conflicting duplicate id -> removed
    ])
    clean, qc = pp.clean_aggtrades(raw, "2024-06-11")
    assert qc["n_raw"] == 10
    assert qc["n_price_nonpositive"] == 1 and qc["n_qty_nonpositive"] == 1 and qc["n_outside_period"] == 1
    assert qc["n_exact_duplicates_removed"] == 1 and qc["n_conflicting_id_duplicates_removed"] == 1
    assert qc["n_bad_tick_flags"] == 1
    assert clean.height == 5 and qc["n_clean"] == 5
    assert clean.filter(pl.col("price") == 150.0).height == 1          # extreme kept, not deleted
    assert clean["ts"].is_sorted()
    # is_buyer_maker == false  <=>  buyer is the aggressor
    assert clean.filter(pl.col("agg_trade_id") == 1)["is_buy"].item() is True
    assert clean.filter(pl.col("agg_trade_id") == 2)["is_buy"].item() is False


def test_unsorted_input_is_sorted():
    raw = _trades([(2, 100.0, 1.0, 2, 2, 2_000, "false"), (1, 100.0, 1.0, 1, 1, 1_000, "false")])
    clean, qc = pp.clean_aggtrades(raw, "2024-06-11")
    assert qc["n_time_decreasing_raw_order"] == 1
    assert clean["agg_trade_id"].to_list() == [1, 2]


# ----------------------------------------------------------------------------- aggregation

def test_aggregate_1min_values_and_grid():
    raw = _trades([
        (1, 100.0, 1.0, 1, 1, 0, "false"),
        (2, 102.0, 3.0, 2, 4, 10_000, "true"),
        (3, 101.0, 2.0, 5, 5, 59_999, "false"),
        (4, 99.0, 30.0, 6, 6, 180_000, "true"),     # minute 3; minutes 1-2 empty
    ])
    clean, _ = pp.clean_aggtrades(raw, "2024-06-11")
    bars = pp.aggregate_1min(clean, "2024-06-11", [10])
    assert bars.height == 1440 and bars["ts"][0] == T0
    b0 = bars.row(0, named=True)
    assert (b0["open"], b0["high"], b0["low"], b0["close"]) == (100.0, 102.0, 100.0, 101.0)
    assert b0["volume"] == 6.0 and b0["buy_volume"] == 3.0 and b0["sell_volume"] == 3.0
    assert b0["buy_trades"] == 2 and b0["sell_trades"] == 1 and b0["n_fills"] == 5
    assert b0["vwap"] == pytest.approx((100 + 306 + 202) / 6)
    b1 = bars.row(1, named=True)
    assert b1["has_trades"] is False and b1["close"] is None and b1["volume"] == 0
    b3 = bars.row(3, named=True)
    assert b3["sell_volume_ge10"] == 30.0 and b3["buy_volume_ge10"] == 0.0
    runs = pp.no_trade_runs(bars.filter(pl.col("ts") < T0 + timedelta(minutes=4)), 1)
    assert runs["minutes"].to_list() == [2]


def test_sample_end_to_end_conservation():
    raw = pp.read_aggtrades(SAMPLE / "ETHUSDT-aggTrades-2024-06-11_0000-0059.zip")
    assert raw.height == 23_032
    clean, qc = pp.clean_aggtrades(raw, "2024-06-11")
    assert qc["time_unit"] == "ms"
    bars = pp.aggregate_1min(clean, "2024-06-11", [1, 10])
    first_hour = bars.filter(pl.col("ts") < T0 + timedelta(hours=1))
    assert first_hour["has_trades"].all()
    assert first_hour["volume"].sum() == pytest.approx(clean["quantity"].sum())
    diff = (first_hour["buy_volume"] + first_hour["sell_volume"] - first_hour["volume"]).abs().max()
    assert diff < 1e-9
    assert (first_hour["high"] >= first_hour["low"]).all()


# ----------------------------------------------------------------------------- order book

def test_book_sample_wide_and_monotone():
    wide, qc = pp.clean_bookdepth(pp.read_bookdepth(SAMPLE / "ETHUSDT-bookDepth-2024-06-11_0000-0059.zip"),
                                  "2024-06-11")
    assert qc["n_incomplete_snapshots"] == 0 and qc["n_snapshots_non_monotone_depth"] == 0
    assert wide.height == qc["n_snapshots"] > 100
    assert {"bid_depth_1", "ask_depth_5", "bid_notional_3"} <= set(wide.columns)


def test_asof_book_has_no_lookahead():
    bars = pl.DataFrame({"ts": [T0, T0 + timedelta(minutes=1)]})
    book = pl.DataFrame({
        "ts": [T0 + timedelta(seconds=59), T0 + timedelta(seconds=60), T0 + timedelta(seconds=150)],
        "bid_depth_1": [1.0, 2.0, 3.0]})
    out = pp.asof_book_to_bars(bars, book, stale_seconds=120)
    # bar [00:00, 00:01) may use the 00:00:59 snapshot, never the one stamped exactly 00:01:00
    assert out["bid_depth_1"].to_list() == [1.0, 2.0]
    assert out["book_age_s"].to_list() == [1, 60]


def test_asof_book_flags_stale():
    bars = pl.DataFrame({"ts": [T0 + timedelta(minutes=10)]})
    book = pl.DataFrame({"ts": [T0], "bid_depth_1": [1.0]})
    out = pp.asof_book_to_bars(bars, book, stale_seconds=120)
    assert out["book_stale"].item() is True and out["bid_depth_1"].item() == 1.0


def test_asof_book_frozen_feed_is_stale():
    # snapshots keep arriving but their values stopped changing at 00:00:30 -> age counts from 00:00:30
    bars = pl.DataFrame({"ts": [T0 + timedelta(minutes=5)]})
    book = pl.DataFrame({"ts": [T0 + timedelta(seconds=30 * k) for k in range(1, 11)],
                         "bid_depth_1": [1.0] * 10,
                         "last_change_ts": [T0 + timedelta(seconds=30)] * 10})
    out = pp.asof_book_to_bars(bars, book, stale_seconds=120)
    assert out["book_age_s"].item() == 330 and out["book_stale"].item() is True
