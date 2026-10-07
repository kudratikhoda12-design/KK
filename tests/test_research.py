"""Unit tests for features, targets, statistics, validation and backtest logic."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ethof import backtest as B  # noqa: E402
from ethof import features as F  # noqa: E402
from ethof import statistics as S  # noqa: E402
from ethof import targets as T  # noqa: E402
from ethof import validation as V  # noqa: E402

T0 = datetime(2024, 6, 11, tzinfo=timezone.utc)


def _bars(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    close = 3000 * np.exp(np.cumsum(rng.normal(0, 5e-4, n)))
    buy = rng.uniform(0, 10, n); sell = rng.uniform(0, 10, n)
    d = {"ts": [T0 + timedelta(minutes=i) for i in range(n)], "open": close, "high": close * 1.0002,
         "low": close * 0.9998, "close": close, "vwap": close, "volume": buy + sell, "notional": (buy + sell) * close,
         "n_trades": rng.integers(1, 50, n), "buy_volume": buy, "sell_volume": sell,
         "buy_trades": rng.integers(1, 25, n), "sell_trades": rng.integers(1, 25, n),
         "buy_volume_ge10": np.zeros(n), "sell_volume_ge10": np.zeros(n), "book_stale": np.zeros(n, bool)}
    for k in range(1, 6):
        d[f"bid_depth_{k}"] = rng.uniform(100, 200, n) * k
        d[f"ask_depth_{k}"] = rng.uniform(100, 200, n) * k
    return pl.DataFrame(d)


def test_order_flow_and_imbalance_definitions():
    b = _bars(200)
    f = F.build_features(b)
    bv, sv = b["buy_volume"].to_numpy(), b["sell_volume"].to_numpy()
    np.testing.assert_allclose(f["ofi_1"].to_numpy(), (bv - sv) / (bv + sv))
    # 15-minute OFI aggregates volumes, not ratios
    i = 100
    exp = (bv[i - 14:i + 1].sum() - sv[i - 14:i + 1].sum()) / (bv[i - 14:i + 1].sum() + sv[i - 14:i + 1].sum())
    assert f["ofi_15"][i] == pytest.approx(exp)
    assert f["ofi_15"][:14].is_null().all()                 # not enough history -> null, never padded
    bd, ad = b["bid_depth_1"].to_numpy(), b["ask_depth_1"].to_numpy()
    np.testing.assert_allclose(f["obi_1pct"].to_numpy(), (bd - ad) / (bd + ad))


def test_imbalance_zero_denominator_is_null():
    b = _bars(50).with_columns(buy_volume=pl.lit(0.0), sell_volume=pl.lit(0.0))
    f = F.build_features(b)
    assert f["ofi_1"].is_null().all()


def test_stale_book_gives_null_book_features():
    b = _bars(50).with_columns(book_stale=pl.Series([True] * 10 + [False] * 40))
    f = F.build_features(b)
    assert f["obi_1pct"][:10].is_null().all() and f["obi_1pct"][10:].is_not_null().all()


def test_rolling_features_are_trailing():
    b = _bars(500)
    full = F.build_features(b)
    cut = F.build_features(b.head(300))
    for c in ("vol_60", "volume_z_1440", "ofi_60", "dist_ma_240", "ret_60"):
        np.testing.assert_allclose(full[c].to_numpy()[:300], cut[c].to_numpy(), equal_nan=True)
    px = b["close"].to_numpy()
    assert full["ret_15"][100] == pytest.approx(np.log(px[100] / px[85]))


def test_forward_fill_only_backward():
    b = _bars(30).with_columns(close=pl.when(pl.int_range(30) == 10).then(None).otherwise(pl.col("close")),
                               vwap=pl.when(pl.int_range(30) == 10).then(None).otherwise(pl.col("vwap")))
    p = F.prepare_price(b)
    assert p["px"][10] == b["close"][9] and p["exec_px"][10] == b["close"][9]
    assert p["exec_px_from_trades"][10] is False


def test_targets():
    df = pl.DataFrame({"ts": [T0 + timedelta(minutes=i) for i in range(10)], "px": [100.0 + i for i in range(10)]})
    t = T.add_targets(df, horizons=(3,), neutral_band=0.025)
    assert t["fwd_ret_3"][0] == pytest.approx(103 / 100 - 1)
    assert t["fwd_ret_3"][7:].is_null().all() and t["up_3"][7:].is_null().all()
    assert t["up_3"][0] == 1
    assert t["cls3_3"][0] == 2 and t["cls3_3"][9] is None      # +3% > band -> UP
    assert t["label_end_3"][0] == T0 + timedelta(minutes=4)


def test_walk_forward_purge_embargo():
    folds = V.walk_forward_folds("2023-03-01", "2024-03-01", "2025-10-01", 3, timedelta(days=1))
    assert len(folds) == 7 and folds[-1].valid_end == V._d("2025-10-01")
    assert all(f.train_end == f.valid_start - timedelta(days=1) for f in folds)
    n = 60 * 24 * 400
    ts = pl.datetime_range(V._d("2023-03-01"), V._d("2023-03-01") + timedelta(minutes=n - 1), "1m",
                           time_zone="UTC", eager=True)
    df = pl.DataFrame({"ts": ts}).with_columns(label_end_15=pl.col("ts") + pl.duration(minutes=16))
    f = V.Fold("x", V._d("2023-03-01"), V._d("2024-01-31"), V._d("2024-02-01"), V._d("2024-03-01"))
    tr, va = V.split(df, f, 15)
    assert tr["label_end_15"].max() <= f.train_end and va["ts"].min() == f.valid_start
    assert va["label_end_15"].max() <= f.valid_end


def test_holdout_guard(tmp_path):
    df = pl.DataFrame({"ts": [V._d("2025-09-30"), V._d("2025-10-02")]})
    assert V.development_only(df, "2025-10-01").height == 1
    with pytest.raises(V.HoldoutLocked):
        V.unlock_holdout(df, "2025-10-01", "2026-10-01", tmp_path / "missing.md", tmp_path / "log.jsonl", "test")
    m = tmp_path / "m.md"; m.write_text("frozen")
    assert V.unlock_holdout(df, "2025-10-01", "2026-10-01", m, tmp_path / "log.jsonl", "test").height == 1
    assert "methodology_sha256" in (tmp_path / "log.jsonl").read_text()


# ----------------------------------------------------------------------------- signals / backtest

def test_signal_rules():
    np.testing.assert_array_equal(B.prob_signal(np.array([0.6, 0.5, 0.4, 0.52, np.nan]), 0.05), [1, 0, -1, 0, 0])
    np.testing.assert_array_equal(B.flow_signal(np.array([1.5, -2, 0.3]), 1.0), [1, -1, 0])
    np.testing.assert_array_equal(B.momentum_signal(np.array([0.1, -0.2, np.nan])), [1, -1, 0])


def test_staggered_position():
    w = B.staggered_position(np.array([1, 1, 1, 0, 0, -1]), H=3)
    np.testing.assert_allclose(w, [1 / 3, 2 / 3, 1, 2 / 3, 1 / 3, -1 / 3])


def test_costs_per_side():
    c = B.Costs(fee_bps=4.5, slippage_bps=1.0, half_spread_bps=0.023)
    assert c.per_side == pytest.approx(5.523e-4)


def test_backtest_pnl_execution_timing_and_costs():
    # execution prices rise 1% per bar; one long signal at bar 0, H = 1
    n = 6
    e = 100 * 1.01 ** np.arange(n)
    df = pl.DataFrame({"ts": [T0 + timedelta(minutes=i) for i in range(n)], "exec_px": e})
    sig = np.array([1.0, 0, 0, 0, 0, 0])
    c = B.Costs(10.0, 0.0, 0.0)
    bt = B.run(df, sig, H=1, costs=c, funding=None)
    # position W_0 = 1 earns e2/e1 - 1 = 1% (entry at bar 1 VWAP, NOT at the signal bar)
    assert bt["gross"][0] == pytest.approx(0.01)
    assert bt["gross"][1:].sum() == pytest.approx(0.0)
    # costs: enter at bar 1 (|dW| = 1) and exit at bar 2 (|dW| = 1) -> 2 x 10 bps
    assert bt["cost"].sum() == pytest.approx(2 * 10e-4)
    assert bt["net"].sum() == pytest.approx(0.01 - 20e-4)
    tr = B.trade_returns(df, sig, 1, c)
    assert tr[0] == pytest.approx(0.01 - 20e-4)
    s = B.summarize(bt, tr)
    assert s["breakeven_cost_bps_per_side"] == pytest.approx(0.01 / 2 * 1e4)


def test_funding_charged_to_position_held_at_settlement():
    n = 10
    ts = [T0 + timedelta(minutes=i) for i in range(n)]
    df = pl.DataFrame({"ts": ts, "exec_px": np.full(n, 100.0)})
    fund = pl.DataFrame({"ts": [T0 + timedelta(minutes=5)], "last_funding_rate": [0.001]})
    bt = B.run(df, np.ones(n), H=1, costs=B.Costs(0, 0, 0), funding=fund)
    # settlement at the start of bar 5 falls in the holding interval of W_3 (bar 4 mid -> bar 5 mid)
    assert bt["funding"][3] == pytest.approx(-0.001) and bt["funding"].sum() == pytest.approx(-0.001)
    bt_short = B.run(df, -np.ones(n), H=1, costs=B.Costs(0, 0, 0), funding=fund)
    assert bt_short["funding"].sum() == pytest.approx(0.001)          # shorts receive positive funding


# ----------------------------------------------------------------------------- statistics

def test_multiple_testing_adjustments():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    np.testing.assert_allclose(S.holm(p), [0.04, 0.09, 0.09, 0.2])
    np.testing.assert_allclose(S.bh(p), [0.04, 0.0533333, 0.0533333, 0.2], rtol=1e-5)


def test_hac_recovers_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = 2.0 * x + rng.normal(size=5000)
    r = S.hac_ols(y, x[:, None], ["x"], maxlags=5)
    assert r["b_x"] == pytest.approx(2.0, abs=0.05) and r["p_x"] < 1e-10
