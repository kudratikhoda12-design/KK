import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data import detect_time_unit, read_kline_zip
from src.quality import audit, build_processed


def _zip(tmp_path, name, body):
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(name.replace(".zip", ".csv"), body)
    return p


def test_time_unit_detection():
    assert detect_time_unit(np.array([1704067200000])) == "ms"
    assert detect_time_unit(np.array([1735689600000000])) == "us"
    with pytest.raises(ValueError):
        detect_time_unit(np.array([1704067200]))          # seconds: refuse to guess


def test_microsecond_file_converts_to_correct_date(tmp_path):
    body = "1735689600000000,1,2,0.5,1.5,10,1735689659999999,15,5,4,6,0\n"
    df = read_kline_zip(_zip(tmp_path, "BTCUSDT-1m-2025-01.zip", body))
    assert df["open_time"].iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")


def _interim(tmp_path):
    rows = []
    base = 1704067200000   # 2024-01-01 ms
    for i in range(100):
        if i in (40, 41, 42):                 # a 3-minute gap
            continue
        o = 100 + i * 0.01
        rows.append(f"{base + i * 60000},{o},{o + 0.05},{o - 0.05},{o + 0.01},2,{base + i * 60000 + 59999},{2 * o},7,1,{o},0")
    rows.append(rows[10])                     # exact duplicate
    bad = rows[20].split(","); bad[4] = "999"; rows.append(",".join(bad))   # conflicting duplicate
    inv = rows[30].split(","); inv[2] = "1"; rows[30] = ",".join(inv)        # high < low: impossible
    df = read_kline_zip(_zip(tmp_path, "BTCUSDT-1m-2024-01.zip", "\n".join(rows) + "\n"))
    return df.sort_values("open_time").reset_index(drop=True)


def test_audit_finds_injected_problems(tmp_path):
    a = audit(_interim(tmp_path))
    assert a["duplicates"]["exact_duplicate_rows"] == 1
    assert len(a["duplicates"]["conflicting_timestamps"]) == 1
    assert int(a["gaps"]["missing_minutes"].max()) == 3
    assert a["ohlc_flags"]["high_lt_low"].sum() == 1


def test_processed_grid_applies_rules_without_filling(tmp_path):
    df = _interim(tmp_path)
    grid, log = build_processed(df)
    assert len(grid) == 100                           # regular grid over the full range
    assert grid["close"].isna().sum() == 5            # 3 gap + 1 conflicting + 1 impossible
    assert dict(zip(log["treatment"], log["count"]))["drop exact duplicate rows"] == 1


def test_close_time_check_is_resolution_safe(tmp_path):
    a = audit(_interim(tmp_path))
    assert a["timestamps"]["bad_close_time"] == 0


def test_funding_lookup_counts_payments_inside_holding_window():
    from src.backtest import funding_between
    f = pd.DataFrame({"funding_time": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:00"], utc=True),
                      "funding_rate": [0.0001, 0.0002]})
    entry = pd.Series(pd.to_datetime(["2023-12-31 23:01", "2024-01-01 01:01"], utc=True))
    exit_ = entry + pd.Timedelta("60min")
    assert np.allclose(funding_between(entry, exit_, f), [0.0001, 0.0])
