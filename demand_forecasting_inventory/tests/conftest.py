"""Shared fixtures.  Data-dependent tests are skipped (not failed) when the prepared real data are absent."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C                                   # noqa: E402
from src.forecast_core import Context                         # noqa: E402
from src.splits import make_split                             # noqa: E402

HAS_PREPARED = (C.DATA_PROC / "panel.parquet").exists() and (C.DATA_RAW / C.RAW_FILES["calendar"]).exists()
HAS_FORECASTS = (C.DATA_PROC / "forecasts.npz").exists() and (C.OUT_MOD / "xgb_H7_final_train_val.json").exists()

needs_prepared = pytest.mark.skipif(not HAS_PREPARED, reason="run `python run_project.py --only prepare` first")
needs_forecasts = pytest.mark.skipif(not (HAS_PREPARED and HAS_FORECASTS), reason="run the forecast stage first")


def make_synthetic_context(n_series: int = 3, n_days: int = 520, seed: int = 0) -> Context:
    """Small weekly-seasonal Poisson panel with a synthetic calendar (SNAP block, a few events, one 'Christmas')."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-02", periods=n_days)           # starts on a Monday
    dow = np.tile([6, 6, 7, 7, 9, 13, 11], n_days // 7 + 1)[:n_days]
    levels = np.array([1.0, 0.4, 0.15])[:n_series]
    Y = np.vstack([rng.poisson(l * dow / 8.0 * 4) for l in levels]).astype(float)
    price = np.vstack([np.full(n_days, 2.0 + i) for i in range(n_series)])
    N_cal = n_days + 28
    cal_dates = pd.date_range(dates[0], periods=N_cal)
    snap = np.vstack([(cal_dates.day <= 10).astype(float) for _ in range(n_series)])
    event = np.zeros(N_cal)
    event[[50, 120, 200, 300, 400]] = 1.0
    xmas = np.zeros(N_cal)
    xmas[[200]] = 1.0
    sel = pd.DataFrame({"series_id": [f"S{i}" for i in range(n_series)], "tier": ["high", "medium", "low"][:n_series],
                        "intermittency": ["regular", "intermittent", "intermittent"][:n_series],
                        "train_sb_class": ["smooth", "intermittent", "lumpy"][:n_series], "cat_id": ["FOODS"] * n_series})
    cal = pd.DataFrame({"date": cal_dates})
    return Context(ids=list(sel["series_id"]), sel=sel, panel=pd.DataFrame(), Y=Y, price=price, dates=dates, split=make_split(dates),
                   calendar=cal, snap_full=snap, event_full=event, christmas_full=xmas)


@pytest.fixture(scope="session")
def synth():
    return make_synthetic_context()
