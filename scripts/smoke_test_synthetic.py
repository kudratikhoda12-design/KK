"""SMOKE TEST ON SYNTHETIC DATA - checks that the pipeline code runs end to end.

Creates a random-walk 1-minute dataset with the same date range and size as
the real BTCUSDT data (plus injected anomalies: microsecond timestamps from
2025, a duplicate, a conflicting duplicate and a gap), points every output
path to a TEMPORARY directory, and runs all stages.

Nothing written by this script is a result about Bitcoin. Outputs never touch
reports/, figures/ or data/ of the repository.

    python scripts/smoke_test_synthetic.py [--days 3331] [--out /tmp/smoke]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src import config  # noqa: E402


def make_interim(days: int, seed: int = 0) -> pd.DataFrame:
    from conftest import make_synthetic_grid
    g = make_synthetic_grid(start="2017-08-17", days=days, beta=0.0, seed=seed)
    df = g.drop(columns=["present"]).reset_index()
    us = df["open_time"] >= pd.Timestamp("2025-01-01", tz="UTC")
    epoch_ms = df["open_time"].astype("int64") // 10**6
    df["raw_open_time"] = np.where(us, epoch_ms * 1000, epoch_ms)
    df["close_time"] = df["open_time"] + np.where(us, pd.Timedelta(microseconds=59_999_999),
                                                  pd.Timedelta(milliseconds=59_999))
    df["ts_unit"] = np.where(us, "us", "ms")
    df["source_file"] = "SYNTH-1m-" + df["open_time"].dt.strftime("%Y-%m") + ".zip"
    df["has_header"] = False
    df["row_in_file"] = df.groupby("source_file").cumcount()
    df = df.drop(index=range(100_000, 100_090))                 # a 90-minute gap
    dup = df.iloc[[5000]].copy()                                 # an exact duplicate
    conf = df.iloc[[6000]].copy(); conf["close"] *= 1.01         # a conflicting duplicate
    df = pd.concat([df, dup, conf]).sort_values("open_time", kind="stable").reset_index(drop=True)
    for c in ["ts_unit", "source_file"]:
        df[c] = df[c].astype("category")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3331)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--from-stage", default=None, help="resume from this stage, re-using data in --out")
    a = ap.parse_args()
    out = Path(a.out or tempfile.mkdtemp(prefix="smoke_synthetic_"))
    # Redirect EVERY output path before the runner is imported
    config.DATA_DIR = out / "data"; config.RAW_DIR = out / "data/raw"
    config.INTERIM_DIR = out / "data/interim"; config.PROCESSED_DIR = out / "data/processed"
    config.REPORTS_DIR = out / "reports"; config.TABLES_DIR = out / "reports/tables"
    config.FIGURES_DIR = out / "figures"; config.PERMUTATION_N = 5
    config.ensure_dirs()
    t0 = time.time()
    if a.from_stage is None:
        df = make_interim(a.days)
        df.to_parquet(config.INTERIM_DIR / f"{config.SYMBOL}_1m_raw.parquet", index=False)
        print(f"synthetic interim: {len(df):,} rows in {time.time() - t0:.0f}s -> {out}")

    import run_pipeline as rp   # noqa: E402  (imported after paths are redirected)
    stages = [s for s in rp.STAGES if not (a.skip_robustness and s == "robustness")]
    if a.from_stage:
        stages = stages[stages.index(a.from_stage):]
    for s in stages:
        t = time.time()
        getattr(rp, f"stage_{s}")(config.SYMBOL)
        print(f"[smoke] stage {s}: {time.time() - t:.0f}s", flush=True)
    print(f"[smoke] all stages ran in {time.time() - t0:.0f}s; outputs in {out}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
