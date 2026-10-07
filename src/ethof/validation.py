"""Chronological splits: expanding walk-forward with purge + embargo, and the holdout guard.

Purge: a training row t is dropped if its label window (tau_t, tau_{t+h}] reaches into the
validation block (label_end_h >= validation start - embargo). Because the windows only expand forward in time,
no training row ever comes after validation data, so only the boundary before the validation block needs purging.

Embargo: an extra gap of `embargo` (default 1 day) between the end of training labels and the
start of validation. It guards against slow-moving features (e.g. 1-day z-scores) and serially dependent
errors carrying training-period information into the first validation minutes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

UTC = timezone.utc


def _d(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: datetime
    train_end: datetime         # exclusive; training rows also need label_end < train_end - embargo
    valid_start: datetime
    valid_end: datetime         # exclusive

    def as_dict(self) -> dict:
        return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in self.__dict__.items()}


def walk_forward_folds(start: str, first_valid: str, end_exclusive: str, months: int = 3,
                       embargo: timedelta = timedelta(days=1)) -> list[Fold]:
    """Expanding-window folds: train [start, valid_start - embargo), validate [valid_start, +months)."""
    folds, vs, i = [], _d(first_valid), 1
    end = _d(end_exclusive)
    while vs < end:
        y, m = vs.year, vs.month + months
        y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
        ve = min(datetime(y, m, 1, tzinfo=UTC), end)
        folds.append(Fold(f"F{i}", _d(start), vs - embargo, vs, ve))
        vs, i = ve, i + 1
    return folds


def split(df: pl.DataFrame, fold: Fold, horizon: int, stride: int = 1) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Training rows: ts in [train_start, train_end) whose label ends before train_end (purge).

    Validation rows: ts in [valid_start, valid_end) whose label ends by valid_end, so validation labels
    never use prices from the next block (and, for the last development fold, never from the holdout).
    `stride` sub-samples training rows (every k-th minute) to reduce near-duplicate overlapping labels.
    """
    le = f"label_end_{horizon}"
    tr = df.filter((pl.col("ts") >= fold.train_start) & (pl.col("ts") < fold.train_end)
                   & (pl.col(le) <= fold.train_end))
    if stride > 1:
        tr = tr.filter((pl.col("ts").dt.epoch("s") // 60) % stride == 0)
    va = df.filter((pl.col("ts") >= fold.valid_start) & (pl.col("ts") < fold.valid_end)
                   & (pl.col(le) <= fold.valid_end))
    return tr, va


# ----------------------------------------------------------------------------- holdout guard

class HoldoutLocked(RuntimeError):
    pass


def methodology_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def development_only(df: pl.DataFrame, holdout_start: str) -> pl.DataFrame:
    """All development work goes through this filter, so the holdout cannot leak in by accident."""
    out = df.filter(pl.col("ts") < _d(holdout_start))
    assert out["ts"].max() < _d(holdout_start)
    return out


def unlock_holdout(df: pl.DataFrame, holdout_start: str, holdout_end_exclusive: str,
                   frozen_methodology: Path, access_log: Path, purpose: str) -> pl.DataFrame:
    """Return holdout rows only if the frozen methodology file exists; every access is logged with the
    SHA-256 of that file, so the log shows that the methodology was not edited after the holdout was opened."""
    if not frozen_methodology.exists():
        raise HoldoutLocked(f"{frozen_methodology} missing: freeze the methodology before touching the holdout")
    entry = {"utc": datetime.now(UTC).isoformat(timespec="seconds"), "purpose": purpose,
             "methodology_sha256": methodology_hash(frozen_methodology)}
    access_log.parent.mkdir(parents=True, exist_ok=True)
    with open(access_log, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return df.filter((pl.col("ts") >= _d(holdout_start)) & (pl.col("ts") < _d(holdout_end_exclusive)))
