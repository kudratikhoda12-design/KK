"""Phase 9 - chronological train / validation / test split and forecast-origin bookkeeping.

Indexing convention (0-based day positions ``j`` on the observed-day axis, N = number of observed days)::

    train = [0, train_end)        validation = [train_end, val_end)        test = [val_end, N)

A *forecast origin* ``t`` is the last day whose sales are known; the forecast covers days ``t+1 ... t+H``
and the target is ``sum(y[t+1..t+H])``.  A target window is *complete inside a period* when ``t+1`` and
``t+H`` both lie in the period.  There is no random splitting anywhere in the project.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass(frozen=True)
class Split:
    n_days: int
    train_end: int
    val_end: int
    dates: pd.DatetimeIndex

    # ---- period boundaries -------------------------------------------------------------
    def bounds(self, period: str) -> tuple[int, int]:
        """Half-open [start, end) day-positions of a period."""
        if period == "train":
            return 0, self.train_end
        if period in ("val", "validation"):
            return self.train_end, self.val_end
        if period == "test":
            return self.val_end, self.n_days
        if period == "train_val":
            return 0, self.val_end
        raise ValueError(period)

    def date_range(self, period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        a, b = self.bounds(period)
        return self.dates[a], self.dates[b - 1]

    # ---- forecast origins ------------------------------------------------------------------
    def eval_origins(self, period: str, H: int) -> np.ndarray:
        """Origins whose complete H-day target window lies inside ``period`` (validation or test)."""
        a, b = self.bounds(period)
        return np.arange(a - 1, b - H, dtype=int)            # t = a-1 ... b-1-H

    def all_origins(self, period: str) -> np.ndarray:
        """Every origin from the last day before the period to the last day of the period.

        Used for producing forecasts (needed daily by the inventory simulation) - windows that run past
        the end of the data have no complete target.
        """
        a, b = self.bounds(period)
        return np.arange(a - 1, b, dtype=int)

    def train_origins(self, H: int, fit_end: int, first: int | None = None) -> np.ndarray:
        """Origins usable to *fit* a direct model: target window must end on or before ``fit_end - 1``."""
        first = (C.MIN_HISTORY - 1) if first is None else first
        return np.arange(first, fit_end - H, dtype=int)       # t + H <= fit_end - 1


def make_split(dates: pd.DatetimeIndex, train_frac: float = C.TRAIN_FRAC, val_frac: float = C.VAL_FRAC) -> Split:
    n = len(dates)
    train_end = int(np.floor(train_frac * n))
    val_end = train_end + int(np.floor(val_frac * n))
    if not (0 < train_end < val_end < n):
        raise ValueError("degenerate split")
    return Split(n_days=n, train_end=train_end, val_end=val_end, dates=pd.DatetimeIndex(dates))


def split_table(split: Split) -> pd.DataFrame:
    rows = []
    for period in ("train", "validation", "test"):
        a, b = split.bounds(period)
        d0, d1 = split.date_range(period)
        rows.append({"period": period, "first_day_index": a, "last_day_index": b - 1, "n_days": b - a,
                     "first_date": d0.date(), "last_date": d1.date(),
                     "share_of_days": round((b - a) / split.n_days, 4)})
    df = pd.DataFrame(rows)
    return df
