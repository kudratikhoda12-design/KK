"""Chronological walk-forward validation and the experiment log.

Why not random K-fold? With time series, a random split lets the model learn
from the future (e.g. train on 2023, test on 2021) and puts highly similar
neighbouring hours on both sides of the split (volatility clustering), so it
overstates skill. Every split here respects time order.

Purging: a training row is dropped if its label window (entry -> exit) ends
after the evaluation block starts; otherwise the last training labels would
overlap the first evaluation period.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from . import config


@dataclass
class Fold:
    name: str
    train_mask: pd.Series
    eval_mask: pd.Series
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp

    def describe(self) -> dict:
        return dict(fold=self.name, train=f"{self.train_start:%Y-%m-%d} -> {self.train_end:%Y-%m-%d}",
                    eval=f"{self.eval_start:%Y-%m-%d} -> {self.eval_end:%Y-%m-%d}",
                    n_train=int(self.train_mask.sum()), n_eval=int(self.eval_mask.sum()))


def walk_forward_folds(data: pd.DataFrame, first_eval_start: pd.Timestamp, end: pd.Timestamp,
                       block_months: int) -> list[Fold]:
    """Expanding-window folds. `data` must have a DatetimeIndex (decision time) and 'exit_time'."""
    idx = data.index
    folds, start = [], first_eval_start
    while start < end:
        stop = min(start + pd.DateOffset(months=block_months), end)
        train = (idx < start) & (data["exit_time"] <= start)          # purge overlapping labels
        ev = (idx >= start) & (idx < stop)
        if ev.any() and train.any():
            folds.append(Fold(name=f"{start:%Y}H{1 if start.month <= 6 else 2}",
                              train_mask=pd.Series(train, index=idx),
                              eval_mask=pd.Series(ev, index=idx),
                              train_start=idx[train].min(), train_end=idx[train].max(),
                              eval_start=idx[ev].min(), eval_end=idx[ev].max()))
        start = stop
    return folds


def development_folds(data: pd.DataFrame) -> list[Fold]:
    return walk_forward_folds(data, config.FIRST_VALIDATION_START, config.DEV_END,
                              config.VALIDATION_BLOCK_MONTHS)


def test_folds(data: pd.DataFrame) -> list[Fold]:
    end = data.index.max() + pd.Timedelta(minutes=1)
    return walk_forward_folds(data, config.TEST_START, end, config.TEST_RETRAIN_MONTHS)


def assert_no_test_leak(fold: Fold) -> None:
    """Hard guard: development folds may never evaluate on the test period."""
    if fold.eval_end >= config.TEST_START and fold.eval_start < config.TEST_START:
        raise AssertionError(f"Fold {fold.name} straddles the test boundary")


class ExperimentLog:
    """Every configuration evaluated is recorded - including failures.

    The count of trials feeds the Deflated Sharpe Ratio, so hiding a trial
    would bias the significance test.
    """

    def __init__(self, path=None):
        self.path = path or (config.TABLES_DIR / "experiment_log.csv")
        self.rows: list[dict] = []

    def add(self, stage: str, model: str, params: dict, features: list[str], horizon: int,
            lag: int, offset: int, train_period: str, eval_period: str, metrics: dict,
            note: str = "") -> None:
        fixed = dict(
            logged_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            stage=stage, model=model, params=json.dumps(params, default=str),
            n_features=len(features), features=",".join(features), horizon_min=horizon,
            lag_min=lag, offset_min=offset, train_period=train_period, eval_period=eval_period)
        # Metric names that collide with fixed fields (e.g. 'model') are prefixed, never dropped
        extra = {(f"metric_{k}" if k in fixed or k == "note" else k):
                 (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()}
        self.rows.append({**fixed, **extra, "note": note})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self) -> None:
        config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
        self.frame().to_csv(self.path, index=False)
