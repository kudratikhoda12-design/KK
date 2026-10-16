"""C-MAPSS loader (Stage F; written after inspection, reports/logs/cmapss_inspection.json).

Framing (D3a): a degradation / remaining-useful-life benchmark. All training engines fail (delta = 1); the official
test set provides true RUL (known outcome, not censoring). Landmark options A and B are both provided and analysed
separately; neither is the approved design (D3b-R unresolved).

Option A: one sample per engine; window [0, s0] (cycles 1..s0); target T - s0; engines with T <= s0 excluded.
Option B: trailing window of w cycles ending at landmark s (cycles s-w+1..s); s = w, w+10, ... <= T-1; u = s/100;
          target T - s. Official test engines: landmark = last observed cycle.
Times inside each window are rescaled to [0, 1]; no input observation is later than the landmark (K4).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import Observed

COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
SENSORS = [f"s{i}" for i in range(1, 22)]


def read_fd(raw_dir, fd="FD001"):
    raw = Path(raw_dir)
    tr = pd.read_csv(raw / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    te = pd.read_csv(raw / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    rul = pd.read_csv(raw / f"RUL_{fd}.txt", sep=r"\s+", header=None)[0].to_numpy()
    return tr, te, rul


def informative_sensors(train_df, units, tol=1e-6):
    """Sensors with training-part standard deviation >= tol (decided per split, on training engines only)."""
    sd = train_df[train_df.unit.isin(units)][SENSORS].std()
    return [s for s in SENSORS if sd[s] >= tol]


def _window_obs(rows, sensors, ids, groups, landmarks, y, include_s, meta):
    L = max(len(r) for r in rows)
    N, D = len(rows), len(sensors)
    times, values = np.zeros((N, L)), np.zeros((N, L, D))
    mask = np.zeros((N, L, D), bool)
    n_obs = np.zeros(N, int)
    for i, r in enumerate(rows):
        c = r["cycle"].to_numpy(float)
        lo, hi = c.min(), c.max()
        times[i, :len(r)] = (c - lo) / (hi - lo) if hi > lo else 0.0
        values[i, :len(r)] = r[sensors].to_numpy(float)
        mask[i, :len(r)] = True
        n_obs[i] = len(r)
    u = np.asarray(landmarks, float)[:, None] / 100.0 if include_s else None
    return Observed(np.asarray(ids), times, values, mask, n_obs, np.asarray(y, float), np.ones(N, bool), u,
                    tuple(sensors), {**meta, "group": np.asarray(groups), "landmark": np.asarray(landmarks),
                                     "max_input_cycle": np.array([r["cycle"].max() for r in rows])})


def option_a(df, units, s0, sensors, life=None):
    """life: optional dict unit -> total life (for official test engines: observed length + true RUL)."""
    rows, ids, groups, lm, y, excluded = [], [], [], [], [], []
    for u_ in units:
        g = df[df.unit == u_].sort_values("cycle")
        T = life[u_] if life is not None else g.cycle.max()
        if T <= s0 or g.cycle.max() < s0:
            excluded.append(int(u_)); continue
        rows.append(g[g.cycle <= s0]); ids.append(int(u_)); groups.append(int(u_)); lm.append(s0); y.append(T - s0)
    obs = _window_obs(rows, sensors, ids, groups, lm, y, False, {"option": "A", "s0": s0})
    return obs, excluded


def option_b(df, units, sensors, w=30, step=10, life=None, last_only=False):
    rows, ids, groups, lm, y = [], [], [], [], []
    for u_ in units:
        g = df[df.unit == u_].sort_values("cycle")
        T = life[u_] if life is not None else g.cycle.max()
        last = g.cycle.max()
        marks = [last] if last_only else list(range(w, int(T), step))
        for s in marks:
            win = g[(g.cycle > s - w) & (g.cycle <= s)]
            if len(win) < w or s >= T:
                continue
            rows.append(win); ids.append(int(u_) * 10000 + int(s)); groups.append(int(u_)); lm.append(s); y.append(T - s)
    return _window_obs(rows, sensors, ids, groups, lm, y, True, {"option": "B", "w": w, "step": step})


def official_test_life(test_df, rul):
    units = sorted(test_df.unit.unique())
    return {u_: int(test_df[test_df.unit == u_].cycle.max() + rul[k]) for k, u_ in enumerate(units)}
