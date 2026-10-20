"""Stacked landmark data set for B3 (D112; docs/design_note_round10.md section 2, B3).

For each landmark t_L and each subject at risk at t_L (observed time y > t_L):
  * history: observations with t_L - W <= t <= t_L only (nothing after t_L), times rescaled to [0, 1] on the window;
  * outcome: time since the landmark, administratively censored at horizon w:
      y* = min(y, t_L + w) - t_L,  d* = d and (y <= t_L + w);
  * the landmark itself is kept as a feature (supermodel edge input) and as the stratum of the baseline.
Subjects with fewer than `min_obs` observations in the window are excluded (default 1; reported in `meta`).
One row per (subject, landmark).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Stacked:
    sid: np.ndarray            # subject index per row
    landmark: np.ndarray       # t_L per row
    times: list                # window-rescaled observation times per row, in [0, 1]
    times_abs: list            # absolute observation times per row (for checks)
    values: list               # (m, D) per row
    z: np.ndarray
    y: np.ndarray              # time since landmark (censored at w)
    d: np.ndarray
    meta: dict = field(default_factory=dict)


def build_stacked(times, values, z, y, d, landmarks, window, horizon, min_obs=1):
    """times: list of (m_i,) absolute times; values: list of (m_i, D); z (n, p); y, d (n,)."""
    y, d = np.asarray(y, float), np.asarray(d, bool)
    rows = dict(sid=[], landmark=[], times=[], times_abs=[], values=[], z=[], y=[], d=[])
    excluded = {}
    for tL in landmarks:
        excluded[float(tL)] = 0
        for i in np.nonzero(y > tL)[0]:
            t = np.asarray(times[i], float)
            keep = (t >= tL - window) & (t <= tL)
            if keep.sum() < min_obs:
                excluded[float(tL)] += 1
                continue
            rows["sid"].append(i)
            rows["landmark"].append(float(tL))
            rows["times_abs"].append(t[keep])
            rows["times"].append((t[keep] - (tL - window)) / window)
            rows["values"].append(np.asarray(values[i])[keep])
            rows["z"].append(np.asarray(z[i], float))
            rows["y"].append(min(y[i], tL + horizon) - tL)
            rows["d"].append(bool(d[i] and y[i] <= tL + horizon))
    return Stacked(np.asarray(rows["sid"], int), np.asarray(rows["landmark"]), rows["times"], rows["times_abs"],
                   rows["values"], np.asarray(rows["z"]).reshape(len(rows["sid"]), -1), np.asarray(rows["y"]),
                   np.asarray(rows["d"]), dict(window=window, horizon=horizon, min_obs=min_obs,
                                               excluded_few_obs=excluded))
