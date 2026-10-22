"""Round-13 real-data layer (D114, D129): longitudinal records per subject -> single-landmark or stacked data sets.

LongData holds, per subject: absolute observation times, values (m, D) with NaN for a missing channel value, baseline
covariates z (numeric), the observed time y from baseline and the event indicator d.
  load_pbcseq(endpoint)  R survival 3.8-12 `pbcseq` (days); channels and log transforms as kanfae.data.pbc.CHANNELS;
                         z = (age, female); E2 death or transplant (primary), E1 death (transplant censored)
  load_aids()            R JM 1.5-2 `aids` (months); channel CD4 as provided (already square-root scale; D128);
                         z = (ddI, male, prevOI AIDS, AZT failure)
  load_prothro()         R JM 1.5-2 `prothro` (years); channel prothrombin index; z = (prednisone)
single_landmark(ld, t_L, window): cohort = subjects with y > t_L and at least one observation in the window
[t_L - W, t_L] (W = t_L: the whole history); history times rescaled to [0, 1] on the window; outcome = residual time
y - t_L (no horizon). This reproduces the Rounds 1-9 PBC cohort (futime > 730, visits with day <= 730, t / 730).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class LongData:
    name: str
    ids: np.ndarray
    times: list
    values: list
    z: np.ndarray
    y: np.ndarray
    d: np.ndarray
    channels: tuple
    z_names: tuple
    unit: str
    meta: dict = field(default_factory=dict)


@dataclass
class Landmarked:
    name: str
    ids: np.ndarray
    times: list              # rescaled to [0, 1] on the window
    values: list             # (m, D) with NaN
    z: np.ndarray
    y: np.ndarray            # residual time after the landmark
    delta: np.ndarray
    channels: tuple
    z_names: tuple
    t_L: float
    window: float
    meta: dict = field(default_factory=dict)


def _rda(path, key):
    import rdata
    df = rdata.read_rda(str(path))[key].copy()
    df.columns = [str(c) for c in df.columns]
    return df


def load_pbcseq(endpoint="E2"):
    from .pbc import CHANNELS
    df = _rda(ROOT / "data/raw/pbc/data/pbc.rda", "pbcseq").sort_values(["id", "day"])
    ids, times, values, z, y, d = [], [], [], [], [], []
    for pid, g in df.groupby("id", sort=True):
        vals = []
        for col, lg in CHANNELS:
            v = g[col].to_numpy(float)
            ok = np.isfinite(v) & ((v > 0) if lg else True)
            vals.append(np.where(ok, np.log(np.where(ok, v, 1.0)) if lg else v, np.nan))
        st = int(g.status.iloc[0])
        ids.append(int(pid)); times.append(g.day.to_numpy(float)); values.append(np.column_stack(vals))
        z.append([float(g.age.iloc[0]), 1.0 if str(g.sex.iloc[0]).lower().startswith("f") else 0.0])
        y.append(float(g.futime.iloc[0])); d.append(st == 2 if endpoint == "E1" else st in (1, 2))
    return LongData(f"pbc_{endpoint}", np.array(ids), times, values, np.array(z), np.array(y), np.array(d),
                    tuple(("log " if lg else "") + c for c, lg in CHANNELS), ("age", "female"), "days",
                    dict(source="R survival 3.8-12 pbcseq", endpoint=endpoint))


def load_aids():
    df = _rda(ROOT / "data/raw/jm/data/aids.rda", "aids").sort_values(["patient", "obstime"])
    ids, times, values, z, y, d = [], [], [], [], [], []
    for pid, g in df.groupby("patient", sort=True, observed=True):
        ids.append(int(pid)); times.append(g.obstime.to_numpy(float)); values.append(g[["CD4"]].to_numpy(float))
        r = g.iloc[0]
        z.append([float(str(r.drug) == "ddI"), float(str(r.gender) == "male"), float(str(r.prevOI) == "AIDS"),
                  float(str(r.AZT) == "failure")])
        y.append(float(r.Time)); d.append(bool(r.death == 1))
    return LongData("aids", np.array(ids), times, values, np.array(z), np.array(y), np.array(d), ("sqrt CD4",),
                    ("ddI", "male", "prevOI_AIDS", "AZT_failure"), "months", dict(source="R JM 1.5-2 aids"))


def load_prothro():
    df = _rda(ROOT / "data/raw/jm/data/prothro.rda", "prothro").sort_values(["id", "time"])
    ids, times, values, z, y, d = [], [], [], [], [], []
    for pid, g in df.groupby("id", sort=True, observed=True):
        ids.append(int(pid)); times.append(g.time.to_numpy(float)); values.append(g[["pro"]].to_numpy(float))
        r = g.iloc[0]
        z.append([float(str(r.treat) == "prednisone")])
        y.append(float(r.Time)); d.append(bool(r.death == 1))
    return LongData("prothro", np.array(ids), times, values, np.array(z), np.array(y), np.array(d),
                    ("prothrombin index",), ("prednisone",), "years", dict(source="R JM 1.5-2 prothro"))


LOADERS = {"pbc_E2": lambda: load_pbcseq("E2"), "pbc_E1": lambda: load_pbcseq("E1"), "aids": load_aids,
           "prothro": load_prothro}


def single_landmark(ld, t_L, window=None):
    W = float(t_L if window is None else window)
    keep, times, values = [], [], []
    for i in range(len(ld.ids)):
        t = np.asarray(ld.times[i], float)
        v = np.asarray(ld.values[i], float)
        m = (t >= t_L - W) & (t <= t_L) & np.isfinite(v).any(1)
        if ld.y[i] > t_L and m.any():
            keep.append(i); times.append((t[m] - (t_L - W)) / W); values.append(v[m])
    keep = np.array(keep)
    return Landmarked(ld.name, ld.ids[keep], times, values, ld.z[keep], ld.y[keep] - t_L, ld.d[keep], ld.channels,
                      ld.z_names, float(t_L), W, dict(ld.meta, n_all=len(ld.ids), n_cohort=len(keep)))
