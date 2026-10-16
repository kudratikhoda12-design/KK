"""PBC sequential data loader (Stage G; R package `survival` 3.8-12, object `pbcseq`).

Endpoints (D9-R unresolved; both analysed as separately labelled sensitivity analyses, neither definitive):
  E1: death; transplant treated as censoring (estimand: survival in a hypothetical world without transplant;
      requires transplant to be conditionally independent censoring, which is doubtful).
  E2: death or transplant, composite (estimand: transplant-free survival).
E3 (competing risks) is NOT implemented.
Landmark (D26): s = 730 days; inputs = visits with day <= 730; cohort = patients with futime > 730.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Observed

CHANNELS = [("bili", True), ("albumin", False), ("alk.phos", True), ("ast", True), ("platelet", False),
            ("protime", True)]          # (column, log-transform)


def read_pbcseq(rda_path) -> pd.DataFrame:
    import rdata
    conv = rdata.read_rda(str(rda_path))
    df = conv["pbcseq"].copy()
    df.columns = [str(c) for c in df.columns]
    return df


def describe(df: pd.DataFrame) -> dict:
    per = df.groupby("id").size()
    pt = df.groupby("id").agg(futime=("futime", "first"), status=("status", "first"),
                              n_fut=("futime", "nunique"), n_st=("status", "nunique"), last_day=("day", "max"))
    return dict(n_rows=int(len(df)), n_patients=int(df.id.nunique()), columns=list(df.columns),
                visits=dict(min=int(per.min()), median=float(per.median()), max=int(per.max())),
                status_counts={int(k): int(v) for k, v in pt.status.value_counts().sort_index().items()},
                futime=dict(min=float(pt.futime.min()), median=float(pt.futime.median()), max=float(pt.futime.max())),
                futime_status_constant_within_patient=bool((pt.n_fut == 1).all() and (pt.n_st == 1).all()),
                visits_after_followup=int((pt.last_day > pt.futime).sum()),
                missing_fraction={c: float(df[c].isna().mean()) for c, _ in CHANNELS + [("chol", False)]})


def landmark_dataset(df: pd.DataFrame, endpoint: str, s: float = 730.0) -> Observed:
    if endpoint not in ("E1", "E2"):
        raise ValueError("endpoint must be 'E1' or 'E2' (E3 competing risks is not implemented)")
    df = df.sort_values(["id", "day"])
    pt = df.groupby("id").agg(futime=("futime", "first"), status=("status", "first"), age=("age", "first"),
                              sex=("sex", "first"))
    cohort = pt[pt.futime > s].index.to_numpy()
    D = len(CHANNELS)
    rows = [df[(df.id == i) & (df.day <= s)] for i in cohort]
    L = max(len(r) for r in rows)
    N = len(cohort)
    times, values, mask = np.zeros((N, L)), np.zeros((N, L, D)), np.zeros((N, L, D), bool)
    n_obs = np.zeros(N, int)
    for k, r in enumerate(rows):
        n = len(r)
        times[k, :n] = r.day.to_numpy(float) / s
        for d, (col, lg) in enumerate(CHANNELS):
            v = r[col].to_numpy(float)
            ok = np.isfinite(v) & ((v > 0) if lg else True)
            values[k, :n, d] = np.where(ok, np.log(np.where(ok, v, 1.0)) if lg else v, 0.0)
            mask[k, :n, d] = ok
        n_obs[k] = n
    status = pt.loc[cohort, "status"].to_numpy(int)
    delta = status == 2 if endpoint == "E1" else np.isin(status, (1, 2))
    sex = pt.loc[cohort, "sex"].astype(str).str.lower().map({"f": 1.0, "m": 0.0}).to_numpy(float)
    u = np.column_stack([pt.loc[cohort, "age"].to_numpy(float), sex])
    y = pt.loc[cohort, "futime"].to_numpy(float) - s
    return Observed(cohort.astype(int), times, values, mask, n_obs, y, delta, u,
                    tuple(("log " if lg else "") + c for c, lg in CHANNELS),
                    {"source": "R survival 3.8-12 pbcseq", "endpoint": endpoint, "landmark_days": s,
                     "status": status, "max_input_day": np.array([r.day.max() for r in rows]),
                     "visits_in_window": n_obs.copy()})
