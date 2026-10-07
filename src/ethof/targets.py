"""Forward-return targets.

    fwd_ret_h(t) = (P[t+h] - P[t]) / P[t],  P = px (last trade price at or before the bar close)

Row t's target uses only prices after the decision time tau_t = ts_t + 1 min, up to tau_{t+h}.
Rows whose horizon runs past the end of the available data, or past a split boundary, have a null
target (handled by `label_end_ts` + purging in validation.py).

Classification:
    binary  : up_h = 1{fwd_ret_h > 0}                       (primary; costs enter in the trading rule)
    3-class : DOWN / NEUTRAL / UP with a neutral band of +-band (fixed a priori = base round-trip cost)
"""
from __future__ import annotations

import polars as pl

HORIZONS = (5, 15, 30, 60)


def add_targets(df: pl.DataFrame, horizons=HORIZONS, neutral_band: float = 0.0011) -> pl.DataFrame:
    px = pl.col("px")
    exprs = []
    for h in horizons:
        fr = px.shift(-h) / px - 1
        exprs += [
            fr.alias(f"fwd_ret_{h}"),
            pl.when(fr.is_null()).then(None).otherwise((fr > 0).cast(pl.Int8)).alias(f"up_{h}"),
            pl.when(fr.is_null()).then(None)
              .when(fr > neutral_band).then(pl.lit(2, pl.Int8))
              .when(fr < -neutral_band).then(pl.lit(0, pl.Int8))
              .otherwise(pl.lit(1, pl.Int8)).alias(f"cls3_{h}"),
            (pl.col("ts") + pl.duration(minutes=h + 1)).alias(f"label_end_{h}"),
        ]
    return df.with_columns(exprs)
