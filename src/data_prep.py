"""
Build a clean quarterly macroeconomic panel from raw FRED downloads and a
single systematic "credit-stress" index Z_t used later to condition rating
migration probabilities on the macro-economic environment.

Series (source: FRED, https://fred.stlouisfed.org, downloaded via the public
fredgraph.csv endpoint, no API key required):
    UNRATE               Civilian unemployment rate, monthly, SA (%)
    A191RL1Q225SBEA      Real GDP growth, quarterly, SAAR (%)
    BAA10Y               Moody's Baa corporate bond yield minus 10Y Treasury
                         yield ("credit spread"), daily (%)
    USREC                NBER recession indicator, monthly (0/1)

Sample window: 1990Q1-2024Q4 (clean, well-documented, avoids thin early
post-war data and avoids unrevised/very-latest data points).
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
START, END = "1990-01-01", "2024-12-31"


def _load(series):
    df = pd.read_csv(DATA_DIR / f"{series}.csv", parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", series: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"]


def _to_quarterly(monthly, how="mean"):
    q = monthly.resample("QE").agg(how)
    q.index = q.index.to_period("Q").to_timestamp("Q")
    return q


def build_macro_panel():
    unrate = _load("UNRATE")
    gdp_growth = _load("A191RL1Q225SBEA")
    spread = _load("BAA10Y")
    rec = _load("USREC")

    unrate_q = _to_quarterly(unrate, "mean")
    spread_q = _to_quarterly(spread, "mean")
    rec_q = _to_quarterly(rec, "max")

    gdp_growth.index = gdp_growth.index.to_period("Q").to_timestamp("Q")
    gdp_q = gdp_growth

    df = pd.DataFrame({
        "unemployment_rate": unrate_q,
        "gdp_growth": gdp_q,
        "credit_spread": spread_q,
        "recession": rec_q,
    })
    df = df.loc[START:END].dropna()

    # Systematic credit-stress index: standardized blend of the YoY change in
    # unemployment (rising unemployment = deteriorating credit conditions)
    # and the level of the Baa-Treasury credit spread (widening = stress).
    d_unrate = df["unemployment_rate"].diff(4)
    z_dunrate = (d_unrate - d_unrate.mean()) / d_unrate.std()
    z_spread = (df["credit_spread"] - df["credit_spread"].mean()) / df["credit_spread"].std()
    stress = 0.5 * z_dunrate.fillna(0) + 0.5 * z_spread
    stress = (stress - stress.mean()) / stress.std()  # re-standardize, mean 0 sd 1
    df["credit_stress_index"] = stress
    # Z: systematic macro factor used in the single-factor migration model.
    # Positive Z = favourable (benign) credit conditions, negative Z = stress.
    # Winsorised at +/-3 standard deviations: the YoY-change construction of
    # the stress index makes single-quarter "shock" observations (e.g. the
    # 2020Q2 unemployment spike) mechanically very extreme, so capping avoids
    # one data point dominating the conditional migration model below.
    df["Z"] = (-df["credit_stress_index"]).clip(-3, 3)

    df.to_csv(DATA_DIR / "macro_quarterly.csv")
    return df


if __name__ == "__main__":
    panel = build_macro_panel()
    print(panel.describe())
    print(panel.tail())
    print(f"\nSaved {len(panel)} quarterly observations to data/macro_quarterly.csv")
