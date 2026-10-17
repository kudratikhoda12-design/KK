"""
Simulate a quarterly, borrower-level credit-rating panel (1990Q1-2024Q4).

No proprietary rating-agency transition dataset was available for this
project (see report, "Data" section), so the panel is generated from a
calibrated, macro-conditional Markov model: the TTC matrix in
markov_models.ttc_quarterly_matrix(), shifted every quarter by the realised
systematic factor Z_t (src/data_prep.py) through the single-factor threshold
model in markov_models.conditional_matrix(). This produces a realistic,
clustered pattern of downgrades/defaults around the 1990-91, 2001, 2008-09
and 2020 macro episodes, which is what the rest of the project analyses.

The portfolio is a replenishing portfolio of N_BORROWERS obligors: a
defaulted obligor is treated as resolved (charged off) and immediately
replaced by a newly originated obligor, so the cross-sectional composition
of the book stays roughly stationary, as in a real revolving loan book.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from markov_models import ttc_quarterly_matrix, conditional_matrix, GOOD, WATCH, DEFAULT

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
N_BORROWERS = 3000
RHO = 0.12
SEED = 42
ORIGINATION_DIST = [0.65, 0.35, 0.0]  # new borrowers: 65% Good, 35% Watchlist


def simulate():
    rng = np.random.default_rng(SEED)
    macro = pd.read_csv(DATA_DIR / "macro_quarterly.csv", parse_dates=["date"]).set_index("date")
    quarters = macro.index
    T = len(quarters)
    P_ttc = ttc_quarterly_matrix()

    state = rng.choice([GOOD, WATCH], size=N_BORROWERS, p=[0.65, 0.35])
    records = np.empty((N_BORROWERS, T), dtype=np.int8)
    records[:, 0] = state

    for t in range(1, T):
        Z = macro["Z"].iloc[t]
        P_cond = conditional_matrix(P_ttc, Z, rho=RHO)
        new_state = state.copy()
        for s in (GOOD, WATCH):
            mask = state == s
            n = mask.sum()
            if n > 0:
                new_state[mask] = rng.choice(3, size=n, p=P_cond[s])
        defaulted = state == DEFAULT
        if defaulted.any():
            new_state[defaulted] = rng.choice(3, size=defaulted.sum(), p=ORIGINATION_DIST)
        state = new_state
        records[:, t] = state

    long_df = pd.DataFrame(records, columns=quarters)
    long_df.insert(0, "borrower_id", np.arange(N_BORROWERS))
    long_df = long_df.melt(id_vars="borrower_id", var_name="quarter", value_name="state")
    long_df["quarter"] = pd.to_datetime(long_df["quarter"])
    long_df = long_df.sort_values(["borrower_id", "quarter"]).reset_index(drop=True)
    long_df["rating"] = long_df["state"].map({GOOD: "Good", WATCH: "Watchlist", DEFAULT: "Default"})

    long_df.to_csv(DATA_DIR / "simulated_rating_panel.csv", index=False)
    return long_df


if __name__ == "__main__":
    df = simulate()
    print(df.head())
    print(f"\n{df['borrower_id'].nunique()} borrowers x {df['quarter'].nunique()} quarters "
          f"= {len(df):,} obligor-quarter observations")
    print("\nOverall rating distribution:")
    print(df["rating"].value_counts(normalize=True))
    print("\nDefault-quarter rate by year (share of obligor-quarters newly defaulting):")
    df["year"] = df["quarter"].dt.year
    new_default = df.sort_values(["borrower_id", "quarter"])
    new_default["prev_state"] = new_default.groupby("borrower_id")["state"].shift(1)
    new_default["new_default"] = (new_default["state"] == DEFAULT) & (new_default["prev_state"] != DEFAULT)
    print(new_default.groupby("year")["new_default"].mean().round(4).to_string())
