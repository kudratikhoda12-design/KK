"""
Estimate the empirical (cohort) transition matrix from the simulated panel,
build multi-period transition/default-probability term structures, and
produce macro-conditional forward default-probability forecasts under a
baseline and a stress macroeconomic scenario (using the ARIMA/SARIMA
forecasts from time_series_models.py).
"""
import numpy as np
import pandas as pd
from pathlib import Path

from markov_models import (
    ttc_quarterly_matrix, implied_annual_matrix, conditional_matrix,
    multiperiod_matrix, cumulative_default_prob,
    estimate_transition_matrix_cohort, STATES, GOOD, WATCH, DEFAULT,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 4, 8, 12, 16, 20]  # quarters


def load_panel():
    df = pd.read_csv(DATA_DIR / "simulated_rating_panel.csv", parse_dates=["quarter"])
    return df


def load_macro():
    return pd.read_csv(DATA_DIR / "macro_quarterly.csv", parse_dates=["date"]).set_index("date")


def term_structure_table(P, label):
    rows = []
    for start_name, start_state in [("Good", GOOD), ("Watchlist", WATCH)]:
        pds = cumulative_default_prob(P, max(HORIZONS), start_state)
        for h in HORIZONS:
            rows.append({"matrix": label, "start_rating": start_name,
                          "horizon_quarters": h, "cum_default_prob": pds[h - 1]})
    return pd.DataFrame(rows)


def build_forward_macro_path(macro, scenario="baseline"):
    """Approximate a forward Z path from the unemployment-rate ARIMA/SARIMA
    forecast, holding the credit spread at its last observed level (no
    spread forecast was produced) and reusing the historical mean/sd used to
    build the historical Z index, so forecast and history are on the same
    scale. 'stress' adds a phased-in +3pp unemployment shock relative to the
    baseline forecast, i.e. a hypothetical moderate recession scenario."""
    unrate_hist = macro["unemployment_rate"]
    spread_last = macro["credit_spread"].iloc[-1]
    d_unrate_hist = unrate_hist.diff(4)
    mu_d, sd_d = d_unrate_hist.mean(), d_unrate_hist.std()
    mu_s, sd_s = macro["credit_spread"].mean(), macro["credit_spread"].std()

    fc = pd.read_csv(DATA_DIR / "unemployment_rate_baseline_forecast.csv",
                      index_col=0, parse_dates=True).iloc[:, 0]
    if scenario == "stress":
        shock = np.minimum(np.arange(1, len(fc) + 1) * 0.75, 3.0)  # ramps to +3pp by Q4
        fc = fc + shock

    # YoY change vs the last 4 *historical* unemployment observations, chained
    # forward for quarters beyond the first year of the forecast.
    full_unrate = pd.concat([unrate_hist, fc])
    d_unrate_fc = full_unrate.diff(4).loc[fc.index]

    z_dunrate = (d_unrate_fc - mu_d) / sd_d
    z_spread = (spread_last - mu_s) / sd_s  # spread held flat at last observed level
    stress_idx = 0.5 * z_dunrate + 0.5 * z_spread
    Z = (-stress_idx).clip(-3, 3)
    return Z.rename("Z")


def forward_default_forecast(P_ttc, Z_path, rho=0.12):
    """Non-homogeneous Markov forecast: apply a different conditional matrix
    each quarter (driven by the forecast Z path) and track the resulting
    state distribution / cumulative default probability starting from each
    rating."""
    horizons = list(range(1, len(Z_path) + 1))
    out = {}
    for start_name, start_state in [("Good", GOOD), ("Watchlist", WATCH)]:
        v = np.zeros(3)
        v[start_state] = 1.0
        cum_default = []
        for Z in Z_path.values:
            P_cond = conditional_matrix(P_ttc, Z, rho=rho)
            v = v @ P_cond
            cum_default.append(v[DEFAULT])
        out[start_name] = cum_default
    df = pd.DataFrame(out, index=Z_path.index)
    df.index.name = "quarter"
    return df


def main():
    panel = load_panel()
    macro = load_macro()
    P_ttc = ttc_quarterly_matrix()

    # --- Cohort estimation on the full sample, vs the calibration input ---
    P_hat, counts = estimate_transition_matrix_cohort(panel)
    pd.DataFrame(P_ttc, index=STATES, columns=STATES).to_csv(OUT_DIR / "P_ttc_input.csv")
    pd.DataFrame(P_hat, index=STATES, columns=STATES).to_csv(OUT_DIR / "P_hat_full_sample.csv")
    pd.DataFrame(counts, index=STATES, columns=STATES).to_csv(OUT_DIR / "transition_counts_full_sample.csv")
    print("Input (calibration) quarterly TTC matrix:\n", np.round(P_ttc, 4))
    print("\nCohort-estimated quarterly matrix (full sample, pooled across macro states):\n",
          np.round(P_hat, 4))
    print("\nImplied annual matrix (from TTC):\n", np.round(implied_annual_matrix(P_ttc), 4))

    # --- Multi-period transition / default term structure (unconditional TTC) ---
    ts_ttc = term_structure_table(P_ttc, "TTC (unconditional)")
    ts_hat = term_structure_table(P_hat, "Cohort-estimated (full sample)")
    term_structure = pd.concat([ts_ttc, ts_hat], ignore_index=True)
    term_structure.to_csv(OUT_DIR / "multiperiod_pd_term_structure.csv", index=False)
    print("\nMulti-period cumulative default probability term structure (TTC matrix):")
    print(ts_ttc.pivot(index="horizon_quarters", columns="start_rating", values="cum_default_prob"))

    # --- Macro-conditional forward scenarios ---
    Z_baseline = build_forward_macro_path(macro, "baseline")
    Z_stress = build_forward_macro_path(macro, "stress")
    fwd_baseline = forward_default_forecast(P_ttc, Z_baseline)
    fwd_stress = forward_default_forecast(P_ttc, Z_stress)

    fwd_baseline.to_csv(OUT_DIR / "forward_cum_default_baseline.csv")
    fwd_stress.to_csv(OUT_DIR / "forward_cum_default_stress.csv")
    pd.DataFrame({"baseline_Z": Z_baseline, "stress_Z": Z_stress}).to_csv(OUT_DIR / "forward_Z_scenarios.csv")

    print("\nForward (8-quarter) cumulative default probability, baseline vs stress scenario:")
    comp = pd.concat([fwd_baseline.add_suffix("_baseline"), fwd_stress.add_suffix("_stress")], axis=1)
    print(comp.round(4))


if __name__ == "__main__":
    main()
