"""
Model validation.

Part A - Backtest of multi-period default forecasts:
  Split the panel at TRAIN_END. Estimate an *unconditional* (static TTC)
  transition matrix on the training window only, and compare its
  matrix-power multi-period default forecast for the test window against
  (i) the macro-conditional forecast (using the *realised* macro path over
  the test window - i.e. an ex-post conditioning check of the mechanism)
  and (ii) the realised empirical cumulative default rate in the test data.

Part B - Baseline comparison vs a conventional default-prediction model:
  One-step-ahead default classification. Compare the macro-conditional
  Markov model's implied P(default next quarter | rating, macro) against a
  logistic regression fitted on rating + raw macro variables (a standard
  reduced-form / scorecard-style default model), both evaluated purely
  out-of-sample on the test window, using AUC, Brier score and log-loss.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from markov_models import (
    ttc_quarterly_matrix, conditional_matrix, multiperiod_matrix,
    estimate_transition_matrix_cohort, GOOD, WATCH, DEFAULT,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2015-12-31"   # train: 1990Q1-2015Q4 (104 qtrs), test: 2016Q1-2024Q4 (36 qtrs)


def load_panel():
    return pd.read_csv(DATA_DIR / "simulated_rating_panel.csv", parse_dates=["quarter"])


def load_macro():
    return pd.read_csv(DATA_DIR / "macro_quarterly.csv", parse_dates=["date"]).set_index("date")


# ---------------------------------------------------------------- Part A --
def backtest_multiperiod(panel, macro):
    train_panel = panel[panel["quarter"] <= TRAIN_END]
    test_panel = panel[panel["quarter"] > TRAIN_END]
    test_quarters = sorted(test_panel["quarter"].unique())
    n_test = len(test_quarters)

    P_hat_train, _ = estimate_transition_matrix_cohort(train_panel)

    # Realised empirical cumulative default rate in the test window, by
    # starting rating as of the first test quarter.
    first_q = test_quarters[0]
    start_state = test_panel[test_panel["quarter"] == first_q].set_index("borrower_id")["state"]
    results = []
    for start_name, s in [("Good", GOOD), ("Watchlist", WATCH)]:
        cohort_ids = start_state[start_state == s].index
        cohort_panel = test_panel[test_panel["borrower_id"].isin(cohort_ids)]
        ever_default = cohort_panel.groupby("borrower_id")["state"].apply(lambda x: (x == DEFAULT).any())
        realised_cum_default = ever_default.mean()

        # Unconditional (static) multi-period forecast using the train-period matrix
        Pn = multiperiod_matrix(P_hat_train, n_test)
        unconditional_pred = Pn[s, DEFAULT]

        # Macro-conditional forecast: apply the realised test-window macro
        # path quarter by quarter to the same train-estimated TTC-shaped
        # matrix (re-expressed via conditional_matrix using actual Z_t).
        P_ttc = ttc_quarterly_matrix()
        v = np.zeros(3)
        v[s] = 1.0
        for q in test_quarters:
            Z = macro.loc[q, "Z"]
            v = v @ conditional_matrix(P_ttc, Z, rho=0.12)
        conditional_pred = v[DEFAULT]

        results.append({
            "start_rating": start_name, "test_horizon_quarters": n_test,
            "realised_cum_default_rate": realised_cum_default,
            "unconditional_markov_pred": unconditional_pred,
            "macro_conditional_markov_pred": conditional_pred,
            "abs_error_unconditional": abs(unconditional_pred - realised_cum_default),
            "abs_error_conditional": abs(conditional_pred - realised_cum_default),
        })
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "backtest_multiperiod.csv", index=False)
    return df


# ---------------------------------------------------------------- Part B --
def _build_onestep_dataset(panel, macro):
    panel = panel.sort_values(["borrower_id", "quarter"]).copy()
    panel["next_state"] = panel.groupby("borrower_id")["state"].shift(-1)
    panel = panel.dropna(subset=["next_state"])
    panel = panel[panel["state"] != DEFAULT]
    panel["target_default"] = (panel["next_state"] == DEFAULT).astype(int)
    panel["is_watchlist"] = (panel["state"] == WATCH).astype(int)
    macro_feats = macro[["unemployment_rate", "gdp_growth", "credit_spread", "Z"]]
    panel = panel.merge(macro_feats, left_on="quarter", right_index=True, how="left")
    return panel


def baseline_comparison(panel, macro):
    ds = _build_onestep_dataset(panel, macro)
    train = ds[ds["quarter"] <= TRAIN_END]
    test = ds[ds["quarter"] > TRAIN_END]

    feat_cols = ["is_watchlist", "unemployment_rate", "gdp_growth", "credit_spread"]
    logit = LogisticRegression(max_iter=1000)
    logit.fit(train[feat_cols], train["target_default"])
    logit_pred = logit.predict_proba(test[feat_cols])[:, 1]

    P_ttc = ttc_quarterly_matrix()
    def markov_pred(row):
        Pc = conditional_matrix(P_ttc, row["Z"], rho=0.12)
        state = WATCH if row["is_watchlist"] else GOOD
        return Pc[state, DEFAULT]
    markov_pred_vals = test.apply(markov_pred, axis=1).values

    y_true = test["target_default"].values
    rows = []
    for name, pred in [("Macro-conditional Markov model", markov_pred_vals),
                        ("Logistic regression (rating + macro)", logit_pred)]:
        rows.append({
            "model": name,
            "auc": roc_auc_score(y_true, pred),
            "brier_score": brier_score_loss(y_true, pred),
            "log_loss": log_loss(y_true, pred, labels=[0, 1]),
            "n_obs": len(y_true), "n_defaults": int(y_true.sum()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "baseline_comparison.csv", index=False)
    return df, logit.coef_, feat_cols


def main():
    panel = load_panel()
    macro = load_macro()

    print("=== Part A: multi-period backtest (train<=2015Q4, test=2016Q1-2024Q4) ===")
    bt = backtest_multiperiod(panel, macro)
    print(bt.to_string(index=False))

    print("\n=== Part B: one-step-ahead default prediction, OOS 2016Q1-2024Q4 ===")
    cmp_df, coefs, feat_cols = baseline_comparison(panel, macro)
    print(cmp_df.to_string(index=False))
    print("\nLogistic regression coefficients:", dict(zip(feat_cols, coefs[0].round(3))))


if __name__ == "__main__":
    main()
