"""
AR / MA / ARIMA / SARIMA forecasting of the two macroeconomic drivers used
later to condition credit migration: the unemployment rate and real GDP
growth. For each series we:
  1. hold out the last HOLDOUT quarters as an out-of-sample test set,
  2. grid-search a small order space for each model family on the training
     data, selecting the in-sample winner within each family by AIC,
  3. forecast across the holdout and score with RMSE / MAE / MAPE,
  4. refit the single best-performing (lowest OOS RMSE) specification on the
     full sample and forecast FORECAST_H quarters beyond 2024Q4 to build a
     baseline and a stress macroeconomic scenario for the Markov model.
"""
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT = 12       # quarters held out for OOS evaluation
FORECAST_H = 8      # quarters forecast beyond end of sample
SERIES = ["unemployment_rate", "gdp_growth"]


def _metrics(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mae = np.mean(np.abs(actual - pred))
    denom = np.where(np.abs(actual) < 1e-6, np.nan, actual)
    mape = np.nanmean(np.abs((actual - pred) / denom)) * 100
    return rmse, mae, mape


def _fit_and_forecast(train, order, seasonal_order=None, h=HOLDOUT):
    if seasonal_order is None:
        model = ARIMA(train, order=order)
    else:
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit()
    fc = fit.get_forecast(steps=h)
    return fit, fc.predicted_mean, fc.conf_int(alpha=0.05)


def _grid_search(train, test, family):
    """Search a small order grid for one model family; return list of
    dict rows (order, aic, rmse/mae/mape on test) and the best row."""
    rows = []
    if family == "AR":
        grid = [((p, 0, 0), None) for p in range(1, 5)]
    elif family == "MA":
        grid = [((0, 0, q), None) for q in range(1, 5)]
    elif family == "ARIMA":
        grid = [((p, d, q), None) for p in range(0, 3) for d in (0, 1)
                for q in range(0, 3) if not (p == 0 and q == 0)]
    elif family == "SARIMA":
        grid = [((p, d, q), (P, D, Q, 4)) for p in (0, 1) for d in (0, 1) for q in (0, 1)
                for P in (0, 1) for D in (0,) for Q in (0, 1)
                if not (p == 0 and q == 0 and P == 0 and Q == 0)]
    else:
        raise ValueError(family)

    for order, seasonal in grid:
        try:
            fit, fc_mean, _ = _fit_and_forecast(train, order, seasonal, h=len(test))
            rmse, mae, mape = _metrics(test, fc_mean)
            rows.append({
                "family": family, "order": str(order),
                "seasonal_order": str(seasonal) if seasonal else "-",
                "aic": fit.aic, "oos_rmse": rmse, "oos_mae": mae, "oos_mape": mape,
            })
        except Exception:
            continue
    if not rows:
        return None, []
    rows_sorted = sorted(rows, key=lambda r: r["aic"])
    best_by_aic = rows_sorted[0]
    return best_by_aic, rows


def run_for_series(series_name, df):
    y = df[series_name].astype(float)
    train, test = y.iloc[:-HOLDOUT], y.iloc[-HOLDOUT:]

    all_rows = []
    family_best = {}
    for family in ["AR", "MA", "ARIMA", "SARIMA"]:
        best, rows = _grid_search(train, test, family)
        all_rows.extend(rows)
        if best is not None:
            family_best[family] = best

    comparison = pd.DataFrame(all_rows).sort_values(["family", "aic"]).reset_index(drop=True)
    comparison.to_csv(OUT_DIR / f"ts_grid_{series_name}.csv", index=False)

    summary = pd.DataFrame(family_best.values())
    summary.insert(0, "series", series_name)
    best_overall = summary.loc[summary["oos_rmse"].idxmin()]

    order = eval(best_overall["order"])
    seasonal = None if best_overall["seasonal_order"] == "-" else eval(best_overall["seasonal_order"])

    # Refit best model on train only -> holdout forecast (for the OOS plot)
    _, test_fc_mean, test_fc_ci = _fit_and_forecast(train, order, seasonal, h=HOLDOUT)

    # Refit on the FULL sample -> genuine future forecast beyond 2024Q4
    fit_full, future_mean, future_ci = _fit_and_forecast(y, order, seasonal, h=FORECAST_H)

    last_date = y.index[-1]
    future_index = pd.date_range(last_date, periods=FORECAST_H + 1, freq="QE")[1:]
    baseline_forecast = pd.Series(future_mean.values, index=future_index, name=series_name)

    test_forecast = pd.Series(test_fc_mean.values, index=test.index, name="forecast")

    return {
        "series": series_name,
        "summary": summary,
        "best_overall": best_overall,
        "train": train, "test": test,
        "test_forecast": test_forecast,
        "test_forecast_ci": pd.DataFrame(test_fc_ci.values, index=test.index,
                                          columns=["lower", "upper"]),
        "baseline_forecast": baseline_forecast,
        "baseline_forecast_ci": pd.DataFrame(future_ci.values, index=future_index,
                                              columns=["lower", "upper"]),
    }


def main():
    df = pd.read_csv(DATA_DIR / "macro_quarterly.csv", parse_dates=["date"]).set_index("date")
    df.index.freq = "QE"

    results = {}
    best_rows = []
    for s in SERIES:
        res = run_for_series(s, df)
        results[s] = res
        best_rows.append(res["best_overall"])
        print(f"\n=== {s} : family leaderboard (best order per family, OOS RMSE) ===")
        print(res["summary"][["family", "order", "seasonal_order", "aic",
                               "oos_rmse", "oos_mae", "oos_mape"]].to_string(index=False))
        print(f"--> selected model: {res['best_overall']['family']} {res['best_overall']['order']} "
              f"{res['best_overall']['seasonal_order']}  OOS RMSE={res['best_overall']['oos_rmse']:.3f}")

    pd.DataFrame(best_rows).to_csv(OUT_DIR / "ts_best_models.csv", index=False)

    # Persist forecasts for the macro-conditional Markov step and for plotting
    for s, res in results.items():
        res["test_forecast"].to_csv(OUT_DIR.parent / "forecasts" if False else DATA_DIR / f"{s}_test_forecast.csv")
        res["baseline_forecast"].to_csv(DATA_DIR / f"{s}_baseline_forecast.csv")
        res["test_forecast_ci"].to_csv(DATA_DIR / f"{s}_test_forecast_ci.csv")
        res["baseline_forecast_ci"].to_csv(DATA_DIR / f"{s}_baseline_forecast_ci.csv")

    return results


if __name__ == "__main__":
    main()
