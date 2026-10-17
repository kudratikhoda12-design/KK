# Stochastic Credit-Rating Migration and Default Forecasting

A stochastic framework for modelling credit-rating migration and multi-period default
probability, combining ARIMA/SARIMA macroeconomic forecasting with a macro-conditional
Markov-chain credit-migration model (Vasicek / Belkin-Suchower-Forest single-factor
approach), validated against a static Markov chain and a conventional logistic-regression
default model.

See `report/Credit_Migration_Report.pdf` for the full write-up (objective, data, methodology,
results, validation, limitations).

## Pipeline

```
python3 src/data_prep.py              # FRED macro data -> data/macro_quarterly.csv
python3 src/simulate_credit_panel.py  # calibrated macro-conditional panel simulation
python3 src/time_series_models.py     # AR/MA/ARIMA/SARIMA grid search + forecasts
python3 src/markov_analysis.py        # transition matrix estimation, PD term structure, scenarios
python3 src/validation.py             # backtest + logistic-regression baseline comparison
python3 src/make_figures.py           # all report figures
python3 report/build_report.py        # assemble + render the PDF report
```

## Layout

- `src/` - all modelling code (see file docstrings)
- `data/` - raw FRED downloads, derived macro panel, simulated credit-rating panel, forecasts
- `outputs/figures/`, `outputs/tables/` - generated figures and result tables
- `report/` - report HTML source and the final PDF deliverable
