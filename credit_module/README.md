# Module A: Credit Risk Modelling & Portfolio Stress Testing (existing project)

This is the **original credit-risk project, kept unchanged**. The market module (Module B) extends its quantitative framework to market data; it does not replace or rewrite it.

| File | What it is |
|---|---|
| `../notebooks/01_credit_risk.ipynb` | The original notebook, exactly as provided (LendingClub 2007–2018Q4) |
| `Credit_Risk_Project_Report.pdf` | The original results write-up |

## Data

LendingClub `accepted_2007_to_2018Q4.csv.gz`, publicly distributed through Kaggle (account required). It is not redistributed here. Place it next to the notebook to re-run.

## Headline results (from the original report; not re-run in this repository)

| Area | Result |
|---|---|
| PD discrimination, 2017 out-of-time sample (logistic regression) | AUC 0.6999, KS 0.2907, Gini 0.3997, Brier 0.1638 |
| Extended OOT run | AUC 0.6992, Brier 0.1639; mean PD 19.91% vs observed default rate 23.13% |
| Calibration | Observed default rate above predicted PD in **all 10** risk deciles (underprediction) |
| 2017 aggregate backtest | Expected 33,719 defaults vs actual 39,169 |
| Vintage AUC | 0.731 (2015), 0.708 (2016), 0.699 (2017), 0.694 (2018) |
| PSI vs 2016 | ≤ 0.027 for every vintage |
| LGD / ECL | Mean realised LGD 93.29%; portfolio ECL ≈ $494.68M (funded amount as EAD proxy) |
| Monte Carlo, ρ = 0.15 (10,000 loans × 2,000 scenarios) | VaR99 $68.54M, ES99 $74.61M; stressed (PD odds ×1.5, LGD +5 pp): VaR99 $83.01M, ES99 $90.33M |
| Dependence sensitivity | VaR99 rises from $30.46M at ρ = 0 to $108.36M at ρ = 0.5 |

## Reproducibility gaps found while integrating (please fix before an interview)

1. **The LightGBM challenger is not in the notebook.** The report and CV cite LightGBM (AUC 0.7080, KS 0.3012), but `01_credit_risk.ipynb` only contains the logistic-regression model. Add the LightGBM code you used, or stop citing those numbers as reproducible.
2. **The notebook has no saved outputs.** Re-run it end to end and save it with outputs, so a reviewer can see the numbers without the 1.3M-row dataset.
3. **Development sample.** The extended validation fits on a 300,000-row random sample of pre-2017 loans. That is a sample of the training period, not a random train/test split, so it is fine, but say so explicitly when asked.
