# CoxKAN: Interpretable Time-to-Event Modelling of Customer Attrition

Customer churn treated as a survival problem. The question is not just whether a customer
will leave, but when, given everything known about them when the account was opened.
The model is a **Cox proportional-hazards model whose log-risk function is a Kolmogorov–Arnold
Network (KAN) with B-spline activations** (CoxKAN). It is benchmarked against **CoxPH** and
**DeepSurv** using the concordance index (C-index).

**Data:** LendingClub consumer loans, 2016 origination vintage. The repository ships a
deterministic **50,000-loan sample** (`data/lendingclub_2016_sample.parquet`).

**Attrition event:** the customer **prepays** the loan (pays it off at least 3 months
before maturity). The bank loses the remaining interest income and the relationship ends.
Loans that are still active, that matured on schedule, or that defaulted are **right-censored**.

> Full write-up with theory, dataset analysis, results and interview prep:
> **[`reports/REPORT.md`](reports/REPORT.md)** (also available as `reports/REPORT.pdf` / `REPORT.html`).

## Results (held-out test set, 10,000 loans)

| Model | Harrell C-index (95% CI) | Uno C | IBS (3–36 m) |
|---|---|---:|---:|
| Kaplan–Meier (no covariates) | 0.500 | 0.500 | 0.1804 |
| CoxPH | 0.6140 (0.606–0.623) | 0.6128 | 0.1698 |
| **CoxKAN [d,1] additive** | **0.6153** (0.606–0.625) | 0.6139 | 0.1692 |
| CoxKAN [d,4,1] | 0.6172 (0.609–0.626) | 0.6158 | 0.1689 |
| DeepSurv | 0.6195 (0.611–0.628) | 0.6183 | 0.1685 |

* The additive CoxKAN matches CoxPH and stays fully transparent. Its closed-form symbolic version keeps C = 0.6147.
* **Top 5 attrition drivers** (HR per +1 SD): loan term 0.78 · open-account ratio 0.84 · interest rate 1.17 ·
  revolving utilisation 0.87 · credit-history length 0.89. The top 4 are identical across CoxPH, CoxKAN and DeepSurv.
* CoxKAN risk quintiles on the test set: 78% vs 46% of customers retained at 24 months (lowest vs highest risk).

![CoxKAN shape functions](reports/figures/13_coxkan_shape_functions.png)

## Project layout

```
sql/
  01_extract_cohort.sql          # vintage filter + deterministic sample (DuckDB over the raw CSV)
  02_build_survival_table.sql    # duration, event flag, censoring reason
  03_feature_engineering.sql     # ratios, CASE parsing, window features (rate spread vs. peers, ...)
src/coxkan_churn/
  config.py      # paths, feature lists, tuned hyper-parameters
  data.py        # runs the SQL, fit-on-train preprocessing, stratified split
  kan.py         # KAN layer: B-spline basis (Cox-de Boor), SiLU residual, sparsity regulariser
  models.py      # Cox partial-likelihood loss (Breslow ties), CoxPH, DeepSurv, CoxKAN
  metrics.py     # Harrell / Uno C-index, Breslow baseline, IPCW Brier score, IBS
  symbolic.py    # symbolic regression of learned spline functions -> closed-form formula
  eda.py         # survival EDA: KM, Nelson-Aalen, hazard, log-rank, VIF, univariate Cox
  plotting.py    # shared figure style
scripts/
  01_extract_and_engineer.py   02_tune.py   03_eda.py   04_train_evaluate.py   05_interpret.py
reports/
  REPORT.md / REPORT.pdf / REPORT.html, figures/, results/ (all CSV outputs)
models/          # trained weights (CoxPH pickle, torch state dicts, preprocessor)
```

## Reproduce

```bash
pip install -r requirements.txt
./run_pipeline.sh                 # SQL -> EDA -> train/benchmark -> interpretation  (~15 min on 4 CPUs)
TUNE=1 ./run_pipeline.sh          # also re-run the validation grid search (~25 min)
./reports/build_report.sh         # re-render REPORT.html / REPORT.pdf (needs pandoc + chromium)
```

To rebuild the 50k sample from the public file (`accepted_2007_to_2018Q4.csv`, available on
Kaggle as *"All Lending Club loan data"* and on Hugging Face as
`codesignal/lending-club-loan-accepted`):

```bash
RAW_CSV=/path/to/accepted_2007_to_2018Q4.csv ./run_pipeline.sh
```
