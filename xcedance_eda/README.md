# Xcedance Financial Dataset – EDA Practice

A **synthetic** retail-lending dataset for the fictional company *Xcedance Financial Services*,
built for practising Exploratory Data Analysis. No real people or companies are represented.

| | |
|---|---|
| File | `xcedance_financial_data.csv` |
| Rows | 5,060 (5,000 applications + 60 duplicate rows) |
| Columns | 50 (1 ID, 1 date, 31 numeric, 16 categorical, 1 target) |
| Target | `loan_default` (1 = defaulted, ~18% positive → imbalanced) |
| Missing cells | ~4.7% overall, 27 columns affected (1% → 62%) |
| Regenerate | `python generate_dataset.py` (fixed seed = 42, reproducible) |

## Data dictionary

| # | Column | Type | Description |
|---|---|---|---|
| 1 | customer_id | ID | Unique customer/application ID (`XCD100000` …) |
| 2 | application_date | Date | Loan application date (2021-01-01 → 2024-12-30) |
| 3 | age | Numeric (int) | Applicant age in years |
| 4 | gender | Categorical | Male / Female |
| 5 | marital_status | Categorical | Single / Married / Divorced / Widowed |
| 6 | dependents | Numeric (int) | Number of dependents |
| 7 | education_level | Ordinal | High School < Diploma < Bachelor < Master < PhD |
| 8 | employment_type | Categorical | Salaried, Self-Employed, Business Owner, Contract, Unemployed, Retired |
| 9 | occupation | Categorical | 14 occupation groups |
| 10 | years_employed | Mixed (text) | Years in current job: `<1`, `1`…`9`, `10+` |
| 11 | annual_income | Numeric | Gross annual income |
| 12 | monthly_income | Numeric | Annual income / 12 |
| 13 | other_income | Numeric | Additional annual income (rent, freelance, …) |
| 14 | region | Categorical | North / South / East / West / Central |
| 15 | city_tier | Ordinal | Tier 1 / Tier 2 / Tier 3 |
| 16 | home_ownership | Categorical | Own / Mortgage / Rent / Living with Family |
| 17 | property_value | Numeric | Value of owned property (owners only) |
| 18 | residence_years | Numeric | Years at current address |
| 19 | credit_score | Numeric (int) | Bureau score, valid range 300–850 |
| 20 | credit_history_years | Numeric | Length of credit history |
| 21 | num_credit_cards | Numeric (int) | Credit cards held |
| 22 | num_open_accounts | Numeric (int) | Open credit lines (cards + loans) |
| 23 | num_bank_accounts | Numeric (int) | Bank accounts held |
| 24 | credit_utilization_ratio | Numeric | Revolving balance / limit, valid range 0–1 |
| 25 | total_credit_limit | Numeric | Sum of credit limits |
| 26 | outstanding_debt | Numeric | Total outstanding debt |
| 27 | debt_to_income_ratio | Numeric | outstanding_debt / annual_income |
| 28 | num_late_payments_12m | Numeric (int) | Late payments in last 12 months |
| 29 | num_delinquencies_2y | Numeric (int) | 30+ day delinquencies in last 2 years |
| 30 | num_credit_inquiries_6m | Numeric (int) | Hard credit inquiries in last 6 months |
| 31 | bankruptcy_flag | Binary | 1 = past bankruptcy |
| 32 | loan_purpose | Categorical | 9 purposes (Debt Consolidation, Car, Home Improvement, …) |
| 33 | loan_amount | Numeric | Requested loan amount |
| 34 | loan_term_months | Discrete | 12 / 24 / 36 / 48 / 60 |
| 35 | interest_rate | Numeric | Annual interest rate (%) |
| 36 | monthly_emi | Numeric | Monthly instalment (amortisation formula) |
| 37 | emi_to_income_ratio | Numeric | monthly_emi / monthly_income |
| 38 | collateral_type | Categorical | No Collateral, Property, Vehicle, Gold, Fixed Deposit, Securities |
| 39 | savings_balance | Numeric | Savings account balance |
| 40 | checking_balance | Numeric | Current account balance (negative = overdraft) |
| 41 | investment_portfolio_value | Numeric | Value of investments held |
| 42 | monthly_expenses | Numeric | Declared monthly expenses |
| 43 | avg_monthly_transactions | Numeric (int) | Average card/bank transactions per month |
| 44 | digital_banking_user | Categorical | Yes / No |
| 45 | customer_segment | Categorical | Retail / Premium / HNI / SME |
| 46 | relationship_tenure_years | Numeric | Years as an Xcedance customer |
| 47 | num_products_held | Numeric (int) | Xcedance products held |
| 48 | insurance_premium_annual | Numeric | Annual insurance premium paid |
| 49 | risk_rating | Ordinal | Internal rating: Low / Medium / High / Very High |
| 50 | loan_default | **Target** (binary) | 1 = customer defaulted on the loan |

## What's hidden in the data (spoilers – try to find them first!)

<details>
<summary>Click to reveal</summary>

**Missing values**
- *Structural*: `property_value` is blank for everyone who rents / lives with family (~62%).
- *MAR*: `annual_income`/`monthly_income` are missing more often for top-10% earners;
  `investment_portfolio_value` is missing far more for High School/Diploma applicants;
  `other_income` is blank mostly when it is really 0.
- *MCAR*: random blanks in ~20 other columns at 1–15% (`insurance_premium_annual` 16%, `monthly_expenses` 12%, `collateral_type` 10%, …).

**Dirty categoricals**
- `gender`: `Male`, `M`, `male`, `Female`, `F`, `female`
- `employment_type`: `Self-Employed` vs `Self Employed`, `salaried`
- `region`: `north`, `EAST`, `" West"`, `"South "` (case + whitespace)
- `digital_banking_user`: `Yes`/`Y`, `No`/`N`

**Type problems**: `years_employed` is text (`<1`, `10+`) and must be converted.

**Outliers / impossible values**
- `age`: 150, 999, 3, -1
- `credit_score`: 9999, 0, 1200, 15 (valid 300–850)
- `credit_utilization_ratio` > 1 (6 rows)
- `loan_amount` negative (5 rows)
- `dependents`: 12, 15, 20, 25
- `annual_income`: 10 extreme values (25× normal)
- Natural right skew in income, balances, investments, `monthly_emi`

**Correlation / multicollinearity**
- `monthly_income` = `annual_income` / 12 (perfect collinearity – drop one)
- `credit_score` ↔ `interest_rate` strongly negative (Spearman ≈ -0.78)
- `credit_score` ↔ `credit_utilization_ratio` negative (≈ -0.68)
- `loan_amount` ↔ `monthly_emi`, `num_credit_cards` ↔ `num_open_accounts`,
  `total_credit_limit` ↔ `outstanding_debt`, `age` ↔ `credit_history_years`
- Note: Pearson correlation of `credit_score` is distorted by its outliers — compare with Spearman.

**Target drivers**: `debt_to_income_ratio`, late payments, delinquencies, `credit_utilization_ratio`,
`interest_rate`, low `credit_score`, bankruptcy, unemployment. `risk_rating` is strongly associated
with default (Low ≈ 2% → Very High ≈ 65%).

**Other**: 60 exact duplicate rows; target imbalance (~18% defaults);
`customer_segment` is derived from income/employment (leaky with income).
</details>

## Suggested EDA exercises
1. Load the data, inspect `shape`, `dtypes`, `info()`, `describe(include="all")`.
2. Find and drop duplicate rows.
3. Standardise the dirty categorical labels; convert `years_employed` to numeric.
4. Build a missing-value table and heatmap (e.g. `missingno`); decide per column whether the
   missingness is MCAR, MAR or structural and choose an imputation strategy.
5. Detect outliers with box plots, IQR and z-scores; separate *impossible* values from *real* extremes.
6. Plot distributions; apply log transforms to skewed money columns.
7. Correlation heatmap (Pearson vs Spearman); identify redundant features (VIF).
8. Bivariate analysis vs `loan_default`: default rate by category, box plots by target.
9. Time analysis: applications and default rate by month/quarter from `application_date`.
10. Feature engineering ideas: income per dependent, savings-to-loan ratio, age bands.
