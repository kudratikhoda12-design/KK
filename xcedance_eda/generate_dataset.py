"""
Generate a synthetic financial dataset for the (fictional) company "Xcedance"
to practise Exploratory Data Analysis (EDA).

Each row is one customer loan application in Xcedance's retail lending book.
The data is fully synthetic -- no real people or companies are represented.

Deliberately built-in EDA challenges:
  * 50 features: numeric (continuous + discrete), categorical (nominal + ordinal),
    binary flags, a date column and a binary target (loan_default)
  * Missing values of three kinds: MCAR (random), MAR (depends on another column)
    and structural (e.g. property_value is blank for renters)
  * Correlated / redundant features (incl. one perfectly collinear pair)
  * Skewed distributions and outliers, plus a few impossible data-entry errors
  * Dirty categorical labels (inconsistent case, spelling, whitespace)
  * A mixed-type column (years_employed contains "<1" and "10+")
  * Duplicate rows and class imbalance in the target

Run:  python generate_dataset.py      ->  writes xcedance_financial_data.csv
"""
import numpy as np
import pandas as pd

SEED = 42
N = 5000
rng = np.random.default_rng(SEED)


def clip(a, lo, hi):
    return np.clip(a, lo, hi)


# --------------------------------------------------------------------------
# Demographics
# --------------------------------------------------------------------------
customer_id = [f"XCD{100000 + i}" for i in range(N)]
application_date = pd.to_datetime("2021-01-01") + pd.to_timedelta(
    rng.integers(0, 365 * 4, N), unit="D"
)
age = clip(rng.normal(40, 11, N).round(), 21, 75).astype(int)
gender = rng.choice(["Male", "Female"], N, p=[0.56, 0.44])
marital_status = rng.choice(
    ["Single", "Married", "Divorced", "Widowed"], N, p=[0.35, 0.50, 0.11, 0.04]
)
dependents = clip(rng.poisson(1.2, N) + (marital_status == "Married"), 0, 6)
education_level = rng.choice(
    ["High School", "Diploma", "Bachelor", "Master", "PhD"],
    N, p=[0.18, 0.17, 0.40, 0.20, 0.05],
)
edu_rank = pd.Series(education_level).map(
    {"High School": 0, "Diploma": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
).to_numpy()
employment_type = rng.choice(
    ["Salaried", "Self-Employed", "Business Owner", "Contract", "Unemployed", "Retired"],
    N, p=[0.55, 0.15, 0.10, 0.10, 0.04, 0.06],
)
employment_type[age >= 62] = rng.choice(["Retired", "Business Owner"], (age >= 62).sum(), p=[0.8, 0.2])
occupation = rng.choice(
    ["IT Professional", "Doctor", "Engineer", "Teacher", "Accountant", "Sales",
     "Lawyer", "Banker", "Consultant", "Trader", "Farmer", "Artist", "Driver", "Other"],
    N,
)
years_employed_num = clip((age - 21) * rng.uniform(0.1, 0.9, N), 0, 40).round(1)
years_employed_num[np.isin(employment_type, ["Unemployed"])] = 0

# --------------------------------------------------------------------------
# Income (log-normal => right-skewed). Depends on education, age, employment.
# --------------------------------------------------------------------------
emp_mult = pd.Series(employment_type).map({
    "Salaried": 1.0, "Self-Employed": 1.15, "Business Owner": 1.6,
    "Contract": 0.8, "Unemployed": 0.25, "Retired": 0.55,
}).to_numpy()
annual_income = np.exp(
    10.6 + 0.18 * edu_rank + 0.012 * (age - 21) + rng.normal(0, 0.45, N)
) * emp_mult
annual_income = annual_income.round(-2)
monthly_income = (annual_income / 12).round(2)           # perfectly collinear
other_income = np.where(rng.random(N) < 0.45, rng.gamma(2, 2500, N), 0).round(0)

region = rng.choice(["North", "South", "East", "West", "Central"], N,
                    p=[0.24, 0.22, 0.18, 0.26, 0.10])
city_tier = rng.choice(["Tier 1", "Tier 2", "Tier 3"], N, p=[0.45, 0.35, 0.20])
home_ownership = np.where(
    rng.random(N) < clip(0.15 + 0.012 * (age - 21), 0, 0.85),
    rng.choice(["Own", "Mortgage"], N, p=[0.4, 0.6]),
    rng.choice(["Rent", "Living with Family"], N, p=[0.8, 0.2]),
)
property_value = (annual_income * rng.uniform(3, 8, N)).round(-3)
residence_years = clip(rng.gamma(2, 3, N), 0, age - 18).round(1)

# --------------------------------------------------------------------------
# Credit profile -- a latent "credit quality" factor drives many columns
# --------------------------------------------------------------------------
quality = (
    0.35 * (np.log(annual_income) - 11) / 0.5
    + 0.25 * (age - 40) / 11
    + 0.15 * (edu_rank - 2)
    + rng.normal(0, 1, N)
)
credit_score = clip(680 + 55 * quality + rng.normal(0, 25, N), 300, 850).round().astype(int)
credit_history_years = clip((age - 18) * rng.uniform(0.3, 1.0, N), 0, 50).round(1)
num_credit_cards = clip(rng.poisson(1.5 + 0.6 * (quality > 0) + annual_income / 80000), 0, 12)
num_open_accounts = clip(num_credit_cards + rng.poisson(2, N), 0, 30)
num_bank_accounts = clip(rng.poisson(2.2, N) + 1, 1, 10)
total_credit_limit = (annual_income * rng.uniform(0.15, 0.6, N) * (1 + num_credit_cards / 5)).round(-2)
credit_utilization_ratio = clip(0.45 - 0.15 * quality + rng.normal(0, 0.15, N), 0, 1).round(3)
outstanding_debt = (total_credit_limit * credit_utilization_ratio
                    + rng.gamma(1.5, 6000, N)).round(2)
debt_to_income_ratio = (outstanding_debt / annual_income).round(3)
num_late_payments_12m = rng.poisson(clip(1.2 - 0.6 * quality, 0.05, 6))
num_delinquencies_2y = rng.poisson(clip(0.6 - 0.4 * quality, 0.02, 4))
num_credit_inquiries_6m = rng.poisson(clip(1.5 - 0.3 * quality, 0.2, 5))
bankruptcy_flag = (rng.random(N) < clip(0.03 - 0.02 * quality, 0.002, 0.2)).astype(int)

# --------------------------------------------------------------------------
# Loan details
# --------------------------------------------------------------------------
loan_purpose = rng.choice(
    ["Home Improvement", "Debt Consolidation", "Car", "Education", "Business",
     "Medical", "Wedding", "Travel", "Personal"],
    N, p=[0.12, 0.25, 0.14, 0.08, 0.10, 0.07, 0.06, 0.05, 0.13],
)
loan_amount = clip(annual_income * rng.uniform(0.1, 0.9, N), 1000, 500000).round(-2)
loan_term_months = rng.choice([12, 24, 36, 48, 60], N, p=[0.10, 0.20, 0.35, 0.15, 0.20])
interest_rate = clip(24 - 0.022 * (credit_score - 300) + rng.normal(0, 1.2, N), 6, 28).round(2)
r = interest_rate / 100 / 12
monthly_emi = (loan_amount * r * (1 + r) ** loan_term_months
               / ((1 + r) ** loan_term_months - 1)).round(2)
emi_to_income_ratio = (monthly_emi / monthly_income).round(3)
collateral_type = rng.choice(
    ["No Collateral", "Property", "Vehicle", "Gold", "Fixed Deposit", "Securities"],
    N, p=[0.45, 0.18, 0.15, 0.10, 0.07, 0.05],
)

# --------------------------------------------------------------------------
# Banking relationship
# --------------------------------------------------------------------------
savings_balance = (np.exp(rng.normal(8.5 + 0.6 * quality, 1.0, N))).round(2)
checking_balance = (rng.normal(monthly_income * 0.8, monthly_income * 0.6)).round(2)  # can be negative (overdraft)
investment_portfolio_value = (np.exp(rng.normal(9 + 0.5 * quality + 0.3 * edu_rank, 1.2, N))).round(2)
monthly_expenses = clip(monthly_income * rng.uniform(0.35, 0.95, N) + 300 * dependents, 300, None).round(2)
avg_monthly_transactions = clip(rng.normal(35, 15, N) + 5 * num_credit_cards, 1, None).round().astype(int)
digital_banking_user = np.where(rng.random(N) < clip(1.1 - age / 80, 0.2, 0.95), "Yes", "No")
customer_segment = np.select(
    [annual_income > 150000, employment_type == "Business Owner", annual_income > 80000],
    ["HNI", "SME", "Premium"], default="Retail",
)
relationship_tenure_years = clip(rng.gamma(2, 2.5, N), 0, age - 18).round(1)
num_products_held = clip(1 + rng.poisson(0.4 * relationship_tenure_years ** 0.5 + 0.5), 1, 10)
insurance_premium_annual = (rng.gamma(2, 400, N) + 0.01 * annual_income + 15 * (age - 21)).round(2)

# --------------------------------------------------------------------------
# Target + derived risk rating
# --------------------------------------------------------------------------
logit = (
    -3.1
    - 0.012 * (credit_score - 680)
    + 2.0 * (debt_to_income_ratio - 0.4)
    + 2.5 * (emi_to_income_ratio - 0.1)
    + 0.30 * num_late_payments_12m
    + 0.35 * num_delinquencies_2y
    + 0.9 * bankruptcy_flag
    + 0.08 * (interest_rate - 15)
    + 0.6 * (employment_type == "Unemployed")
    - 0.3 * (collateral_type != "No Collateral")
)
loan_default = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

risk_score = (credit_score - 680) / 55 - 2 * debt_to_income_ratio - 0.5 * num_late_payments_12m
risk_rating = pd.cut(risk_score, [-np.inf, *np.quantile(risk_score, [0.10, 0.30, 0.65]), np.inf],
                     labels=["Very High", "High", "Medium", "Low"]).astype(str)

df = pd.DataFrame({
    "customer_id": customer_id,
    "application_date": application_date.strftime("%Y-%m-%d"),
    "age": age,
    "gender": gender,
    "marital_status": marital_status,
    "dependents": dependents,
    "education_level": education_level,
    "employment_type": employment_type,
    "occupation": occupation,
    "years_employed": years_employed_num.astype(object),
    "annual_income": annual_income,
    "monthly_income": monthly_income,
    "other_income": other_income,
    "region": region,
    "city_tier": city_tier,
    "home_ownership": home_ownership,
    "property_value": property_value,
    "residence_years": residence_years,
    "credit_score": credit_score,
    "credit_history_years": credit_history_years,
    "num_credit_cards": num_credit_cards,
    "num_open_accounts": num_open_accounts,
    "num_bank_accounts": num_bank_accounts,
    "credit_utilization_ratio": credit_utilization_ratio,
    "total_credit_limit": total_credit_limit,
    "outstanding_debt": outstanding_debt,
    "debt_to_income_ratio": debt_to_income_ratio,
    "num_late_payments_12m": num_late_payments_12m,
    "num_delinquencies_2y": num_delinquencies_2y,
    "num_credit_inquiries_6m": num_credit_inquiries_6m,
    "bankruptcy_flag": bankruptcy_flag,
    "loan_purpose": loan_purpose,
    "loan_amount": loan_amount,
    "loan_term_months": loan_term_months,
    "interest_rate": interest_rate,
    "monthly_emi": monthly_emi,
    "emi_to_income_ratio": emi_to_income_ratio,
    "collateral_type": collateral_type,
    "savings_balance": savings_balance,
    "checking_balance": checking_balance,
    "investment_portfolio_value": investment_portfolio_value,
    "monthly_expenses": monthly_expenses,
    "avg_monthly_transactions": avg_monthly_transactions,
    "digital_banking_user": digital_banking_user,
    "customer_segment": customer_segment,
    "relationship_tenure_years": relationship_tenure_years,
    "num_products_held": num_products_held,
    "insurance_premium_annual": insurance_premium_annual,
    "risk_rating": risk_rating,
    "loan_default": loan_default,
})
assert df.shape[1] == 50

# --------------------------------------------------------------------------
# Inject data-quality problems
# --------------------------------------------------------------------------
# 1) Mixed-type column: "<1" and "10+" strings in years_employed
ye = df["years_employed"].astype(float)
df["years_employed"] = np.where(ye < 1, "<1", np.where(ye >= 10, "10+", ye.round().astype(int).astype(str)))

# 2) Dirty categorical labels
def dirty(col, mapping, frac):
    idx = rng.random(N) < frac
    df.loc[idx, col] = df.loc[idx, col].map(lambda v: mapping.get(v, v))

dirty("gender", {"Male": "M", "Female": "F"}, 0.05)
dirty("gender", {"Male": "male", "Female": "female"}, 0.03)
dirty("employment_type", {"Self-Employed": "Self Employed"}, 0.30)
dirty("employment_type", {"Salaried": "salaried"}, 0.04)
dirty("region", {"North": "north", "South": "South ", "East": "EAST", "West": " West"}, 0.06)
dirty("digital_banking_user", {"Yes": "Y", "No": "N"}, 0.05)

# 3) Structural missingness: property_value is blank for non-owners
df.loc[~df["home_ownership"].isin(["Own", "Mortgage"]), "property_value"] = np.nan

# 4) MAR: high earners more often skip income; investments blank for low-education
p_inc = clip(0.03 + 0.10 * (df["annual_income"] > df["annual_income"].quantile(0.9)), 0, 1)
inc_missing = rng.random(N) < p_inc
df.loc[inc_missing, ["annual_income", "monthly_income"]] = np.nan
p_inv = np.where(edu_rank <= 1, 0.55, 0.20)
df.loc[rng.random(N) < p_inv, "investment_portfolio_value"] = np.nan
df.loc[(df["other_income"] == 0) & (rng.random(N) < 0.35), "other_income"] = np.nan

# 5) MCAR: random blanks across many columns at different rates
mcar_rates = {
    "gender": 0.01, "marital_status": 0.02, "dependents": 0.03, "education_level": 0.04,
    "occupation": 0.08, "years_employed": 0.05, "region": 0.01, "residence_years": 0.06,
    "credit_score": 0.05, "credit_history_years": 0.03, "num_credit_inquiries_6m": 0.07,
    "credit_utilization_ratio": 0.04, "outstanding_debt": 0.02, "loan_purpose": 0.02,
    "collateral_type": 0.10, "savings_balance": 0.09, "checking_balance": 0.05,
    "monthly_expenses": 0.12, "avg_monthly_transactions": 0.03,
    "digital_banking_user": 0.02, "insurance_premium_annual": 0.15,
    "relationship_tenure_years": 0.04,
}
for col, rate in mcar_rates.items():
    df.loc[rng.random(N) < rate, col] = np.nan

# 6) Outliers and impossible values (data-entry errors)
err = rng.choice(N, 8, replace=False)
df.loc[err[:4], "age"] = [150, 999, 3, -1]
df.loc[err[4:], "credit_score"] = [9999, 0, 1200, 15]
df.loc[rng.choice(N, 6, replace=False), "credit_utilization_ratio"] = [1.35, 2.1, 1.8, 4.5, 1.2, 3.3]
big = rng.choice(N, 10, replace=False)
df.loc[big, "annual_income"] = df.loc[big, "annual_income"] * 25   # extreme incomes
df.loc[big, "monthly_income"] = df.loc[big, "annual_income"] / 12
df.loc[rng.choice(N, 5, replace=False), "loan_amount"] = -df["loan_amount"].sample(5, random_state=SEED).to_numpy()
df.loc[rng.choice(N, 4, replace=False), "dependents"] = [15, 20, 12, 25]

# 7) Duplicate rows (exact copies appended and shuffled in)
dups = df.sample(60, random_state=SEED)
df = pd.concat([df, dups]).sample(frac=1, random_state=SEED).reset_index(drop=True)

df.to_csv("xcedance_financial_data.csv", index=False)
print(f"Saved xcedance_financial_data.csv  shape={df.shape}")
print(f"Default rate: {df['loan_default'].mean():.2%}")
print(f"Missing cells: {df.isna().sum().sum()} ({df.isna().mean().mean():.2%})")
