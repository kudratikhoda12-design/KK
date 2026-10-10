"""Central configuration: paths, feature lists and hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = ROOT / "sql"
DATA_DIR = ROOT / "data"
SAMPLE_PATH = DATA_DIR / "lendingclub_2016_sample.parquet"   # 50k-row cohort (committed)
MODEL_TABLE_PATH = DATA_DIR / "model_table.parquet"           # output of the SQL pipeline
FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "reports" / "results"

SEED = 42
N_SAMPLE = 50_000

DURATION = "duration_months"
EVENT = "event"

# Numeric features. Heavily right-skewed money/count variables get a log1p transform.
# Engineered in SQL but dropped for multicollinearity (VIF > 10, see eda.py):
VIF_DROPPED = ["loan_to_income", "income_vs_state", "loan_vs_purpose_avg", "total_acc", "bc_util"]
NUMERIC = [
    "loan_amnt", "term_months", "int_rate", "rate_spread", "annual_inc", "emp_years",
    "dti", "fico", "credit_history_months", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "mort_acc",
    "tot_cur_bal", "acc_open_past_24mths", "percent_bc_gt_75",
    "num_tl_op_past_12m", "payment_to_income", "revol_to_income", "open_acc_ratio",
]
LOG1P = [
    "loan_amnt", "annual_inc", "revol_bal", "tot_cur_bal", "delinq_2yrs", "pub_rec",
    "inq_last_6mths", "mort_acc", "loan_to_income", "revol_to_income", "income_vs_state",
    "payment_to_income",
]
BINARY = ["whole_loan", "joint_app", "ever_delinquent"]
CATEGORICAL = ["home_ownership", "purpose", "verification_status"]

# Training hyper-parameters (chosen on the validation split, see REPORT.md)
COXKAN_ADDITIVE = dict(width=[None, 1], grid=8, k=3, lr=1e-2, epochs=600, lamb_l1=1e-3, lamb_coef=1e-4)
COXKAN_DEEP = dict(width=[None, 4, 1], grid=5, k=3, lr=1e-2, epochs=600, lamb_l1=1e-3, lamb_coef=1e-4)
DEEPSURV = dict(hidden=[128, 64], dropout=0.3, lr=3e-3, weight_decay=1e-4, epochs=600)
COXPH_PENALIZER = 0.01
