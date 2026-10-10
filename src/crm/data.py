"""Raw data loading, target construction and feature engineering.

Only information available at origination is used as a model input. Any
field populated after the loan was booked (payments, recoveries, last FICO,
hardship / settlement flags, ...) is used only to build targets, LGD and EAD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Application-time inputs.
APP_COLS = [
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment", "grade",
    "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "addr_state", "dti", "delinq_2yrs",
    "earliest_cr_line", "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "initial_list_status",
    "application_type", "collections_12_mths_ex_med",
    "mths_since_last_major_derog", "acc_now_delinq", "tot_coll_amt",
    "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths", "avg_cur_bal",
    "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths", "delinq_amnt",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc",
    "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
    "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
]
# Post-origination fields: targets / LGD / EAD only, never model inputs.
OUTCOME_COLS = [
    "id", "issue_d", "loan_status", "total_rec_prncp", "total_rec_int",
    "recoveries", "collection_recovery_fee", "last_pymnt_d", "out_prncp",
]

BAD_STATUS = {
    "Charged Off", "Default", "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
}
TERMINAL_STATUS = {
    "Fully Paid", "Charged Off", "Default",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
}


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_FILE, usecols=APP_COLS + OUTCOME_COLS,
                     low_memory=False)
    df = df[df["loan_status"].notna() & df["issue_d"].notna()].copy()
    return df


def _months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["issue_year"] = df["issue_dt"].dt.year
    df["issue_q"] = df["issue_dt"].dt.to_period("Q").astype(str)
    df["term_m"] = df["term"].str.extract(r"(\d+)").astype(float)
    df["policy_exception"] = df["loan_status"].str.startswith("Does not meet").astype(int)

    bad = df["loan_status"].isin(BAD_STATUS)
    # Month-on-book of the first missed instalment. When no recovery cash was
    # received, last_pymnt_d is the last borrower payment -> first miss = +1.
    # Otherwise last_pymnt_d may be a post-charge-off recovery, so we count paid
    # instalments; ceil(paid)+1 reproduces the date-based value with mean bias
    # -0.1 months and 98% agreement on the 12-month flag (validated on the
    # 104k charge-offs without recoveries).
    lp = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y")
    mob_lp = _months_between(lp, df["issue_dt"]) + 1
    paid_inst = (df["total_rec_prncp"] + df["total_rec_int"]) / df["installment"]
    mob_pay = np.ceil(paid_inst.fillna(0)) + 1
    use_lp = (df["recoveries"] == 0) & mob_lp.notna()
    mob = np.where(use_lp, mob_lp, mob_pay)
    df["default_mob"] = np.where(bad, np.clip(mob, 1, df["term_m"]), np.nan)
    df["bad_lifetime"] = bad.astype(int)

    # 12-month PD target: default (90+ dpd / charge-off) with first missed
    # instalment within 12 months on book. Defined for every loan with a full
    # observation window, regardless of current status -> no survivorship bias.
    observable = df["issue_dt"] <= pd.Timestamp(config.LAST_OBSERVABLE_ISSUE)
    df["obs_12m"] = observable.astype(int)
    df["default_12m"] = (bad & (df["default_mob"] <= config.HORIZON_M)).astype(int)

    # Original (replication) design: terminal loans only, lifetime default.
    df["terminal"] = df["loan_status"].isin(TERMINAL_STATUS).astype(int)

    # Loss-given-default ingredients (defaulted loans only).
    df["ead_default"] = (df["funded_amnt"] - df["total_rec_prncp"]).clip(lower=0)
    net_rec = (df["recoveries"] - df["collection_recovery_fee"]).clip(lower=0)
    df["lgd_realised"] = np.where(
        bad & (df["ead_default"] > 0),
        (1 - net_rec / df["ead_default"].where(df["ead_default"] > 0)).clip(0, 1),
        np.nan)
    df["default_dt"] = df["issue_dt"] + pd.to_timedelta(
        (df["default_mob"].fillna(0) * 30.44).round(), unit="D")
    df.loc[~bad, "default_dt"] = pd.NaT
    df["months_since_default"] = _months_between(
        pd.Series(pd.Timestamp(config.SNAPSHOT), index=df.index), df["default_dt"])
    return df


EMP_MAP = {"< 1 year": 0, "1 year": 1, "10+ years": 10}
EMP_MAP.update({f"{i} years": i for i in range(2, 10)})


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    ecl = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_hist_m"] = _months_between(df["issue_dt"], ecl)
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")
    df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")
    df["emp_len"] = df["emp_length"].map(EMP_MAP)
    df["emp_missing"] = df["emp_length"].isna().astype(int)
    grade_order = sorted(df["sub_grade"].dropna().unique())
    df["sub_grade_n"] = df["sub_grade"].map({g: i + 1 for i, g in enumerate(grade_order)})
    inc = df["annual_inc"].where(df["annual_inc"] > 0)
    df["log_inc"] = np.log1p(inc)
    df["loan_to_inc"] = df["loan_amnt"] / inc
    df["inst_to_inc"] = df["installment"] * 12 / inc
    df["revol_to_inc"] = df["revol_bal"] / inc
    df["dti"] = df["dti"].where(df["dti"].between(0, 100))
    df["joint"] = (df["application_type"] == "Joint App").astype(int)
    df["whole_loan"] = (df["initial_list_status"] == "w").astype(int)
    df["home_ownership"] = df["home_ownership"].replace({"ANY": "OTHER", "NONE": "OTHER"})
    return df


CAT_FEATURES = ["home_ownership", "verification_status", "purpose", "addr_state"]


def prepare() -> pd.DataFrame:
    df = load_raw()
    df = build_targets(df)
    df = engineer_features(df)
    # Policy-exception loans (2007-2010, "does not meet the credit policy")
    # come from a different underwriting regime; they are excluded.
    df = df[df["policy_exception"] == 0].copy()
    for c in CAT_FEATURES + ["grade", "sub_grade", "loan_status"]:
        df[c] = df[c].astype("category")
    drop = ["term", "emp_length", "earliest_cr_line", "issue_d",
            "fico_range_low", "fico_range_high", "application_type",
            "initial_list_status", "last_pymnt_d"]
    return df.drop(columns=drop)
