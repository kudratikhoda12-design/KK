"""
Developer Attrition-Risk Analytics
Unit 2 — Data extraction, filtering, and preprocessing.

Source: Stack Overflow Annual Developer Survey, 2019
(https://survey.stackoverflow.co/ — archived results, ODbL 1.0 licence).

Pipeline:
  1. Load raw public responses for 2019.
  2. Filter to professional, full-time developers.
  3. Keep the 5 largest countries by respondent count.
  4. Clean/derive the numeric and outcome variables used downstream.
  5. Draw a reproducible random subsample (n ~ 8000).
  6. Write the analysis-ready table and a missingness/cleaning report.

Run: python3 scripts/01_clean_data.py
"""
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/survey_results_public_2019.csv"
OUT_PATH = "data/processed/clean_sample.csv"
REPORT_PATH = "tables/table0_cleaning_report.txt"
SEED = 42
TARGET_N = 8000

JOBSAT_MAP = {
    "Very dissatisfied": 1,
    "Slightly dissatisfied": 2,
    "Neither satisfied nor dissatisfied": 3,
    "Slightly satisfied": 4,
    "Very satisfied": 5,
}

# Explicit outcome mapping, stated up front per the brief:
#   "actively looking" and "open to new opportunities" are both job-seeking
#   risk (the retention-relevant group); only explicit disinterest is coded 0.
JOBSEEK_MAP = {
    "I am actively looking for a job": 1,
    "I’m not actively looking, but I am open to new opportunities": 1,
    "I am not interested in new job opportunities": 0,
}


def years_to_numeric(s):
    s = s.replace({"Less than 1 year": "0.5", "More than 50 years": "51"})
    return pd.to_numeric(s, errors="coerce")


def main():
    cols = [
        "Respondent", "MainBranch", "Employment", "Country", "Hobbyist",
        "EdLevel", "OrgSize", "YearsCodePro", "WorkWeekHrs", "Age",
        "JobSat", "CodeRevHrs", "ConvertedComp", "JobSeek",
    ]
    df = pd.read_csv(RAW_PATH, usecols=cols, low_memory=False)
    n_raw = len(df)

    report = []
    report.append(f"Raw 2019 responses loaded: {n_raw}")

    # --- Step 1: professional, full-time filter -----------------------------
    df = df[
        (df["MainBranch"] == "I am a developer by profession")
        & (df["Employment"] == "Employed full-time")
    ].copy()
    report.append(
        f"After MainBranch='developer by profession' & Employment='Employed "
        f"full-time': {len(df)}"
    )

    # --- Step 2: top-5 countries by count -----------------------------------
    top5 = df["Country"].value_counts().head(5).index.tolist()
    df = df[df["Country"].isin(top5)].copy()
    report.append(f"Top-5 countries kept: {top5}")
    report.append(f"After country filter: {len(df)}")

    # --- Step 3: missingness on raw filtered set (before derived cleaning) --
    miss = (
        df[["ConvertedComp", "YearsCodePro", "WorkWeekHrs", "Age", "JobSat",
            "CodeRevHrs", "JobSeek", "EdLevel", "Hobbyist", "OrgSize"]]
        .isna().mean().mul(100).round(2).sort_values(ascending=False)
    )
    report.append("\nMissingness (%) on filtered sample, pre-cleaning:")
    report.append(miss.to_string())

    # --- Step 4: derive / clean variables -----------------------------------
    df["YearsCodePro"] = years_to_numeric(df["YearsCodePro"])
    df["JobSat_num"] = df["JobSat"].map(JOBSAT_MAP)
    df["JobSeek_bin"] = df["JobSeek"].map(JOBSEEK_MAP)

    # Outlier handling, stated explicitly:
    #  - WorkWeekHrs capped at 100 (brief's instruction); values <=0 treated missing.
    #  - Age restricted to a plausible working-adult range [16, 75].
    #  - CodeRevHrs capped at the 99th percentile (long right tail, many self-reports > 40h/wk implausible).
    #  - ConvertedComp: require > 0 for log transform; cap at 99th percentile to tame reporting/currency outliers.
    df.loc[df["WorkWeekHrs"] <= 0, "WorkWeekHrs"] = np.nan
    df["WorkWeekHrs"] = df["WorkWeekHrs"].clip(upper=100)

    df.loc[(df["Age"] < 16) | (df["Age"] > 75), "Age"] = np.nan

    crh_cap = df["CodeRevHrs"].quantile(0.99)
    df["CodeRevHrs"] = df["CodeRevHrs"].clip(upper=crh_cap)

    df.loc[df["ConvertedComp"] <= 0, "ConvertedComp"] = np.nan
    comp_cap = df["ConvertedComp"].quantile(0.99)
    df["ConvertedComp"] = df["ConvertedComp"].clip(upper=comp_cap)
    df["logComp"] = np.log(df["ConvertedComp"])

    report.append(f"\nWorkWeekHrs capped at 100; CodeRevHrs capped at 99th pct "
                   f"({crh_cap:.1f} h); ConvertedComp capped at 99th pct "
                   f"(${comp_cap:,.0f}); Age restricted to [16,75].")

    numeric_cols = ["logComp", "YearsCodePro", "WorkWeekHrs", "Age",
                     "JobSat_num", "CodeRevHrs"]

    # --- Step 5: complete-case on analysis variables ------------------------
    needed = numeric_cols + ["JobSeek_bin", "EdLevel", "Hobbyist", "OrgSize",
                              "Country"]
    before_cc = len(df)
    df_cc = df.dropna(subset=needed).copy()
    report.append(
        f"\nComplete-case drop on analysis variables: {before_cc} -> "
        f"{len(df_cc)} ({100*(1-len(df_cc)/before_cc):.1f}% dropped)"
    )

    # --- Step 6: reproducible subsample -------------------------------------
    n_final = min(TARGET_N, len(df_cc))
    df_final = df_cc.sample(n=n_final, random_state=SEED).reset_index(drop=True)
    report.append(f"\nFinal subsample drawn with seed={SEED}: N = {len(df_final)}")

    bal = df_final["JobSeek_bin"].value_counts(normalize=True).mul(100).round(1)
    report.append(
        f"\nJobSeek class balance in final sample: "
        f"seeking/open = {bal.get(1,0)}%, not interested = {bal.get(0,0)}%"
    )
    report.append("\nCountry counts in final sample:")
    report.append(df_final["Country"].value_counts().to_string())

    keep_cols = [
        "Respondent", "Country", "EdLevel", "Hobbyist", "OrgSize",
        "ConvertedComp", "logComp", "YearsCodePro", "WorkWeekHrs", "Age",
        "JobSat", "JobSat_num", "CodeRevHrs", "JobSeek", "JobSeek_bin",
    ]
    df_final[keep_cols].to_csv(OUT_PATH, index=False)

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(str(x) for x in report))

    print("\n".join(str(x) for x in report))
    print(f"\nWrote cleaned sample -> {OUT_PATH}")
    print(f"Wrote cleaning report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
