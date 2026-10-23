"""
Unit 2 (cont.) — Exploratory analysis.

Produces:
  Table 1  : descriptives of the 5 numeric DVs by JobSeek group
  tables/table1b_corr_matrix.csv : correlation matrix of the numeric block
                                    (motivates PCA in Unit 4)
  figures/fig0_distributions.png : distribution check for the numeric block

Run: python3 scripts/02_eda.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/clean_sample.csv"
NUMERIC = ["logComp", "YearsCodePro", "WorkWeekHrs", "Age", "JobSat_num"]
NUMERIC_PLUS = NUMERIC + ["CodeRevHrs"]


def main():
    df = pd.read_csv(DATA_PATH)
    df["JobSeek_lbl"] = df["JobSeek_bin"].map({1: "Seeking/Open", 0: "Not interested"})

    # Table 1: descriptives by group
    rows = []
    for col in NUMERIC_PLUS:
        g = df.groupby("JobSeek_lbl")[col].agg(["mean", "std", "count"])
        rows.append(pd.concat({col: g}, names=["variable"]))
    table1 = pd.concat(rows)
    table1 = table1.round(3)
    table1.to_csv("tables/table1_descriptives_by_jobseek.csv")
    print("Table 1 — Descriptives by JobSeek group")
    print(table1)

    # Overall N and class balance
    print("\nN =", len(df))
    print(df["JobSeek_lbl"].value_counts())
    print(df["JobSeek_lbl"].value_counts(normalize=True).round(3))

    # Correlation matrix of numeric block (for PCA motivation)
    corr = df[NUMERIC_PLUS].corr().round(3)
    corr.to_csv("tables/table1b_corr_matrix.csv")
    print("\nCorrelation matrix of numeric block:")
    print(corr)

    # Missingness already reported in Table 0; distribution figure for numerics
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, NUMERIC_PLUS):
        ax.hist(df[col].dropna(), bins=30, color="#4C72B0", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    fig.suptitle("Figure 0. Distributions of numeric variables (final sample, N=8000)")
    fig.tight_layout()
    fig.savefig("figures/fig0_distributions.png", dpi=150)
    print("\nSaved figures/fig0_distributions.png")


if __name__ == "__main__":
    main()
