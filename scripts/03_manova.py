"""
Unit 3 — MANOVA.

One-way MANOVA : DVs = [logComp, YearsCodePro, WorkWeekHrs, Age, JobSat_num]
                 factor = JobSeek (seeking/open vs. not interested)
Two-way MANOVA : same DVs, factors = JobSeek x Country (5 countries)
Assumption checks: Box's M (equal covariances), Mardia's test (per-group MVN
                    on group-centered residuals), Levene's test per DV.

Outputs:
  tables/table2_manova_oneway.txt
  tables/table2_manova_twoway.txt
  tables/table2b_assumptions.txt
  tables/table2c_univariate_anova.csv
  figures/fig1_parallel_coords.png

Run: python3 scripts/03_manova.py
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/clean_sample.csv"
DVS = ["logComp", "YearsCodePro", "WorkWeekHrs", "Age", "JobSat_num"]
DV_LABELS = {
    "logComp": "log(Comp)", "YearsCodePro": "YearsCodePro",
    "WorkWeekHrs": "WorkWeekHrs", "Age": "Age", "JobSat_num": "JobSat",
}


def box_m_test(df, dvs, group_col):
    groups = df[group_col].unique()
    p = len(dvs)
    covs, ns = [], []
    for g in groups:
        X = df.loc[df[group_col] == g, dvs].values
        covs.append(np.cov(X, rowvar=False, ddof=1))
        ns.append(len(X))
    ns = np.array(ns)
    g_n = len(groups)
    pooled = sum((n - 1) * c for n, c in zip(ns, covs)) / (ns.sum() - g_n)
    sign, logdet_pooled = np.linalg.slogdet(pooled)
    M = (ns.sum() - g_n) * logdet_pooled
    for n, c in zip(ns, covs):
        sign, logdet_c = np.linalg.slogdet(c)
        M -= (n - 1) * logdet_c
    c1 = (np.sum(1.0 / (ns - 1)) - 1.0 / (ns.sum() - g_n))
    c1 *= (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g_n - 1))
    chi2_stat = M * (1 - c1)
    df_chi2 = p * (p + 1) * (g_n - 1) / 2
    p_val = 1 - stats.chi2.cdf(chi2_stat, df_chi2)
    return {"M": M, "chi2": chi2_stat, "df": df_chi2, "p": p_val}


def mardia_test(X):
    """Mardia's multivariate skewness & kurtosis test. X: n x p array."""
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = np.cov(Xc, rowvar=False, ddof=0)
    Sinv = np.linalg.pinv(S)
    D = Xc @ Sinv @ Xc.T  # n x n Mahalanobis cross-products

    b1p = np.sum(D**3) / n**2
    A = n * b1p / 6
    df_skew = p * (p + 1) * (p + 2) / 6
    p_skew = 1 - stats.chi2.cdf(A, df_skew)

    b2p = np.trace(D**2) / n if False else np.sum(np.diag(D) ** 2) / n
    mean_b2 = p * (p + 2)
    var_b2 = 8 * p * (p + 2) / n
    Z = (b2p - mean_b2) / np.sqrt(var_b2)
    p_kurt = 2 * (1 - stats.norm.cdf(abs(Z)))

    return {
        "skew_stat": A, "skew_df": df_skew, "skew_p": p_skew,
        "kurt_z": Z, "kurt_p": p_kurt, "n": n,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["JobSeek_f"] = df["JobSeek_bin"].map({1: "SeekingOpen", 0: "NotInterested"})
    lines_one, lines_two, lines_assum = [], [], []

    # ---------------- One-way MANOVA ---------------------------------------
    formula = " + ".join(DVS) + " ~ C(JobSeek_f)"
    m1 = MANOVA.from_formula(formula, data=df)
    res1 = m1.mv_test()
    lines_one.append("One-way MANOVA: DVs ~ JobSeek")
    lines_one.append(str(res1))
    print(lines_one[-1][:2000])

    # Univariate follow-up ANOVAs (run regardless, reported conditional on
    # the one-way MANOVA being significant — see report text)
    anova_rows = []
    for dv in DVS:
        g1 = df.loc[df["JobSeek_bin"] == 1, dv]
        g0 = df.loc[df["JobSeek_bin"] == 0, dv]
        fstat, pval = stats.f_oneway(g1, g0)
        anova_rows.append({
            "DV": dv, "F": round(fstat, 3), "p": pval,
            "mean_seeking": round(g1.mean(), 3), "mean_notint": round(g0.mean(), 3),
        })
    anova_tbl = pd.DataFrame(anova_rows)
    anova_tbl.to_csv("tables/table2c_univariate_anova.csv", index=False)
    print("\nUnivariate follow-up ANOVAs:\n", anova_tbl)

    # ---------------- Two-way MANOVA: JobSeek x Country ---------------------
    formula2 = " + ".join(DVS) + " ~ C(JobSeek_f) * C(Country)"
    m2 = MANOVA.from_formula(formula2, data=df)
    res2 = m2.mv_test()
    lines_two.append("Two-way MANOVA: DVs ~ JobSeek * Country")
    lines_two.append(str(res2))
    print(lines_two[-1][:3000])

    with open("tables/table2_manova_oneway.txt", "w") as f:
        f.write("\n".join(lines_one))
    with open("tables/table2_manova_twoway.txt", "w") as f:
        f.write("\n".join(lines_two))

    # ---------------- Assumption checks -------------------------------------
    bm = box_m_test(df, DVS, "JobSeek_f")
    lines_assum.append(
        f"Box's M (equality of covariance matrices across JobSeek groups): "
        f"M={bm['M']:.2f}, chi2({bm['df']:.0f})={bm['chi2']:.2f}, p={bm['p']:.4g}"
    )

    lines_assum.append("\nMardia's test on group-mean-centered residuals, by JobSeek group:")
    for g, sub in df.groupby("JobSeek_f"):
        X = sub[DVS].values
        X = X - X.mean(axis=0)  # residuals under the one-way model
        mt = mardia_test(X)
        lines_assum.append(
            f"  {g} (n={mt['n']}): skewness chi2({mt['skew_df']:.0f})="
            f"{mt['skew_stat']:.2f}, p={mt['skew_p']:.4g}; "
            f"kurtosis Z={mt['kurt_z']:.2f}, p={mt['kurt_p']:.4g}"
        )

    lines_assum.append("\nLevene's test (equal variance) per DV, JobSeek groups:")
    for dv in DVS:
        g1 = df.loc[df["JobSeek_bin"] == 1, dv]
        g0 = df.loc[df["JobSeek_bin"] == 0, dv]
        lstat, lp = stats.levene(g1, g0)
        lines_assum.append(f"  {dv}: W={lstat:.3f}, p={lp:.4g}")

    with open("tables/table2b_assumptions.txt", "w") as f:
        f.write("\n".join(lines_assum))
    print("\n".join(lines_assum))

    # ---------------- Figure 1: parallel coordinates of group means --------
    means = df.groupby("JobSeek_f")[DVS].mean()
    z = (means - df[DVS].mean()) / df[DVS].std()  # standardized for comparability

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DVS))
    for g in z.index:
        ax.plot(x, z.loc[g, DVS].values, marker="o", linewidth=2, label=g)
    ax.set_xticks(x)
    ax.set_xticklabels([DV_LABELS[d] for d in DVS])
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Standardized group mean (z-score)")
    ax.set_title("Figure 1. Standardized mean profiles by job-seeking status")
    ax.legend(title="JobSeek group")
    fig.tight_layout()
    fig.savefig("figures/fig1_parallel_coords.png", dpi=150)
    print("\nSaved figures/fig1_parallel_coords.png")


if __name__ == "__main__":
    main()
