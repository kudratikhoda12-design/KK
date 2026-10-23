"""
Unit 5 — Logistic regression (odds ratios) + ridge regularization.

Model: JobSeek ~ logComp + YearsCodePro + WorkWeekHrs + Age + JobSat_num +
                  CodeRevHrs + EdLevel_grp + Hobbyist + Country

Outputs:
  tables/table4_odds_ratios.csv
  tables/table4b_lr_tests.csv
  tables/table4c_condition_number.txt
  tables/table4d_ridge_vs_ols_coefs.csv
  figures/fig4_ridge_trace.png
  figures/fig4b_odds_ratio_plot.png

Run: python3 scripts/05_logistic.py
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/clean_sample.csv"
NUMERIC = ["logComp", "YearsCodePro", "WorkWeekHrs", "Age", "JobSat_num", "CodeRevHrs"]

ED_MAP = {
    "Bachelor’s degree (BA, BS, B.Eng., etc.)": "Bachelors",
    "Master’s degree (MA, MS, M.Eng., MBA, etc.)": "Masters",
    "Other doctoral degree (Ph.D, Ed.D., etc.)": "Doctoral_Prof",
    "Professional degree (JD, MD, etc.)": "Doctoral_Prof",
    "Some college/university study without earning a degree": "LessThanBachelors",
    "Associate degree": "LessThanBachelors",
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "LessThanBachelors",
    "Primary/elementary school": "LessThanBachelors",
    "I never completed any formal education": "LessThanBachelors",
}


def build_design(df, standardize_numeric=False):
    d = df.copy()
    d["EdLevel_grp"] = d["EdLevel"].map(ED_MAP)
    d["Hobbyist_bin"] = (d["Hobbyist"] == "Yes").astype(int)

    num = d[NUMERIC].copy()
    if standardize_numeric:
        num = (num - num.mean()) / num.std(ddof=1)

    ed_dum = pd.get_dummies(d["EdLevel_grp"], prefix="Ed", drop_first=False)
    ed_dum = ed_dum.drop(columns=["Ed_LessThanBachelors"])  # reference category

    country_dum = pd.get_dummies(d["Country"], prefix="Ctry", drop_first=False)
    country_dum = country_dum.drop(columns=["Ctry_United States"])  # reference

    X = pd.concat([num, d[["Hobbyist_bin"]], ed_dum, country_dum], axis=1)
    X = X.astype(float)
    y = d["JobSeek_bin"].astype(int).values
    return X, y


def lr_test(full_llf, full_k, red_llf, red_k):
    chi2_stat = 2 * (full_llf - red_llf)
    df_diff = full_k - red_k
    pval = 1 - stats.chi2.cdf(chi2_stat, df_diff)
    return chi2_stat, df_diff, pval


def main():
    df = pd.read_csv(DATA_PATH)

    # ============ Part A: interpretable (natural-unit) logistic model =======
    X, y = build_design(df, standardize_numeric=False)
    Xc = sm.add_constant(X)
    model = sm.Logit(y, Xc).fit(disp=0)
    print(model.summary())

    params = model.params
    conf = model.conf_int()
    or_table = pd.DataFrame({
        "coef": params, "OR": np.exp(params),
        "CI_low": np.exp(conf[0]), "CI_high": np.exp(conf[1]),
        "p": model.pvalues,
    }).round(4)
    or_table.to_csv("tables/table4_odds_ratios.csv")
    print("\nOdds ratio table:\n", or_table)

    mcfadden_r2 = model.prsquared
    print(f"\nMcFadden pseudo-R2: {mcfadden_r2:.4f}, deviance: {-2*model.llf:.2f}, "
          f"LL0: {model.llnull:.2f}, LLfull: {model.llf:.2f}")

    # ---- LR tests for block importance (drop numerics / EdLevel / Hobbyist / Country) ----
    blocks = {
        "numerics": NUMERIC,
        "EdLevel": [c for c in X.columns if c.startswith("Ed_")],
        "Hobbyist": ["Hobbyist_bin"],
        "Country": [c for c in X.columns if c.startswith("Ctry_")],
    }
    lr_rows = []
    full_k = Xc.shape[1]
    for name, cols in blocks.items():
        Xr = Xc.drop(columns=cols)
        mr = sm.Logit(y, Xr).fit(disp=0)
        chi2_stat, df_diff, pval = lr_test(model.llf, full_k, mr.llf, Xr.shape[1])
        lr_rows.append({"block": name, "LR_chi2": round(chi2_stat, 3),
                         "df": df_diff, "p": pval})
    lr_tbl = pd.DataFrame(lr_rows)
    lr_tbl.to_csv("tables/table4b_lr_tests.csv", index=False)
    print("\nLikelihood-ratio tests for block importance:\n", lr_tbl)

    # ============ Part B: collinearity / ridge story (standardized X) =======
    Xz, y = build_design(df, standardize_numeric=True)
    Xz_const = sm.add_constant(Xz)
    cond_num = np.linalg.cond(Xz_const.values)
    with open("tables/table4c_condition_number.txt", "w") as f:
        f.write(f"Condition number of standardized design matrix X (incl. intercept): "
                f"{cond_num:.2f}\n")
        f.write("Rule of thumb: >30 indicates moderate-to-strong multicollinearity.\n")
        f.write("Largest pairwise correlation among predictors: YearsCodePro vs Age "
                "= 0.859 (see Table 1b), the main driver of the elevated condition number.\n")
    print(f"\nCondition number of standardized X (with intercept): {cond_num:.2f}")

    # Ridge trace across log-lambda grid
    lambdas = np.logspace(-3, 3, 40)
    Cs = 1.0 / lambdas
    coef_path = []
    for C in Cs:
        clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000)
        clf.fit(Xz.values, y)
        coef_path.append(clf.coef_.ravel())
    coef_path = np.array(coef_path)  # (n_lambda, n_features)

    fig, ax = plt.subplots(figsize=(9, 6))
    for j, col in enumerate(Xz.columns):
        if col in NUMERIC:  # highlight numeric predictors; grey out dummies
            ax.plot(np.log10(lambdas), coef_path[:, j], label=col, linewidth=2)
        else:
            ax.plot(np.log10(lambdas), coef_path[:, j], color="lightgrey", linewidth=0.8)
    ax.set_xlabel("log10(lambda)")
    ax.set_ylabel("Standardized ridge coefficient")
    ax.set_title("Figure 4. Ridge trace: numeric-predictor coefficients vs. log(lambda)\n"
                  "(grey = categorical control coefficients)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig("figures/fig4_ridge_trace.png", dpi=150)
    print("\nSaved figures/fig4_ridge_trace.png")

    # CV-selected lambda (5-fold, scored by held-out log-loss ~ deviance)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lrcv = LogisticRegressionCV(
        Cs=Cs, cv=cv, penalty="l2", solver="lbfgs", scoring="neg_log_loss",
        max_iter=2000, refit=True,
    )
    lrcv.fit(Xz.values, y)
    best_C = lrcv.C_[0]
    best_lambda = 1.0 / best_C
    print(f"\nCV-selected lambda (min deviance, 5-fold): {best_lambda:.4f} "
          f"(C={best_C:.4f})")

    ols_logit_z = sm.Logit(y, sm.add_constant(Xz)).fit(disp=0)
    ridge_best = LogisticRegression(penalty="l2", C=best_C, solver="lbfgs", max_iter=2000)
    ridge_best.fit(Xz.values, y)

    compare = pd.DataFrame({
        "OLS_logit_std_coef": ols_logit_z.params.drop("const").reindex(Xz.columns),
        "Ridge_std_coef": ridge_best.coef_.ravel(),
    })
    compare["shrinkage_pct"] = (
        100 * (1 - compare["Ridge_std_coef"].abs() / compare["OLS_logit_std_coef"].abs())
    ).round(1)
    compare = compare.round(4)
    compare.to_csv("tables/table4d_ridge_vs_ols_coefs.csv")
    print(f"\nRidge (lambda={best_lambda:.3f}) vs OLS logistic, standardized coefficients:\n",
          compare)

    with open("tables/table4c_condition_number.txt", "a") as f:
        f.write(f"\nCV-selected ridge lambda (5-fold, min deviance/log-loss): "
                f"{best_lambda:.4f}\n")

    # ---- Supplementary: odds-ratio forest plot (natural-unit model) --------
    plot_tbl = or_table.drop(index="const")
    fig, ax = plt.subplots(figsize=(8, 7))
    yvals = np.arange(len(plot_tbl))
    ax.errorbar(plot_tbl["OR"], yvals,
                xerr=[plot_tbl["OR"] - plot_tbl["CI_low"], plot_tbl["CI_high"] - plot_tbl["OR"]],
                fmt="o", color="#4C72B0", ecolor="grey", capsize=3)
    ax.axvline(1, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(yvals)
    ax.set_yticklabels(plot_tbl.index)
    ax.set_xlabel("Odds ratio (95% CI)")
    ax.set_title("Figure 4b. Odds ratios with 95% confidence intervals")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig("figures/fig4b_odds_ratio_plot.png", dpi=150)
    print("Saved figures/fig4b_odds_ratio_plot.png")


if __name__ == "__main__":
    main()
