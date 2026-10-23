"""
Unit 4 — Principal Component Analysis on the numeric block.

Numerics: logComp, YearsCodePro, WorkWeekHrs, Age, JobSat_num, CodeRevHrs
(standardized). Retention rule: eigenvalue > 1 AND cumulative variance >= 70%.

Outputs:
  tables/table3_pca_loadings.csv
  tables/table3b_pca_eigen_summary.csv
  figures/fig2_scree.png
  figures/fig3_biplot.png
  data/processed/pc_scores.csv   (PC scores appended, for Unit 7 clustering)

Run: python3 scripts/04_pca.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/clean_sample.csv"
NUMERIC = ["logComp", "YearsCodePro", "WorkWeekHrs", "Age", "JobSat_num", "CodeRevHrs"]
LABELS = {
    "logComp": "log(Comp)", "YearsCodePro": "YearsCodePro",
    "WorkWeekHrs": "WorkWeekHrs", "Age": "Age", "JobSat_num": "JobSat",
    "CodeRevHrs": "CodeRevHrs",
}
EIG_RULE_CUMVAR = 0.70


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC].values
    Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    corr = pd.DataFrame(Xz, columns=NUMERIC).corr()
    corr.round(3).to_csv("tables/table_corr_for_pca.csv")
    print("Correlation matrix used for PCA:\n", corr.round(3))

    # Eigendecomposition of the correlation matrix
    eigvals, eigvecs = np.linalg.eigh(corr.values)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    var_explained = eigvals / eigvals.sum()
    cum_var = np.cumsum(var_explained)

    n_by_eig1 = int(np.sum(eigvals > 1))
    n_by_cum = int(np.argmax(cum_var >= EIG_RULE_CUMVAR) + 1)
    n_retain = max(n_by_eig1, 1)
    # State rule explicitly: eigenvalue>1 AND cum var >= 70%; if eigenvalue>1
    # count alone doesn't reach 70%, extend to the component that does.
    n_retain = max(n_by_eig1, n_by_cum) if n_by_eig1 < n_by_cum else n_by_eig1
    print(f"\nComponents with eigenvalue>1: {n_by_eig1}; "
          f"components needed for >=70% cum. var: {n_by_cum}; retaining {n_retain}")

    eig_summary = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(eigvals))],
        "eigenvalue": eigvals.round(3),
        "pct_variance": (var_explained * 100).round(2),
        "cum_pct_variance": (cum_var * 100).round(2),
    })
    eig_summary.to_csv("tables/table3b_pca_eigen_summary.csv", index=False)
    print("\n", eig_summary)

    # Loadings = eigenvectors scaled by sqrt(eigenvalue) (correlation-scale loadings)
    loadings = eigvecs * np.sqrt(eigvals)
    loadings_df = pd.DataFrame(
        loadings[:, :n_retain],
        index=[LABELS[c] for c in NUMERIC],
        columns=[f"PC{i+1}" for i in range(n_retain)],
    ).round(3)
    loadings_df.to_csv("tables/table3_pca_loadings.csv")
    print("\nLoadings (retained components):\n", loadings_df)

    # PC scores for ALL components we might use downstream (retain n_retain for clustering)
    pc_scores_all = Xz @ eigvecs  # standardized-data scores, unit-eigenvector basis
    for i in range(n_retain):
        df[f"PC{i+1}"] = pc_scores_all[:, i]
    df.to_csv("data/processed/pc_scores.csv", index=False)
    print(f"\nSaved PC scores (first {n_retain} PCs) -> data/processed/pc_scores.csv")

    # ---------------- Figure 2: scree plot ----------------------------------
    fig, ax1 = plt.subplots(figsize=(7, 5))
    x = np.arange(1, len(eigvals) + 1)
    ax1.plot(x, eigvals, "o-", color="#4C72B0", label="Eigenvalue")
    ax1.axhline(1, color="grey", linestyle="--", linewidth=1, label="Eigenvalue = 1")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Eigenvalue", color="#4C72B0")
    ax1.set_xticks(x)
    ax2 = ax1.twinx()
    ax2.plot(x, cum_var * 100, "s-", color="#DD8452", label="Cumulative % variance")
    ax2.axhline(70, color="#DD8452", linestyle=":", linewidth=1)
    ax2.set_ylabel("Cumulative % variance explained", color="#DD8452")
    fig.suptitle("Figure 2. Scree plot: eigenvalues and cumulative variance explained")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="center right")
    fig.tight_layout()
    fig.savefig("figures/fig2_scree.png", dpi=150)
    print("Saved figures/fig2_scree.png")

    # ---------------- Figure 3: biplot (PC1 vs PC2) --------------------------
    fig, ax = plt.subplots(figsize=(7.5, 7))
    colors = df["JobSeek_bin"].map({1: "#DD8452", 0: "#4C72B0"})
    ax.scatter(df["PC1"], df["PC2"], c=colors, alpha=0.25, s=10)
    scale = 3.0
    for i, var in enumerate(NUMERIC):
        ax.arrow(0, 0, eigvecs[i, 0] * scale, eigvecs[i, 1] * scale,
                  color="black", width=0.01, head_width=0.12, length_includes_head=True)
        ax.text(eigvecs[i, 0] * scale * 1.15, eigvecs[i, 1] * scale * 1.15,
                LABELS[NUMERIC[i]], fontsize=10, ha="center", color="black")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% var)")
    ax.set_title("Figure 3. PCA biplot (PC1 vs PC2), points colored by JobSeek status")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#DD8452", label="Seeking/Open", markersize=8),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", label="Not interested", markersize=8)]
    ax.legend(handles=handles)
    fig.tight_layout()
    fig.savefig("figures/fig3_biplot.png", dpi=150)
    print("Saved figures/fig3_biplot.png")


if __name__ == "__main__":
    main()
