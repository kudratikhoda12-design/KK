"""
Unit 7 — Clustering: K-means and Ward hierarchical clustering.

Clustering is performed on the 3 retained PC scores from Unit 4 (PC1-PC3,
which already capture ~70% of the numeric block's variance and remove the
YearsCodePro/Age collinearity that would otherwise dominate Euclidean
distance). k chosen via elbow (within-cluster SS) + gap statistic.

Outputs:
  tables/table6_cluster_profiles.csv
  tables/table6b_cluster_vs_jobseek_country.csv
  tables/table6c_gap_statistic.csv
  tables/table6d_ari.txt
  figures/fig6_dendrogram.png
  figures/fig7_kmeans_pc1_pc2.png
  figures/fig6b_elbow_gap.png

Run: python3 scripts/07_clustering.py
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/pc_scores.csv"
NUMERIC = ["logComp", "YearsCodePro", "WorkWeekHrs", "Age", "JobSat_num", "CodeRevHrs"]
PCS = ["PC1", "PC2", "PC3"]
SEED = 42
K_RANGE = range(1, 11)
N_REF_GAP = 15


def gap_statistic(X, k_range, n_ref=N_REF_GAP, seed=SEED):
    rng = np.random.default_rng(seed)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    gaps, sk = [], []
    wk_real_all = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
        km.fit(X)
        wk_real = np.log(km.inertia_)
        wk_real_all.append(wk_real)

        wk_refs = []
        for b in range(n_ref):
            X_ref = rng.uniform(mins, maxs, size=X.shape)
            km_ref = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed + b)
            km_ref.fit(X_ref)
            wk_refs.append(np.log(km_ref.inertia_))
        wk_refs = np.array(wk_refs)
        gap = wk_refs.mean() - wk_real
        sd = wk_refs.std() * np.sqrt(1 + 1.0 / n_ref)
        gaps.append(gap)
        sk.append(sd)
    return np.array(wk_real_all), np.array(gaps), np.array(sk)


def main():
    df = pd.read_csv(DATA_PATH)
    Xpc = df[PCS].values

    # ---------------- Elbow + gap statistic ----------------------------------
    inertias = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED)
        km.fit(Xpc)
        inertias.append(km.inertia_)

    wk_log, gaps, sk = gap_statistic(Xpc, K_RANGE)
    gap_tbl = pd.DataFrame({
        "k": list(K_RANGE), "inertia": inertias, "log_Wk": wk_log,
        "gap": gaps, "s_k": sk,
    })
    # Standard rule: smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
    chosen_k = None
    for i in range(len(gap_tbl) - 1):
        if gap_tbl["gap"][i] >= gap_tbl["gap"][i + 1] - gap_tbl["s_k"][i + 1]:
            chosen_k = gap_tbl["k"][i]
            break
    if chosen_k is None:
        chosen_k = int(gap_tbl.loc[gap_tbl["gap"].idxmax(), "k"])
    gap_tbl.to_csv("tables/table6c_gap_statistic.csv", index=False)
    print("Gap statistic table:\n", gap_tbl)
    print(f"\nGap statistic selects k = {chosen_k}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(list(K_RANGE), inertias, "o-", color="#4C72B0")
    ax1.set_xlabel("k"); ax1.set_ylabel("Within-cluster SS (inertia)")
    ax1.set_title("Elbow plot")
    ax2.errorbar(list(K_RANGE), gaps, yerr=sk, fmt="o-", color="#DD8452")
    ax2.axvline(chosen_k, color="grey", linestyle="--", label=f"chosen k={chosen_k}")
    ax2.set_xlabel("k"); ax2.set_ylabel("Gap statistic")
    ax2.set_title("Gap statistic")
    ax2.legend()
    fig.suptitle("Figure 6b. K selection: elbow and gap statistic")
    fig.tight_layout()
    fig.savefig("figures/fig6b_elbow_gap.png", dpi=150)

    k = chosen_k

    # ---------------- Final k-means (k-means++, multiple restarts) ----------
    km_final = KMeans(n_clusters=k, init="k-means++", n_init=25, random_state=SEED)
    km_labels = km_final.fit_predict(Xpc)
    df["kmeans_cluster"] = km_labels
    print(f"\nFinal k-means (k={k}) within-cluster SS: {km_final.inertia_:.2f}")

    # ---------------- Hierarchical (Ward, Euclidean) -------------------------
    Z = linkage(Xpc, method="ward", metric="euclidean")
    hc_labels = fcluster(Z, t=k, criterion="maxclust")
    df["hc_cluster"] = hc_labels

    ari = adjusted_rand_score(km_labels, hc_labels)
    with open("tables/table6d_ari.txt", "w") as f:
        f.write(f"Adjusted Rand Index between k-means and Ward hierarchical "
                f"partitions (k={k}): {ari:.4f}\n")
    print(f"\nAdjusted Rand Index (k-means vs. Ward hierarchical): {ari:.4f}")

    # Dendrogram (truncated for readability)
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, truncate_mode="lastp", p=40, show_leaf_counts=True, ax=ax)
    ax.axhline(Z[-(k - 1), 2], color="red", linestyle="--",
               label=f"cut for k={k}")
    ax.set_title(f"Figure 6. Ward hierarchical dendrogram (truncated), cut at k={k}")
    ax.set_xlabel("Cluster (or sample count in truncated node)")
    ax.set_ylabel("Ward linkage distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/fig6_dendrogram.png", dpi=150)
    print("Saved figures/fig6_dendrogram.png")

    # ---------------- Figure 7: PC1-PC2 scatter colored by k-means cluster --
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")
    for c in range(k):
        sub = df[df["kmeans_cluster"] == c]
        ax.scatter(sub["PC1"], sub["PC2"], s=10, alpha=0.4, color=cmap(c),
                   label=f"Cluster {c} (n={len(sub)})")
    centers_pc = km_final.cluster_centers_
    ax.scatter(centers_pc[:, 0], centers_pc[:, 1], marker="X", s=200,
               color="black", label="Centroids", zorder=5)
    ax.set_xlabel("PC1 (seniority/earnings)")
    ax.set_ylabel("PC2 (workload/code-review)")
    ax.set_title(f"Figure 7. K-means clusters (k={k}) in PC1-PC2 space")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/fig7_kmeans_pc1_pc2.png", dpi=150)
    print("Saved figures/fig7_kmeans_pc1_pc2.png")

    # ---------------- Table 6: cluster profiles -------------------------------
    profile = df.groupby("kmeans_cluster")[NUMERIC].mean().round(2)
    profile["n"] = df.groupby("kmeans_cluster").size()
    profile.to_csv("tables/table6_cluster_profiles.csv")
    print("\nTable 6 — Cluster profiles (k-means):\n", profile)

    cross = pd.crosstab(df["kmeans_cluster"], df["JobSeek"], normalize="index").mul(100).round(1)
    cross["n"] = df.groupby("kmeans_cluster").size()
    country_cross = pd.crosstab(df["kmeans_cluster"], df["Country"], normalize="index").mul(100).round(1)
    combined = pd.concat({"JobSeek_pct": cross, "Country_pct": country_cross}, axis=1)
    combined.to_csv("tables/table6b_cluster_vs_jobseek_country.csv")
    print("\nCluster vs JobSeek / Country cross-tabs:\n", combined)

    df.to_csv("data/processed/clustered_sample.csv", index=False)
    print("\nSaved data/processed/clustered_sample.csv")


if __name__ == "__main__":
    main()
