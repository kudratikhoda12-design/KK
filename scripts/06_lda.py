"""
Unit 6 — Linear Discriminant Analysis vs. Logistic Regression.

LDA/QDA/Logistic all fit on the SAME feature space (the 6 standardized
numerics) so the classifier comparison is apples-to-apples. Cross-validation
scheme: 5-fold stratified CV, shuffled, random_state=42; out-of-fold
predictions are pooled into a single confusion matrix per model.

Outputs:
  tables/table5_confusion_matrices.csv
  tables/table5b_classifier_metrics.csv
  figures/fig5_roc_curves.png

Run: python3 scripts/06_lda.py
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/clean_sample.csv"
NUMERIC = ["logComp", "YearsCodePro", "WorkWeekHrs", "Age", "JobSat_num", "CodeRevHrs"]


def metrics_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn)  # recall for class 1 (seeking/open)
    spec = tn / (tn + fp)
    err = (fp + fn) / cm.sum()
    return sens, spec, err


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC].values
    Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    y = df["JobSeek_bin"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "Logistic": LogisticRegression(max_iter=1000),
    }

    cm_rows, metric_rows, roc_data = [], [], {}
    for name, clf in models.items():
        y_pred = cross_val_predict(clf, Xz, y, cv=cv, method="predict")
        y_proba = cross_val_predict(clf, Xz, y, cv=cv, method="predict_proba")[:, 1]
        cm = confusion_matrix(y, y_pred)
        sens, spec, err = metrics_from_cm(cm)
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        acc = accuracy_score(y, y_pred)

        cm_rows.append({
            "model": name, "TN": cm[0, 0], "FP": cm[0, 1],
            "FN": cm[1, 0], "TP": cm[1, 1],
        })
        metric_rows.append({
            "model": name, "accuracy": round(acc, 4), "error_rate": round(err, 4),
            "sensitivity_recall_seeking": round(sens, 4),
            "specificity": round(spec, 4), "ROC_AUC": round(roc_auc, 4),
        })
        roc_data[name] = (fpr, tpr, roc_auc)
        print(f"{name}: confusion matrix\n{cm}")
        print(f"  accuracy={acc:.4f} error={err:.4f} sens={sens:.4f} "
              f"spec={spec:.4f} AUC={roc_auc:.4f}\n")

    cm_tbl = pd.DataFrame(cm_rows)
    cm_tbl.to_csv("tables/table5_confusion_matrices.csv", index=False)
    metric_tbl = pd.DataFrame(metric_rows)
    metric_tbl.to_csv("tables/table5b_classifier_metrics.csv", index=False)
    print("Table 5 — Confusion matrices (5-fold CV, pooled out-of-fold):\n", cm_tbl)
    print("\nClassifier comparison metrics:\n", metric_tbl)

    # ---- Box's M already computed in Unit 3 (script 03) on the DV set used
    # for MANOVA; here we re-state the LDA-relevant covariance check on the
    # 6-variable numeric block used for classification.
    def box_m(df, cols, group_col):
        groups = df[group_col].unique()
        p = len(cols)
        covs, ns = [], []
        for g in groups:
            Xg = df.loc[df[group_col] == g, cols].values
            covs.append(np.cov(Xg, rowvar=False, ddof=1))
            ns.append(len(Xg))
        ns = np.array(ns)
        gN = len(groups)
        pooled = sum((n - 1) * c for n, c in zip(ns, covs)) / (ns.sum() - gN)
        _, logdet_pooled = np.linalg.slogdet(pooled)
        M = (ns.sum() - gN) * logdet_pooled
        for n, c in zip(ns, covs):
            _, logdet_c = np.linalg.slogdet(c)
            M -= (n - 1) * logdet_c
        c1 = (np.sum(1.0 / (ns - 1)) - 1.0 / (ns.sum() - gN))
        c1 *= (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (gN - 1))
        chi2_stat = M * (1 - c1)
        dfc = p * (p + 1) * (gN - 1) / 2
        pval = 1 - stats.chi2.cdf(chi2_stat, dfc)
        return M, chi2_stat, dfc, pval

    df["JobSeek_f"] = df["JobSeek_bin"].map({1: "SeekingOpen", 0: "NotInterested"})
    M, chi2s, dfc, pval = box_m(df, NUMERIC, "JobSeek_f")
    note = (
        f"Box's M on the 6-variable numeric block (LDA predictors): "
        f"chi2({dfc:.0f})={chi2s:.2f}, p={pval:.4g}. "
        "With n~8000, Box's M is highly sensitive and virtually always "
        "significant; a formal reading would favor QDA, but we select between "
        "LDA and QDA empirically using cross-validated AUC/error rate below "
        "rather than the Box's M p-value alone."
    )
    print("\n" + note)
    with open("tables/table5c_boxm_note.txt", "w") as f:
        f.write(note)

    # ---------------- Figure 5: ROC curves ----------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"LDA": "#4C72B0", "QDA": "#55A868", "Logistic": "#DD8452"}
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=colors[name], linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Figure 5. ROC curves: LDA, QDA, and logistic regression\n"
                  "(5-fold CV, numeric predictor block)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("figures/fig5_roc_curves.png", dpi=150)
    print("\nSaved figures/fig5_roc_curves.png")


if __name__ == "__main__":
    main()
