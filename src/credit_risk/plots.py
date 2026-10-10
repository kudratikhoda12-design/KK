"""Figures for the report (matplotlib only, one idea per chart)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

PALETTE = {"lr": "#4C78A8", "rf": "#E45756", "hgb": "#54A24B", "altman": "#7F7F7F", "ink": "#222222"}
LABELS = {"lr": "Logistic regression", "rf": "Random forest", "hgb": "Gradient boosting (challenger)",
          "altman": "Altman Z'' (benchmark)"}


def _finish(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def roc_plot(y, scores: dict, aucs: dict, path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for k, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=PALETTE[k], lw=2, ls="--" if k == "altman" else "-",
                label=f"{LABELS[k]} (AUC {aucs[k]:.3f})")
    ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves - hold-out test set")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    _finish(fig, path)


def calibration_plot(y, curves: dict, path: Path):
    """curves: label -> reliability DataFrame (mean_pd, observed_rate)."""
    fig, ax = plt.subplots(figsize=(5.6, 5))
    top = max(max(c["mean_pd"].max(), c["observed_rate"].max()) for c in curves.values()) * 1.05
    ax.plot([0, top], [0, top], color="#BBBBBB", lw=1, label="Perfect calibration")
    for (label, c), color in zip(curves.items(), ["#BBBBBB", PALETTE["rf"], PALETTE["lr"]]):
        ax.plot(c["mean_pd"], c["observed_rate"], marker="o", ms=4, lw=1.8, color=color, label=label)
    ax.set(xlabel="Mean predicted PD (decile)", ylabel="Observed default rate",
           title="Reliability - hold-out test set")
    ax.legend(fontsize=8, frameon=False)
    _finish(fig, path)


def cost_curve_plot(curve, chosen: float, others: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    c = curve[np.isfinite(curve["threshold"])]
    ax.plot(c["threshold"], c["cost"] / c["cost"].max(), color=PALETTE["rf"], lw=2)
    ax.axvline(chosen, color=PALETTE["ink"], lw=1.5, label=f"cost-optimal  PD >= {chosen:.3f}")
    for (label, t), ls in zip(others.items(), [":", "--"]):
        ax.axvline(t, color="#888888", lw=1.2, ls=ls, label=f"{label}  PD >= {t:.3f}")
    ax.set(xlim=(0, 0.6), xlabel="PD threshold (flag if PD >= t)", ylabel="Total cost (relative to maximum)",
           title="Cost-sensitive threshold (out-of-fold, training firms)")
    ax.legend(fontsize=8, frameon=False)
    _finish(fig, path)


def capture_plot(curves: dict, path: Path):
    """curves: label -> (fractions, captured share)."""
    fig, ax = plt.subplots(figsize=(5.6, 5))
    colors = {"EL rank": PALETTE["rf"], "PD rank": PALETTE["lr"], "Random": "#BBBBBB"}
    for label, (q, c) in curves.items():
        ax.plot(q, c, lw=2, color=colors.get(label, PALETTE["ink"]), label=label, ls=":" if label == "Random" else "-")
    ax.set(xlabel="Share of portfolio reviewed (top-ranked first)", ylabel="Share of realised loss captured",
           title="Watchlist back-test - hold-out portfolio", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    _finish(fig, path)


def sqc_plot(report, path: Path, top: int = 20):
    r = report[report["status"] == "kept"].sort_values("lift", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 6))
    y = np.arange(len(r))
    ax.barh(y + 0.2, r["detection_rate"] * 100, height=0.4, color=PALETTE["rf"], label="Defaulters beyond limit (detection)")
    ax.barh(y - 0.2, r["false_alarm_rate"] * 100, height=0.4, color="#BBBBBB", label="Healthy firms beyond limit (false alarm)")
    ax.set_yticks(y, [f"{a} ({t})" for a, t in zip(r["ratio"], r["adverse_tail"])], fontsize=8)
    ax.set(xlabel="% of firms outside the adverse-tail control limit",
           title=f"SQC screening - top {top} kept ratios by lift")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _finish(fig, path)


def watchlist_plot(wl, path: Path, top: int = 25):
    t = wl.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    sc = ax.barh(np.arange(len(t)), t["expected_loss"], color=plt.cm.OrRd(np.clip(t["pd"] / 0.6, 0.15, 1)))
    ax.set_yticks(np.arange(len(t)), [f"firm {i}" + (" *" if d == 1 else "") for i, d in
                                       zip(t["firm_id"], t.get("realized_default", np.zeros(len(t))))], fontsize=7)
    ax.set(xlabel="Expected loss (source currency units)",
           title=f"Top {top} watchlist by Expected Loss (darker = higher PD; * = defaulted)")
    _finish(fig, path)


def horizon_plot(table, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for k in ("lr", "rf", "hgb", "altman"):
        ax.plot(table.index, table[k], marker="o", lw=2, color=PALETTE[k], ls="--" if k == "altman" else "-", label=LABELS[k])
    ax.set(xlabel="Years before bankruptcy at which ratios are observed", ylabel="Hold-out AUC-ROC",
           title="Discrimination by forecast horizon")
    ax.set_xticks(table.index)
    ax.legend(fontsize=8, frameon=False)
    _finish(fig, path)
