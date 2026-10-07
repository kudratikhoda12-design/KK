"""Figures for the formal report, drawn ONLY from saved outputs (no data, no models).

    python scripts/make_report_figures.py

Reads reports/tables/validation_folds.csv, test_per_block.csv, test_drift.csv and
src/config.py; writes three PNGs to figures/:
  report_architecture.png  schematic of the research pipeline (no data)
  report_walk_forward.png  the actual walk-forward and test blocks used
  report_drift.png         test AUC by half-year and volatility-feature PSI
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.eda import ACCENT, ACCENT2, INK, MUTED  # noqa: E402  (also sets the shared style)

T, F = config.TABLES_DIR, config.FIGURES_DIR
LIGHT = "#d9dee6"


def architecture(path: Path) -> None:
    steps = [
        ("Binance Vision archive", "110 monthly files\nSHA-256 verified"),
        ("Audit + 1-min grid", "ms/µs units, gaps,\nphase-shifted candles"),
        ("16 features", "closed candles only;\nleakage perturbation test"),
        ("Walk-forward validation", "2019–2023, 10 purged folds;\nLR vs LightGBM rule"),
        ("Frozen selection", "committed to git\nbefore the test"),
        ("Untouched test", "2024-01 → 2026-09,\nrun once"),
        ("Costed backtest", "next-minute fills,\n7 bp/side + funding"),
        ("Risk & kill tests", "VaR/ES, stress, 23 variants,\nETH replication"),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 4); ax.set_ylim(0, 2); ax.axis("off")
    w, h = 0.84, 0.66
    # snake layout: top row left->right, bottom row right->left
    centers = [(i + 0.5, 1.48) for i in range(4)] + [(3.5 - i, 0.52) for i in range(4)]
    for k, ((title, sub), (cx, cy)) in enumerate(zip(steps, centers)):
        face = "#eef3fa" if k < 4 else "#fdf1ea"
        edge = ACCENT if k < 4 else ACCENT2
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=face, ec=edge, lw=1.2))
        ax.text(cx, cy + 0.15, title, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
        ax.text(cx, cy - 0.12, sub, ha="center", va="center", fontsize=7.5, color=MUTED)
    arrow = dict(arrowstyle="-|>", color=MUTED, lw=1.0)
    for i in range(3):
        ax.annotate("", xy=(i + 1.5 - w / 2, 1.48), xytext=(i + 0.5 + w / 2, 1.48), arrowprops=arrow)
        ax.annotate("", xy=(2.5 - i + w / 2, 0.52), xytext=(3.5 - i - w / 2, 0.52), arrowprops=arrow)
    ax.annotate("", xy=(3.5, 0.52 + h / 2), xytext=(3.5, 1.48 - h / 2), arrowprops=arrow)
    ax.text(0.02, 1.95, "Development (data before 2024)", fontsize=8, color=ACCENT, va="top")
    ax.text(0.02, 0.0, "Evaluation (2024-01 → 2026-09, after the design was frozen)", fontsize=8, color=ACCENT2, va="bottom")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def walk_forward(path: Path) -> None:
    folds = pd.read_csv(T / "validation_folds.csv")
    rows = []
    for _, r in folds.iterrows():
        tr0, tr1 = [pd.Timestamp(x.strip()) for x in r["train"].split("->")]
        ev0, ev1 = [pd.Timestamp(x.strip()) for x in r["eval"].split("->")]
        rows.append(dict(label=f"val {r['fold']}", tr0=tr0, tr1=tr1, ev0=ev0, ev1=ev1, kind="val"))
    blocks = pd.read_csv(T / "test_per_block.csv")["block"].tolist()
    start = config.TEST_START.tz_localize(None)
    data_end = pd.Timestamp("2026-09-30 23:00")
    for b in blocks:
        ev1 = min(start + pd.DateOffset(months=6) - pd.Timedelta(hours=1), data_end)
        rows.append(dict(label=f"test {b}", tr0=pd.Timestamp("2017-08-17"), tr1=start - pd.Timedelta(hours=1),
                         ev0=start, ev1=ev1, kind="test"))
        start = start + pd.DateOffset(months=6)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, r in enumerate(rows[::-1]):
        ax.barh(i, (r["tr1"] - r["tr0"]).days, left=r["tr0"], height=0.62, color=LIGHT, edgecolor="none")
        ax.barh(i, (r["ev1"] - r["ev0"]).days + 1, left=r["ev0"], height=0.62,
                color=ACCENT if r["kind"] == "val" else ACCENT2, edgecolor="none")
    ax.set_axisbelow(True)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["label"] for r in rows[::-1]], fontsize=7.5)
    ax.axvline(config.TEST_START.tz_localize(None), color=INK, lw=0.9, ls="--")
    ax.text(config.TEST_START.tz_localize(None), len(rows) - 0.3, "  design frozen; test starts 2024-01-01",
            fontsize=8, color=INK, va="bottom")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=LIGHT, label="training window (expanding, purged)"),
                       Patch(color=ACCENT, label="validation block (model and threshold choice)"),
                       Patch(color=ACCENT2, label="untouched test block (frozen model refitted)")],
              frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    ax.set_title("Walk-forward design actually used (from validation_folds.csv and test blocks)")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def drift(path: Path) -> None:
    blk = pd.read_csv(T / "test_per_block.csv")
    psi = pd.read_csv(T / "test_drift.csv")
    labels = [b.replace("2026H2", "2026H2*") for b in blk["block"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.set_axisbelow(True)
    a1.plot(range(len(blk)), blk["auc"], "o-", color=ACCENT, ms=6, lw=2)
    a1.axhline(0.5, color=MUTED, lw=0.9)
    a1.text(len(blk) - 1, 0.5015, "no skill (0.5)", ha="right", fontsize=8, color=MUTED)
    for i, v in enumerate(blk["auc"]):
        a1.text(i, v + 0.0025, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    a1.set_xticks(range(len(blk))); a1.set_xticklabels(labels, fontsize=8)
    a1.set_ylim(0.49, 0.58); a1.set_ylabel("test AUC"); a1.set_title("Test AUC by half-year")
    a2.set_axisbelow(True); a2.grid(axis="x", visible=False)
    a2.bar(range(len(psi)), psi["log_rv_24h"], color=ACCENT2, width=0.6)
    a2.axhline(0.25, color=MUTED, lw=0.9)
    a2.text(-0.4, 3.2, "grey line: PSI = 0.25, the conventional\n'large shift' threshold", fontsize=8, color=MUTED, va="top")
    for i, v in enumerate(psi["log_rv_24h"]):
        a2.text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    a2.set_xticks(range(len(psi))); a2.set_xticklabels(labels, fontsize=8)
    a2.set_ylabel("PSI vs development period"); a2.set_title("Drift of 24-hour realised volatility (PSI)")
    fig.text(0.01, -0.02, "* 2026H2 covers July–September 2026 only (2,207 test hours).", fontsize=7.5, color=MUTED)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    F.mkdir(parents=True, exist_ok=True)
    architecture(F / "report_architecture.png")
    walk_forward(F / "report_walk_forward.png")
    drift(F / "report_drift.png")
    print("wrote", [p.name for p in sorted(F.glob("report_*.png"))])
