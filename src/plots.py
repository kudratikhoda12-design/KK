"""Result figures. Each answers one question; static PNGs for the report.

Palette: reference categorical slots (blue, orange, aqua) validated for CVD
separation and contrast on the light surface; recessive solid grid, one y-axis.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eda import ACCENT, ACCENT2, INK, MUTED  # noqa: F401  (shared rcParams are set in eda)

AQUA = "#1baf7a"


def equity_curves(curves: dict[str, pd.Series], path, title: str) -> None:
    """Q: How did the strategy grow vs buy-and-hold, gross vs net of costs?"""
    colors = [ACCENT, ACCENT2, AQUA]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ends = []
    for (name, net), col in zip(curves.items(), colors):
        eq = (1 + net).cumprod()
        ax.plot(eq.index, eq.to_numpy(), color=col, lw=1.2, label=name)
        ends.append([name, eq.index[-1], float(eq.iloc[-1])])
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_yscale("log"); ax.set_ylabel("growth of 1 (log scale)"); ax.set_title(title)
    # Direct end labels, nudged apart in log space so they never overlap
    ends.sort(key=lambda e: e[2])
    min_gap = 0.06 * (np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))
    placed = []
    for name, x, y in ends:
        ly = np.log10(y)
        if placed and ly - placed[-1] < min_gap:
            ly = placed[-1] + min_gap
        placed.append(ly)
        ax.annotate(name, xy=(x, y), xytext=(x + pd.Timedelta(days=5), 10 ** ly), color=INK,
                    fontsize=8, va="center", annotation_clip=False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(curves))
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def calibration(cal: pd.DataFrame, path, title: str) -> None:
    """Q: When the model says P(up)=p, does price go up a fraction p of the time?"""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.fill_between(cal["mean_p"], cal["ci_low"], cal["ci_high"], color=ACCENT, alpha=0.18, lw=0,
                    label="observed 95% CI")
    ax.plot(cal["mean_p"], cal["observed_up"], "o-", color=ACCENT, ms=5, label="observed up-rate")
    lo = min(cal["mean_p"].min(), cal["ci_low"].min()) - 0.005
    hi = max(cal["mean_p"].max(), cal["ci_high"].max()) + 0.005
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, label="perfect calibration")
    ax.set_xlabel("mean predicted P(up), by decile"); ax.set_ylabel("observed share up")
    ax.set_title(title); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def decile_returns(cal: pd.DataFrame, path, title: str, round_trip_cost_bp: float) -> None:
    """Q: Does a higher predicted probability come with a higher average forward return,
    and is the spread between deciles larger than the trading cost?"""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(cal["decile"], cal["mean_fwd_ret_bp"], color=ACCENT, width=0.7)
    ax.axhline(0, color=MUTED, lw=0.8)
    for y in (round_trip_cost_bp, -round_trip_cost_bp):
        ax.axhline(y, color=ACCENT2, lw=1.0)
    lim = max(1.25 * round_trip_cost_bp, 1.1 * cal["mean_fwd_ret_bp"].abs().max())
    ax.set_ylim(-lim, lim)
    ax.text(0.6, round_trip_cost_bp, f"+/- round-trip cost ({round_trip_cost_bp:.0f} bp)", color=INK,
            fontsize=8, va="bottom", ha="left")
    ax.set_xlabel("predicted-probability decile (1 = most bearish)")
    ax.set_ylabel("mean forward return (bp)"); ax.set_title(title)
    ax.set_xticks(range(1, 11))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def sharpe_bars(table: pd.DataFrame, label_col: str, value_col: str, path, title: str,
                highlight: str | None = None) -> None:
    """Q: How does net Sharpe change across scenarios / variants? (horizontal bars, zero line)"""
    t = table[[label_col, value_col]].dropna().iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(t) + 1.2))
    colors = [ACCENT2 if highlight and lab == highlight else ACCENT for lab in t[label_col]]
    ax.barh(t[label_col], t[value_col], color=colors, height=0.65)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(title + ("\n(orange = frozen selection)" if highlight else ""))
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def mc_histogram(paths: pd.DataFrame, path, title: str) -> None:
    """Q: How uncertain is a one-year outcome given the realised return process?"""
    r = 100 * paths["one_year_return"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(r, bins=60, color=ACCENT, edgecolor="#fcfcfb", linewidth=0.5)
    ax.axvline(0, color=MUTED, lw=0.8)
    q5 = np.quantile(r, 0.05)
    ax.axvline(q5, color=ACCENT2, lw=1.2)
    ax.text(q5, ax.get_ylim()[1] * 0.92, f" 5th pct {q5:.1f}%", color=INK, fontsize=8)
    ax.set_xlabel("simulated one-year net return (%)"); ax.set_ylabel("paths"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
