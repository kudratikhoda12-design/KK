"""Generate all figures used in the final report."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from markov_models import ttc_quarterly_matrix, conditional_matrix, STATES, GOOD, WATCH, DEFAULT

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TAB_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"
FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_GOOD, C_WATCH, C_DEFAULT = "#2E7D32", "#F9A825", "#C62828"
C_BASE, C_STRESS = "#1565C0", "#C62828"
C_ACTUAL, C_FC = "#263238", "#1565C0"

plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False,
})


def shade_recessions(ax, macro):
    rec = macro["recession"]
    in_rec, start = False, None
    for dt, val in rec.items():
        if val >= 0.5 and not in_rec:
            in_rec, start = True, dt
        elif val < 0.5 and in_rec:
            ax.axvspan(start, dt, color="grey", alpha=0.15, lw=0)
            in_rec = False
    if in_rec:
        ax.axvspan(start, rec.index[-1], color="grey", alpha=0.15, lw=0)


def fig_macro_series(macro):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, col, title, color in zip(
        axes, ["unemployment_rate", "gdp_growth", "Z"],
        ["Unemployment rate (%, quarterly avg)", "Real GDP growth (% SAAR)",
         "Systematic credit factor Z (+ = benign, − = stressed)"],
        [C_ACTUAL, C_ACTUAL, "#6A1B9A"]):
        shade_recessions(ax, macro)
        ax.plot(macro.index, macro[col], color=color, lw=1.3)
        ax.axhline(0, color="black", lw=0.6, alpha=0.4) if col in ("gdp_growth", "Z") else None
        ax.set_title(title, loc="left")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Macroeconomic indicators, 1990-2024 (shaded = NBER recessions)", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_macro_series.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_ts_forecast(series_name, title, ylabel):
    train = pd.read_csv(DATA_DIR.parent / "data" / f"macro_quarterly.csv", parse_dates=["date"]).set_index("date")[series_name]
    test_fc = pd.read_csv(DATA_DIR / f"{series_name}_test_forecast.csv", index_col=0, parse_dates=True).iloc[:, 0]
    test_ci = pd.read_csv(DATA_DIR / f"{series_name}_test_forecast_ci.csv", index_col=0, parse_dates=True)
    base_fc = pd.read_csv(DATA_DIR / f"{series_name}_baseline_forecast.csv", index_col=0, parse_dates=True).iloc[:, 0]
    base_ci = pd.read_csv(DATA_DIR / f"{series_name}_baseline_forecast_ci.csv", index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    recent = train.loc["2005-01-01":]
    ax.plot(recent.index, recent.values, color=C_ACTUAL, lw=1.3, label="Actual")
    ax.plot(test_fc.index, test_fc.values, color=C_FC, lw=1.6, ls="--", label="Out-of-sample forecast (holdout)")
    ax.fill_between(test_ci.index, test_ci["lower"], test_ci["upper"], color=C_FC, alpha=0.15)
    ax.plot(base_fc.index, base_fc.values, color="#2E7D32", lw=1.6, ls="--", label="Forward forecast (beyond 2024Q4)")
    ax.fill_between(base_ci.index, base_ci["lower"], base_ci["upper"], color="#2E7D32", alpha=0.15)
    ax.axvline(test_fc.index[0], color="grey", lw=0.8, ls=":")
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"02_forecast_{series_name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_transition_heatmaps():
    P_ttc = pd.read_csv(TAB_DIR / "P_ttc_input.csv", index_col=0)
    P_hat = pd.read_csv(TAB_DIR / "P_hat_full_sample.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, P, title in zip(axes, [P_ttc, P_hat],
                             ["Calibration input (TTC)", "Cohort-estimated (full sample)"]):
        im = ax.imshow(P.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(STATES)
        ax.set_yticks(range(3)); ax.set_yticklabels(STATES)
        ax.set_title(title, loc="left", fontsize=11)
        for i in range(3):
            for j in range(3):
                v = P.values[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=9)
        ax.grid(False)
    fig.suptitle("Quarterly transition matrices: calibration vs. cohort re-estimation", y=1.03)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03, label="Probability")
    fig.savefig(FIG_DIR / "03_transition_heatmaps.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_pd_term_structure():
    df = pd.read_csv(TAB_DIR / "multiperiod_pd_term_structure.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for rating, color in [("Good", C_GOOD), ("Watchlist", C_WATCH)]:
        sub = df[(df["matrix"] == "TTC (unconditional)") & (df["start_rating"] == rating)]
        ax.plot(sub["horizon_quarters"], sub["cum_default_prob"] * 100, "o-", color=color,
                label=f"Starting rating: {rating}")
    ax.set_xlabel("Horizon (quarters)")
    ax.set_ylabel("Cumulative default probability (%)")
    ax.set_title("Multi-period cumulative default probability term structure (TTC matrix)", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_pd_term_structure.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_macro_sensitivity():
    P_ttc = ttc_quarterly_matrix()
    zs = np.linspace(-3, 3, 61)
    good_d = [conditional_matrix(P_ttc, z)[GOOD, DEFAULT] for z in zs]
    watch_d = [conditional_matrix(P_ttc, z)[WATCH, DEFAULT] for z in zs]
    watch_down = [conditional_matrix(P_ttc, z)[GOOD, WATCH] for z in zs]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(zs, np.array(good_d) * 100, color=C_GOOD, lw=1.8, label="Good → Default")
    axes[0].plot(zs, np.array(watch_d) * 100, color=C_WATCH, lw=1.8, label="Watchlist → Default")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Systematic factor Z  (− = stressed, + = benign)")
    axes[0].set_ylabel("Quarterly default probability (%, log scale)")
    axes[0].set_title("Default-probability sensitivity to macro factor Z", loc="left", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].axvline(0, color="grey", lw=0.7, ls=":")

    axes[1].plot(zs, np.array(watch_down) * 100, color="#6A1B9A", lw=1.8)
    axes[1].set_xlabel("Systematic factor Z  (− = stressed, + = benign)")
    axes[1].set_ylabel("Quarterly downgrade probability (%)")
    axes[1].set_title("Good → Watchlist downgrade probability vs. Z", loc="left", fontsize=11)
    axes[1].axvline(0, color="grey", lw=0.7, ls=":")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_macro_sensitivity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_forward_scenarios():
    base = pd.read_csv(TAB_DIR / "forward_cum_default_baseline.csv", index_col=0, parse_dates=True)
    stress = pd.read_csv(TAB_DIR / "forward_cum_default_stress.csv", index_col=0, parse_dates=True)
    z = pd.read_csv(TAB_DIR / "forward_Z_scenarios.csv", index_col=0, parse_dates=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for rating, color in [("Good", C_GOOD), ("Watchlist", C_WATCH)]:
        axes[0].plot(base.index, base[rating] * 100, color=color, lw=1.8, ls="-",
                     label=f"{rating} – baseline")
        axes[0].plot(stress.index, stress[rating] * 100, color=color, lw=1.8, ls="--",
                     label=f"{rating} – stress")
    axes[0].set_ylabel("Cumulative default probability (%)")
    axes[0].set_title("Forward multi-period default probability:\nbaseline vs. stress macro scenario", loc="left", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-Q%q") if False else mdates.DateFormatter("%Y"))

    axes[1].plot(z.index, z["baseline_Z"], color=C_BASE, lw=1.8, label="Baseline")
    axes[1].plot(z.index, z["stress_Z"], color=C_STRESS, lw=1.8, ls="--", label="Stress")
    axes[1].axhline(0, color="grey", lw=0.7, ls=":")
    axes[1].set_title("Forecast systematic factor Z path", loc="left", fontsize=11)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_forward_scenarios.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_backtest():
    df = pd.read_csv(TAB_DIR / "backtest_multiperiod.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df["realised_cum_default_rate"] * 100, w, label="Realised", color="#455A64")
    ax.bar(x, df["unconditional_markov_pred"] * 100, w, label="Unconditional Markov (static)", color="#B0BEC5")
    ax.bar(x + w, df["macro_conditional_markov_pred"] * 100, w, label="Macro-conditional Markov", color=C_BASE)
    ax.set_xticks(x); ax.set_xticklabels(df["start_rating"])
    ax.set_ylabel("Cumulative default rate, 2016Q1-2024Q4 (%)")
    ax.set_title("Backtest: realised vs. predicted 36-quarter cumulative default rate", loc="left")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_backtest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_baseline_comparison():
    df = pd.read_csv(TAB_DIR / "baseline_comparison.csv")
    metrics = ["auc", "brier_score", "log_loss"]
    titles = ["AUC (higher better)", "Brier score (lower better)", "Log-loss (lower better)"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    colors = [C_BASE, "#8D6E63"]
    for ax, m, t in zip(axes, metrics, titles):
        ax.bar(df["model"], df[m], color=colors)
        ax.set_title(t, fontsize=10.5, loc="left")
        ax.set_xticklabels([lbl.replace(" (", "\n(") for lbl in df["model"]], fontsize=8)
        for i, v in enumerate(df[m]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("One-step-ahead default prediction: structural model vs. logistic-regression baseline\n(out-of-sample, 2016Q1-2024Q4)", y=1.08)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_baseline_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    macro = pd.read_csv(DATA_DIR / "macro_quarterly.csv", parse_dates=["date"]).set_index("date")
    fig_macro_series(macro)
    fig_ts_forecast("unemployment_rate", "Unemployment rate: actual, out-of-sample forecast and 2025-2026 forward forecast", "%")
    fig_ts_forecast("gdp_growth", "Real GDP growth: actual, out-of-sample forecast and 2025-2026 forward forecast", "% SAAR")
    fig_transition_heatmaps()
    fig_pd_term_structure()
    fig_macro_sensitivity()
    fig_forward_scenarios()
    fig_backtest()
    fig_baseline_comparison()
    print("All figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
