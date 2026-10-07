"""Figures for forecasting, residuals, uncertainty and inventory (Phase 35).  Every chart answers one analytical question; the
computed answer is stated in the subtitle.  Companion CSV tables with the plotted numbers are written by the pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats
from statsmodels.tsa.stattools import acf

from . import config as C
from . import viz
from .forecast_core import ForecastBook
from .viz import BLUE, ORANGE, AQUA, INK, INK2, MUTED, AXIS

FAMILY_LEGEND = [("machine learning", "XGBoost (ML)"), ("SARIMA", "SARIMA"), ("exponential smoothing", "Exponential smoothing"), ("baseline", "Baselines")]


def _family_handles():
    return [plt.Line2D([], [], marker="o", ls="", color=viz.FAMILY_COLOR[k], label=lbl, markersize=6) for k, lbl in FAMILY_LEGEND]


# ======================================================================================
# Forecast accuracy
# ======================================================================================
def fig_forecast_vs_actual(book: ForecastBook, rep: pd.DataFrame, horizons=(7, 28), models=("XGBoost", "SARIMA", "MovingAverage")) -> str:
    ctx = book.ctx
    fig, axes = plt.subplots(3, len(horizons), figsize=(6.2 * len(horizons), 8.0), squeeze=False)
    colors = {"XGBoost": BLUE, "SARIMA": ORANGE, "MovingAverage": MUTED, "SeasonalNaive": MUTED, "HoltWinters": AQUA}
    for r, row in enumerate(rep.itertuples()):
        for c, H in enumerate(horizons):
            ax = axes[r, c]
            t, y, _ = book.eval_arrays("XGBoost", "test", H)
            d = ctx.dates[t]
            ax.plot(d, y[row.series_idx], color=INK, lw=1.2, label="actual")
            for m in models:
                _, _, f = book.eval_arrays(m, "test", H)
                ax.plot(d, f[row.series_idx], color=colors[m], lw=1.1, alpha=0.95, label=m)
            ax.set_title(f"{row.series_id} ({row.tier}) - {H}-day demand")
            ax.set_ylabel(f"units over next {H} days")
            if r == 0 and c == 0:
                ax.legend(loc="upper left", ncol=2)
    viz.title_block(fig, "Do the forecasts track the demand actually realised in the TEST period?",
                    "Each point is the sum of demand over the next H days from a daily forecast origin; actual (black) vs XGBoost, SARIMA and the validation-selected moving average.")
    return viz.save(fig, "fc_01_forecast_vs_actual.png")


def fig_model_comparison(pooled_test: pd.DataFrame, pooled_val: pd.DataFrame, horizons=C.PRIMARY_HORIZONS) -> tuple[str, str]:
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.0 * len(horizons), 4.6), sharey=False)
    lines = []
    for ax, H in zip(np.atleast_1d(axes), horizons):
        g = pooled_test[pooled_test["H"] == H].sort_values("WAPE", ascending=True).reset_index(drop=True)
        v = pooled_val[pooled_val["H"] == H].set_index("model")["WAPE"]
        ypos = np.arange(len(g))[::-1]
        ax.hlines(ypos, 0, g["WAPE"], color=viz.GRID, lw=1.0)
        ax.scatter(g["WAPE"], ypos, s=46, c=[viz.model_color(m) for m in g["model"]], edgecolor=viz.SURFACE, linewidth=1.2, zorder=3)
        ax.scatter(v.reindex(g["model"]).to_numpy(), ypos, s=26, facecolor="none", edgecolor=INK2, linewidth=0.9, zorder=2)
        ax.set_yticks(ypos)
        ax.set_yticklabels(g["model"])
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.set_xlabel("WAPE (%) - lower is better")
        ax.set_title(f"{H}-day horizon")
        ax.set_xlim(0, max(g["WAPE"].max(), v.max()) * 1.12)
        lines.append(f"H={H}: {g.iloc[0]['model']} {g.iloc[0]['WAPE']:.1f}%")
    handles = _family_handles() + [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK2, label="validation WAPE", markersize=5)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    viz.title_block(fig, "Which forecasting model is most accurate at each horizon?",
                    "TEST-period pooled WAPE (filled) with validation WAPE (hollow). Best per horizon - " + "; ".join(lines) + ".")
    return viz.save(fig, "fc_02_model_comparison_wape.png", rect=(0, 0.06, 1, 0.88)), "; ".join(lines)


def fig_error_comparison(pooled_test: pd.DataFrame, horizons=C.PRIMARY_HORIZONS, ref: str = "MovingAverage") -> str:
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.05]})
    axes, cax = axs[:2], axs[2]
    models = [m for m in pooled_test["model"].unique()]
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        piv = pooled_test[pooled_test["H"].isin(horizons)].pivot(index="model", columns="H", values=metric).loc[models]
        skill = 1.0 - piv.div(piv.loc[ref], axis=1)                # > 0 : better than the reference
        lim = max(0.05, float(np.nanmax(np.abs(skill.to_numpy()))))
        im = ax.imshow(skill.to_numpy(), cmap=viz.DIV, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(skill.columns)))
        ax.set_xticklabels([f"{h}-day" for h in skill.columns])
        ax.set_yticks(range(len(skill.index)))
        ax.set_yticklabels(skill.index)
        ax.grid(False)
        for i in range(skill.shape[0]):
            for j in range(skill.shape[1]):
                ax.text(j, i, f"{100 * skill.iat[i, j]:+.0f}%", ha="center", va="center", fontsize=8, color=INK)
        ax.set_title(f"{metric} improvement vs {ref}")
    fig.colorbar(im, cax=cax, label="relative error reduction (+ = better than reference)")
    viz.title_block(fig, "How much does each model reduce error relative to the validation-selected moving average?",
                    "TEST-period pooled MAE and RMSE; positive blue-side values mean lower error than the reference, negative red-side values mean higher error.")
    return viz.save(fig, "fc_03_error_comparison_skill.png")


def fig_series_level_wape(by_series_test: pd.DataFrame, H: int = 7, models=("SeasonalNaive", "MovingAverage", "HoltWinters", "SARIMA", "XGBoost")) -> str:
    g = by_series_test[(by_series_test["H"] == H) & by_series_test["model"].isin(models)]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for k, m in enumerate(models):
        d = g[g["model"] == m]
        x = np.full(len(d), k) + np.random.default_rng(C.SEED).uniform(-0.12, 0.12, len(d))
        ax.scatter(x, d["WAPE"].clip(upper=300), s=26, color=viz.model_color(m), alpha=0.85, edgecolor=viz.SURFACE, linewidth=0.8)
        ax.hlines(d["WAPE"].median(), k - 0.25, k + 0.25, color=INK, lw=1.6)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel(f"WAPE (%) of {H}-day demand, per series (capped at 300)")
    med = g.groupby("model")["WAPE"].median().sort_values()
    viz.title_block(fig, f"How consistent is each model across the 18 series ({H}-day horizon)?",
                    "TEST period; each dot is one series, black bar = median across series. Lowest median WAPE: " + f"{med.index[0]} ({med.iloc[0]:.0f}%).")
    return viz.save(fig, f"fc_04_series_level_wape_H{H}.png")


# ======================================================================================
# Residual diagnostics
# ======================================================================================
def fig_residuals_stat(arrays: dict, rep: pd.DataFrame) -> str:
    row = rep[rep["tier"] == "high"].iloc[0]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    for r, (model, col) in enumerate((("SARIMA", ORANGE), ("HoltWinters", AQUA))):
        res = np.asarray(arrays[row.series_id].get(model, []), float)
        if len(res) == 0:
            continue
        axes[r, 0].plot(res, color=col, lw=0.6)
        axes[r, 0].axhline(0, color=AXIS, lw=0.8)
        axes[r, 0].set_title(f"{model}: one-step residuals (TRAIN)")
        axes[r, 1].hist(res, bins=40, density=True, color=col, alpha=0.8)
        xs = np.linspace(res.min(), res.max(), 200)
        axes[r, 1].plot(xs, stats.norm.pdf(xs, res.mean(), res.std()), color=INK, lw=1.0)
        axes[r, 1].set_title(f"distribution (skew {stats.skew(res):.2f}, ex.kurt {stats.kurtosis(res):.1f})")
        (osm, osr), (slope, icpt, _) = stats.probplot(res, dist="norm")
        axes[r, 2].scatter(osm, osr, s=5, color=col, alpha=0.6)
        axes[r, 2].plot(osm, slope * osm + icpt, color=INK, lw=1.0)
        axes[r, 2].set_title("normal Q-Q plot")
        a = acf(res, nlags=28, fft=True)
        axes[r, 3].vlines(range(1, 29), 0, a[1:], color=col, lw=1.4)
        axes[r, 3].axhline(1.96 / np.sqrt(len(res)), color=AXIS, lw=0.8)
        axes[r, 3].axhline(-1.96 / np.sqrt(len(res)), color=AXIS, lw=0.8)
        axes[r, 3].set_title(f"residual ACF (mean resid {res.mean():.2f})")
        axes[r, 3].set_xlabel("lag (days)")
    viz.title_block(fig, "Do the statistical models leave structure in their residuals?",
                    f"One-step in-sample residuals on TRAIN for {row.series_id} (representative high-demand series): mean, variance stability, autocorrelation and distribution. "
                    "Summary over all series in outputs/tables/residuals_statistical_models.csv.")
    return viz.save(fig, "resid_01_statistical_models.png")


def fig_residuals_xgb(res: dict, H: int) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    lv = res["by_level"]
    axes[0, 0].bar(range(len(lv)), lv["bias"], color=BLUE, width=0.6)
    axes[0, 0].axhline(0, color=AXIS, lw=0.8)
    axes[0, 0].set_xticks(range(len(lv)))
    axes[0, 0].set_xticklabels([f"{m:.0f}" for m in lv["mean_actual"]])
    axes[0, 0].set_xlabel("mean actual demand of the window bin (units)")
    axes[0, 0].set_ylabel("mean error (forecast - actual)")
    axes[0, 0].set_title("Bias by demand level (bins of the actual sum)")
    bt = res["by_tier"]
    axes[0, 1].bar(bt["tier"], bt["WAPE"], color=[viz.TIER_COLOR[t] for t in bt["tier"]], width=0.55)
    axes[0, 1].set_ylabel("WAPE (%)")
    axes[0, 1].set_title("Error by demand tier")
    bm = res["by_month"]
    axes[1, 0].plot(bm["month"], bm["bias"], color=BLUE, marker="o", ms=3)
    axes[1, 0].axhline(0, color=AXIS, lw=0.8)
    axes[1, 0].set_xticks(range(0, len(bm), 2))
    axes[1, 0].set_xticklabels(bm["month"].iloc[::2], rotation=45, ha="right", fontsize=7.5)
    axes[1, 0].set_ylabel("mean error by origin month")
    axes[1, 0].set_title("Temporal pattern of the bias")
    bw = res["by_weekday"]
    axes[1, 1].bar(bw["weekday"].str[:3], bw["bias"], color=BLUE, width=0.6)
    axes[1, 1].axhline(0, color=AXIS, lw=0.8)
    axes[1, 1].set_ylabel("mean error by origin weekday")
    axes[1, 1].set_title("Bias by weekday of the forecast origin")
    o = res["overall"]
    viz.title_block(fig, f"Is the {H}-day XGBoost forecast biased for high or low demand, or at certain times?",
                    f"TEST period. Overall bias {o['bias(mean f-y)']:+.2f} units ({o['bias_pct_of_mean_actual']:+.1f}% of mean actual {o['mean_actual']:.1f}); positive = over-forecast.")
    return viz.save(fig, f"resid_02_xgboost_H{H}.png")


# ======================================================================================
# Uncertainty
# ======================================================================================
def fig_prediction_interval(arr_by_series: dict, rep: pd.DataFrame, ctx, H: int, model: str, method: str, level: float) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.6), sharex=True)
    for ax, row in zip(axes, rep.itertuples()):
        t, y, f, lo, hi = arr_by_series
        d = ctx.dates[t]
        s = row.series_idx
        ax.fill_between(d, lo[s], hi[s], color=BLUE, alpha=0.18, lw=0, label=f"{int(level * 100)}% interval ({method})")
        ax.plot(d, f[s], color=BLUE, lw=1.3, label=f"{model} forecast")
        ax.plot(d, y[s], color=INK, lw=1.0, label="actual")
        cov = float(np.mean((y[s] >= lo[s]) & (y[s] <= hi[s])))
        ax.set_title(f"{row.series_id} ({row.tier}): observed coverage {100 * cov:.0f}%")
        ax.set_ylabel(f"units over {H} days")
    axes[0].legend(loc="upper left", ncol=3)
    viz.title_block(fig, f"Does the {int(level * 100)}% prediction interval for the {H}-day sum contain the realised demand?",
                    f"TEST period, {model}; interval = forecast + empirical quantiles of VALIDATION residuals ({method}), lower bound clipped at 0.")
    return viz.save(fig, f"unc_01_prediction_interval_H{H}.png")


def fig_coverage(cov: pd.DataFrame, models=("XGBoost", "SARIMA", "MovingAverage"), horizons=C.PRIMARY_HORIZONS) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, level in zip(axes, C.INTERVAL_LEVELS):
        g = cov[(cov["level"] == level) & cov["model"].isin(models) & cov["H"].isin(horizons)]
        labels = []
        for k, (m, H) in enumerate([(m, H) for H in horizons for m in models]):
            for off, meth, col in ((-0.17, "gaussian", MUTED), (0.17, "empirical", BLUE)):
                d = g[(g["model"] == m) & (g["H"] == H) & (g["method"] == meth)]
                if len(d):
                    ax.bar(k + off, 100 * d["coverage"].iloc[0], width=0.3, color=col, edgecolor=viz.SURFACE, linewidth=1.0)
            labels.append(f"{m}\n{H}d")
        ax.axhline(100 * level, color=INK, lw=1.0)
        ax.text(len(labels) - 0.45, 100 * level + 1, f"nominal {int(level * 100)}%", ha="right", fontsize=8, color=INK2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0, 100)
        ax.set_title(f"{int(level * 100)}% intervals")
    axes[0].set_ylabel("observed TEST coverage (%)")
    axes[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MUTED, label="Gaussian (z x sigma)"), plt.Rectangle((0, 0), 1, 1, color=BLUE, label="empirical quantiles")], loc="lower right")
    viz.title_block(fig, "Are the prediction intervals calibrated out of sample?",
                    "Share of TEST-period actuals inside the interval vs the nominal level; sigma and quantiles estimated from VALIDATION residuals only.")
    return viz.save(fig, "unc_02_coverage_calibration.png")


def fig_qq_standardised(book: ForecastBook, model: str = "XGBoost", H: int = 7) -> str:
    from .uncertainty import val_residuals
    e = val_residuals(book, model, H)
    z = (e / np.sqrt(np.mean(e ** 2, axis=1, keepdims=True))).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(z, bins=60, density=True, color=BLUE, alpha=0.8)
    xs = np.linspace(z.min(), z.max(), 300)
    axes[0].plot(xs, stats.norm.pdf(xs), color=INK, lw=1.0)
    axes[0].set_title("standardised validation residuals vs N(0,1)")
    (osm, osr), (slope, icpt, _) = stats.probplot(z, dist="norm")
    axes[1].scatter(osm, osr, s=5, color=BLUE, alpha=0.5)
    axes[1].plot(osm, slope * osm + icpt, color=INK, lw=1.0)
    axes[1].set_title("normal Q-Q plot")
    viz.title_block(fig, f"Is the Gaussian assumption reasonable for {model} {H}-day forecast errors?",
                    f"Pooled standardised VALIDATION residuals: skewness {stats.skew(z):.2f}, excess kurtosis {stats.kurtosis(z):.2f} (0 and 0 for a normal distribution).")
    return viz.save(fig, f"unc_03_residual_normality_{model}_H{H}.png")


# ======================================================================================
# Inventory
# ======================================================================================
def fig_inventory_trajectory(ledgers: dict, ctx, series_idx: int, policy_a: str, policy_b: str, L: int, SL: float, method: str = "gaussian") -> str:
    dates = ctx.dates[ctx.split.val_end:]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True, sharey=True)
    for ax, (label, pol, col) in zip(axes, (("Policy A - historical baseline", policy_a, MUTED), (f"Policy B - {policy_b} forecasts", policy_b, BLUE))):
        led = ledgers[(pol, L, SL, "gaussian", series_idx)]
        ax.fill_between(dates, led.ending_inventory, color=col, alpha=0.18, lw=0)
        ax.plot(dates, led.ending_inventory, color=col, lw=1.2, label="ending inventory")
        ax.step(dates, led.rop, where="post", color=INK, lw=0.9, label="reorder point")
        so = led.stockout > 0
        ax.scatter(dates[so], np.zeros(so.sum()), marker="v", s=22, color=ORANGE, label="stockout day", zorder=4)
        ordd = led.orders_placed > 0
        ax.scatter(dates[ordd], led.ending_inventory[ordd], marker="o", s=14, color=INK, label="order placed", zorder=4)
        ax.set_title(f"{label}: {int(so.sum())} stockout days, {int(ordd.sum())} orders, avg inventory {led.ending_inventory.mean():.1f}")
        ax.set_ylabel("units")
    axes[0].legend(loc="upper right", ncol=4)
    viz.title_block(fig, "How does inventory evolve under the two policies?",
                    f"{ctx.ids[series_idx]}, lead time {L} days, target service level {int(SL * 100)}%, TEST period; orders arrive L+1 days after the end-of-day order.")
    return viz.save(fig, f"inv_01_trajectory_{ctx.ids[series_idx]}.png")


def fig_stockout_timeline(ledgers: dict, ctx, policy_a: str, policy_b: str, L: int, SL: float) -> str:
    dates = ctx.dates[ctx.split.val_end:]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    tot = {}
    for ax, (label, pol) in zip(axes, (("Policy A - historical baseline", policy_a), (f"Policy B - {policy_b} forecasts", policy_b))):
        M = np.vstack([(ledgers[(pol, L, SL, "gaussian", s)].stockout > 0) for s in range(ctx.n_series)]).astype(float)
        tot[pol] = M.sum()
        ax.imshow(M, aspect="auto", cmap=ListedColormap([viz.SURFACE, ORANGE]), interpolation="nearest",
                  extent=[0, len(dates), ctx.n_series, 0])
        ax.set_yticks(np.arange(ctx.n_series) + 0.5)
        ax.set_yticklabels(ctx.ids, fontsize=6.5)
        ax.grid(False)
        ax.set_title(f"{label}: {int(M.sum())} stockout series-days")
    xt = np.linspace(0, len(dates) - 1, 8).astype(int)
    axes[1].set_xticks(xt)
    axes[1].set_xticklabels([str(dates[i].date()) for i in xt], rotation=30, ha="right")
    viz.title_block(fig, "When and where do stockouts occur?",
                    f"Orange = day with unmet demand; lead time {L} days, service level {int(SL * 100)}%, TEST period. Total stockout series-days: A {int(tot[policy_a])} vs B {int(tot[policy_b])}.")
    return viz.save(fig, "inv_02_stockout_periods.png")


def fig_service_vs_cost(sens: pd.DataFrame, scenario: str = "MEDIUM") -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=False)
    for ax, L in zip(axes, C.LEAD_TIMES):
        for grp, col, lab in (("A_historical", MUTED, "A: historical"), ("B_forecast_driven", BLUE, "B: forecast-driven")):
            g = sens[(sens["policy_group"] == grp) & (sens["L"] == L) & (sens["scenario"] == scenario)].sort_values("SL")
            ax.plot(100 * g["SL"], g["total_cost"], color=col, marker="o", ms=6, mec=viz.SURFACE, mew=1.2, label=lab)
            for _, r in g.iterrows():
                ax.annotate(f"{r['total_cost']:,.0f}", (100 * r["SL"], r["total_cost"]), textcoords="offset points", xytext=(0, 7 if grp == "B_forecast_driven" else -13),
                            ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks([100 * s for s in C.SERVICE_LEVELS])
        ax.set_xlabel("target service level (%)")
        ax.set_title(f"lead time {L} days")
        ax.margins(y=0.18)
    axes[0].set_ylabel(f"total cost, {scenario} stockout penalty (USD, 18 series)")
    axes[0].legend(loc="best")
    viz.title_block(fig, "What does a higher service level cost, and does forecasting change the trade-off?",
                    "Pooled total cost (holding + ordering + stockout, HYPOTHETICAL costs) over the TEST period vs target service level, by lead time.")
    return viz.save(fig, "inv_03_service_level_vs_cost.png")


def fig_cost_vs_fill_rate(sens: pd.DataFrame, scenario: str = "MEDIUM") -> str:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for grp, col, lab in (("A_historical", MUTED, "A: historical"), ("B_forecast_driven", BLUE, "B: forecast-driven")):
        for L, ls in zip(C.LEAD_TIMES, ("-", "-", "-")):
            g = sens[(sens["policy_group"] == grp) & (sens["L"] == L) & (sens["scenario"] == scenario)].sort_values("SL")
            ax.plot(100 * g["fill_rate"], g["total_cost"], color=col, marker="o", ms=5, mec=viz.SURFACE, mew=1.0, lw=1.2, alpha=0.9)
            viz.direct_label(ax, 100 * g["fill_rate"].iloc[-1], g["total_cost"].iloc[-1], f"L={L}", color=INK2, dx=5, fontsize=7.5)
    ax.set_xlabel("achieved fill rate (% of demanded units served)")
    ax.set_ylabel(f"total cost, {scenario} penalty (USD)")
    ax.legend(handles=[plt.Line2D([], [], color=MUTED, marker="o", label="A: historical"), plt.Line2D([], [], color=BLUE, marker="o", label="B: forecast-driven")], loc="best")
    ax.grid(True, axis="both")
    viz.title_block(fig, "Which policy delivers a given fill rate at the lowest cost?",
                    "Each line connects target service levels 90/95/98% for one lead time; points further left/lower are cheaper for the same fill rate.")
    return viz.save(fig, "inv_04_cost_vs_fill_rate.png")


def fig_total_cost_components(runs: pd.DataFrame, L: int = 7, SL: float = 0.95, scenario: str = "MEDIUM", method: str = "gaussian") -> str:
    sub = runs[(runs["L"] == L) & (runs["SL"] == SL) & ((runs["method"] == method))]
    g = sub.groupby("policy").agg(holding=("holding_cost", "sum"), ordering=("ordering_cost", "sum"), stockout=(f"stockout_cost_{scenario}", "sum")).reset_index()
    g["total"] = g[["holding", "ordering", "stockout"]].sum(axis=1)
    g = g.sort_values("total").reset_index(drop=True)
    labels = ["A: historical" if p == "A_historical" else f"B: {p}" for p in g["policy"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    bottom = np.zeros(len(g))
    for comp, col in (("holding", BLUE), ("ordering", ORANGE), ("stockout", AQUA)):
        ax.bar(range(len(g)), g[comp], bottom=bottom, color=col, width=0.62, edgecolor=viz.SURFACE, linewidth=1.4, label=f"{comp} cost")
        bottom += g[comp].to_numpy()
    for i, t in enumerate(g["total"]):
        ax.text(i, t, f"{t:,.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("total cost over TEST period (USD, 18 series)")
    ax.legend(loc="upper left")
    best = g.iloc[0]
    viz.title_block(fig, "Which policy has the lowest total inventory cost, and where does the cost come from?",
                    f"Lead time {L} days, target service level {int(SL * 100)}%, {scenario} stockout penalty (HYPOTHETICAL costs). Cheapest: {labels[0]} ({best['total']:,.0f}).")
    return viz.save(fig, "inv_05_total_cost_components.png")


def fig_sensitivity_heatmaps(sens: pd.DataFrame) -> str:
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 0.045]})
    axes, cax = axs[:3], axs[3]
    vals = {}
    for scn in C.SCENARIOS:
        a = sens[(sens["policy_group"] == "A_historical") & (sens["scenario"] == scn)].pivot(index="L", columns="SL", values="total_cost")
        b = sens[(sens["policy_group"] == "B_forecast_driven") & (sens["scenario"] == scn)].pivot(index="L", columns="SL", values="total_cost")
        vals[scn] = 100 * (b / a - 1)
    lim = max(5.0, max(float(np.nanmax(np.abs(v.to_numpy()))) for v in vals.values()))
    for ax, scn in zip(axes, C.SCENARIOS):
        v = vals[scn]
        im = ax.imshow(v.to_numpy(), cmap=viz.DIV, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{int(100 * s)}%" for s in v.columns])
        ax.set_yticks(range(3)); ax.set_yticklabels([f"{l} d" for l in v.index])
        ax.set_xlabel("target service level"); ax.set_title(f"{scn} stockout penalty")
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{v.iat[i, j]:+.1f}%", ha="center", va="center", fontsize=9, color=INK)
    axes[0].set_ylabel("lead time")
    fig.colorbar(im, cax=cax, label="total cost of B vs A (%; blue = B cheaper)")
    viz.title_block(fig, "Is the forecast-driven policy's cost advantage robust to lead time, service level and penalty cost?",
                    "Relative change in pooled total cost, Policy B (validation-selected forecasting model per lead time) vs Policy A, TEST period, HYPOTHETICAL costs.")
    return viz.save(fig, "inv_06_sensitivity_heatmaps.png")


def fig_accuracy_vs_cost(avc: pd.DataFrame, summ: pd.DataFrame, SL: float = 0.95, scenario: str = "MEDIUM") -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    notes = []
    offsets = [(6, 6), (6, -11), (-6, 7), (-6, -12), (8, 0)]
    for ax, L in zip(axes, C.LEAD_TIMES):
        g = avc[(avc["L"] == L) & (avc["SL"] == SL) & (avc["scenario"] == scenario)].copy()
        main = g[g["model"] != "Naive"].sort_values("test_WAPE").reset_index(drop=True)
        ax.scatter(main["test_WAPE"], main["total_cost"], s=60, c=[viz.model_color(m) for m in main["model"]], edgecolor=viz.SURFACE, linewidth=1.2, zorder=3)
        for k, r in main.iterrows():                                   # staggered label offsets avoid collisions
            dx, dy = offsets[k % len(offsets)]
            ax.annotate(r["model"], (r["test_WAPE"], r["total_cost"]), textcoords="offset points", xytext=(dx, dy), fontsize=7.5, color=INK2,
                        ha="left" if dx > 0 else "right")
        lo, hi = main["total_cost"].min(), main["total_cost"].max()
        pad = max((hi - lo) * 0.25, 0.02 * hi)
        ax.set_ylim(lo - pad, hi + pad)
        nv = g[g["model"] == "Naive"]
        if len(nv):
            ax.text(0.98, 0.97, f"Naive off-scale: WAPE {nv['test_WAPE'].iloc[0]:.0f}%, cost {nv['total_cost'].iloc[0]:,.0f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color=MUTED)
        row = summ[(summ["L"] == L) & (summ["SL"] == SL) & (summ["scenario"] == scenario)].iloc[0]
        ax.set_title(f"lead time {L} d: rho = {row['spearman_testWAPE_vs_cost']:+.2f} (excl. Naive {row['spearman_testWAPE_vs_cost_excl_Naive']:+.2f})")
        ax.set_xlabel(f"TEST WAPE of the {L}-day forecast (%)")
        ax.grid(True, axis="both")
        notes.append(f"L={L}: {row['spearman_testWAPE_vs_cost']:+.2f} / {row['spearman_testWAPE_vs_cost_excl_Naive']:+.2f}")
    axes[0].set_ylabel(f"total inventory cost ({scenario} penalty, USD)")
    fig.legend(handles=_family_handles(), loc="lower center", ncol=4, frameon=False)
    viz.title_block(fig, "Does the model with the lowest forecast error also give the lowest inventory cost?",
                    f"One dot per forecasting model; service level {int(SL * 100)}%, {scenario} penalty. Spearman rank correlation between forecast error and cost "
                    "(with / without the degenerate Naive model): " + "; ".join(notes) + ". A correlation near 1 would mean accuracy ranking = cost ranking.")
    return viz.save(fig, "inv_07_accuracy_vs_cost.png", rect=(0, 0.07, 1, 0.86))


def fig_robustness(forecast_by_group: pd.DataFrame, inv_by_group: pd.DataFrame, H: int = 7, L: int = 7) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    groups = ["tier=high", "tier=medium", "tier=low", "regular", "intermittent"]
    models = ["SeasonalNaive", "MovingAverage", "SARIMA", "XGBoost"]
    width = 0.8 / len(models)
    ax = axes[0]
    for k, m in enumerate(models):
        vals = [float(forecast_by_group[(forecast_by_group["group"] == g) & (forecast_by_group["model"] == m) & (forecast_by_group["H"] == H)]["WAPE"].iloc[0]) for g in groups]
        shade = {"SeasonalNaive": MUTED, "MovingAverage": AXIS}.get(m, viz.model_color(m))      # two distinguishable greys for the two baselines
        ax.bar(np.arange(len(groups)) + k * width, vals, width=width * 0.92, color=shade, edgecolor=viz.SURFACE, linewidth=1.0, label=m)
    ax.set_xticks(np.arange(len(groups)) + 0.4 - width / 2)
    n = {g: int(forecast_by_group[forecast_by_group["group"] == g]["n_series"].iloc[0]) for g in groups}
    ax.set_xticklabels([f"{g}\n(n={n[g]})" for g in groups], fontsize=8)
    ax.set_ylabel("WAPE (%)")
    ax.set_title(f"{H}-day forecast error by group (TEST)")
    ax.legend(loc="upper left", fontsize=7.5)
    ax = axes[1]
    d = inv_by_group.copy()
    d["label"] = d["group"] + "\n(n=" + d["n_series"].astype(str) + ")"
    ax.bar(range(len(d)), d["B_vs_A_total_cost_pct"], color=[BLUE if v < 0 else ORANGE for v in d["B_vs_A_total_cost_pct"]], width=0.55)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["label"], fontsize=8)
    ax.set_ylabel("total cost of B vs A (%)")
    ax.set_title(f"Inventory: B ({d['B_model'].iloc[0]}) vs A, L={L}, 95%, MEDIUM")
    viz.title_block(fig, "Do the conclusions hold across demand tiers and for intermittent vs regular demand?",
                    "Subgroup results with 4-14 series each: descriptive, not statistically powered. Blue bar = forecast-driven policy cheaper, orange = more expensive.")
    return viz.save(fig, "robust_01_tiers_and_regularity.png")
