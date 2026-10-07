"""Phase 7 - exploratory data analysis (TRAIN period only for all statistics used to draw conclusions).

Representative series: for each demand tier the series whose TRAIN mean is closest to the tier's median (a rule, not a choice).
All numbers in reports/eda.md are computed here; wording is descriptive - correlation / pattern, never causation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

from . import config as C
from . import stat_models as sm
from . import viz
from .features import trailing_stat
from .forecast_core import Context

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def representative_series(ctx: Context) -> pd.DataFrame:
    sel = ctx.sel
    rows = []
    for tier in C.TIER_ORDER:
        g = sel[sel["tier"] == tier]
        med = g["train_mean"].median()
        k = (g["train_mean"] - med).abs().idxmin()
        rows.append({"tier": tier, "series_idx": int(k), "series_id": sel.loc[k, "series_id"], "train_mean": float(sel.loc[k, "train_mean"]),
                     "sb_class": sel.loc[k, "train_sb_class"]})
    return pd.DataFrame(rows)


def series_frame(ctx: Context, i: int) -> pd.DataFrame:
    return pd.DataFrame({"date": ctx.dates, "y": ctx.Y[i], "price": ctx.price[i]}).set_index("date")


# --------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------
def summary_stats(ctx: Context) -> pd.DataFrame:
    a = ctx.split.train_end
    strat = sm.stationarity_table(ctx)[["series_id", "seasonal_strength_Fs", "acf_lag7", "adf_p_level", "kpss_p_level", "level_conclusion"]]
    rows = []
    for i, sid in enumerate(ctx.ids):
        y = pd.Series(ctx.Y[i, :a], index=ctx.dates[:a])
        px = ctx.price[i, :a]
        wk = y.resample("W").sum()
        t = np.arange(len(wk))
        lr = stats.linregress(t, wk.to_numpy())
        dow_idx = y.groupby(y.index.dayofweek).mean() / y.mean()
        nz = y[y > 0]
        fence = nz.quantile(0.75) + 3 * (nz.quantile(0.75) - nz.quantile(0.25)) if len(nz) else np.nan
        yv = y.to_numpy(float)
        d7 = yv[7:] - yv[:-7]                                        # seasonal difference (1 - B^7) y
        a_d, p_d = acf(d7, nlags=28, fft=True), pacf(d7, nlags=28, method="ywm")
        m28, s28 = trailing_stat(yv, 28, "mean"), trailing_stat(yv, 28, "std")
        ok = np.isfinite(m28) & np.isfinite(s28)
        rows.append({
            "series_id": sid, "tier": ctx.sel.loc[i, "tier"], "cat_id": ctx.sel.loc[i, "cat_id"], "store_id": ctx.sel.loc[i, "store_id"],
            "mean": y.mean(), "median": y.median(), "std": y.std(), "cv": y.std() / y.mean(), "min": y.min(), "max": y.max(),
            "zero_pct": 100 * (y == 0).mean(), "adi": ctx.sel.loc[i, "train_adi"], "cv2": ctx.sel.loc[i, "train_cv2"], "sb_class": ctx.sel.loc[i, "train_sb_class"],
            "trend_pct_of_mean_per_year": 100 * lr.slope * 52.18 / wk.mean() if wk.mean() > 0 else np.nan, "trend_p_value": lr.pvalue,
            "dow_peak_day": DOW[int(dow_idx.idxmax())], "dow_peak_index": float(dow_idx.max()), "dow_trough_index": float(dow_idx.min()),
            "n_price_changes": int((np.diff(px) != 0).sum()), "price_cv": float(np.std(px) / np.mean(px)),
            "price_min": float(px.min()), "price_max": float(px.max()),
            "n_outliers_tukey3": int((y > fence).sum()) if np.isfinite(fence) else 0,
            "rolling28_mean_range_ratio": float(y.rolling(28).mean().max() / max(y.rolling(28).mean().min(), 1e-9)),
            "d7_acf_lag7": float(a_d[7]), "d7_pacf_lag7": float(p_d[7]), "d7_pacf_lag14": float(p_d[14]),
            "d7_pacf_lag21": float(p_d[21]), "d7_pacf_lag28": float(p_d[28]),
            "corr_roll28_mean_std": float(np.corrcoef(m28[ok], s28[ok])[0, 1]) if ok.sum() > 2 and np.std(s28[ok]) > 0 else np.nan,
        })
    df = pd.DataFrame(rows).merge(strat, on="series_id")
    return df


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig_daily_and_rolling(ctx: Context, rep: pd.DataFrame) -> str:
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.2), gridspec_kw={"width_ratios": [2.3, 1.3]})
    for r, row in enumerate(rep.itertuples()):
        f = series_frame(ctx, row.series_idx)
        col = viz.TIER_COLOR[row.tier]
        ax = axes[r, 0]
        ax.plot(f.index, f["y"], color=viz.AXIS, lw=0.5)
        ax.plot(f.index, f["y"].rolling(28).mean(), color=col, lw=1.6)
        viz.shade_periods(ax, ctx.split, label=(r == 0))
        ax.set_title(f"{row.series_id} ({row.tier} demand, {row.sb_class})")
        ax.set_ylabel("units / day")
        if r == 0:
            ax.legend(handles=[plt.Line2D([], [], color=viz.AXIS, lw=1), plt.Line2D([], [], color=col, lw=1.6)],
                      labels=["daily sales", "28-day rolling mean"], loc="upper left")
        ax2 = axes[r, 1]
        ax2.plot(f.index, f["y"].rolling(28).std(), color=col, lw=1.4)
        viz.shade_periods(ax2, ctx.split)
        ax2.set_title("28-day rolling standard deviation")
        ax2.set_ylabel("units / day")
    viz.title_block(fig, "Is demand trending, volatile or seasonal over time?",
                    "Daily sales (grey), 28-day rolling mean (colour) and rolling standard deviation; grey bands mark the validation and test periods.")
    return viz.save(fig, "eda_01_daily_rolling.png")


def fig_weekly_monthly(ctx: Context, rep: pd.DataFrame) -> str:
    fig, axes = plt.subplots(3, 2, figsize=(12, 7.8))
    for r, row in enumerate(rep.itertuples()):
        f = series_frame(ctx, row.series_idx)
        col = viz.TIER_COLOR[row.tier]
        w = f["y"].resample("W").sum().iloc[1:-1]
        m = f["y"].resample("MS").sum().iloc[:-1]
        axes[r, 0].plot(w.index, w, color=col, lw=1.0)
        axes[r, 0].set_title(f"{row.series_id} - weekly demand")
        axes[r, 0].set_ylabel("units / week")
        axes[r, 1].plot(m.index, m, color=col, lw=1.4, marker="o", ms=3)
        axes[r, 1].set_title("monthly demand")
        axes[r, 1].set_ylabel("units / month")
        for ax in axes[r]:
            viz.shade_periods(ax, ctx.split)
    viz.title_block(fig, "How does demand look when aggregated to weeks and months?", "Aggregation smooths the daily noise; level shifts and seasonal swings become visible.")
    return viz.save(fig, "eda_02_weekly_monthly.png")


def fig_distribution(ctx: Context, rep: pd.DataFrame) -> str:
    a = ctx.split.train_end
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, row in zip(axes, rep.itertuples()):
        y = ctx.Y[row.series_idx, :a].astype(int)
        hi = int(np.quantile(y, 0.99)) + 1
        counts = np.bincount(np.clip(y, 0, hi), minlength=hi + 1) / len(y)
        ax.bar(np.arange(hi + 1), counts, color=viz.TIER_COLOR[row.tier], width=0.75)
        ax.set_title(f"{row.series_id}")
        ax.set_xlabel("units sold in a day (last bar = capped at P99)")
        ax.set_ylabel("share of days")
        ax.text(0.97, 0.93, f"zero days: {100 * (y == 0).mean():.0f}%\nmean {y.mean():.2f}, sd {y.std():.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=viz.INK2)
    viz.title_block(fig, "What does the daily demand distribution look like?", "Right-skewed counts with many zeros - a Gaussian error model is a simplification, tested later on the forecast residuals.")
    return viz.save(fig, "eda_03_distribution.png")


def fig_dow_month(ctx: Context, rep: pd.DataFrame) -> str:
    a = ctx.split.train_end
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for row in rep.itertuples():
        y = pd.Series(ctx.Y[row.series_idx, :a], index=ctx.dates[:a])
        col = viz.TIER_COLOR[row.tier]
        d = y.groupby(y.index.dayofweek).mean() / y.mean()
        m = y.groupby(y.index.month).mean() / y.mean()
        axes[0].plot(DOW, d.to_numpy(), color=col, marker="o", ms=4, label=f"{row.tier}: {row.series_id}")
        axes[1].plot(m.index, m.to_numpy(), color=col, marker="o", ms=4)
        viz.direct_label(axes[0], 6, d.iloc[-1], row.tier, dx=6)
        viz.direct_label(axes[1], 12, m.iloc[-1], row.tier, dx=6)
    for ax in axes:
        ax.axhline(1.0, color=viz.AXIS, lw=0.8)
        ax.set_ylabel("index (series mean = 1)")
    axes[0].set_title("Day-of-week pattern (TRAIN)")
    axes[1].set_title("Monthly seasonality (TRAIN)")
    axes[1].set_xticks(range(1, 13))
    axes[0].legend(loc="upper left")
    viz.title_block(fig, "Is there a weekly or yearly seasonal pattern?", "Mean sales by weekday and by month relative to each series' own mean (descriptive, TRAIN period).")
    return viz.save(fig, "eda_04_dow_monthly_seasonality.png")


def fig_acf_pacf(ctx: Context, rep: pd.DataFrame) -> str:
    a = ctx.split.train_end
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for c, row in enumerate(rep.itertuples()):
        y = ctx.Y[row.series_idx, :a].astype(float)
        col = viz.TIER_COLOR[row.tier]
        plot_acf(y, lags=42, ax=axes[0, c], color=col, vlines_kwargs={"colors": col, "linewidth": 1.0}, title="")
        plot_pacf(y, lags=28, ax=axes[1, c], method="ywm", color=col, vlines_kwargs={"colors": col, "linewidth": 1.0}, title="")
        axes[0, c].set_title(f"ACF - {row.series_id}")
        axes[1, c].set_title("PACF")
        for ax in axes[:, c]:
            ax.set_xlabel("lag (days)")
            for line in ax.lines:
                if line.get_marker() == "o":
                    line.set_markersize(3)
    a7 = {row.tier: acf(ctx.Y[row.series_idx, :a].astype(float), nlags=7, fft=True)[7] for row in rep.itertuples()}
    viz.title_block(fig, "How strongly is demand correlated with its own past (lags 7, 14, 21, ...)?",
                    "Autocorrelation at lag 7 (representative series): " + ", ".join(f"{k} {v:.2f}" for k, v in a7.items()) +
                    ". Positive correlation at multiples of 7 would indicate weekly seasonality (s = 7); shaded band = 95% bounds under no autocorrelation.")
    return viz.save(fig, "eda_05_acf_pacf.png")


def fig_acf_pacf_differenced(ctx: Context, rep: pd.DataFrame) -> str:
    """ACF/PACF of the seasonally differenced (lag 7) series: evidence for the seasonal MA / AR terms of the SARIMA candidates."""
    a = ctx.split.train_end
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for c, row in enumerate(rep.itertuples()):
        y = ctx.Y[row.series_idx, :a].astype(float)
        d7 = y[7:] - y[:-7]
        col = viz.TIER_COLOR[row.tier]
        plot_acf(d7, lags=35, ax=axes[0, c], color=col, vlines_kwargs={"colors": col, "linewidth": 1.0}, title="")
        plot_pacf(d7, lags=28, ax=axes[1, c], method="ywm", color=col, vlines_kwargs={"colors": col, "linewidth": 1.0}, title="")
        a7, a14 = acf(d7, nlags=14, fft=True)[[7, 14]]
        axes[0, c].set_title(f"ACF of (1-B^7)y - {row.series_id}\nlag 7: {a7:+.2f}, lag 14: {a14:+.2f}")
        axes[1, c].set_title("PACF of (1-B^7)y")
        for ax in axes[:, c]:
            ax.set_xlabel("lag (days)")
    viz.title_block(fig, "What does the seasonally differenced series suggest for the SARIMA structure?",
                    "A lag-7 autocorrelation near -0.5 after seasonal differencing is the signature of an MA(1) at lag 7. It arises either from a genuine seasonal-MA structure or from "
                    "over-differencing a series whose weekly pattern is weak or deterministic (consistent with the low seasonal strength F_s). The candidate set therefore contains "
                    "models with and without seasonal differencing, and validation error - not these plots - picks the structure.")
    return viz.save(fig, "eda_09_acf_pacf_seasonally_differenced.png")


def fig_intermittency(ctx: Context) -> str:
    sel = ctx.sel
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.grid(True, axis="both")
    for tier in C.TIER_ORDER:
        g = sel[sel["tier"] == tier]
        ax.scatter(g["train_adi"], g["train_cv2"], s=48, color=viz.TIER_COLOR[tier], edgecolor=viz.SURFACE, linewidth=1.2, label=f"{tier} demand", zorder=3)
    ax.axvline(C.ADI_CUTOFF, color=viz.AXIS, lw=1)
    ax.axhline(C.CV2_CUTOFF, color=viz.AXIS, lw=1)
    xmax = float(sel["train_adi"].max()) * 1.08
    ymax = max(float(sel["train_cv2"].max()) * 1.1, 0.7)
    ax.set_xlim(1.0, xmax)
    ax.set_ylim(0, ymax)
    for x, y, t in ((1.02, 0.02, "smooth"), (1.02, ymax * 0.97, "erratic"), (xmax * 0.99, 0.02, "intermittent"), (xmax * 0.99, ymax * 0.97, "lumpy")):
        ax.text(x, y, t, color=viz.MUTED, fontsize=8.5, ha="left" if x < 1.5 else "right", va="bottom" if y < 0.1 else "top")
    ax.set_xlabel("ADI: average days between non-zero sales (TRAIN)")
    ax.set_ylabel("CV$^2$ of non-zero demand sizes")
    ax.legend(loc="center right")
    counts = sel["train_sb_class"].value_counts().to_dict()
    viz.title_block(fig, "How intermittent is the selected demand?", f"Syntetos-Boylan classes of the 18 series: {counts}. Intermittent/lumpy demand dominates.")
    return viz.save(fig, "eda_06_intermittency_map.png")


def fig_price(ctx: Context, rep: pd.DataFrame) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True)
    for ax, row in zip(axes, rep.itertuples()):
        f = series_frame(ctx, row.series_idx)
        ax.step(f.index, f["price"], where="post", color=viz.TIER_COLOR[row.tier], lw=1.3)
        viz.shade_periods(ax, ctx.split, label=(row.tier == "high"))
        ax.set_ylabel("USD")
        ax.set_title(f"{row.series_id}: selling price")
    viz.title_block(fig, "How much does the selling price vary?", "Weekly prices are piecewise constant; most items change price rarely, so price is a weak, sparse signal (descriptive, no causal claim).")
    return viz.save(fig, "eda_07_price_variation.png")


def fig_trend_heatmap(ctx: Context) -> str:
    a = ctx.split.train_end
    q = pd.DataFrame(ctx.Y.T, index=ctx.dates, columns=ctx.ids).resample("QS").mean().iloc[:-1]
    idx = q / pd.Series(ctx.Y[:, :a].mean(axis=1), index=ctx.ids)
    order = ctx.sel.sort_values(["tier", "store_id"], key=lambda s: s.map({t: i for i, t in enumerate(C.TIER_ORDER)}) if s.name == "tier" else s)["series_id"]
    idx = idx[list(order)]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    im = ax.imshow(idx.T.to_numpy(), aspect="auto", cmap=viz.SEQ, vmin=0, vmax=min(2.5, float(np.nanmax(idx.to_numpy()))))
    ax.set_yticks(range(len(idx.columns)))
    ax.set_yticklabels([f"{c} ({ctx.sel.set_index('series_id').loc[c, 'tier']})" for c in idx.columns], fontsize=7.5)
    ax.set_xticks(range(0, len(idx), 2))
    ax.set_xticklabels([str(d.date()) for d in idx.index[::2]], rotation=45, ha="right", fontsize=7.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("quarterly mean demand / TRAIN mean")
    viz.title_block(fig, "Do the series share trends, seasonality or regime changes?", "Quarterly mean demand relative to each series' TRAIN mean; dark = above average, light = near zero (including long dead spells).")
    return viz.save(fig, "eda_08_quarterly_index_heatmap.png")


def run_eda(ctx: Context) -> dict:
    rep = representative_series(ctx)
    stats_df = summary_stats(ctx)
    stats_df.to_csv(C.OUT_TAB / "eda_summary_stats.csv", index=False)
    rep.to_csv(C.OUT_TAB / "eda_representative_series.csv", index=False)
    figs = [fig_daily_and_rolling(ctx, rep), fig_weekly_monthly(ctx, rep), fig_distribution(ctx, rep), fig_dow_month(ctx, rep),
            fig_acf_pacf(ctx, rep), fig_acf_pacf_differenced(ctx, rep), fig_intermittency(ctx), fig_price(ctx, rep), fig_trend_heatmap(ctx)]
    write_eda_report(ctx, rep, stats_df)
    return {"rep": rep, "stats": stats_df, "figures": figs}


def write_eda_report(ctx: Context, rep: pd.DataFrame, st: pd.DataFrame) -> None:
    a = ctx.split.train_end
    sig_trend = st[st["trend_p_value"] < 0.05]
    neg_trend = st[(st["trend_p_value"] < 0.05) & (st["trend_pct_of_mean_per_year"] < 0)]
    strong_seas = st[st["seasonal_strength_Fs"] > 0.64]
    wk_peak = st["dow_peak_day"].value_counts().to_dict()
    pc = st["n_price_changes"].describe()
    fs_med, fs_max = st["seasonal_strength_Fs"].median(), st["seasonal_strength_Fs"].max()
    acf_pos = int((st["acf_lag7"] > 2 / np.sqrt(a)).sum())
    seas_word = "strong" if fs_med > 0.64 else ("moderate" if fs_med > 0.3 else "weak")
    # monthly index range of representative series (TRAIN)
    mrange = []
    for row in rep.itertuples():
        y = pd.Series(ctx.Y[row.series_idx, :a], index=ctx.dates[:a])
        m = y.groupby(y.index.month).mean() / y.mean()
        mrange.append(f"{row.tier}: {m.min():.2f}-{m.max():.2f}")
    d_idx = []
    for row in rep.itertuples():
        y = pd.Series(ctx.Y[row.series_idx, :a], index=ctx.dates[:a])
        d = y.groupby(y.index.dayofweek).mean() / y.mean()
        d_idx.append(f"{row.tier}: {d.min():.2f}-{d.max():.2f}")
    L = ["# Exploratory data analysis (Phase 7)", "",
         f"All statistics use the **TRAIN period only** ({ctx.dates[0].date()} to {ctx.dates[a - 1].date()}, {a} days) so that EDA does not "
         "leak validation/test information into modelling decisions. Figures are in `outputs/figures/eda_*.png`; "
         "the full numbers are in `outputs/tables/eda_summary_stats.csv`. Statements below describe patterns; **no causal claims** are made from correlations.", "",
         "## Representative series (rule: per tier, the series closest to the tier's median TRAIN mean)", "",
         rep.round(3).to_markdown(index=False), "",
         "## Summary statistics (TRAIN, all 18 series)", "",
         st[["series_id", "tier", "mean", "median", "std", "cv", "min", "max", "zero_pct", "adi", "cv2", "sb_class"]].round(2).to_markdown(index=False), "",
         "## Findings", "",
         f"* **Volume and zeros:** daily means range from {st['mean'].min():.2f} to {st['mean'].max():.2f} units; the share of zero-sales days ranges from "
         f"{st['zero_pct'].min():.0f}% to {st['zero_pct'].max():.0f}% (median {st['zero_pct'].median():.0f}%). "
         f"{(st['adi'] >= C.ADI_CUTOFF).sum()} of 18 series have ADI >= {C.ADI_CUTOFF} (intermittent or lumpy).",
         f"* **Variability:** the coefficient of variation ranges from {st['cv'].min():.2f} to {st['cv'].max():.2f}; "
         f"median CV by tier: high {st[st.tier=='high']['cv'].median():.2f}, medium {st[st.tier=='medium']['cv'].median():.2f}, low {st[st.tier=='low']['cv'].median():.2f}.",
         f"* **Weekly pattern: {seas_word}.** Autocorrelation at lag 7 is above the 95% noise bound for {acf_pos} of 18 series (mean {st['acf_lag7'].mean():.2f}, max {st['acf_lag7'].max():.2f}). "
         f"Robust-STL seasonal strength F_s (period 7): median {fs_med:.2f}, max {fs_max:.2f}; {len(strong_seas)} of 18 exceed 0.64 (the usual threshold for suggesting seasonal differencing). "
         f"Day-of-week indices (min-max, series mean = 1) of the representative series: {'; '.join(d_idx)}; busiest weekday over all series: {wk_peak}.",
         f"* **Yearly pattern:** monthly indices of the representative series range {'; '.join(mrange)} (TRAIN has only ~3.7 years, so each month effect rests on 3-4 observations - indicative only).",
         f"* **Trend:** an OLS slope of weekly sales on time is significant (p < 0.05) for {len(sig_trend)} of 18 series, and *negative* for {len(neg_trend)} of them "
         f"(median trend over all series {st['trend_pct_of_mean_per_year'].median():.1f}% of the mean per year). Because eligibility required the item to be listed since d_1, "
         "the sample consists of long-established items; their decline is consistent with product life-cycle effects, but the data cannot establish the cause. "
         "Long zero spells (up to ~1 year) behave like regime changes (see `zero_sales_analysis.md`) rather than a smooth trend.",
         "* **Stationarity (ADF/KPSS on levels, TRAIN):** " + ", ".join(f"{k}: {v}" for k, v in st["level_conclusion"].value_counts().items()) +
         ". 'Conflict' (ADF rejects a unit root but KPSS rejects level-stationarity) is the classic signature of a slowly moving mean or structural breaks; "
         "it motivates SARIMA candidates both with and without differencing.",
         f"* **Prices:** the median number of weekly price changes over TRAIN is {pc['50%']:.0f} (range {pc['min']:.0f}-{pc['max']:.0f}); prices are piecewise constant, so variation is sparse.",
         f"* **Outliers:** Tukey far-out fence (Q3 + 3 IQR of non-zero demand) flags {int(st['n_outliers_tukey3'].sum())} series-days "
         f"({100 * st['n_outliers_tukey3'].sum() / (18 * a):.2f}% of TRAIN series-days); they are retained (see `data_cleaning.md`).", "",
         "## Implications for modelling",
         f"* Weekly seasonality is {seas_word}: a seasonal period of 7 is still the natural choice for seasonal baselines, Holt-Winters and SARIMA, but gains from modelling it should be modest.",
         "* Intermittency and skewness -> Gaussian assumptions are approximate; Croston-SBA is included as a baseline and prediction intervals are checked empirically.",
         "* Level shifts, declines and dead spells -> slow-adapting models (small smoothing parameters) can lag; long moving averages are a strong baseline.",
         "* Sparse price variation -> price features are expected to matter little; SHAP will show whether that is the case.", ""]
    (C.REPORTS / "eda.md").write_text("\n".join(L) + "\n")
