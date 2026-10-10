"""Exploratory data analysis of the survival table: every figure / table used in REPORT.md."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import multivariate_logrank_test

from . import config as C
from .plotting import INK2, SERIES, plt, save

F, R = C.FIG_DIR, C.RES_DIR
T, E = C.DURATION, C.EVENT


def km_by(df, col, ax, title, order=None):
    if order is None:
        order = (list(df[col].cat.categories) if isinstance(df[col].dtype, pd.CategoricalDtype)
                 else sorted(df[col].dropna().unique()))
    groups = order
    for i, g in enumerate(groups):
        sub = df[df[col] == g]
        km = KaplanMeierFitter().fit(sub[T], sub[E], label=f"{g} (n={len(sub):,})")
        km.plot_survival_function(ax=ax, ci_show=False, color=SERIES[i % len(SERIES)])
    lr = multivariate_logrank_test(df[T], df[col], df[E])
    ax.set_title(f"{title}\nlog-rank chi2={lr.test_statistic:,.0f}, p={lr.p_value:.1e}")
    ax.set_xlabel("months on book"); ax.set_ylabel("P(not yet prepaid)")
    ax.set_ylim(0, 1.02)
    return dict(feature=col, chi2=float(lr.test_statistic), dof=int(len(groups) - 1), p=float(lr.p_value))


def run(df: pd.DataFrame) -> dict:
    F.mkdir(parents=True, exist_ok=True); R.mkdir(parents=True, exist_ok=True)
    summary = {}

    # ---------------------------------------------------------------- 1. outcome structure
    counts = df["exit_reason"].value_counts()
    summary["n"] = int(len(df))
    summary["exit_reason"] = counts.to_dict()
    summary["event_rate"] = float(df[E].mean())
    summary["censoring_rate"] = float(1 - df[E].mean())
    summary["duration"] = df[T].describe().round(2).to_dict()
    summary["duration_by_exit"] = df.groupby("exit_reason")[T].median().to_dict()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].barh(counts.index[::-1], counts.values[::-1], color=SERIES[0], height=0.6)
    for i, v in enumerate(counts.values[::-1]):
        ax[0].text(v, i, f"  {v:,} ({v / len(df):.1%})", va="center", fontsize=8, color=INK2)
    ax[0].set_title("How loans leave the book"); ax[0].set_xlim(0, counts.max() * 1.35); ax[0].grid(axis="y")
    for i, g in enumerate(counts.index):
        ax[1].hist(df.loc[df.exit_reason == g, T], bins=np.arange(0.5, 39.5), histtype="step",
                   linewidth=1.8, color=SERIES[i], label=g)
    ax[1].set_title("Observed duration by exit reason"); ax[1].set_xlabel("months on book"); ax[1].legend()
    save(fig, F / "01_outcome_structure.png")

    # ---------------------------------------------------------------- 2. KM / NA / hazard
    km = KaplanMeierFitter().fit(df[T], df[E])
    na = NelsonAalenFitter().fit(df[T], df[E])
    life = (df.groupby(T).agg(events=(E, "sum"), exits=(E, "size")).reindex(range(1, 39), fill_value=0))
    life["at_risk"] = life["exits"][::-1].cumsum()[::-1]
    life["smm"] = life["events"] / life["at_risk"]                 # single-monthly-mortality = monthly hazard
    life["cpr"] = 1 - (1 - life["smm"]) ** 12                       # annualised conditional prepayment rate
    life.to_csv(R / "life_table.csv")
    summary["km_survival"] = {int(m): float(km.predict(m)) for m in [6, 12, 18, 24, 30, 36]}
    summary["km_median"] = float(km.median_survival_time_)
    summary["mean_smm"] = float(life.loc[1:36, "smm"].mean())

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    km.plot_survival_function(ax=ax[0], color=SERIES[0], legend=False)
    ax[0].set_title("Kaplan-Meier S(t) (95% CI)"); ax[0].set_ylabel("P(not yet prepaid)"); ax[0].set_ylim(0, 1.02)
    na.plot_cumulative_hazard(ax=ax[1], color=SERIES[1], legend=False)
    ax[1].set_title("Nelson-Aalen cumulative hazard H(t)")
    ax[2].bar(life.index, life["smm"] * 100, color=SERIES[2], width=0.75)
    ax[2].set_title("Monthly prepayment hazard (SMM)"); ax[2].set_ylabel("% of at-risk loans prepaying")
    for a in ax:
        a.set_xlabel("months on book")
    save(fig, F / "02_km_na_hazard.png")

    # hazard by term (the PH-violation story)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, term in enumerate([36, 60]):
        sub = df[df.term_months == term]
        lt = sub.groupby(T).agg(ev=(E, "sum"), n=(E, "size")).reindex(range(1, 39), fill_value=0)
        lt["risk"] = lt["n"][::-1].cumsum()[::-1]
        ax.plot(lt.index, (lt.ev / lt.risk).rolling(3, center=True, min_periods=1).mean() * 100,
                color=SERIES[i], label=f"{term}-month loans")
    ax.set_title("Monthly prepayment hazard by term (3-mo smoothed)"); ax.set_xlabel("months on book")
    ax.set_ylabel("% prepaying"); ax.legend()
    save(fig, F / "03_hazard_by_term.png")

    # ---------------------------------------------------------------- 3. KM by segments + log-rank
    df = df.copy()
    df["fico_band"] = pd.cut(df.fico, [0, 680, 720, 760, 900], labels=["<680", "680-719", "720-759", "760+"])
    df["income_band"] = pd.qcut(df.annual_inc, 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    df["rate_spread_band"] = pd.cut(df.rate_spread, [-99, -0.25, 0.25, 99], labels=["below peers", "at peers", "above peers"])
    specs = [("term_months", "By term"), ("grade", "By grade"), ("home_ownership", "By home ownership"),
             ("purpose", "By purpose"), ("verification_status", "By income verification"),
             ("fico_band", "By FICO band"), ("income_band", "By income quartile"),
             ("rate_spread_band", "By rate spread vs. peers")]
    fig, axes = plt.subplots(4, 2, figsize=(12, 15))
    lr_rows = [km_by(df, c, a, t) for (c, t), a in zip(specs, axes.ravel())]
    for a in axes.ravel():
        a.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, F / "04_km_by_segment.png")
    pd.DataFrame(lr_rows).sort_values("chi2", ascending=False).to_csv(R / "logrank_tests.csv", index=False)

    # segment-level prepayment rates (business table)
    seg = []
    for c, _ in specs:
        g = df.groupby(c, observed=True).agg(n=(E, "size"), prepay_rate=(E, "mean"), median_months=(T, "median"))
        g["segment"] = c
        seg.append(g.reset_index().rename(columns={c: "level"}))
    pd.concat(seg).to_csv(R / "segment_rates.csv", index=False)

    # ---------------------------------------------------------------- 4. feature distributions
    feats = ["int_rate", "annual_inc", "loan_amnt", "fico", "dti", "rate_spread", "revol_util",
             "credit_history_months", "payment_to_income"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for a, f in zip(axes.ravel(), feats):
        lo, hi = df[f].quantile([0.01, 0.99])
        bins = np.linspace(lo, hi, 40)
        for i, (lab, m) in enumerate([("prepaid (event)", df[E] == 1), ("censored", df[E] == 0)]):
            a.hist(df.loc[m, f].clip(lo, hi), bins=bins, density=True, histtype="step", linewidth=1.6,
                   color=SERIES[i], label=lab)
        a.set_title(f); a.set_yticks([])
    axes[0, 0].legend()
    fig.tight_layout()
    save(fig, F / "05_feature_distributions.png")

    desc = df[C.NUMERIC + C.VIF_DROPPED].describe(percentiles=[0.01, 0.5, 0.99]).T
    desc["missing_%"] = df[C.NUMERIC + C.VIF_DROPPED].isna().mean() * 100
    desc["skew"] = df[C.NUMERIC + C.VIF_DROPPED].skew()
    desc.round(3).to_csv(R / "descriptive_stats.csv")
    miss = df.isna().mean().sort_values(ascending=False)
    summary["missing"] = {k: round(float(v) * 100, 2) for k, v in miss[miss > 0].items()}

    # ---------------------------------------------------------------- 5. correlations + VIF
    num = df[C.NUMERIC + C.VIF_DROPPED].astype(float)
    corr = num.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    ax.grid(False); fig.colorbar(im, shrink=0.7, label="Spearman rho")
    ax.set_title("Spearman correlation of numeric features")
    save(fig, F / "06_correlation.png")

    z = num.fillna(num.median())
    for c in C.LOG1P:
        if c in z:
            z[c] = np.log1p(z[c].clip(lower=0))
    z = z.clip(z.quantile(0.005), z.quantile(0.995), axis=1)
    vif_all = pd.Series(np.diag(np.linalg.inv(np.corrcoef(z.values.T))), index=z.columns)
    vif_kept = pd.Series(np.diag(np.linalg.inv(np.corrcoef(z[C.NUMERIC].values.T))), index=C.NUMERIC)
    pd.DataFrame({"VIF_all_features": vif_all, "VIF_after_drop": vif_kept}).round(2) \
        .sort_values("VIF_all_features", ascending=False).to_csv(R / "vif.csv")

    # ---------------------------------------------------------------- 6. univariate Cox screen
    rows = []
    for f in C.NUMERIC:
        x = z[f] if f in z else df[f]
        x = (x - x.mean()) / x.std()
        tmp = pd.DataFrame({"x": x, T: df[T], E: df[E]})
        cph = CoxPHFitter().fit(tmp, T, E)
        s = cph.summary.loc["x"]
        rows.append(dict(feature=f, HR_per_SD=s["exp(coef)"], lower=s["exp(coef) lower 95%"],
                         upper=s["exp(coef) upper 95%"], p=s["p"], c_index=cph.concordance_index_))
    uni = pd.DataFrame(rows).sort_values("c_index", ascending=False)
    uni.round(4).to_csv(R / "univariate_cox.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    u = uni.sort_values("HR_per_SD")
    ax.errorbar(u.HR_per_SD, range(len(u)), xerr=[u.HR_per_SD - u.lower, u.upper - u.HR_per_SD],
                fmt="o", color=SERIES[0], markersize=5, capsize=0, elinewidth=1.5)
    ax.axvline(1, color=INK2, linewidth=1)
    ax.set_yticks(range(len(u)), u.feature, fontsize=8)
    ax.set_xlabel("hazard ratio per +1 SD (univariate)"); ax.set_title("Univariate Cox screen: prepayment hazard")
    save(fig, F / "07_univariate_hr.png")

    # ---------------------------------------------------------------- 7. vintage view
    df["issue_q"] = df.issue_date.dt.to_period("Q").astype(str)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, (q, sub) in enumerate(df.groupby("issue_q")):
        KaplanMeierFitter().fit(sub[T], sub[E], label=q).plot_survival_function(ax=ax, ci_show=False, color=SERIES[i])
    ax.set_title("Vintage curves by issue quarter (2016)"); ax.set_xlabel("months on book")
    ax.set_ylabel("P(not yet prepaid)")
    save(fig, F / "08_vintage_curves.png")

    with open(R / "eda_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary
