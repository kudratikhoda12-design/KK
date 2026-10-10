"""Additional figures for the LaTeX technical report (reports/latex)."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from crm import config
from crm.plots import GRID, INK2, SERIES, plt, save

T, F = config.TAB_DIR, config.FIG_DIR
M = 1e6

# 1) Data overview: volume and 12m default rate by issue year (two panels, no dual axis)
c = pd.read_csv(T / "01_cohort_summary.csv", index_col=0)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].bar(c.index, c["loans"] / 1e3, color=SERIES[0], width=0.7)
axs[0].set(title="Loans issued per year", ylabel="Loans (thousands)")
d = c.dropna(subset=["dr_12m_all_loans"])
axs[1].plot(d.index, d["dr_12m_all_loans"] * 100, color=SERIES[0], marker="o", ms=5)
axs[1].set(title="12-month default rate by issue year", ylabel="Default rate (%)")
for ax in axs:
    ax.set_xticks(c.index, c.index, rotation=45)
save(fig, F / "08_data_overview.png")

# 2) Survivorship bias of the terminal-only design (36m loans)
b = pd.read_csv(T / "07_terminal_bias_by_year.csv", index_col=0)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].plot(b.index, b.share_terminal * 100, color=SERIES[0], marker="o", ms=5)
axs[0].set(title="Share of 36m loans closed at 2019-03 snapshot", ylabel="%")
axs[1].plot(b.index, b.dr_terminal * 100, color=SERIES[1], marker="o", ms=5, label="Terminal loans only")
axs[1].plot(b.index, b.dr_all_loans_incl_open * 100, color=SERIES[0], marker="o", ms=5, label="All loans")
axs[1].set(title="Lifetime default rate, 36m loans", ylabel="%")
axs[1].legend()
save(fig, F / "08_survivorship_bias.png")

# 3) Information value
iv = pd.read_csv(T / "02_information_value.csv", index_col=0)["iv"].head(30)[::-1]
fig, ax = plt.subplots(figsize=(7, 7))
ax.barh(iv.index, iv.values, color=[SERIES[0] if v >= 0.02 else "#b5b3ad" for v in iv.values], height=0.7)
ax.axvline(0.02, color=INK2, lw=1, ls="--")
ax.annotate(" IV = 0.02 cut-off", (0.02, 1), fontsize=8, color=INK2)
ax.set(title="Information value (top 30, development 2007-2014)", xlabel="IV")
ax.tick_params(axis="y", labelsize=8)
save(fig, F / "08_information_value.png")

# 4) LightGBM gain importance
imp = pd.read_csv(T / "02_lgb_gain_importance.csv", index_col=0).iloc[:, 0]
imp = (imp / imp.sum()).head(20)[::-1]
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(imp.index, imp.values * 100, color=SERIES[0], height=0.7)
ax.set(title="LightGBM feature importance (share of total gain)", xlabel="% of gain")
ax.tick_params(axis="y", labelsize=8)
save(fig, F / "08_lgb_importance.png")

# 5) KS chart and score distribution - final PD, 2017
v = pd.read_parquet(config.DATA_DIR / "pd_final_2017.parquet")
s = np.sort(v.pd_final.unique())
grid = np.quantile(v.pd_final, np.linspace(0, 1, 401))
bad = v.pd_final[v.default_12m == 1].to_numpy()
good = v.pd_final[v.default_12m == 0].to_numpy()
cb = np.searchsorted(np.sort(bad), grid, side="right") / len(bad)
cg = np.searchsorted(np.sort(good), grid, side="right") / len(good)
k = np.argmax(cg - cb)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
axs[0].plot(grid * 100, cg * 100, color=SERIES[0], label="Non-defaults (cum.)")
axs[0].plot(grid * 100, cb * 100, color=SERIES[1], label="Defaults (cum.)")
axs[0].vlines(grid[k] * 100, cb[k] * 100, cg[k] * 100, color=INK2, lw=1.2, ls="--")
axs[0].annotate(f" KS = {cg[k]-cb[k]:.3f}", (grid[k] * 100, (cb[k] + cg[k]) * 50), fontsize=9)
axs[0].set(xscale="log", xlabel="Predicted 12m PD (%) [log scale]", ylabel="Cumulative %", title="KS chart - 2017")
axs[0].legend(fontsize=8)
bins = np.logspace(np.log10(v.pd_final.min()), np.log10(v.pd_final.max()), 60)
axs[1].hist(good * 100, bins=bins * 100, density=True, color=SERIES[0], alpha=0.65, label="Non-defaults")
axs[1].hist(bad * 100, bins=bins * 100, density=True, color=SERIES[1], alpha=0.6, label="Defaults")
axs[1].set(xscale="log", xlabel="Predicted 12m PD (%) [log scale]", ylabel="Density", title="Score separation - 2017")
axs[1].legend(fontsize=8)
save(fig, F / "08_ks_score_distribution.png")

# 6) Backtest by decile and grade with 95% Jeffreys intervals
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
for ax, f, lab in [(axs[0], "03_backtest_decile_2017.csv", "pd_decile"), (axs[1], "03_backtest_grade_2017.csv", "grade")]:
    t = pd.read_csv(T / f)
    lo = stats.beta.ppf(0.025, t.defaults + 0.5, t.n - t.defaults + 0.5)
    hi = stats.beta.ppf(0.975, t.defaults + 0.5, t.n - t.defaults + 0.5)
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.pd_mean * 100, width=0.4, color=SERIES[0], label="Mean predicted PD")
    ax.bar(x + 0.2, t.odr * 100, width=0.4, color=SERIES[1], label="Observed DR")
    ax.errorbar(x + 0.2, t.odr * 100, yerr=[(t.odr - lo) * 100, (hi - t.odr) * 100], fmt="none",
                ecolor="#0b0b0b", elinewidth=1, capsize=2)
    ax.set_xticks(x, t[lab].astype(str))
    ax.set(xlabel="PD decile" if lab == "pd_decile" else "Grade", ylabel="%")
    ax.legend(fontsize=8)
axs[0].set_title("Backtest by PD decile - 2017 (95% Jeffreys CI)")
axs[1].set_title("Backtest by grade - 2017 (95% Jeffreys CI)")
save(fig, F / "08_backtest.png")

# 7) PSI by year
p = pd.read_csv(T / "03_psi_score_by_year.csv")
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.bar(p.year.astype(str), p.psi_score, color=SERIES[0], width=0.6)
ax.axhline(0.10, color=SERIES[3], lw=1.2, ls="--")
ax.axhline(0.25, color=SERIES[7], lw=1.2, ls="--")
ax.annotate("0.10 monitor", (-0.4, 0.103), fontsize=8)
ax.annotate("0.25 significant shift", (-0.4, 0.253), fontsize=8)
ax.set(title="Score PSI vs development sample", ylabel="PSI", ylim=(0, 0.28))
save(fig, F / "08_psi.png")

# 8) LGD distribution and by default year
loans = pd.read_parquet(config.DATA_DIR / "loans.parquet",
                        columns=["loan_status", "lgd_realised", "months_since_default", "default_dt"])
co = loans[loans.loan_status.isin(["Charged Off", "Default"]) & loans.lgd_realised.notna()
           & (loans.months_since_default >= 18)]
ly = pd.read_csv(T / "04_lgd_by_default_year.csv", index_col=0)
ly = ly[ly["size"] >= 1000]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].hist(co.lgd_realised, bins=50, color=SERIES[0])
axs[0].axvline(co.lgd_realised.mean(), color=SERIES[1], lw=1.5, ls="--")
axs[0].annotate(f" mean {co.lgd_realised.mean():.3f}", (co.lgd_realised.mean(), axs[0].get_ylim()[1] * 0.85),
                fontsize=8, ha="right")
axs[0].set(title="Realised LGD, mature charge-offs", xlabel="LGD", ylabel="Loans")
axs[1].plot(ly.index, ly["mean"], color=SERIES[0], marker="o", ms=5)
axs[1].set(title="Mean LGD by default year (n >= 1,000)", ylabel="LGD", xlabel="Default year")
save(fig, F / "08_lgd.png")

# 9) ECL by grade
g = pd.read_csv(T / "05_ecl_by_grade_2017.csv", index_col=0)
fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(len(g))
ax.bar(x - 0.2, g.ecl_12m / M, width=0.4, color=SERIES[0], label="12-month ECL")
ax.bar(x + 0.2, g.ecl_life / M, width=0.4, color=SERIES[1], label="Lifetime ECL")
ax.set_xticks(x, g.index)
ax.set(title="ECL by grade - 2017 cohort", ylabel="$ million", xlabel="Grade")
ax.legend()
save(fig, F / "08_ecl_grade.png")

# 10) Monte Carlo convergence
cv = pd.read_csv(T / "06_mc_convergence.csv")
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.errorbar(cv.n_scen, cv.VaR99 / M, yerr=[(cv.VaR99 - cv.VaR99_lo) / M, (cv.VaR99_hi - cv.VaR99) / M],
            fmt="o-", color=SERIES[0], ecolor=SERIES[0], capsize=3, ms=5)
ax.set(xscale="log", xlabel="Number of scenarios", ylabel="99% VaR ($M)",
       title="Monte Carlo convergence of 99% VaR (95% order-statistic CI)")
save(fig, F / "08_mc_convergence.png")

# 11) Stress test results
st = pd.read_csv(T / "06_stress_results.csv").dropna(subset=["ES99"])
st = st.iloc[::-1]
lab = st.scenario.str.replace(r" \(.*\)", "", regex=True).str.replace("Combined adverse: ", "Combined: ")
y = np.arange(len(st))
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.barh(y + 0.27, st.EL_mc / M, height=0.27, color=SERIES[2], label="Expected loss")
ax.barh(y, st.VaR99 / M, height=0.27, color=SERIES[0], label="99% VaR")
ax.barh(y - 0.27, st.ES99 / M, height=0.27, color=SERIES[1], label="99% ES")
ax.set_yticks(y, lab, fontsize=8)
ax.set(xlabel="$ million", title="Stress testing - 2017 portfolio, 12-month horizon")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=3)
save(fig, F / "08_stress.png")

# 12) Term structure of default: 12m vs lifetime CDR by grade (36m)
ts = pd.read_csv(T / "05_pd_term_structure.csv")
ts36 = ts[ts.term_m == 36]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
x = np.arange(len(ts36))
ax.bar(x - 0.2, ts36.cdr_12m * 100, width=0.4, color=SERIES[0], label="12-month CDR")
ax.bar(x + 0.2, ts36.cdr_life * 100, width=0.4, color=SERIES[1], label="Lifetime CDR")
ax.set_xticks(x, ts36.grade)
ax.set(title="Matured 36m vintages 2012-2015: 12m vs lifetime default", ylabel="%", xlabel="Grade")
ax.legend()
save(fig, F / "08_term_structure.png")
print("figures written")
