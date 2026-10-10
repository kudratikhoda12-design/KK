"""One-factor Gaussian Monte Carlo portfolio loss, model validation and
stress testing for the full 2017 cohort.

1. Asset correlation rho estimated by MLE from quarterly cohort 12m default
   rates vs model PDs (2009Q1-2017Q4), CI by moving-block bootstrap.
2. Loan-level MC (all 2017 loans), common random numbers across scenarios.
3. Checks: MC EL vs analytic EL, MC vs ASRF analytic quantiles, MC error.
4. Backtest: percentile of the realised 2017 12m loss in the predicted distribution.
5. Stress tests: PD odds, downturn LGD, correlation, historical factor shock,
   combined adverse, and a reverse stress test.
"""
import json
import pickle
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import optimize, stats

from crm import config, models, portfolio as pf
from crm.calibration import expit, logit
from crm.plots import SERIES, plt, save

T, F = config.TAB_DIR, config.FIG_DIR
N_SCEN = 10_000
SEED = config.SEED
res = {}

# ---------------------------------------------------------------------------
# 1) Asset correlation
# ---------------------------------------------------------------------------
with open(config.MODEL_DIR / "pd_models.pkl", "rb") as f:
    M = pickle.load(f)
with open(config.MODEL_DIR / "recalibrators.pkl", "rb") as f:
    R = pickle.load(f)
score = {"LightGBM": lambda X: M["lgb"].predict(X),
         "XGBoost": lambda X: models.predict_xgb(M["xgb"], X),
         "LogReg_WOE": lambda X: M["lr"].predict(X)}
loans = pd.read_parquet(config.DATA_DIR / "loans.parquet")
obs = loans[(loans.obs_12m == 1) & (loans.issue_year >= 2009)].copy()
champ = R["champion"] if R["champion"] in score else "LightGBM"
obs["pd"] = R["recal"].predict(score[champ](obs[models.FEATURES]))
q = obs.groupby("issue_q").agg(n=("id", "size"), dr=("default_12m", "mean"), pd=("pd", "mean"))
q = q[q.n >= 500]
fit = pf.fit_rho_mle(q.dr.to_numpy(), q.pd.to_numpy())
# Moving-block bootstrap (block = 4 quarters: 12m windows of adjacent quarters overlap).
rng = np.random.default_rng(SEED)
nq, blk = len(q), 4
boots = []
for _ in range(1000):
    idx = np.concatenate([np.arange(s, s + blk) for s in rng.integers(0, nq - blk + 1, nq // blk + 1)])[:nq]
    boots.append(pf.fit_rho_mle(q.dr.to_numpy()[idx], q.pd.to_numpy()[idx])["rho"])
q["z_t"] = fit["z_t"]
q.to_csv(T / "06_quarterly_cohorts_rho.csv")
rho = float(fit["rho"])
rho_hi = float(np.quantile(boots, 0.975))
# Unconditional (grade-level, PD fixed per grade) estimate for comparison.
gq = obs.groupby(["grade", "issue_q"], observed=True).agg(n=("id", "size"), dr=("default_12m", "mean"))
gq = gq[gq.n >= 300].reset_index()
pooled = pf.fit_rho_pooled({g: d.dr.to_numpy() for g, d in gq.groupby("grade", observed=True)})
port = pd.read_parquet(config.DATA_DIR / "portfolio_2017.parquet")
res["rho"] = {"model_adjusted_mle": rho, "wald_lo": fit["lo"], "wald_hi": fit["hi"],
              "block_boot_lo": float(np.quantile(boots, 0.025)), "block_boot_hi": rho_hi,
              "n_quarters": int(nq), "grade_pooled_unconditional": pooled["rho"],
              "grade_pooled_lo": pooled["lo"], "grade_pooled_hi": pooled["hi"],
              "basel_retail_at_mean_pd": float(pf.basel_retail_rho(port.pd_final.mean())),
              "worst_quarter": q.z_t.idxmax(), "worst_z": float(q.z_t.max())}
print(json.dumps(res["rho"], indent=2, default=float))

# ---------------------------------------------------------------------------
# 2) Base Monte Carlo on the full 2017 portfolio
# ---------------------------------------------------------------------------
PD, LGD, EAD = port.pd_final.to_numpy(), port.lgd.to_numpy(), port.ead_12m.to_numpy()
EL = float(np.sum(PD * LGD * EAD))
Z = np.random.default_rng(SEED).standard_normal(N_SCEN)   # common random numbers


def run(pd_, lgd, rho_, label, seed=SEED):
    t0 = time.time()
    L = pf.simulate_losses(pd_, lgd, EAD, rho_, z=Z, seed=seed)
    r = {"scenario": label, "EL_analytic": float(np.sum(pd_ * lgd * EAD)), "EL_mc": float(L.mean()),
         "EL_mc_se": float(L.std(ddof=1) / np.sqrt(len(L))), "sd": float(L.std(ddof=1))}
    for qq in (0.99, 0.999):
        ve = pf.var_es(L, qq)
        tag = "99" if qq == 0.99 else "99.9"
        r.update({f"VaR{tag}": ve["VaR"], f"VaR{tag}_lo": ve["VaR_lo"], f"VaR{tag}_hi": ve["VaR_hi"],
                  f"ES{tag}": ve["ES"], f"ES{tag}_lo": ve["ES_lo"], f"ES{tag}_hi": ve["ES_hi"],
                  f"ASRF_VaR{tag}": pf.asrf_quantile_loss(pd_, lgd, EAD, rho_, qq)})
    r["UL_capital_99.9"] = r["VaR99.9"] - r["EL_analytic"]
    print(f"{label:<38s} EL={r['EL_mc']/1e6:7.1f}M VaR99={r['VaR99']/1e6:7.1f}M ES99={r['ES99']/1e6:7.1f}M "
          f"VaR99.9={r['VaR99.9']/1e6:7.1f}M ({time.time()-t0:.0f}s)")
    return r, L


base, L_base = run(PD, LGD, rho, "Base")
# MC validity checks
res["checks"] = {
    "EL_z": (base["EL_mc"] - EL) / base["EL_mc_se"],
    "EL_p": float(2 * stats.norm.sf(abs((base["EL_mc"] - EL) / base["EL_mc_se"]))),
    "VaR99_mc_vs_asrf_pct": base["VaR99"] / base["ASRF_VaR99"] - 1,
    "VaR99.9_mc_vs_asrf_pct": base["VaR99.9"] / base["ASRF_VaR99.9"] - 1,
}
# Convergence of VaR99 with scenario count
conv = [{"n_scen": n, "VaR99": float(np.quantile(L_base[:n], 0.99)),
         "VaR99_lo": pf.var_es(L_base[:n], 0.99)["VaR_lo"], "VaR99_hi": pf.var_es(L_base[:n], 0.99)["VaR_hi"]}
        for n in (500, 1000, 2000, 5000, 10000)]
pd.DataFrame(conv).to_csv(T / "06_mc_convergence.csv", index=False)

# Original project set-up (10,000 loans x 2,000 scenarios) for comparison.
rs = np.random.default_rng(1)
idx = rs.choice(len(PD), 10_000, replace=False)
L_small = pf.simulate_losses(PD[idx], LGD[idx], EAD[idx], rho, n_scen=2000, seed=2)
vs = pf.var_es(L_small, 0.99)
res["original_setup_10k_loans_2k_scen"] = {"VaR99": vs["VaR"], "VaR99_lo": vs["VaR_lo"], "VaR99_hi": vs["VaR_hi"],
                                          "ES99": vs["ES"], "EAD": float(EAD[idx].sum()),
                                          "VaR99_ci_width_pct": (vs["VaR_hi"] - vs["VaR_lo"]) / vs["VaR"]}

# ---------------------------------------------------------------------------
# 3) Backtest: realised 2017 12m loss within the predicted distribution
# ---------------------------------------------------------------------------
realised = float((port.ead_default * port.default_12m * LGD).sum())
pct = float((L_base <= realised).mean())
res["backtest_2017"] = {"realised_loss": realised, "predicted_EL": EL, "percentile_in_mc": pct,
                        "two_sided_p": float(2 * min(pct, 1 - pct)),
                        "implied_z_2017": float(q.loc[q.index.str.startswith("2017"), "z_t"].mean())}
print(res["backtest_2017"])

# ---------------------------------------------------------------------------
# 4) Stress tests (same Z draws -> differences are not MC noise)
# ---------------------------------------------------------------------------
lgd_dt = json.load(open(T / "04_lgd_summary.json"))["lgd_downturn"]
odds = lambda p, m: expit(logit(p) + np.log(m))
rows = [base]
scen = {
    "PD odds x1.5": (odds(PD, 1.5), LGD, rho),
    "PD odds x2.0": (odds(PD, 2.0), LGD, rho),
    f"Downturn LGD ({lgd_dt:.3f})": (PD, np.full_like(LGD, lgd_dt), rho),
    "LGD +5pp": (PD, np.minimum(LGD + 0.05, 1), rho),
    f"Correlation stress (rho={rho_hi:.3f}, 97.5% boot)": (PD, LGD, rho_hi),
    "Combined: PD x1.5 + downturn LGD + rho stress": (odds(PD, 1.5), np.full_like(LGD, lgd_dt), rho_hi),
}
for lab, (p_, l_, r_) in scen.items():
    rows.append(run(p_, l_, r_, lab)[0])
# Historical scenario: re-run the worst observed systematic factor (deterministic Z).
zw = res["rho"]["worst_z"]
Lh = pf.simulate_losses(PD, LGD, EAD, rho, z=np.full(2000, zw), seed=7)
rows.append({"scenario": f"Historical worst quarter {res['rho']['worst_quarter']} (Z={zw:.2f})",
             "EL_analytic": float(np.sum(pf.cond_pd(PD, rho, zw) * LGD * EAD)), "EL_mc": float(Lh.mean()),
             "VaR99": float(np.quantile(Lh, 0.99))})
st = pd.DataFrame(rows)
st.to_csv(T / "06_stress_results.csv", index=False)
print(st[["scenario", "EL_mc", "VaR99", "ES99", "VaR99.9", "ES99.9"]].assign(
    **{c: lambda d, c=c: d[c] / 1e6 for c in ["EL_mc", "VaR99", "ES99", "VaR99.9", "ES99.9"]}).round(1).to_string(index=False))

# Reverse stress: PD odds multiplier at which expected loss = base 99.9% VaR.
mult = optimize.brentq(lambda m: np.sum(odds(PD, m) * LGD * EAD) - base["VaR99.9"], 1.0, 20.0)
res["reverse_stress_pd_odds_multiplier_EL_eq_base_VaR99.9"] = float(mult)
res["base"] = base

fig, ax = plt.subplots(figsize=(7.5, 4.2))
L_c = pf.simulate_losses(*scen["Combined: PD x1.5 + downturn LGD + rho stress"][:2], EAD,
                         scen["Combined: PD x1.5 + downturn LGD + rho stress"][2], z=Z, seed=SEED)
bins = np.linspace(min(L_base.min(), L_c.min()), max(L_base.max(), L_c.max()), 120) / 1e6
ax.hist(L_base / 1e6, bins=bins, color=SERIES[0], alpha=0.75, label="Base")
ax.hist(L_c / 1e6, bins=bins, color=SERIES[1], alpha=0.6, label="Combined adverse")
for v_, c_ in [(base["VaR99"], SERIES[0]), (rows[6]["VaR99"], SERIES[1])]:
    ax.axvline(v_ / 1e6, color=c_, lw=1.2, ls="--")
ax.axvline(realised / 1e6, color="#0b0b0b", lw=1.4)
ax.annotate("Realised 2017", (realised / 1e6, ax.get_ylim()[1] * 0.9), fontsize=8, ha="right")
ax.set(xlabel="12-month portfolio loss ($M)", ylabel="Scenarios",
       title="2017 cohort 12m loss distribution (dashed = 99% VaR)")
ax.legend()
save(fig, F / "06_loss_distribution.png")

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(range(len(q)), q.z_t, color=SERIES[0], marker="o", ms=3)
ax.axhline(0, color="#b5b3ad", lw=1)
ax.set_xticks(range(0, len(q), 4), q.index[::4], rotation=45, fontsize=8)
ax.set(ylabel="Implied systematic factor Z", title=f"Implied credit-cycle factor by issue quarter (rho={rho:.3f})")
save(fig, F / "06_systematic_factor.png")

json.dump(res, open(T / "06_portfolio_summary.json", "w"), indent=2, default=float)
