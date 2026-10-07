"""Phases 39-41 - interview defence guide, 60-second / 2-minute / 5-minute project stories and resume bullets.

Every project-specific number is read from outputs/tables through ``reporting.Facts``; the conceptual explanations are static text.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import config as C
from .reporting import Facts, PRIMARY, sgn


def _qa(n: int, q: str, short: str, detail: str, math: str | None, follow: str, follow_ans: str) -> str:
    out = [f"### Q{n}. {q}", "", f"**1. Short interview answer.** {short}", "", f"**2. Detailed technical answer.** {detail}", ""]
    out.append("**3. Mathematical explanation.**")
    out.append("")
    out.append(math if math else "Not applicable (conceptual question).")
    out += ["", f"**4. Likely follow-up:** *{follow}*", "", f"> {follow_ans}", ""]
    return "\n".join(out)


class Numbers:
    """Frequently used project numbers (all computed)."""

    def __init__(self, f: Facts):
        self.f = f
        t, v = f.test, f.val
        self.best = {H: f.best_row(H) for H in C.HORIZONS}
        self.hB = f.headB
        self.g = {L: f.cost_gain_vs_A(self.hB[L], L) for L in C.LEAD_TIMES}
        self.ma_gain7 = f.cost_gain_vs_A("MovingAverage", 7)
        p = f.pair[(f.pair.scenario == "MEDIUM") & (f.pair.SL == 0.95)].set_index("L")
        self.ci = {L: (100 * p.loc[L, "boot_ci_low"], 100 * p.loc[L, "boot_ci_high"]) for L in C.LEAD_TIMES}
        self.n_cheaper = int((f.pair.rel_diff_pooled < 0).sum())
        self.n_cfg = len(f.pair)
        self.ratio = t.merge(v, on=["model", "H"], suffixes=("_t", "_v")).assign(r=lambda d: d.RMSE_t / d.RMSE_v)
        self.a95 = f.inv("A_historical", 7, 0.95)
        self.b95 = f.inv("B_forecast_driven", 7, 0.95)
        s = f.avc_s
        self.rho = {sc: float(s[s.scenario == sc]["spearman_testWAPE_vs_cost_excl_Naive"].mean()) for sc in C.SCENARIOS}
        self.same_best = int(s.same_best_test_accuracy_and_cost.sum())
        sh = f.shap
        self.roll = {H: 100 * sh[(sh.H == H) & sh.feature.isin(["rolling_mean_7", "rolling_mean_14", "rolling_mean_28"])].share_of_total.sum() for H in (7, 28)}
        s7 = sh[sh.H == 7].set_index("feature")
        nd = sh[(sh.H == 7) & ~sh.feature.str.startswith(("rolling_", "lag_"))].sort_values("share_of_total", ascending=False).head(2)
        label = {"snap_days_window": "SNAP-day counts", "price": "price"}
        self.top_nondemand = " and ".join(f"{label.get(r.feature, r.feature)} ({100 * r.share_of_total:.1f}%)" for r in nd.itertuples())
        self.snap = 100 * float(s7.loc["snap_days_window", "share_of_total"])
        self.price = 100 * float(s7.loc["price", "share_of_total"])
        self.cnt = f.dm_counts("XGBoost")
        self.sig_worse = [m for H in PRIMARY for m in self.cnt[H]["worse"]]
        self.nonnormal = float(f.norm[f.norm.model == "XGBoost"].pooled_excess_kurtosis.abs().max())
        self.sel = f.sel
        self.n_ci_excl0 = int(((f.pair.boot_ci_high < 0) | (f.pair.boot_ci_low > 0)).sum())
        mr = t.merge(v, on=["model", "H"], suffixes=("_t", "_v")).assign(r=lambda d: d.MAE_t / d.MAE_v)
        self.mae_ratio = (float(mr.r.min()), float(mr.r.max()))
        self.xgb_best_H = [H for H in C.HORIZONS if self.best[H]["best_by_test_WAPE"] == "XGBoost"]
        self.sarima_rank = {H: int(t[t.H == H].sort_values("WAPE").reset_index(drop=True).query("model == 'SARIMA'").index[0] + 1) for H in PRIMARY}
        self.sarima_val_best_H = [H for H in PRIMARY if self.best[H]["best_by_validation_WAPE"] == "SARIMA"]
        self.chosen_struct = f.sar_s.chosen.str.split(":", n=1).str[1].value_counts().to_dict()
        self.n_seasdiff = int(f.sar_s.chosen.str.startswith(("S3", "S4", "S5")).sum())
        self.depths = f.xgb_ch.set_index("H").max_depth.astype(int).to_dict()
        sa = f.ets[(f.ets.phase == "val") & f.ets.aicc.notna()].pivot(index="series_id", columns="model", values="aicc")
        self.hw_better = int((sa.HoltWinters < sa.SES).sum())
        self.lw = lambda x: ("lowered" if x < 0 else "raised") + f" simulated total cost by {abs(x):.1f}%"
        # ---- additional computed facts (so that no explanation is a hard-coded claim about the results)
        self.conc7 = f.concentration(7)
        self.emp = f.emp_vs_gauss()
        self.xgb_best_primary = [H for H in PRIMARY if self.best[H]["best_by_test_WAPE"] == "XGBoost"]
        self.xgb_never_better = not any(self.cnt[H]["better"] for H in PRIMARY)
        self.worse7 = sorted(self.cnt[7]["worse"])
        self.worse_long = sorted({m for H in (14, 28) for m in self.cnt[H]["worse"]})
        self.beaters_of_naive = [m for m in ("XGBoost", "SARIMA") if f.beats_naive(m)]
        e = f.eda
        band = 1.96 / np.sqrt(f.split.train_end - 7)
        pc = [f"d7_pacf_lag{k}" for k in (7, 14, 21, 28)]
        self.pacf_d = [float(e[c].median()) for c in pc]
        self.pacf_all_out = int((e[pc].abs() > band).all(axis=1).sum())
        self.pacf_gradual = all(abs(a) > abs(b) for a, b in zip(self.pacf_d, self.pacf_d[1:]))
        self.acf_d7 = (float(e.d7_acf_lag7.median()), float(e.d7_acf_lag7.min()), float(e.d7_acf_lag7.max()))
        self.corr_mean_std = float(e.corr_roll28_mean_std.median())
        self.rho_price_level = float(stats.spearmanr(self.sel.train_mean_price, self.sel.train_mean).statistic)
        self.nd_b, self.nd_a, self.nd_days = f.lost_nov_dec(7)
        self.same_pf7 = f.ma_same_point_forecast_as_A(7)
        self.tier_best = {H: {g: f.best_in_group(g, H) for g in ("tier=high", "tier=medium", "tier=low", "regular", "intermittent")} for H in (14, 28)}
        self.sz = f.smape_zero_story(28)
        self.n_nonnaive = int(f.test[f.test.H == 7].model.nunique() - 1)
        self.n_grp = f.rob_best[f.rob_best.H == 28].set_index('group').n_series.astype(int).to_dict()
        self.tune_gain = [100 * (1 - g.val_MAE.min() / g[g.is_default].val_MAE.iloc[0]) for _, g in f.xgb_log.groupby("H")]


def build_interview_guide(f: Facts) -> None:
    N = Numbers(f)
    sp = f.split
    L = ["# Interview defence guide", "",
         "How to use: every question has (1) a 20-30 second answer, (2) the technical depth behind it, (3) the maths, (4) the follow-up an interviewer is likely to ask, with an answer. "
         "Project-specific numbers are generated from the actual result tables (re-run `python run_project.py` and they update).", "",
         "**Honesty rules to repeat in any interview:** costs are hypothetical; XGBoost did *not* clearly win; the test period was harder than validation; the evidence for the inventory gain is suggestive (wide intervals).", "",
         "---", "", "## DATA", ""]
    n = 0

    def add(*a):
        nonlocal n
        n += 1
        L.append(_qa(n, *a))

    add("Why this dataset?",
        "M5 is the standard public retail-demand benchmark: real Walmart item-store daily sales with prices, calendar events and SNAP flags. It is realistic (intermittent, noisy, promotional) and big enough to compare statistical and ML methods fairly.",
        f"M5 has 30,490 item-store series over {sp.n_days:,} days in 3 states and 10 stores, plus weekly prices and an event calendar. It lets me test ML with price/event features and tie forecasts to inventory decisions. "
        "The official Kaggle files need credentials, so I used two independent public Hugging Face mirrors and verified every file by SHA-256 (identical in both mirrors) plus structural checks "
        f"(30,490 series x {sp.n_days:,} days, 6,841,121 price rows, validation file = prefix of evaluation file). I state the limitation that I could not compare with Kaggle's own checksums.",
        None, "How do you know the mirror was not altered?",
        "I cannot prove byte-equality with Kaggle, but two independently uploaded copies have identical hashes and the structure matches the documented release; I report that as a limitation rather than claim certainty.")
    add("What is the target variable?",
        "Observed daily unit sales of one item in one store, y_t. For modelling I forecast the sum of demand over the next H days (7, 14, 28) from each forecast origin.",
        "At origin t (end of day t) the model predicts Target_H(t) = sum of the next H days. Every model - baselines, ETS, SARIMA, XGBoost - is evaluated on exactly this quantity, so the comparison is like for like. "
        "Statistical models forecast the daily path and I sum it (negative forecasts clipped at zero); XGBoost uses the *direct* strategy, one model per horizon. H = 3 is supplementary, needed only for the 3-day lead time of the inventory layer.",
        "```\nTarget_H(t) = y[t+1] + y[t+2] + ... + y[t+H]        (H = 7, 14, 28; H = 3 supplementary)\n```",
        "Why predict the sum rather than each day?",
        "Because inventory needs *lead-time demand* (a sum), sums are less intermittent than daily values, and it avoids error accumulation of recursive multi-step forecasts.")
    add("Why daily demand?",
        "Lead times and replenishment decisions are in days, the data are daily, and weekly patterns (SNAP, weekends) live at the daily level. I aggregate to H-day sums for evaluation.",
        f"Daily data keep the weekly cycle and let the simulation run day by day (orders, arrivals, stockouts). The downside is intermittency ({int((N.sel.train_adi >= C.ADI_CUTOFF).sum())} of 18 series have ADI >= 1.32), which is why I added a Croston baseline and evaluate on sums.",
        None, "Would weekly forecasting be better?",
        "Possibly for slow movers; but weekly buckets would blur lead times of 3 days and the weekday structure. I would test it, scoring both on the same lead-time sums.")
    add("Why did you select these products?",
        "To get a reproducible, representative mix of high, medium and low demand using only training information: one store per state, demand tiers by TRAIN-mean percentile bands, seeded random draws. Nothing was chosen for good test performance.",
        f"Steps: chronological split first; one store per state with the largest TRAIN volume ({', '.join(f.log['stores'])}); eligibility = listed every TRAIN day from d_1 ({f.log['n_eligible_listed_all_train_days']:,} series); "
        f"tiers from percentile bands of TRAIN mean sales; 2 random draws (seed {C.SEED}) per store x tier, different categories preferred and no repeated item -> 18 series, 18 distinct products, categories {N.sel.cat_id.value_counts().to_dict()}. "
        "Selection ran once; nothing was repeated after seeing any forecasting result. I also verified (not selected on) that the eligible items stay listed in validation/test.",
        None, "Isn't 18 series too few?",
        f"Yes, it limits power: bootstrap and DM intervals are wide and subgroup results ({min(N.n_grp.values())}-{max(N.n_grp.values())} series) are descriptive. It was a deliberate trade-off for transparency; the pipeline scales by changing the config.")
    add("What are the limitations of the data?",
        "Sales are not demand: stockouts are invisible, so observed sales may underestimate true demand. Also no real costs or lead times, and my sample is old, continuously listed items.",
        f"(1) Zero sales are {100 * float(f.cleaning_checks['zero_share_overall']):.0f}% of cells; I can identify structural zeros (no price row) and closures, but not true zero demand vs stockout; "
        f"{int((f.zero_s.longest_zero_run_open_days >= 90).sum())} of 18 series have >= 90-day zero spells while listed. (2) No inventory records, costs or supplier lead times - all assumed and labelled hypothetical. "
        f"(3) Items listed since 2011: {int(((f.eda.trend_p_value < 0.05) & (f.eda.trend_pct_of_mean_per_year < 0)).sum())} of 18 series have a significantly negative trend (median trend {f.eda.trend_pct_of_mean_per_year.median():.0f}% of the mean per year). (4) One {sp.n_days - sp.val_end}-day test period.",
        None, "What would you do about censored demand in a real company?",
        "Use stock records to flag stockout days, treat them as censored observations (Tobit / Kaplan-Meier style estimators) or exclude them from training, and measure lost sales explicitly.")

    L += ["", "## TIME SERIES", ""]
    add("What is stationarity?",
        "A series is (weakly) stationary if its mean, variance and autocovariance structure do not change over time. Many classical models, like ARMA, assume it.",
        "Weak stationarity: constant mean, constant finite variance, autocovariance that depends only on the lag. Trends, level shifts and changing variance break it. Differencing or modelling the trend restores it. "
        f"In my data the ADF/KPSS pair gives: {', '.join(f'{k}: {v}' for k, v in f.stat.level_conclusion.value_counts().items())} - i.e. {int(f.stat.level_conclusion.str.startswith('conflict').sum())} of 18 series show 'ADF says no unit root but KPSS rejects level-stationarity', "
        "which is typical of a slowly moving mean or structural breaks.",
        "```\nE[y_t] = mu,   Var(y_t) = sigma^2 < inf,   Cov(y_t, y_{t-k}) = gamma(k)   for all t\n```",
        "What did you do about the conflict?",
        "I kept both d = 0 and d = 1 (and D = 0 / 1) candidates and let validation error choose; AIC cannot compare different differencing orders.")
    add("What is the ADF test?",
        "The Augmented Dickey-Fuller test checks the null hypothesis of a unit root (non-stationarity). A small p-value means the series looks stationary.",
        "It regresses the first difference on the lagged level plus lagged differences (lag order by AIC); a significantly negative coefficient on the lagged level rejects the unit root. "
        "It has low power against slow mean reversion, and structural breaks make it reject too rarely or too often, so I pair it with KPSS.",
        "```\nDelta y_t = a + gamma * y_{t-1} + sum_{i=1..p} delta_i * Delta y_{t-i} + e_t\nH0: gamma = 0 (unit root)   vs   H1: gamma < 0 (stationary)\n```",
        "What if ADF and KPSS disagree?",
        "Then the evidence is inconclusive: ADF rejects a unit root (H0) while KPSS rejects stationarity (its H0). That points to a trend/regime change; I treat differencing as a modelling choice to be validated.")
    add("What is the KPSS test?",
        "KPSS reverses the null: H0 is that the series is stationary around a mean (or trend). A small p-value says non-stationary. I use it with ADF for a two-sided view.",
        "It builds the partial sums of residuals from the series' mean and compares their variability (LM statistic) with a long-run variance estimate; large partial-sum wandering means a unit-root-like component.",
        "```\nS_t = sum_{i<=t} e_i,   KPSS = (1/T^2) * sum_t S_t^2 / sigma_LR^2      (reject stationarity if large)\n```",
        "Why use both tests?",
        f"Because their nulls are opposite: agreement gives confidence ({int(f.stat.level_conclusion.str.startswith('stationary').sum())} 'stationary' and {int(f.stat.level_conclusion.str.startswith('unit root').sum())} 'unit root' series here); disagreement flags structure that needs modelling.")
    add("What is the ACF?",
        "The autocorrelation function shows the correlation of the series with its own lags. Spikes at lag 7, 14, 21 suggest weekly seasonality.",
        f"ACF(k) = corr(y_t, y_(t-k)); under no autocorrelation the 95% band is about +-1.96/sqrt(n). In my TRAIN data lag-7 autocorrelation exceeds the band for {int((f.eda.acf_lag7 > 2 / np.sqrt(sp.train_end)).sum())} of 18 series "
        f"(mean {f.eda.acf_lag7.mean():.2f}, max {f.eda.acf_lag7.max():.2f}) - present but modest, matching the low seasonal strength (median F_s {f.eda.seasonal_strength_Fs.median():.2f}).",
        "```\nrho(k) = Cov(y_t, y_{t-k}) / Var(y_t)        band = +-1.96 / sqrt(n)\n```",
        "How did you use it for SARIMA?",
        "After seasonal differencing the ACF has a spike near -0.5 at lag 7, the signature of a seasonal MA(1) term - or of over-differencing a weakly seasonal series; hence candidates with and without D = 1.")
    add("What is the PACF?",
        "The partial autocorrelation is the correlation at lag k after removing the effect of the intermediate lags. It suggests the AR order: a sharp cut-off after lag p indicates AR(p).",
        "PACF(k) is the coefficient of y_(t-k) in an AR(k) regression. For a pure MA process the PACF " + ("decays gradually instead of cutting off" if N.pacf_gradual else "does not cut off cleanly") + ": "
        f"the seasonally differenced TRAIN series show median PACF {N.pacf_d[0]:.2f}, {N.pacf_d[1]:.2f}, {N.pacf_d[2]:.2f}, {N.pacf_d[3]:.2f} at the seasonal lags 7, 14, 21, 28 "
        f"(outside the 95% band at all four lags in {N.pacf_all_out} of 18 series; `eda_09_acf_pacf_seasonally_differenced.png`), which points to a seasonal MA term.",
        "```\nPACF(k) = last coefficient phi_kk of the AR(k) fit  y_t = phi_k1 y_{t-1} + ... + phi_kk y_{t-k} + e_t\n```",
        "Why look at PACF if you pick the model by validation?",
        "It narrows the candidate set to a handful of structures (small and fast) so I do not search blindly; validation error then decides.")
    add("Why seasonal period 7?",
        "Daily retail data have a weekly cycle, so the natural seasonal period is 7. I confirm it with the ACF and with Holt-Winters AICc rather than assume it.",
        f"AICc of Holt-Winters is lower than SES for {N.hw_better} of 18 series. "
        f"But seasonal strength is weak (median {f.eda.seasonal_strength_Fs.median():.2f}), and for 7/14/28-day *sums* the additive seasonal indices cancel, so weekly seasonality barely changes those forecasts "
        f"(test WAPE SES {f.wape('SES', 7):.1f}% vs Holt-Winters {f.wape('HoltWinters', 7):.1f}% at 7 days).",
        "```\nseasonal period m = 7 : s_{t+h} = s_{t+h-7*ceil(h/7)}  (additive weekly index)\n```",
        "What about yearly seasonality?",
        "With only ~3.7 training years, month effects rest on 3-4 observations; the model sees month/week-of-year/Christmas features instead, and I flag the estimates as indicative.")
    add("What is differencing?",
        "Subtracting the previous (or seasonal) value to remove trend or seasonality. d is the number of ordinary differences, D the number of seasonal ones.",
        "First difference (1-B)y removes a stochastic trend; seasonal difference (1-B^7)y removes a stable weekly pattern. Over-differencing injects negative autocorrelation (an MA(1) with theta = -1): independent noise differenced at lag 7 has autocorrelation -0.5 at lag 7. "
        f"The seasonally differenced TRAIN series have a lag-7 autocorrelation of {N.acf_d7[0]:.2f} (range {N.acf_d7[1]:.2f} to {N.acf_d7[2]:.2f}); but removing a stable weekly pattern from noisy data gives the same -0.5, so the spike alone cannot tell a needed seasonal difference from an unnecessary one - validation error has to decide.",
        "```\n(1 - B) y_t = y_t - y_{t-1}        (1 - B^7) y_t = y_t - y_{t-7}\nIf y_t is white noise: corr((1-B^7)y_t, (1-B^7)y_{t-7}) = -0.5\n```",
        "Which differencing did the data choose?",
        f"Validation error picked structures per series: {', '.join(f'{k} x{v}' for k, v in N.chosen_struct.items())}; {N.n_seasdiff} of 18 use seasonal differencing although seasonal strength is weak (median F_s {f.eda.seasonal_strength_Fs.median():.2f}); with a seasonal MA(1) term the model behaves like a seasonal exponential smoother, so the weekly pattern is learned gradually rather than assumed fixed.")
    add("What is SARIMA?",
        "SARIMA(p,d,q)(P,D,Q)_s combines autoregressive and moving-average terms for the series and for its seasonal lags, with differencing; it captures autocorrelation and weekly seasonality in one linear model.",
        f"I used five small candidates with s = 7 and chose per series on validation error (AIC/BIC logged). SARIMA was the best model on validation at horizons {N.sarima_val_best_H} but ranked {N.sarima_rank[7]} / {N.sarima_rank[14]} / {N.sarima_rank[28]} of 9 on test at 7 / 14 / 28 days (test WAPE {f.wape('SARIMA', 7):.1f}% / {f.wape('SARIMA', 14):.1f}% / {f.wape('SARIMA', 28):.1f}%), "
        "illustrating validation optimism. Parameters are fixed after fitting and the Kalman state is updated each day (`extend`), verified identical to a full re-filter.",
        "```\nPhi(B^s) phi(B) (1-B)^d (1-B^s)^D y_t = c + Theta(B^s) theta(B) e_t,   s = 7\n```",
        "How did you avoid leakage when rolling the model?",
        "Parameters are estimated on the fitting window only; each day the state absorbs one new observation; a perturbation test shows forecasts at origin t are unchanged if later data are replaced.")
    add("What is exponential smoothing?",
        "A family of models whose forecast is a weighted average of past values with exponentially decaying weights: SES (level), Holt (level + trend), Holt-Winters (level + trend + seasonality).",
        f"I use SES, Holt with damped trend (undamped trends explode over 28 steps) and additive Holt-Winters (multiplicative is impossible with zeros). Parameters are small (median alpha {f.ets[(f.ets.phase == 'val') & (f.ets.model == 'SES')].alpha.median():.2f}): slow-adapting. "
        "A Holt-Winters -> Holt -> SES fallback exists (0 fallbacks occurred).",
        "```\nl_t = alpha*y_t + (1-alpha)*l_{t-1}                (SES: forecast = l_t)\nHolt (damped): b_t = beta*(l_t-l_{t-1}) + (1-beta)*phi*b_{t-1};  yhat_{t+h} = l_t + (phi+...+phi^h) b_t\nHW additive: yhat_{t+h} = l_t + (phi+..+phi^h) b_t + s_{t+h-7*ceil(h/7)}\n```",
        "Why is SES competitive here?",
        f"A plausible reason: demand is noisy with weak weekly seasonality (median seasonal strength {f.eda.seasonal_strength_Fs.median():.2f}), so a slowly adapting level (median SES alpha {f.ets[(f.ets.phase == 'val') & (f.ets.model == 'SES')].alpha.median():.2f}) is hard to beat; "
        f"{N.best[7]['best_by_test_WAPE']} had the lowest test WAPE at 7 days ({N.best[7]['runner_up_test_WAPE']} second, {abs(N.best[7]['runner_up_WAPE'] - N.best[7]['test_WAPE']):.2f} points behind).")
    add("What is Croston's method and why SBA?",
        "Croston forecasts intermittent demand by smoothing demand sizes and inter-demand intervals separately; the Syntetos-Boylan correction removes its bias.",
        f"{int((N.sel.train_adi >= C.ADI_CUTOFF).sum())} of 18 series are intermittent/lumpy (ADI >= 1.32), so I included Croston-SBA (alpha = 0.1) as a baseline. It did not beat ETS or the moving average here "
        f"(test WAPE {f.wape('Croston', 7):.1f}% / {f.wape('Croston', 28):.1f}% at 7 / 28 days).",
        "```\nz_t = z_{t-1} + a*(y_t - z_{t-1}),  p_t = p_{t-1} + a*(q_t - p_{t-1})   (only when y_t > 0)\nforecast = (1 - a/2) * z_t / p_t           (SBA bias correction)\n```",
        "Why did it not win?",
        "Its constant-rate forecast ignores the level shifts and the weekly/SNAP pattern, and 28-day sums are far less intermittent than daily values.")

    L += ["", "## MACHINE LEARNING", ""]
    add("Why XGBoost?",
        "It handles non-linear effects and interactions between lags, calendar and price on tabular data, is fast and robust with little tuning, and has an exact, fast explainer (TreeSHAP). Deep learning was unnecessary for 18 series.",
        "Gradient-boosted trees were the backbone of the M5 winners. I use one *global* model per horizon (pooled over series) so splits are shared. Honest result: it is competitive, not dominant - "
        + (("best pooled test WAPE at " + " and ".join(f"{H} days ({f.wape('XGBoost', H):.1f}%)" for H in N.xgb_best_primary)) if N.xgb_best_primary else "never the best pooled test WAPE at 7/14/28 days")
        + f", but {'never' if N.xgb_never_better else 'only sometimes'} significantly better than the other non-naive models in the Diebold-Mariano tests"
        + (f", and significantly worse than {', '.join(N.worse7)} at 7 days" if N.worse7 else ", and not significantly different from them at 7 days")
        + (f" (and than {', '.join(N.worse_long)} at 14/28 days)." if N.worse_long else "."),
        "```\nF(x) = sum_{m=1..M} eta * f_m(x),   f_m = regression tree fitted to the gradient of the squared-error loss\n```",
        "Why not a neural network?",
        "No compelling methodological reason: small data, a need for explainability and a risk of unreproducible gains. I would revisit with thousands of series.")
    add("Why regression?",
        "The target is a continuous non-negative quantity (sum of units), so regression with a squared-error objective gives the conditional mean, which is what inventory planning needs.",
        "Squared error estimates E[Target | features]; MAE would estimate the median; quantile loss would give a service-level quantile (an extension I recommend). Predictions are clipped at zero.",
        "```\nminimise sum_i (Target_i - F(x_i))^2   ->   F(x) ~ E[Target | x]\n```",
        "Would a Tweedie/Poisson objective be better?",
        "Plausibly for intermittent counts; I kept squared error for interpretability, consistency with RMSE and SHAP in units; it is a natural next experiment.")
    add("How were the lag features created?",
        "As the sales k days before the first forecast day: lag_1 is the last observed day, lag_7 the same weekday last week, up to lag_28.",
        "Row = (series, forecast origin t). lag_k(t) = y[t+1-k], so lags are measured from the first forecast day t+1. A unit test compares 200 random rows with explicit loops; the audit re-checks 400 rows on the real panel.",
        "```\nlag_k(t) = y[t + 1 - k],  k in {1, 7, 14, 21, 28}\n```",
        "Why measure lags from t+1 rather than t?",
        "Because then lag_7 aligns with the weekday of the first forecast day (same as seasonal naive) and the origin day's own value is available, which the rolling windows also include.")
    add("What are rolling features?",
        "Summary statistics of the most recent window: mean and standard deviation over the last 7/14/28 days, ending at the forecast origin.",
        "rolling_mean_w(t) = mean(y[t-w+1..t]) and the sample standard deviation (ddof = 1), computed with exact sliding windows. They carry level and volatility; SHAP shows they "
        f"{'dominate' if min(N.roll.values()) > 50 else 'matter for'} the model ({N.roll[7]:.0f}% of attribution at 7 days, {N.roll[28]:.0f}% at 28 days).",
        "```\nrolling_mean_w(t) = (1/w) * sum_{j=0..w-1} y[t-j]        rolling_std_w(t) = sqrt( sum (y[t-j]-mean)^2 / (w-1) )\n```",
        "Why is the std useful?",
        "It tells the model how volatile the item is, which matters when demand is lumpy. "
        f"SHAP: rolling_std_28 carries {100 * float(f.shap[(f.shap.H == 28) & (f.shap.feature == 'rolling_std_28')]['share_of_total'].iloc[0]):.1f}% of the 28-day model's attribution "
        f"(correlation of feature value and SHAP {float(f.shap[(f.shap.H == 28) & (f.shap.feature == 'rolling_std_28')]['corr(feature value, shap)'].iloc[0]):+.2f}) but "
        f"{100 * float(f.shap[(f.shap.H == 7) & (f.shap.feature == 'rolling_std_28')]['share_of_total'].iloc[0]):.1f}% at 7 days (correlation {float(f.shap[(f.shap.H == 7) & (f.shap.feature == 'rolling_std_28')]['corr(feature value, shap)'].iloc[0]):+.2f}). "
        f"For count data the standard deviation rises with the mean (median within-series correlation of the two 28-day windows {N.corr_mean_std:.2f}), so credit is shared with the level and I do not read it as a separate effect.")
    add("How can rolling features cause leakage?",
        "If the window includes future days (centred windows, off-by-one) or statistics are computed on the whole series, the feature secretly contains the target.",
        "Typical bugs: centred rolling windows (t-3..t+3), forgetting to shift so that today's value predicts today, scaling or encoding with global statistics, building targets whose windows overlap the validation period. "
        "I wrote a negative-control unit test: a deliberately centred-window feature is caught by the perturbation test, which proves the audit has power.",
        "```\nSAFE:    mean(y[t-6..t])      LEAKY:  mean(y[t-3..t+3])  (uses y[t+1], y[t+2], y[t+3])\n```",
        "How do you know there is no leakage in your final model?",
        f"{int(f.audit.status.eq('PASS').sum())}/{len(f.audit)} automated checks pass, including perturbation tests: replace all data after a random cut-off with wild values and verify that features, model forecasts, sigma and policy decisions up to the cut-off are identical.")
    add("How did you prevent leakage overall?",
        "Chronological split, features that only use data up to the forecast origin, training rows cut so their targets end before the next period, hyper-parameters chosen on validation only, sigma from validation residuals, and automated perturbation tests.",
        "(1) Split by date; (2) history features end at the origin; (3) the only features with later timestamps are calendar counts (SNAP, events, Christmas) that are deterministic functions of the published calendar and verified independent of sales; "
        "(4) XGBoost rows satisfy origin + H <= fit_end - 1; (5) ETS/SARIMA/XGBoost are refit on train+validation only after selection; (6) the test period is evaluated once; (7) the inventory policy uses only information at the end of day t.",
        None, "Is using future calendar information not leakage?",
        "No: holidays and SNAP days are published long in advance, exactly like a retailer's planning calendar; I documented them as 'known-ahead' features and tested that they do not depend on sales. Using *future prices or sales* would be leakage.")
    add("Direct or recursive multi-step forecasting?",
        "Direct: one model per horizon, trained on the target 'sum of the next H days'. It avoids error accumulation and matches the inventory need.",
        f"Recursive forecasting would feed predictions back as lags and compound errors; for sums it is also awkward. Direct models cost more training runs (4 here) but each is tuned for its horizon (chosen max_depth by horizon: {N.depths}).",
        None, "Why not one model with horizon as a feature?",
        "Possible, and it shares data across horizons, but separate models are easier to explain and tune; with 4 horizons the cost is small.")
    add("How did you tune XGBoost without overfitting?",
        "Modestly and only on validation: 24 configurations per horizon (default + random draws), early stopping on validation MAE, then a refit on train+validation with the chosen number of trees. The test period was never touched.",
        f"Validation MAE improved by {', '.join(f'{100 * (1 - g.val_MAE.min() / g[g.is_default].val_MAE.iloc[0]):.1f}% (H={H})' for H, g in f.xgb_log.groupby('H'))} versus the default configuration, so the tuning gains were {'modest' if max(N.tune_gain) < 10 else 'material'}; "
        "the validation score of the winner is optimistic because it is the minimum of 24 draws.",
        None, "Why not cross-validation?",
        "Time-series CV needs expanding windows with gaps; with one pooled model and a few thousand rows a single chronological hold-out is simpler and sufficient - I accept the extra variance.")

    L += ["", "## EVALUATION", ""]
    add("Why MAE?",
        "It is in the units of demand, easy to explain, and robust to occasional spikes. It is optimal for the median.",
        f"MAE answers 'on average how many units am I off?'. At 28 days pooled test MAE is {f.metric('XGBoost', 28, 'MAE'):.1f} units for XGBoost and {f.metric('MovingAverage', 28, 'MAE'):.1f} for the 28-day moving average.",
        "```\nMAE = (1/n) * sum |y_i - f_i|\n```",
        "Why do you also report RMSE?",
        "Because the two can rank models differently when spikes matter - and in this project they do.")
    add("Why RMSE?",
        "It squares errors, so it punishes large misses (the demand spikes that cause stockouts) and is the natural scale for safety stock, since sigma is a root-mean-square error.",
        f"RMSE rewards the conditional mean. Rankings differ from MAE: at 7 days the best RMSE is {N.best[7]['best_by_test_RMSE']} while the best WAPE is {N.best[7]['best_by_test_WAPE']}. "
        "My sigma_L is the RMSE of validation errors, so a biased model gets a bigger safety stock.",
        "```\nRMSE = sqrt( (1/n) * sum (y_i - f_i)^2 ) ;  RMSE^2 = Var(error) + Bias^2\n```",
        "When is RMSE misleading?",
        f"When a few huge misses dominate: test RMSE is {N.ratio.r.min():.1f}-{N.ratio.r.max():.1f}x validation RMSE while MAE grows only {N.mae_ratio[0]:.1f}-{N.mae_ratio[1]:.1f}x. "
        f"At 7 days the worst 5% of windows produce {100 * N.conc7['test']['worst5'][0]:.0f}-{100 * N.conc7['test']['worst5'][1]:.0f}% of the test squared error against {100 * N.conc7['val']['worst5'][0]:.0f}-{100 * N.conc7['val']['worst5'][1]:.0f}% in validation"
        + (f"; the lost units of the forecast-driven policy are concentrated in November-December ({N.nd_b:.0f}% of them in {N.nd_days:.0f}% of the test days)." if N.nd_b > 1.25 * N.nd_days else "."))
    add("Why sMAPE?",
        "It is a scale-free percentage error that is bounded and handles zeros better than MAPE. I report it but do not rely on it.",
        f"sMAPE = mean(200*|e|/(|y|+|f|)) with 0/0 := 0. Weakness for intermittent data: a forecast of exactly zero when actual is zero scores perfectly, so methods that output exact zeros collect free perfect scores - at 28 days the best sMAPE belongs to {N.sz['model']} ({N.sz['smape']:.0f}%), which ranks {N.sz['wape_rank']} of {N.sz['n_models']} on WAPE; it forecasts exactly zero in {100 * N.sz['share_zero_forecast']:.1f}% of the 28-day test windows (the actual is zero in {100 * N.sz['share_actual_zero']:.1f}%) whereas {', '.join(N.sz['never_zero']) or 'no model'} never do.",
        "```\nsMAPE = (100/n) * sum 2|y_i - f_i| / (|y_i| + |f_i|)      (0/0 := 0; range 0..200)\n```",
        "Then why report it?",
        "Because it is standard in forecasting competitions and lets readers compare; I flag its bias towards zero forecasts.")
    add("Why WAPE?",
        "It is total absolute error divided by total actual demand - a volume-weighted, zero-safe percentage that business people understand ('we miss about a third of volume').",
        f"WAPE = MAE / mean demand. It is my headline metric: {N.best[7]['test_WAPE']:.1f}% / {N.best[14]['test_WAPE']:.1f}% / {N.best[28]['test_WAPE']:.1f}% for the best models at 7/14/28 days. Because it is volume-weighted it favours models that do well on the high-volume series (best 28-day model there: {N.tier_best[28]['tier=high']}), "
        "which is why I also run an equal-weighted DM test.",
        "```\nWAPE = 100 * sum |y_i - f_i| / sum |y_i|\n```",
        "What if total demand is zero?",
        "WAPE is undefined for that group (the code returns a missing value instead of a number); I only compute it on pooled groups with positive demand.")
    add("Why not MAPE?",
        "MAPE divides by the actual, which is zero on many days (68% of cells), so it is undefined or explosive, and it penalises over-forecasts more than under-forecasts.",
        "|y - f|/|y| is unbounded for over-forecasts and capped at 100% for under-forecasts, which biases model selection toward under-forecasting; sMAPE/WAPE/MASE avoid division by small actuals.",
        "```\nMAPE = (100/n) * sum |y_i - f_i| / |y_i|     undefined when y_i = 0\n```",
        "Would MASE be better?",
        "MASE (scaled by in-sample naive error) is a good scale-free alternative and I would add it; WAPE was chosen for business readability.")
    add("Why chronological validation?",
        "Because time series are ordered and autocorrelated: shuffling lets the model see the future, inflating accuracy. Chronological splits mimic real deployment.",
        f"Train {sp.date_range('train')[0].date()}-{sp.date_range('train')[1].date()}, validation {sp.date_range('validation')[0].date()}-{sp.date_range('validation')[1].date()}, test {sp.date_range('test')[0].date()}-{sp.date_range('test')[1].date()}. "
        f"Within each period I use *rolling origins* (every day), so each model is scored on {min(len(sp.eval_origins(ph, H)) for ph in ('validation', 'test') for H in C.HORIZONS)}-{max(len(sp.eval_origins(ph, H)) for ph in ('validation', 'test') for H in C.HORIZONS)} overlapping windows per series. Overlap means the effective sample is about n/H, which the DM test accounts for with a HAC variance (lag H-1).",
        "```\nOrigins t = a-1 ... b-1-H  (window t+1..t+H inside the period)\nDM = mean(d) / sqrt(LRV(d)/n),  d_t = L(e_A,t) - L(e_B,t),  LRV with lag H-1\n```",
        "What did the DM test tell you?",
        f"XGBoost is {'not significantly better' if N.xgb_never_better else 'sometimes significantly better'} than ETS/SARIMA/MA at 14/28 days (smallest Holm p = {min(f.dm_xgb(H, m)['p_holm'] for H in (14, 28) for m in ['MovingAverage', 'SES', 'Holt', 'HoltWinters', 'SARIMA']):.2f}) and is "
        + (f"significantly worse than {', '.join(N.worse7)} at 7 days; " if N.worse7 else "not significantly different from them at 7 days; ")
        + (f"{' and '.join(N.beaters_of_naive)} beat{'s' if len(N.beaters_of_naive) == 1 else ''} naive and seasonal naive significantly at all three horizons." if N.beaters_of_naive else "neither reference model beats both naive baselines significantly at every horizon."))

    L += ["", "## EXPLAINABILITY", ""]
    add("What is SHAP?",
        "SHAP assigns each feature a contribution to one prediction using Shapley values from game theory, so contributions add up exactly to the difference between the prediction and the average prediction.",
        "For trees, TreeSHAP computes them exactly and fast. I explain the final train+validation models on test-period origins (4,000 sampled rows) and verify additivity (max error "
        f"{f.shap_chk.additivity_max_abs_error.max():.1e}). Findings: rolling means dominate ({N.roll[7]:.0f}% at 7 days); SNAP-day counts {N.snap:.1f}%; price {N.price:.1f}%.",
        "```\nphi_i = sum_{S subset F\\{i}} |S|! (|F|-|S|-1)! / |F|!  *  [ f(S u {i}) - f(S) ]\nsum_i phi_i + base_value = prediction\n```",
        "Why SHAP rather than feature importance?",
        "Impurity importances are global and biased; SHAP is local (per prediction), additive and consistent, and supports dependence plots and individual waterfalls.")
    add("What does a positive SHAP value mean?",
        "It means that, for that prediction, the feature's value pushes the forecast above the average prediction (the base value); a negative value pushes it below.",
        f"Example: for the 7-day model the base value is about {f.shap_chk.base_value.iloc[0]:.1f} units; a high 28-day rolling mean gives a large positive SHAP (correlation of feature value and SHAP {f.shap[(f.shap.H == 7) & (f.shap.feature == 'rolling_mean_28')]['corr(feature value, shap)'].iloc[0]:.2f}). "
        "The magnitude is in units of the forecast.",
        "```\nprediction = base_value + sum_i phi_i ;   phi_i > 0 raises, phi_i < 0 lowers the prediction\n```",
        "Can a positive SHAP value be read as 'the feature increases demand'?",
        "No - it describes how the *model* uses the feature, not what happens in the world if the feature is changed.")
    add("Is SHAP causal?",
        "No. SHAP explains the fitted model's behaviour. Correlated features share credit, and a feature can proxy for something unobserved.",
        f"Example in this project: `price` has a negative SHAP relation (corr {f.shap[(f.shap.H == 7) & (f.shap.feature == 'price')]['corr(feature value, shap)'].iloc[0]:+.2f}), but item prices change rarely (median {f.eda.n_price_changes.median():.0f} changes in TRAIN), so it "
        + (f"most likely separates cheap, fast-selling items from expensive, slow ones (Spearman correlation of mean price and mean sales across the 18 series {N.rho_price_level:+.2f})" if N.rho_price_level < -0.3
           else f"may reflect between-item differences (Spearman correlation of mean price and mean sales across the 18 series {N.rho_price_level:+.2f})")
        + " - not a price elasticity. Causal claims would need experiments or causal modelling.",
        None, "How would you estimate price elasticity?",
        "With within-item price variation and controls (or a randomised price test) - e.g. a log-log regression with item fixed effects - rather than from SHAP.")

    L += ["", "## INVENTORY", ""]
    add("What is safety stock?",
        "Extra stock held to protect against uncertainty in demand (and lead time) during the replenishment lead time.",
        "SS = z * sigma_L, where sigma_L is the standard deviation of the lead-time demand forecast error. In my Policy B, sigma_L is the RMSE of the model's out-of-sample (validation) errors of the L-day sum; "
        "in Policy A it is the textbook sqrt(L) x trailing daily std. Using real forecast errors captures autocorrelation that the i.i.d. formula misses.",
        "```\nSS = z * sigma_L ;   i.i.d. daily demand:  sigma_L = sqrt(L) * sigma_daily\n```",
        "What did using validation errors change?",
        (f"With the same 28-day-mean point forecast as Policy A, only the sigma estimate changed and total cost moved {sgn(N.ma_gain7)} at L=7 (95%, MEDIUM) - {100 * N.ma_gain7 / N.g[7]:.0f}% of the headline change ({sgn(N.g[7])})."
         if N.same_pf7 else
         f"The moving-average model (window {f.ma_windows()[7]} days at H = 7 versus Policy A's 28) moved total cost {sgn(N.ma_gain7)} at L=7 (95%, MEDIUM) against {sgn(N.g[7])} for the headline model, "
         "so the sigma estimate and the point forecast cannot be separated cleanly."))
    add("What is the reorder point (ROP)?",
        "The inventory position at which you place a new order: expected demand during the lead time plus safety stock.",
        "I check the inventory position (on hand + on order) at the end of each day and order up to S = ROP + Q when it falls to the ROP. Orders arrive L+1 days after the end-of-day order so the protection interval is exactly the L forecast days.",
        "```\nROP_t = mu_L(t) + z * sigma_L ;  order if IP_t <= ROP_t, quantity = ceil(ROP_t + Q_t - IP_t)\n```",
        "Why inventory position and not on-hand?",
        "Because outstanding orders already cover part of the demand; using on-hand alone would order again and again during the lead time.")
    add("Why use z?",
        "z converts a target service level (a probability) into a number of standard deviations of safety stock, assuming the lead-time error is roughly normal.",
        "z is the normal quantile: 90% -> 1.282, 95% -> 1.645, 98% -> 2.054 (prescribed in the project). Normality is an approximation: for the H-day sums the standardised validation residuals have skewness "
        f"{f.norm[f.norm.model.isin(['XGBoost', 'SARIMA', 'MovingAverage'])].pooled_skew.min():.2f} to {f.norm[f.norm.model.isin(['XGBoost', 'SARIMA', 'MovingAverage'])].pooled_skew.max():.2f}; the larger problem was that sigma estimated on validation understated test errors, "
        f"so the realised cycle service level of a nominal 95% was {100 * N.b95.Cycle_Service_Level:.0f}% (B) / {100 * N.a95.Cycle_Service_Level:.0f}% (A).",
        "```\nP(D_L <= mu_L + z*sigma_L) = alpha  =>  z = Phi^{-1}(alpha)\nz(90%) = 1.282, z(95%) = 1.645, z(98%) = 2.054\n```",
        "Do empirical quantiles fix that?",
        f"Not systematically: I ran the empirical-quantile variant for every model, lead time and service level ({len(N.emp)} cells). Total cost changed by a median of {N.emp.cost_pct.abs().median():.1f}% (range {N.emp.cost_pct.min():+.1f}% to {N.emp.cost_pct.max():+.1f}%), "
        f"fill rate by at most {N.emp.fill_pp.abs().max():.1f} points and cycle service level by a median of {N.emp.cycle_pp.abs().median():.1f} points (mean change {N.emp.cycle_pp.mean():+.1f}), so it did not repair the shortfall - the dominant issue is the shift in error size between validation and test.")
    add("What is service level?",
        "The probability (or share) of demand you can satisfy from stock. It is a design target, not a guarantee.",
        f"I report several: cycle service level (share of replenishment cycles without a stockout - what z targets), fill rate (units served / units demanded), and in-stock day rate. At 95% target and L=7: cycle service level {100 * N.a95.Cycle_Service_Level:.0f}% (A) / {100 * N.b95.Cycle_Service_Level:.0f}% (B), "
        f"fill rate {100 * N.a95.Fill_Rate:.1f}% / {100 * N.b95.Fill_Rate:.1f}%, in-stock days {100 * N.a95.In_Stock_Day_Rate:.1f}% / {100 * N.b95.In_Stock_Day_Rate:.1f}%.",
        "```\nType-1 (cycle) SL = P(no stockout in a cycle) ;  Type-2 (fill rate) = 1 - E[shortage per cycle] / E[demand per cycle]\n```",
        "Which should the business use?",
        "Fill rate is closer to customer experience and cost; cycle service level is what the z-formula controls. Choose from the cost of a lost sale, then monitor the realised value.")
    add("What is EOQ?",
        "The Economic Order Quantity balances ordering cost against holding cost to minimise total cost per unit time: Q* = sqrt(2 D S / H).",
        "Derivation: cost per year = (D/Q)*S + (Q/2)*H; minimising over Q gives Q*. I use D = 365 x (28-day forecast)/28, S = $1.00 per order, H = 25% of unit cost per year (hypothetical), "
        f"giving cycle lengths of {f.eoq.eoq_cycle_days.min():.0f}-{f.eoq.eoq_cycle_days.max():.0f} days on TRAIN data. Assumptions: constant demand, no shortages in the EOQ itself - the ROP/safety stock handles uncertainty.",
        "```\nTC(Q) = (D/Q)*S + (Q/2)*H   =>   dTC/dQ = 0   =>   Q* = sqrt(2*D*S/H)\n```",
        "Why EOQ with variable demand?",
        "It is a standard (s, S)/(s, Q) building block; with a forecast that updates daily, Q* updates too. A more exact approach would optimise (s, S) numerically or use a cost-aware simulation.")
    add("What happens when lead time increases?",
        "Both the expected lead-time demand and its uncertainty grow, so the reorder point and safety stock rise and total cost increases.",
        f"mu_L grows ~ L and sigma_L ~ sqrt(L) for i.i.d. demand (faster if errors are autocorrelated). In the simulation the mean total cost of Policy B (averaged over other factors) is "
        f"{f.sens[(f.sens.policy_group == 'B_forecast_driven') & (f.sens.L == 3)].total_cost.mean():,.0f} / {f.sens[(f.sens.policy_group == 'B_forecast_driven') & (f.sens.L == 7)].total_cost.mean():,.0f} / {f.sens[(f.sens.policy_group == 'B_forecast_driven') & (f.sens.L == 14)].total_cost.mean():,.0f} for L = 3 / 7 / 14 days "
        f"- a {f.factor_ranges()['lead time']:.0f}% range, versus {f.factor_ranges()['service level']:.0f}% across the three service levels.",
        "```\nmu_L = L * mu_d ;  sigma_L = sqrt(L) * sigma_d (i.i.d.) ;  ROP = mu_L + z*sigma_L  -> increases with L\n```",
        "What can a retailer do about it?",
        "Shorten or stabilise the lead time (supplier agreements, closer sourcing). "
        + (f"In my results the lead-time range of mean cost ({f.factor_ranges()['lead time']:.0f}%) exceeded both the service-level range ({f.factor_ranges()['service level']:.0f}%) and the largest cost difference between the forecast-driven and the historical policy ({max(abs(x) for x in N.g.values()):.0f}%)."
           if f.factor_ranges()['lead time'] > max(f.factor_ranges()['service level'], max(abs(x) for x in N.g.values())) else "In my results the lead-time effect was not larger than the other levers."))
    add("What happens when the service level increases?",
        "z rises, safety stock and holding cost increase, and stockouts fall - with diminishing returns because z grows faster at high service levels.",
        f"z goes 1.282 -> 1.645 -> 2.054 for 90/95/98%; the extra safety stock per point of service increases. Whether it pays depends on the stockout penalty: in my simulation the cost-minimising level on the 90/95/98% grid is: {f.optimal_sl_text()}. "
        f"Mean total cost of Policy B moves only {100 * (1 - f.sens[(f.sens.policy_group == 'B_forecast_driven') & (f.sens.SL == 0.98)].total_cost.mean() / f.sens[(f.sens.policy_group == 'B_forecast_driven') & (f.sens.SL == 0.9)].total_cost.mean()):.0f}% from 90% to 98%.",
        "```\nSS(alpha) = z(alpha) * sigma_L ;  d SS / d alpha = sigma_L / phi(z)  (rises steeply as alpha -> 1)\n```",
        "How would you pick the service level?",
        "From the critical ratio: alpha* ~ p / (p + h*T_cycle) where p is the cost of a lost unit and h*T_cycle the holding cost of a leftover unit for one cycle - then verify by simulation and monitor the realised level.")

    L += ["", "## BUSINESS", ""]
    add("Why does better forecasting not necessarily mean lower inventory cost?",
        "Because forecast accuracy metrics are symmetric and average over all periods, while inventory cost is asymmetric (lost sales vs holding) and driven by bias and tail errors during lead times, plus how uncertainty is turned into safety stock.",
        f"In my results the lowest-WAPE model was the cheapest in only {N.same_best} of {len(f.avc_s)} cells; the Spearman correlation between error rank and cost rank was {N.rho['LOW']:+.2f} (LOW penalty), {N.rho['MEDIUM']:+.2f} (MEDIUM), {N.rho['HIGH']:+.2f} (HIGH). Reasons: "
        "(1) asymmetric costs favour quantile-type forecasts; (2) noisy forecasters get large sigma and hold more stock; "
        f"(3) bias ({'all models under-forecast on average in the test year' if f.under_forecast_cells()[0] == f.under_forecast_cells()[1] else str(f.under_forecast_cells()[0]) + ' of ' + str(f.under_forecast_cells()[1]) + ' model-horizon cells under-forecast on average in the test year'}) "
        f"goes with lost sales (Spearman {f.bias_vs_lost(7)[0]:+.2f} between test bias and lost units across models at L=7, p = {f.bias_vs_lost(7)[1]:.3f}; an association across {N.n_nonnaive} non-naive models, not proof); "
        + (f"(4) sigma estimated on validation is optimistic for the models with the smallest validation sigma ({', '.join(f.winners_curse(7)[1])}), which are also the three with the largest test/validation RMSE ratio at 7 days (a winner's-curse pattern, not proof of cause)."
           if f.winners_curse(7)[0] else f"(4) validation-based sigma {'understates' if N.ratio.r.min() > 1 else 'can misstate'} test-period errors (test RMSE is {N.ratio.r.min():.2f}-{N.ratio.r.max():.2f}x validation RMSE)."),
        "```\nOptimal order-up-to for asymmetric cost (newsvendor): S* = F^{-1}( p / (p + h) )   -> a QUANTILE of demand, not the mean\n```",
        "So should you ignore accuracy?",
        "No - accuracy is necessary, but the objective should be the decision: evaluate with the inventory simulation, train quantile/cost-aware models, and control bias.")
    add("What assumptions did you make?",
        "Hypothetical costs and lead times, lost sales, constant lead time, no minimum order size or perishability, observed sales as demand, independence across items, constant sigma over the test period.",
        f"Costs: unit cost 70% of price, holding 25%/year, order cost $1, stockout penalty 0.25x / 1x / 3x unit margin (LOW/MEDIUM/HIGH); lead times 3/7/14 days; initial inventory cover L+14 days. EOQ cycles on TRAIN data: {f.eoq.eoq_cycle_days.min():.0f}-{f.eoq.eoq_cycle_days.max():.0f} days; "
        "the critical-ratio argument that set the scenarios used TRAIN data only. I never present these as real business costs.",
        None, "How sensitive are conclusions to them?",
        "Strongly for dollar amounts, less so for direction: B is cheaper than A in "
        f"{N.n_cheaper} of {N.n_cfg} grid cells; the factors rank by how much they move the mean cost of B: "
        + ", then ".join(f"{k} ({v:.0f}%)" for k, v in sorted(f.factor_ranges().items(), key=lambda kv: -kv[1])) + ".")
    add("What would you change in a real company?",
        "Use real costs and lead-time distributions, stockout/inventory records to handle censored demand, promotions and price plans as drivers, rolling re-estimation of sigma, quantile/cost-aware forecasts and a controlled pilot before rollout.",
        "Concretely: (1) estimate the cost of a lost sale from margin and substitution data; (2) model stochastic lead times; (3) re-estimate sigma weekly on rolling errors and monitor realised service; (4) train quantile models for the lead-time demand; "
        "(5) respect pack sizes/MOQs/shelf life; (6) forecast hierarchically (store-department-item) to help intermittent items; (7) A/B-test the policy on a subset of stores.",
        None, "What is the first thing you would do?",
        (lambda pen, med, hi: f"Measure the real stockout cost and lead-time distribution: in my simulation moving between the LOW and HIGH penalty scenario changed the mean total cost of Policy B by {pen:.0f}%, whereas the choice among the non-naive forecasting models "
                              f"changed it by a median of {med:.0f}% (MEDIUM penalty; up to {hi:.0f}% with the HIGH penalty)" + (" - the penalty assumption mattered more than the model choice." if pen > hi else "."))
        (100 * float(f.factor[(f.factor.policy_group == 'B_forecast_driven') & f.factor.factor.str.startswith('stockout-penalty')].relative_range_of_mean_total_cost.iloc[0]),
         f.model_cost_spread('MEDIUM')[0], f.model_cost_spread('HIGH')[1]))
    add("What are the limitations of the project?",
        "Hypothetical costs, censored demand, only 18 series and one test year, and wide uncertainty on the inventory gain. " + ("The ML model did not clearly beat simple methods." if N.xgb_never_better else "The ML model's advantage over simple methods was only partial."),
        f"Specific numbers: bootstrap intervals for B vs A include zero in {N.n_cfg - N.n_ci_excl0} of {N.n_cfg} cells; the test year was harder than validation for {'every' if N.ratio.r.min() > 1 else 'most'} model (RMSE ratio {N.ratio.r.min():.2f}-{N.ratio.r.max():.2f}x); "
        f"the model with the best validation WAPE was the cheapest on test in {f.val_choice_cheapest()[0]} of {f.val_choice_cheapest()[1]} cells (median regret {f.val_choice_cheapest()[2]:.0f}% with the MEDIUM penalty); "
        f"the realised cycle service level at the 95% target and L = 7 was {100 * N.b95.Cycle_Service_Level:.0f}% (B) / {100 * N.a95.Cycle_Service_Level:.0f}% (A). Mitigations: chronological protocol, leakage audit, honest reporting.",
        None, "What are you most proud of?",
        "The discipline: a leakage audit with a negative control, a frozen protocol, " + ("reporting results that went against the 'ML wins' narrative, " if N.xgb_never_better else "reporting results as they came out, ") + "and a clear answer that accuracy is not business value.")
    add("Why did you use the Diebold-Mariano test?",
        "To check whether differences in forecast accuracy between two models are statistically meaningful rather than noise.",
        "It tests whether the mean loss differential is zero, with a long-run variance that accounts for overlapping forecast errors (lag H-1), the Harvey-Leybourne-Newbold small-sample correction and Holm adjustment across comparisons. "
        "I aggregate across series with equal weights (scaled losses) to respect cross-sectional dependence. Power is low at long horizons because the effective sample is about n/H.",
        "```\nd_t = |e_A,t|/scale - |e_B,t|/scale ;  DM = mean(d)/sqrt(LRV/n) ;  HLN: DM* = DM * sqrt((n+1-2H+H(H-1)/n)/n),  t_{n-1}\n```",
        "Why can XGBoost lead on WAPE but not on DM?",
        f"WAPE is volume-weighted (busy series dominate) while the DM panel loss gives each series equal weight. By group, the best 28-day model is {N.tier_best[28]['tier=high']} for the {N.n_grp['tier=high']} high-volume series, "
        f"{N.tier_best[28]['tier=medium']} for the {N.n_grp['tier=medium']} medium and {N.tier_best[28]['tier=low']} for the {N.n_grp['tier=low']} low-volume ones ({N.tier_best[28]['regular']} for the {N.n_grp['regular']} regular and "
        f"{N.tier_best[28]['intermittent']} for the {N.n_grp['intermittent']} intermittent series), so the ranking depends on which series carry the weight.")

    (C.REPORTS / "interview_guide.md").write_text("\n".join(L) + "\n")


def _resume_numbers(f: Facts) -> dict:
    N = Numbers(f)
    return {"N": N}


def build_project_story(f: Facts) -> None:
    N = Numbers(f)
    sp = f.split
    best = N.best
    ma_w28 = f.wape("MovingAverage", 28)
    sn_w28 = f.wape("SeasonalNaive", 28)
    p7 = f.pair[(f.pair.scenario == "MEDIUM") & (f.pair.L == 7) & (f.pair.SL == 0.95)].iloc[0]
    n_checks = int(f.audit.status.eq("PASS").sum())
    cnt_sar = f.dm_counts("SARIMA")
    indist_14_28 = not any(N.cnt[H]["better"] or cnt_sar[H]["better"] for H in (14, 28))
    sigma_share = 100 * N.ma_gain7 / N.g[7] if N.g[7] != 0 else float("nan")
    sigma_main = N.same_pf7 and np.isfinite(sigma_share) and sigma_share > 50
    ri = f.rob_i.set_index("group")
    L = ["# Project story, interview explanations and resume bullets", "",
         "*All numbers are generated from `outputs/tables/`; costs are hypothetical.*", "",
         "## 60-second explanation", "",
         f"**PROBLEM** - Retailers need to decide how much stock to hold; I asked whether accurate, explainable demand forecasts really lead to better inventory decisions. "
         f"**DATA** - Walmart's public M5 data: I modelled 18 representative item-store series (3 stores, high/medium/low demand, mostly intermittent) over {sp.n_days:,} days. "
         f"**METHOD** - I compared baselines, exponential smoothing, SARIMA and XGBoost on 7/14/28-day demand, then fed forecasts into a reorder-point / EOQ policy simulated day by day. "
         f"**VALIDATION** - strictly chronological with rolling origins, leakage-audited ({n_checks} automated checks), choices on validation and one test evaluation. "
         f"**RESULT** - best test WAPE was {best[7]['test_WAPE']:.1f}% / {best[14]['test_WAPE']:.1f}% / {best[28]['test_WAPE']:.1f}% at 7/14/28 days ({best[7]['best_by_test_WAPE']}, {best[14]['best_by_test_WAPE']}, {best[28]['best_by_test_WAPE']}), but the leading models were statistically hard to separate. "
         f"**BUSINESS IMPACT** - the forecast-driven policy {N.lw(N.g[7])} at a 7-day lead time versus a historical baseline (hypothetical costs, wide interval)"
         + (", mostly through better uncertainty estimates, " if sigma_main else ", ")
         + (f"and the most accurate model was usually not the cheapest ({N.same_best} of {len(f.avc_s)} cells) - so I optimise the decision, not just the forecast." if N.same_best < len(f.avc_s) / 2
            else f"and the most accurate model was also the cheapest in {N.same_best} of {len(f.avc_s)} cells."), "",
         "## 2-minute explanation", "",
         "**Problem.** Better forecasts only matter if they improve the stocking decision: too little stock loses sales, too much ties up capital. My question was whether forecasts can be converted into better inventory decisions by balancing stockout risk and holding cost.", "",
         f"**Data.** The M5 Walmart dataset (30,490 item-store series, daily, with prices and calendar/SNAP). The official download needs Kaggle credentials, so I used two verified public mirrors (identical SHA-256). I picked a reproducible sample using training information only: one store per state, three demand tiers, "
         f"seeded draws -> 18 series; {int((N.sel.train_adi >= C.ADI_CUTOFF).sum())} are intermittent or lumpy, and zero sales cannot be told apart from stockouts, which I state as a key limitation.", "",
         "**Method.** Everything is evaluated on the same target - the sum of demand over the next 7/14/28 days from daily forecast origins - with baselines (naive, seasonal naive, moving average, Croston), SES/Holt/Holt-Winters, SARIMA and a global XGBoost with leakage-safe lag, rolling, calendar, SNAP and price features. "
         "Uncertainty comes from validation residuals. The decision layer is a reorder-point + EOQ policy with safety stock z*sigma, simulated chronologically with order lead times of 3/7/14 days, comparing a historical baseline (A) with a forecast-driven policy (B).", "",
         f"**Validation.** Train/validation/test by date ({sp.date_range('train')[0].date()}-{sp.date_range('train')[1].date()}, then {sp.date_range('validation')[0].date()}-{sp.date_range('validation')[1].date()}, then {sp.date_range('test')[0].date()}-{sp.date_range('test')[1].date()}); all selections on validation, refit on train+validation, test evaluated once. "
         f"{n_checks} automated leakage checks, including perturbation tests and a negative control, all pass.", "",
         f"**Results.** Test WAPE: {best[7]['best_by_test_WAPE']} {best[7]['test_WAPE']:.1f}% (7 days), {best[14]['best_by_test_WAPE']} {best[14]['test_WAPE']:.1f}% (14), {best[28]['best_by_test_WAPE']} {best[28]['test_WAPE']:.1f}% (28); the 28-day moving average gets {ma_w28:.1f}% and seasonal naive {sn_w28:.1f}%. "
         f"Diebold-Mariano tests show XGBoost is {'never significantly better than' if not any(N.cnt[H]['better'] for H in PRIMARY) else 'only sometimes better than'} ETS/SARIMA/MA at 14-28 days and is {('significantly worse than ' + ', '.join(sorted(set(N.sig_worse))) + ' at 7 days') if N.sig_worse else 'not significantly different from them at 7 days'}. SHAP shows XGBoost is mainly a smoothed recent-demand model ({N.roll[7]:.0f}% of attribution from the 7/14/28-day rolling means).", "",
         f"**Business impact.** In the simulation B changes total cost vs A by {sgn(N.g[3])} / {sgn(N.g[7])} / {sgn(N.g[14])} at L = 3 / 7 / 14 (95%, medium penalty); the bootstrap interval at L=7 is {100 * p7.boot_ci_low:+.0f}% to {100 * p7.boot_ci_high:+.0f}%, so it is {'suggestive rather than conclusive' if p7.boot_ci_low < 0 < p7.boot_ci_high else 'statistically distinguishable from zero (hypothetical costs)'}. "
         + (f"{sigma_share:.0f}% of the L=7 change comes from replacing sqrt(L) sigma by validation-based sigma with the same point forecast. " if N.same_pf7
            else f"The moving-average model (window {f.ma_windows()[7]} days) with validation-based sigma gives {sgn(N.ma_gain7)} at L=7, against {sgn(N.g[7])} for the headline model. ")
         + (f"The most accurate model was usually not the cheapest (rank correlation {N.rho['MEDIUM']:+.2f} for the medium penalty), " if N.same_best < len(f.avc_s) / 2
            else f"The most accurate model was often the cheapest (rank correlation {N.rho['MEDIUM']:+.2f} for the medium penalty), ")
         + (f"and realised cycle service ({100 * N.b95.Cycle_Service_Level:.0f}%) fell short of the 95% target. " if N.b95.Cycle_Service_Level < 0.95 else f"and realised cycle service ({100 * N.b95.Cycle_Service_Level:.0f}%) met the 95% target. ")
         + f"Recommendation: simple exponential-smoothing forecasts, calibrated sigma, a cost-aware service level ({f.optimal_sl_text()} on the simulated grid) and shorter lead times.", "",
         "## 5-minute technical explanation", "",
         "1. **Framing.** FORECAST -> DECISION -> BUSINESS IMPACT. The target is Target_H(t) = sum_{k=1..H} y_{t+k}; the same target scores every model and equals the lead-time demand the inventory policy needs.",
         f"2. **Data and honesty about it.** M5 from verified mirrors; PROBLEM -> DIAGNOSIS -> ACTION -> REASON cleaning log; zero analysis: {100 * f.zero_all.iloc[1]['share_of_zero_cells']:.1f}% of zero cells are structural (no price row), {100 * f.zero_all.iloc[2]['share_of_zero_cells']:.2f}% closures (Christmas and two outages), "
         f"{100 * f.zero_all.iloc[3]['share_of_zero_cells']:.1f}% ambiguous -> observed sales can underestimate demand.",
         "3. **Sampling.** TRAIN-only selection algorithm (store per state, eligibility, percentile tiers, seeded draws), 18 series, nothing tuned to results.",
         "4. **Validation protocol.** Chronological split; daily rolling origins with complete windows inside the period; selections on validation; refit on train+validation; test once. Training rows are cut so that origin + H <= fit_end - 1.",
         f"5. **Leakage control.** Features at the origin (lag_k = y[t+1-k], windows end at t); calendar 'known-ahead' counts proven independent of sales; perturbation tests replace everything after a random cut-off and require identical features, forecasts, sigma and policy decisions up to the cut-off ({n_checks}/{len(f.audit)} pass); a deliberately leaky feature is detected (negative control).",
         f"6. **Models.** Baselines (MA window by validation: {f.ma_window_text()}), Croston-SBA, SES/Holt(damped)/Holt-Winters with a fallback chain, SARIMA candidates with ADF/KPSS/ACF/PACF-motivated structures chosen on validation, direct XGBoost per horizon with {C.XGB_SEARCH_ITER}-config validation search (validation MAE gains of {min(N.tune_gain):.0f}-{max(N.tune_gain):.0f}% over the defaults).",
         f"7. **Accuracy results.** Test WAPE table; best per horizon {best[7]['best_by_test_WAPE']}/{best[14]['best_by_test_WAPE']}/{best[28]['best_by_test_WAPE']}; validation choice matched test winner at {int(f.best[f.best.primary_horizon].validation_choice_equals_test_best.sum())}/3 horizons; test RMSE {N.ratio.r.min():.2f}-{N.ratio.r.max():.2f}x validation RMSE. DM tests (HAC, HLN, Holm) show " + ("no significant XGBoost advantage" if N.xgb_never_better else "only a partial XGBoost advantage") + "; the volume-weighted (pooled WAPE) versus equal-weighted (DM panel) views can rank models differently.",
         f"8. **Explainability.** TreeSHAP on the final models: rolling means {N.roll[7]:.0f}%/{N.roll[28]:.0f}% of attribution, SNAP {N.snap:.1f}%, price {N.price:.1f}% (most likely a between-item effect, not an elasticity); additivity verified; three rule-selected waterfalls.",
         f"9. **Uncertainty.** sigma_L = validation RMSE of the L-day sum; Gaussian vs empirical intervals compared by split-half calibration on validation ({f.interval_method().capitalize()} won); test coverage of the chosen intervals was within 5 points of nominal in {f.coverage_close()[0]} of {f.coverage_close()[1]} model/horizon/level cases, "
        f"but test errors were larger than validation errors (RMSE ratio {N.ratio.r.min():.2f}-{N.ratio.r.max():.2f}x), which later lowers realised service.",
         "10. **Inventory model.** (s, S) with ROP = mu_L + z*sigma_L and EOQ order quantity; order placed at end of day t arrives at start of t+L+1; lost sales; daily ledger with receive -> demand -> fulfil -> stockout -> position -> order; unit-tested balance, timing, costs and causality. Policy A: trailing mean, sqrt(L) x trailing std; Policy B: model forecast + validation sigma.",
         f"11. **Results.** B vs A cost {sgn(N.g[3])}/{sgn(N.g[7])}/{sgn(N.g[14])}; {N.n_cheaper}/{N.n_cfg} cells cheaper, bootstrap intervals include zero in {N.n_cfg - N.n_ci_excl0} of {N.n_cfg} cells; "
         + (f"the sigma method alone accounts for {sigma_share:.0f}% of the L=7 change; " if N.same_pf7 else f"the moving-average model with validation-based sigma gives {sgn(N.ma_gain7)} at L=7; ")
         + (f"the gain is concentrated in the {ri.loc['intermittent', 'n_series']:.0f} intermittent series ({ri.loc['intermittent', 'B_vs_A_total_cost_pct']:+.1f}%) while the {ri.loc['regular', 'n_series']:.0f} regular ones show {ri.loc['regular', 'B_vs_A_total_cost_pct']:+.1f}% (L=7, 95%, MEDIUM). "
            if ri.loc['intermittent', 'B_vs_A_total_cost_pct'] < ri.loc['regular', 'B_vs_A_total_cost_pct'] - 2 else "no clear difference between intermittent and regular series. ")
         + f"Cost-minimising service level: {f.optimal_sl_text()}; ranges of mean cost: " + ", ".join(f"{k} {v:.0f}%" for k, v in sorted(f.factor_ranges().items(), key=lambda kv: -kv[1])) + ".",
         f"12. **The key insight.** Accuracy rank != cost rank (only {N.same_best}/{len(f.avc_s)} cells agree; mean rho LOW {N.rho['LOW']:+.2f}, MEDIUM {N.rho['MEDIUM']:+.2f}, HIGH {N.rho['HIGH']:+.2f}) because of asymmetric costs, bias, the safety-stock mechanism and validation optimism.",
         "13. **Limitations and next steps.** Hypothetical costs, censored demand, 18 series, one test year; next: quantile/cost-aware forecasts, rolling sigma, real costs and lead-time distributions, promotions, hierarchical forecasting, pilot test.", "",
         "## Resume bullets (numbers from the result tables)", "",
         f"* Forecasted 7/14/28-day retail demand for 18 M5 (Walmart) item-store series with nine models (naive/seasonal-naive/moving-average/Croston baselines, SES/Holt/Holt-Winters, SARIMA, XGBoost) in a chronological rolling-origin framework; "
         f"best test WAPE {best[7]['test_WAPE']:.1f}% / {best[14]['test_WAPE']:.1f}% / {best[28]['test_WAPE']:.1f}% ({best[7]['best_by_test_WAPE']} / {best[14]['best_by_test_WAPE']} / {best[28]['best_by_test_WAPE']}) vs {f.wape('SeasonalNaive', 7):.1f}% / {f.wape('SeasonalNaive', 14):.1f}% / {sn_w28:.1f}% for seasonal naive; "
         + ("Diebold-Mariano tests showed the leading models statistically indistinguishable at 14-28 days." if indist_14_28
            else "Diebold-Mariano tests found some significant differences between the leading models at 14-28 days (see the final report)."),
         f"* Built leakage-safe lag/rolling/calendar/SNAP/price features validated by {n_checks} automated audit checks (perturbation tests plus a negative control) and used SHAP to show that 7/14/28-day rolling demand levels drive {N.roll[7]:.0f}% of XGBoost's 7-day attributions, "
         f"with {N.top_nondemand} the largest non-demand features.",
         f"* Converted forecasts into reorder-point/EOQ policies and simulated 292 days x 18 series: the forecast-driven policy {N.lw(N.g[7]).replace('simulated total cost', 'pooled total cost')} (L=7, 95% target, hypothetical medium penalty; 95% bootstrap CI {100 * p7.boot_ci_low:+.0f}% to {100 * p7.boot_ci_high:+.0f}%, {'i.e. suggestive, not statistically significant' if p7.boot_ci_low < 0 < p7.boot_ci_high else 'excluding zero'}) and lifted realised cycle service from "
         f"{100 * N.a95.Cycle_Service_Level:.0f}% to {100 * N.b95.Cycle_Service_Level:.0f}% versus a historical baseline, and showed that forecast-accuracy rank did not predict cost rank (Spearman {N.rho['MEDIUM']:+.2f} for the medium penalty).", ""]
    (C.REPORTS / "project_story_and_resume.md").write_text("\n".join(L) + "\n")
