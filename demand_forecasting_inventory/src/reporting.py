"""Phase 36 / 42 - report generation.  EVERY number in the generated markdown is read from outputs/tables (computed by the
pipeline); qualitative wording that depends on a result (who wins, is a difference significant, does the correlation have a
sign) is produced by conditionals on those numbers, so re-running the pipeline keeps the text honest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .features import FEATURES
from .splits import make_split

T = C.OUT_TAB
FIG = "../outputs/figures/"
PRIMARY = list(C.PRIMARY_HORIZONS)
FAM = {"Naive": "baseline", "SeasonalNaive": "baseline", "MovingAverage": "baseline", "Croston": "baseline", "SES": "exp. smoothing",
       "Holt": "exp. smoothing", "HoltWinters": "exp. smoothing", "SARIMA": "SARIMA", "XGBoost": "ML"}


def rd(name: str, **k) -> pd.DataFrame:
    return pd.read_csv(T / name, **k)


def pct(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}%"


def md(df: pd.DataFrame, floatfmt: str = ".2f", index: bool = False) -> str:
    return df.to_markdown(index=index, floatfmt=floatfmt)


def sgn(x: float, d: int = 1) -> str:
    return f"{x:+.{d}f}%"


@dataclass
class Facts:
    """All result tables plus derived numbers used by the narrative."""
    sel: pd.DataFrame = field(default_factory=lambda: pd.read_csv(C.DATA_PROC / "selected_series.csv"))

    def __post_init__(self):
        self.split_tab = rd("split_table.csv")
        self.val = rd("forecast_summary_validation.csv")
        self.test = rd("forecast_summary_test.csv")
        self.by_series = rd("forecast_results.csv")
        self.best = rd("best_model_by_horizon.csv")
        self.dm = rd("dm_tests_panel.csv")
        self.dm_ps = rd("dm_tests_per_series.csv")
        self.ma = rd("ma_window_selection.csv")
        self.ets = rd("ets_fit_log.csv")
        self.sar_c = rd("sarima_candidates.csv")
        self.sar_s = rd("sarima_selection.csv")
        self.stat = rd("stationarity_tests.csv")
        self.xgb_log = rd("xgb_tuning_log.csv")
        self.xgb_ch = rd("xgb_chosen_params.csv")
        self.res_stat = rd("residuals_statistical_models_summary.csv")
        self.res_xgb = rd("residuals_xgboost_overall.csv")
        self.shap = rd("shap_importance.csv")
        self.shap_chk = rd("shap_checks.csv")
        self.shap_cases = rd("shap_individual_cases.csv")
        self.sigma = rd("sigma_table.csv")
        self.norm = rd("uncertainty_normality.csv")
        self.calib = rd("uncertainty_calibration_validation.csv")
        self.calib_choice = rd("uncertainty_method_choice.csv")
        self.cov = rd("uncertainty_coverage_test.csv")
        self.inv_final = rd("final_inventory_results.csv")
        self.inv_all = rd("inventory_all_policies_aggregated.csv")
        self.sens = rd("sensitivity_headline_policies.csv")
        self.opt = rd("sensitivity_optimal_service_level.csv")
        self.factor = rd("sensitivity_factor_importance.csv")
        self.pair = rd("inventory_paired_bootstrap_B_vs_A.csv")
        self.pair_all = rd("inventory_paired_bootstrap_all_models_MEDIUM.csv")
        self.avc = rd("accuracy_vs_cost_detail.csv")
        self.avc_s = rd("accuracy_vs_cost_summary.csv")
        self.rob_f = rd("robustness_forecast_by_group.csv")
        self.rob_best = rd("robustness_best_model_by_group.csv")
        self.rob_i = rd("robustness_inventory_by_group.csv")
        self.headB = rd("headline_policy_B_models.csv").set_index("lead_time")["model"].to_dict()
        self.eoq = rd("eoq_cycle_sanity_train.csv")
        self.ledger = rd("inventory_ledger_headline_L7_SL95.csv", parse_dates=["date"])
        self.audit = rd("leakage_audit.csv")
        self.eda = rd("eda_summary_stats.csv")
        self.conc = rd("error_concentration.csv")
        self.zf = rd("zero_forecast_shares.csv")
        self.zero_all = rd("zero_cells_dataset_wide.csv")
        self.zero_s = rd("zero_sales_by_series.csv")
        self.ver = rd("raw_data_verification.csv")
        self.cleaning_checks = rd("data_inspection_checks.csv").set_index("check")["value"].to_dict()
        self.log = json.loads((T / "subset_selection_log.json").read_text())
        self.split = make_split(pd.DatetimeIndex(sorted(pd.read_parquet(C.DATA_PROC / "panel.parquet", columns=["date"])["date"].unique())))

    # ------------------------------------------------------------------ helpers
    def wape(self, model: str, H: int, phase: str = "test") -> float:
        d = self.test if phase == "test" else self.val
        return float(d[(d.model == model) & (d.H == H)]["WAPE"].iloc[0])

    def metric(self, model: str, H: int, met: str, phase: str = "test") -> float:
        d = self.test if phase == "test" else self.val
        return float(d[(d.model == model) & (d.H == H)][met].iloc[0])

    def best_row(self, H: int) -> pd.Series:
        return self.best[self.best.H == H].iloc[0]

    def inv(self, policy_group: str, L: int, SL: float, scenario: str = "MEDIUM") -> pd.Series:
        d = self.inv_final
        return d[(d.Policy == policy_group) & (d.Lead_Time_days == L) & (d.Service_Level == SL) & (d.Cost_Scenario == scenario)].iloc[0]

    def agg(self, policy: str, L: int, SL: float, scenario: str = "MEDIUM", method: str = "gaussian") -> pd.Series:
        d = self.inv_all
        return d[(d.policy == policy) & (d.L == L) & (d.SL == SL) & (d.scenario == scenario) & (d.method == method)].iloc[0]

    def dm_xgb(self, H: int, model_b: str, loss: str = "abs") -> pd.Series:
        d = self.dm
        return d[(d.ref == "XGBoost") & (d.H == H) & (d.model_b == model_b) & (d.loss == loss)].iloc[0]

    def dm_counts(self, ref: str = "XGBoost") -> dict:
        d = self.dm[(self.dm.ref == ref) & (self.dm.loss == "abs") & (~self.dm.model_b.isin(["Naive", "SeasonalNaive"]))]
        out = {}
        for H in PRIMARY:
            g = d[d.H == H]
            out[H] = {"better": g[(g.p_holm < 0.05) & (g.mean_d < 0)]["model_b"].tolist(), "worse": g[(g.p_holm < 0.05) & (g.mean_d > 0)]["model_b"].tolist(),
                      "n": len(g), "min_p_holm": float(g["p_holm"].min())}
        return out

    def beats_naive(self, ref: str) -> bool:
        d = self.dm[(self.dm.ref == ref) & (self.dm.loss == "abs") & self.dm.model_b.isin(["Naive", "SeasonalNaive"]) & self.dm.H.isin(PRIMARY)]
        return bool(((d.p_holm < 0.05) & (d.mean_d < 0)).all())

    def per_series_tally(self, ref: str = "XGBoost", others=("MovingAverage", "HoltWinters", "SARIMA")) -> tuple[int, int, int]:
        d = self.dm_ps[(self.dm_ps.ref == ref) & self.dm_ps.model_b.isin(others)]
        better = int(((d.p_value < 0.05) & (d.mean_d < 0)).sum())
        worse = int(((d.p_value < 0.05) & (d.mean_d > 0)).sum())
        return better, worse, len(d)

    def bias_vs_lost(self, L: int, SL: float = 0.95, drop_naive: bool = True) -> tuple[float, float]:
        g = self.inv_all[(self.inv_all.scenario == "MEDIUM") & (self.inv_all.method == "gaussian") & (self.inv_all.SL == SL) & (self.inv_all.L == L) &
                         (self.inv_all.policy != "A_historical")].set_index("policy")
        b = self.test[self.test.H == L].set_index("model")["bias"]
        d = pd.DataFrame({"bias": b, "lost": g["stockout_units"]}).dropna()
        if drop_naive:
            d = d.drop(index="Naive", errors="ignore")
        r = stats.spearmanr(d.bias, d.lost)
        return float(r.statistic), float(r.pvalue)

    def cost_gain_vs_A(self, policy: str, L: int, SL: float = 0.95, scenario: str = "MEDIUM") -> float:
        a = self.agg("A_historical", L, SL, scenario)["total_cost"]
        b = self.agg(policy, L, SL, scenario)["total_cost"]
        return 100.0 * (b / a - 1.0)

    # ---- derived facts used by several documents (so the final report and the interview guide cannot disagree) ----------
    def concentration(self, H: int = 7) -> dict:
        """Ranges across models of the error-concentration diagnostics at horizon H (validation vs test)."""
        c = self.conc[self.conc.H == H]
        out = {}
        for ph in ("val", "test"):
            g = c[c.phase == ph]
            out[ph] = {k: (float(g[col].min()), float(g[col].max())) for k, col in
                       (("rmse_mae", "RMSE_over_MAE"), ("worst5", "worst5pct_share_of_squared_error"), ("kurt", "excess_kurtosis"))}
        return out

    def smape_zero_story(self, H: int = 28) -> dict:
        """Which model has the best test sMAPE at horizon H, how does it rank on WAPE, and how often does it (and do the others) forecast exactly zero."""
        t = self.test[self.test.H == H].sort_values("sMAPE").reset_index(drop=True)
        best = str(t.model.iloc[0])
        wape_rank = int(self.test[self.test.H == H].sort_values("WAPE").reset_index(drop=True).query("model == @best").index[0] + 1)
        z = self.zf[self.zf.H == H].set_index("model")
        never = sorted(m for m in z.index if z.loc[m, "share_zero_forecast"] == 0)
        return {"model": best, "smape": float(t.sMAPE.iloc[0]), "wape_rank": wape_rank, "n_models": len(t),
                "share_zero_forecast": float(z.loc[best, "share_zero_forecast"]), "share_both_zero": float(z.loc[best, "share_both_zero"]),
                "share_actual_zero": float(z.loc[best, "share_actual_zero"]), "never_zero": never}

    def emp_vs_gauss(self) -> pd.DataFrame:
        """Empirical-quantile minus Gaussian safety stock for every Policy-B model x L x service level (MEDIUM penalty, pooled over series)."""
        d = self.inv_all[(self.inv_all.scenario == "MEDIUM") & (self.inv_all.policy != "A_historical")]
        g = d[d.method == "gaussian"].set_index(["policy", "L", "SL"])
        e = d[d.method == "empirical"].set_index(["policy", "L", "SL"])
        return pd.DataFrame({"cost_pct": 100 * (e.total_cost / g.total_cost - 1), "fill_pp": 100 * (e.fill_rate - g.fill_rate),
                             "cycle_pp": 100 * (e.cycle_service_level - g.cycle_service_level)})

    def under_forecast_cells(self) -> tuple[int, int]:
        """(model x horizon cells whose test bias is negative, all cells)."""
        t = self.test[self.test.H.isin(C.HORIZONS)]
        return int((t.bias < 0).sum()), len(t)

    def lost_nov_dec(self, L: int = 7) -> tuple[float, float, float]:
        """Share (%) of lost units that fall in Nov-Dec for the headline B model and for A, and Nov-Dec's share (%) of the test days."""
        led = self.ledger

        def share(pol):
            g = led[led.policy == pol]
            m = g.date.dt.month.isin([11, 12])
            return 100 * g[m].stockout.sum() / max(g.stockout.sum(), 1e-9), 100 * m.mean()
        nd_b, nd_days = share(self.headB[L])
        nd_a, _ = share("A_historical")
        return nd_b, nd_a, nd_days

    def ma_windows(self) -> dict[int, int]:
        """Moving-average window chosen on validation WAPE, per horizon."""
        sel = self.ma[self.ma.selected]
        return {int(h): int(w) for h, w in zip(sel.H, sel.window)}

    def ma_window_text(self) -> str:
        w = self.ma_windows()
        vals = sorted(set(w.values()))
        return f"{vals[0]} days at every horizon" if len(vals) == 1 else ", ".join(f"{v} days at H = {h}" for h, v in w.items())

    def ma_same_point_forecast_as_A(self, L: int) -> bool:
        """Policy A uses a 28-day trailing mean (mu_L = L x mean, annual demand = 365 x mean); the MA model in B is identical iff the
        validation-chosen windows for H = L and H = 28 are both 28 days."""
        w = self.ma_windows()
        return w.get(L) == 28 and w.get(28) == 28

    def optimal_sl_text(self, policy_group: str = "B_forecast_driven") -> str:
        """Cost-minimising service level on the grid, per penalty scenario (across lead times)."""
        o = self.opt[self.opt.policy_group == policy_group]
        top = max(C.SERVICE_LEVELS)
        parts = []
        for sc in C.SCENARIOS:
            vals = sorted(set(o[o.scenario == sc].cost_minimising_SL))
            parts.append(f"{sc} {'/'.join(f'{100 * v:.0f}%' for v in vals)}" + (" (the grid edge)" if vals == [top] else ""))
        return ", ".join(parts)

    def factor_ranges(self, policy_group: str = "B_forecast_driven") -> dict[str, float]:
        """Relative range (%) of the mean total cost across the levels of each factor (the other factors averaged out)."""
        d = self.factor[self.factor.policy_group == policy_group]
        name = lambda t: "stockout-penalty scenario" if "penalty" in t else "lead time" if "lead" in t else "service level"
        return {name(r.factor): 100 * float(r.relative_range_of_mean_total_cost) for r in d.itertuples()}

    def n_nonnaive(self) -> int:
        return int(self.test[self.test.H == 7].model.nunique() - 1)

    def winners_curse(self, H: int = 7) -> tuple[bool, list[str]]:
        """True if the three models with the smallest mean validation sigma are also the three with the largest test/validation RMSE ratio."""
        m = self.test.merge(self.val, on=["model", "H"], suffixes=("_t", "_v"))
        r = m[m.H == H].assign(r=lambda d: d.RMSE_t / d.RMSE_v).set_index("model")["r"]
        sig = self.sigma[self.sigma.H == H].groupby("model").sigma_rmse.mean()
        low_sigma = list(sig.sort_values().index[:3])
        return set(low_sigma) == set(r.sort_values(ascending=False).index[:3]), low_sigma

    def intermittent_vs_regular(self) -> tuple[float, float, int, int]:
        """B-vs-A total-cost change (%) at L=7/95%/MEDIUM for intermittent and regular series, and the group sizes."""
        ri = self.rob_i.set_index("group")
        return (float(ri.loc["intermittent", "B_vs_A_total_cost_pct"]), float(ri.loc["regular", "B_vs_A_total_cost_pct"]),
                int(ri.loc["intermittent", "n_series"]), int(ri.loc["regular", "n_series"]))

    def cheapest_ex_post(self, L: int = 7, SL: float = 0.95, scenario: str = "MEDIUM") -> str:
        d = self.avc[(self.avc.scenario == scenario) & (self.avc.L == L) & (self.avc.SL == SL)]
        return str(d.sort_values("total_cost").iloc[0].model)

    def n_cheapest(self, model: str) -> tuple[int, int]:
        return int((self.avc_s.best_by_total_cost == model).sum()), len(self.avc_s)

    def ets_vs_best(self) -> tuple[float, bool]:
        """(largest WAPE gap in points between the best ETS-family model and the overall best, over the primary horizons;
        True if XGBoost is never significantly better than an ETS-family model in the DM tests)."""
        gaps = []
        for H in PRIMARY:
            t = self.test[self.test.H == H]
            gaps.append(float(t[t.model.isin(["SES", "Holt", "HoltWinters"])].WAPE.min() - t.WAPE.min()))
        cnt = self.dm_counts("XGBoost")
        never = not any(m in cnt[H]["better"] for H in PRIMARY for m in ("SES", "Holt", "HoltWinters"))
        return max(gaps), never

    def critical_ratio_text(self) -> tuple[str, bool]:
        """Median newsvendor critical ratios (TRAIN only) per scenario and whether all of them lie inside the simulated service-level grid."""
        cr = {sc: 100 * float(self.eoq[f"critical_ratio_{sc}"].median()) for sc in C.SCENARIOS}
        inside = all(100 * min(C.SERVICE_LEVELS) <= v <= 100 * max(C.SERVICE_LEVELS) for v in cr.values())
        return " / ".join(f"{v:.0f}%" for v in cr.values()), inside

    def interval_method(self) -> str:
        return str(self.calib_choice.sort_values("abs_coverage_error").iloc[0]["method"])

    def coverage_close(self, tol: float = 0.05) -> tuple[int, int]:
        """(model x horizon x level cases whose TEST coverage of the validation-chosen interval method is within +-tol of nominal, all cases)."""
        c = self.cov[self.cov.method == self.interval_method()]
        return int((c.abs_coverage_error <= tol).sum()), len(c)

    def best_in_group(self, group: str, H: int) -> str:
        g = self.rob_best[(self.rob_best.group == group) & (self.rob_best.H == H)]
        return str(g.best_model.iloc[0])

    def val_choice_cheapest(self) -> tuple[int, int, float]:
        """(cells where the model with the best validation WAPE is also the cheapest, all cells, median regret % in the MEDIUM penalty)."""
        s = self.avc_s
        med = s[s.scenario == "MEDIUM"]["regret_of_val_choice_pct"].median()
        return int((s.best_by_val_WAPE == s.best_by_total_cost).sum()), len(s), float(med)

    def model_cost_spread(self, scenario: str = "MEDIUM") -> tuple[float, float]:
        """Spread (%) of pooled total cost between the dearest and cheapest non-naive forecasting model: (median, max) over L x service level."""
        d = self.avc[(self.avc.scenario == scenario) & (self.avc.model != "Naive")]
        r = 100 * d.groupby(["L", "SL"]).total_cost.agg(lambda x: x.max() / x.min() - 1)
        return float(r.median()), float(r.max())


# ======================================================================================
# validation_strategy.md  (Phase 9)
# ======================================================================================
def build_validation_strategy(f: Facts) -> None:
    sp = f.split
    tab = f.split_tab.copy()
    rows = []
    for H in C.HORIZONS:
        rows.append({"horizon H": H, "role": "primary" if H in PRIMARY else "supplementary (lead time L=3)",
                     "validation origins": len(sp.eval_origins("validation", H)), "test origins": len(sp.eval_origins("test", H)),
                     "XGBoost tuning rows (train)": int(f.xgb_ch[f.xgb_ch.H == H]["train_rows"].iloc[0]),
                     "XGBoost final-fit rows (train+val)": int(f.xgb_ch[f.xgb_ch.H == H]["final_fit_rows"].iloc[0])})
    L = ["# Validation strategy (Phase 9)", "",
         "## 1. Chronological split - never random", "",
         md(tab), "",
         f"The {sp.n_days:,} observed days are split by *date*: {C.TRAIN_FRAC:.0%} train / {C.VAL_FRAC:.0%} validation / remainder test "
         f"(`floor(0.70 x N) = {sp.train_end}` and `floor(0.15 x N) = {sp.val_end - sp.train_end}` days; the test period gets the remaining {sp.n_days - sp.val_end}). "
         "No observation is ever shuffled; a static check (`leakage_audit.check_no_random_splitting`) confirms that no random-split API is used.", "",
         "## 2. Rolling-origin (daily) evaluation of H-day demand sums", "",
         "* A **forecast origin** `t` is the last day whose sales are known. The forecast is for `sum(y[t+1..t+H])`.",
         "* Every day of the validation (test) period that leaves the *complete* H-day window inside the period is an origin, so each model is scored on hundreds of "
         "overlapping windows (consecutive windows share H-1 days; this is why the Diebold-Mariano test uses a HAC variance with lag H-1).",
         "* Origin `t = train_end - 1` (last training day) is the first validation origin; `t = val_end - 1` is the first test origin.", "",
         md(pd.DataFrame(rows)), "",
         "## 3. What is chosen on which data", "",
         "| decision | made on | never uses |", "|---|---|---|",
         "| moving-average window (7/14/28) | validation WAPE (per horizon) | test |",
         "| ETS variant (Holt-Winters with / without damped trend) | AICc on the training window | validation, test |",
         "| SARIMA structure (5 candidates, per series) | validation error of the H = 7, 14, 28 sums (AIC/BIC reported; AIC is not comparable across differencing orders) | test |",
         "| XGBoost hyper-parameters, number of trees (early stopping) | validation rows only (24 configurations x 4 horizons) | test |",
         "| sigma_L of the safety stock; prediction-interval method | VALIDATION residuals of the model trained on TRAIN | test |",
         "| headline Policy-B forecasting model per lead time | lowest validation WAPE at H = L | test cost |",
         "| cost parameters, scenarios, service levels, lead times | fixed a priori (config.py; EOQ cycle lengths sanity-checked on TRAIN only) | any result |", "",
         "## 4. Two-phase protocol (identical for every model family)", "",
         "1. **Validation phase:** fit on `y[:train_end]`; produce forecasts for validation origins (parameters frozen, state updated day by day); select / calibrate.",
         "2. **Test phase:** with the structure chosen in phase 1, **refit on train + validation** (`y[:val_end]`); produce forecasts for test origins; evaluate **once**.",
         "3. XGBoost training rows are cut so that their target windows end before the period being predicted (`origin + H <= fit_end - 1`; asserted in code and in the audit).", "",
         "## 5. Why not k-fold or a longer rolling re-fit?",
         "k-fold would shuffle time. A fully rolling re-fit (re-estimating every day) would multiply the cost by ~290 for SARIMA/XGBoost without changing the protocol's logic; "
         "instead parameters are re-estimated once per phase and the *states* (ETS level/seasonals, SARIMA Kalman state) and *features* (lags, rolling means) are updated daily - the way "
         "such models run in production between periodic re-fits.", "",
         "## 6. Known limitations of the design",
         "* One validation and one test period (292 days, one calendar year of seasonality): results describe this period, not all futures.",
         "* Overlapping windows make errors strongly autocorrelated; the effective number of independent windows is roughly n / H.",
         f"* The test period turned out to be harder than validation for *every* model (pooled RMSE ratio test/validation {f.test.merge(f.val, on=['model', 'H'], suffixes=('_t', '_v')).assign(r=lambda d: d.RMSE_t / d.RMSE_v)['r'].min():.2f} to "
         f"{f.test.merge(f.val, on=['model', 'H'], suffixes=('_t', '_v')).assign(r=lambda d: d.RMSE_t / d.RMSE_v)['r'].max():.2f}), which matters for validation-based safety stocks (section 15 of the final report).", ""]
    (C.REPORTS / "validation_strategy.md").write_text("\n".join(L) + "\n")


# ======================================================================================
# final_report.md  (Phase 36)
# ======================================================================================
def _dm_table(f: Facts) -> str:
    rows = []
    for H in PRIMARY:
        row = {"horizon": H}
        for m in ["MovingAverage", "Croston", "SES", "Holt", "HoltWinters", "SARIMA"]:
            r = f.dm_xgb(H, m)
            sym = "better" if (r["p_holm"] < 0.05 and r["mean_d"] < 0) else ("WORSE" if (r["p_holm"] < 0.05 and r["mean_d"] > 0) else "n.s.")
            row[m] = f"{sym} (p_Holm={r['p_holm']:.3f})"
        for m in ["SeasonalNaive", "Naive"]:
            r = f.dm_xgb(H, m)
            sym = "better" if (r["p_holm"] < 0.05 and r["mean_d"] < 0) else ("WORSE" if (r["p_holm"] < 0.05 and r["mean_d"] > 0) else "n.s.")
            row[m] = f"{sym} (p_Holm={r['p_holm']:.3f})"
        rows.append(row)
    return md(pd.DataFrame(rows))


def _pivot(df: pd.DataFrame, metric: str, horizons=None, order_by: int = 28) -> pd.DataFrame:
    horizons = horizons or C.HORIZONS
    p = df[df.H.isin(horizons)].pivot(index="model", columns="H", values=metric)
    p.columns = [f"{c}-day" for c in p.columns]
    return p.sort_values(f"{order_by}-day").reset_index().rename(columns={"model": "Model"})


def build_final_report(f: Facts) -> None:
    sp = f.split
    sel = f.sel
    best = {H: f.best_row(H) for H in C.HORIZONS}
    cnt = f.dm_counts()
    L = []
    A = L.append

    # ---- derived headline numbers --------------------------------------------------------------
    hB = f.headB
    g7 = f.cost_gain_vs_A(hB[7], 7)
    g14 = f.cost_gain_vs_A(hB[14], 14)
    g3 = f.cost_gain_vs_A(hB[3], 3)
    p7 = f.pair[(f.pair.scenario == "MEDIUM") & (f.pair.L == 7) & (f.pair.SL == 0.95)].iloc[0]
    n_cheaper = int((f.pair.rel_diff_pooled < 0).sum())
    n_cfg = len(f.pair)
    n_ci_excl0 = int(((f.pair.boot_ci_high < 0) | (f.pair.boot_ci_low > 0)).sum())
    ma_gain7 = f.cost_gain_vs_A("MovingAverage", 7)                     # same mu as policy A: isolates the sigma method
    bestcost7 = f.avc[(f.avc.scenario == "MEDIUM") & (f.avc.L == 7) & (f.avc.SL == 0.95)].sort_values("total_cost").iloc[0]
    rho_med = f.avc_s[f.avc_s.scenario == "MEDIUM"]["spearman_testWAPE_vs_cost_excl_Naive"].mean()
    rho_low = f.avc_s[f.avc_s.scenario == "LOW"]["spearman_testWAPE_vs_cost_excl_Naive"].mean()
    rho_high = f.avc_s[f.avc_s.scenario == "HIGH"]["spearman_testWAPE_vs_cost_excl_Naive"].mean()
    ratio = f.test.merge(f.val, on=["model", "H"], suffixes=("_t", "_v")).assign(r=lambda d: d.RMSE_t / d.RMSE_v)
    a95 = f.inv("A_historical", 7, 0.95)
    b95 = f.inv("B_forecast_driven", 7, 0.95)
    mae_ratio = f.test.merge(f.val, on=["model", "H"], suffixes=("_t", "_v")).assign(r=lambda d: d.MAE_t / d.MAE_v)
    conc7 = f.concentration(7)
    sz = f.smape_zero_story(28)

    # ====================== 1. Executive summary ======================
    A("# Explainable Demand Forecasting & Inventory Optimization - Final Report")
    A("")
    A("*Every number below is read from `outputs/tables/` (produced by `python run_project.py`); costs are **hypothetical** assumptions, not observed business costs.*")
    A("")
    A("## 1. Executive Summary")
    A("")
    A("**Question.** Can accurate and explainable retail demand forecasts be converted into better inventory decisions by balancing stockout risk against holding cost?")
    A("")
    A(f"**Data and design.** M5 Walmart unit sales (30,490 item-store series, {sp.n_days:,} days) - a reproducible, TRAIN-information-only sample of **18 item-store series** "
      f"(3 stores x 3 demand tiers x 2 items; {int((sel.train_adi >= C.ADI_CUTOFF).sum())} of 18 are intermittent/lumpy). Chronological split: train {sp.date_range('train')[0].date()}-{sp.date_range('train')[1].date()}, "
      f"validation {sp.date_range('validation')[0].date()}-{sp.date_range('validation')[1].date()}, test {sp.date_range('test')[0].date()}-{sp.date_range('test')[1].date()}. "
      f"Nine forecasting models (naive, seasonal naive, moving average, Croston-SBA, SES, Holt, Holt-Winters, SARIMA, XGBoost) are scored on the sum of demand over the next 7/14/28 days "
      f"from daily rolling origins; forecasts then drive a reorder-point + EOQ inventory policy simulated day by day over the {sp.n_days - sp.val_end} test days.")
    A("")
    A("**Main findings**")
    A("")
    wrow = lambda H: f"{best[H]['best_by_test_WAPE']} ({best[H]['test_WAPE']:.1f}% WAPE)"
    n_x_better = sum(len(cnt[H]["better"]) for H in PRIMARY)
    n_x_cmp = sum(cnt[H]["n"] for H in PRIMARY)
    worse_txt = "; ".join(f"H={H}: {', '.join(cnt[H]['worse'])}" for H in PRIMARY if cnt[H]["worse"]) or "none"
    cnt_s = f.dm_counts("SARIMA")
    pmin_long = min(f.dm_xgb(H, m)["p_holm"] for H in (14, 28) for m in ["MovingAverage", "SES", "Holt", "HoltWinters", "SARIMA"])
    naive_txt = ("XGBoost and SARIMA are significantly better than naive and seasonal-naive at every primary horizon" if f.beats_naive("XGBoost") and f.beats_naive("SARIMA")
                 else "XGBoost and SARIMA do not beat naive / seasonal-naive significantly at every primary horizon")
    A(f"1. **Forecast accuracy (test, pooled WAPE).** Best model per horizon: 7-day {wrow(7)}, 14-day {wrow(14)}, 28-day {wrow(28)}. "
      f"Differences between the leading models are small. In Holm-adjusted Diebold-Mariano tests (equal-weighted scaled loss) XGBoost is significantly better than another non-naive model in {n_x_better} of {n_x_cmp} comparisons "
      f"and significantly worse in {sum(len(cnt[H]['worse']) for H in PRIMARY)} ({worse_txt}); at 14 and 28 days the smallest Holm p-value against ETS, SARIMA or the 28-day moving average is {pmin_long:.2f}. {naive_txt}.")
    A(f"2. **Validation vs test.** The model chosen on validation was the test winner at {int(f.best[f.best.primary_horizon].validation_choice_equals_test_best.sum())} of 3 primary horizons; "
      f"every model's test RMSE is {ratio.r.min():.2f}-{ratio.r.max():.2f}x its validation RMSE (the test year is harder, and selection on validation is optimistic).")
    A(f"3. **Explainability.** SHAP shows that XGBoost is essentially an adaptive recent-demand model: the 7/14/28-day rolling means carry "
      f"{100 * f.shap[(f.shap.H == 7) & f.shap.feature.isin(['rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28'])].share_of_total.sum():.0f}% of the attribution at 7 days and "
      f"{100 * f.shap[(f.shap.H == 28) & f.shap.feature.isin(['rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28'])].share_of_total.sum():.0f}% at 28 days; SNAP-day counts, price and the calendar add the rest.")
    A(f"4. **Inventory (hypothetical costs, MEDIUM stockout penalty, 95% target, pooled over 18 series).** The forecast-driven policy B (validation-selected model per lead time) changes total cost "
      f"vs the historical baseline A by {sgn(g3)} (L=3, {hB[3]}), {sgn(g7)} (L=7, {hB[7]}) and {sgn(g14)} (L=14, {hB[14]}). "
      f"B is cheaper in {n_cheaper} of {n_cfg} scenario x lead-time x service-level configurations, but the bootstrap 95% interval over the 18 series excludes zero in {'none' if n_ci_excl0 == 0 else ('only ' if n_ci_excl0 < n_cfg / 2 else '') + str(n_ci_excl0)} of them "
      f"(L=7: CI {100 * p7.boot_ci_low:+.0f}% to {100 * p7.boot_ci_high:+.0f}%) - **suggestive, not conclusive**.")
    spread_med, spread_max = f.model_cost_spread("MEDIUM")
    same_pf = f.ma_same_point_forecast_as_A(7)
    A("5. **Where the gain comes from.** "
      + ("Using the *same* point forecast as A (a 28-day moving average)" if same_pf else f"Using a moving-average point forecast (window {f.ma_windows()[7]} days at H = 7; policy A uses 28 days)")
      + f" but a safety stock based on validation forecast errors instead of sqrt(L) x trailing std changes cost by {sgn(ma_gain7)} at L=7 (headline {sgn(g7)}). "
      f"For orientation, the pooled cost of the non-naive forecasting models spans a median of {spread_med:.0f}% (up to {spread_max:.0f}%) from the cheapest to the dearest model at the MEDIUM penalty. "
      "Point forecast and uncertainty estimate cannot be separated cleanly, because every model brings its own validation sigma.")
    wape_best7 = f.avc_s[(f.avc_s.scenario == 'MEDIUM') & (f.avc_s.L == 7) & (f.avc_s.SL == 0.95)].best_by_test_WAPE.iloc[0]
    A(f"6. **Accuracy is not business value.** The lowest-WAPE model is {'not ' if wape_best7 != bestcost7['model'] else ''}the lowest-cost model (e.g. L=7, MEDIUM, 95%: lowest test WAPE {wape_best7}, "
      f"lowest cost {bestcost7['model']}). The mean Spearman correlation between forecast-error rank and cost rank (excluding the degenerate Naive model) is {rho_low:+.2f} (LOW penalty), "
      f"{rho_med:+.2f} (MEDIUM) and {rho_high:+.2f} (HIGH)" + (": the more asymmetric the cost, the less a symmetric accuracy metric tells about cost." if rho_low > rho_med > rho_high else " (no monotone relation with the penalty)."))
    A(f"7. **Service levels are not delivered as designed.** At a 95% target and L=7 the realised cycle service level is {100 * a95.Cycle_Service_Level:.0f}% for A and {100 * b95.Cycle_Service_Level:.0f}% for B "
      f"(the {hB[7]} forecast errors in the test year are {float(ratio[(ratio.model == hB[7]) & (ratio.H == 7)].r.iloc[0]):.2f}x larger (RMSE) than the validation errors used to set sigma).")
    A("")
    ets_gap, xgb_never_beats_ets = f.ets_vs_best()
    hi_best28 = f.best_in_group("tier=high", 28)
    A("**Recommendation (details in section 22).** Use a simple exponential-smoothing forecast (SES / Holt) as the default"
      + f" - its test WAPE is within {ets_gap:.1f} points of the best model at every primary horizon" + (" and XGBoost is never significantly better than it in the Diebold-Mariano tests" if xgb_never_beats_ets else "")
      + ", estimate sigma from out-of-sample lead-time errors rather than sqrt(L) scaling, re-estimate it on a rolling basis, and choose the service level from the cost asymmetry "
      f"(cost-minimising target on the simulated 90/95/98% grid: {f.optimal_sl_text()}). XGBoost was not shown to be worth its extra complexity here"
      + (f"; its best use is high-volume items at horizons >= 14 days (the lowest WAPE in the high-volume tier at 28 days was {hi_best28})." if hi_best28 == "XGBoost" else f" (the lowest 28-day WAPE in the high-volume tier was {hi_best28})."))
    A("")

    # ====================== 2-3 ======================
    A("## 2. Business Problem")
    A("")
    A("A retailer must decide *how much to hold* of each item in each store. Too little stock loses sales (and customers); too much ties up capital and shelf space. Demand forecasts are only useful if they improve "
      "this decision, so the project follows **FORECAST -> DECISION -> BUSINESS IMPACT** rather than treating forecasting as a competition.")
    A("")
    A("## 3. Research Question")
    A("")
    A("> *Can accurate and explainable retail demand forecasts be converted into better inventory decisions by balancing stockout risk against inventory holding cost?*")
    A("")
    A("Sub-questions: (a) which forecasting approach is most accurate at 7/14/28 days and is the difference real? (b) what drives the ML forecasts? (c) how uncertain are forecasts? "
      "(d) does a forecast-driven reorder-point policy beat a historical-demand baseline on service, inventory and total cost? (e) does the most accurate model give the lowest cost? "
      "(f) how robust is this to costs, service level, lead time and demand type?")
    A("")

    # ====================== 4. Dataset ======================
    A("## 4. Dataset")
    A("")
    A("**M5 Forecasting - Accuracy** (Walmart item-store daily unit sales 2011-01-29 to 2016-05-22, calendar with events/SNAP, weekly sell prices). The official Kaggle download requires credentials (HTTP 401; no credentials in this environment), "
      f"so the four files were taken from two independent public Hugging Face mirrors (`{C.PRIMARY_MIRROR}`, `{C.CROSSCHECK_MIRROR}`); all files match the SHA-256 hashes published by both mirrors "
      "(`reports/data_source.md`; Kaggle's own checksums are not accessible, which is stated as a limitation of the integrity evidence).")
    A("")
    A(f"Structure checks: {int(float(f.cleaning_checks['n_series'])):,} series x {int(float(f.cleaning_checks['n_days']))} days, no missing/negative sales, {int(float(f.cleaning_checks['price_rows'])):,} price rows, "
      f"the validation file is a strict prefix of the evaluation file ({f.cleaning_checks['validation_file_is_prefix_of_evaluation']}), overall zero share {100 * float(f.cleaning_checks['zero_share_overall']):.1f}%.")
    A("")
    A("**Selected sample (TRAIN information only; algorithm in `reports/subset_selection.md`).** One store per state with the largest TRAIN volume "
      f"({', '.join(f.log['stores'])}); eligible = listed every TRAIN day from d_1 ({f.log['n_eligible_listed_all_train_days']:,} series); demand tiers by TRAIN-mean percentile bands; "
      f"2 random draws (seed {C.SEED}) per store x tier, preferring different categories and distinct items.")
    A("")
    A(md(sel[["series_id", "tier", "cat_id", "train_mean", "train_zero_share", "train_adi", "train_cv2", "train_sb_class"]].rename(columns={
        "train_mean": "mean/day", "train_zero_share": "zero share", "train_adi": "ADI", "train_cv2": "CV^2", "train_sb_class": "class"})))
    A("")
    A("**Chronological split**")
    A("")
    A(md(f.split_tab))
    A("")

    # ====================== 5. Data preparation ======================
    A("## 5. Data Preparation")
    A("")
    za = f.zero_all.set_index("category")
    A("Full PROBLEM -> DIAGNOSIS -> ACTION -> REASON log: `reports/data_cleaning.md`. Highlights:")
    A("")
    A("* Wide-to-long conversion for the selected series, merged with calendar (events, state-specific SNAP) and weekly price; sorted by item, store, date. No duplicates, no missing dates/sales, no negative sales.")
    A(f"* **Zero sales** are {100 * float(f.cleaning_checks['zero_share_overall']):.0f}% of all cells. Dataset-wide, {100 * za.iloc[1]['share_of_zero_cells']:.1f}% of zeros are *structural* (item not listed that week - identifiable because sales never occur without a price row), "
      f"{100 * za.iloc[2]['share_of_zero_cells']:.2f}% are store closures/outages (Dec 25 in every store and year plus two unexplained outages: WI_1 2011-02-02, TX_2 2015-03-24) and {100 * za.iloc[3]['share_of_zero_cells']:.1f}% are *ambiguous* "
      "(true zero demand or a stockout). Nothing is deleted or imputed.")
    A(f"* {int((f.zero_s.longest_zero_run_open_days >= 90).sum())} of the 18 selected series contain zero spells of >= 90 consecutive store-open days in TRAIN while the item stays listed - evidence that the price table does not reveal every unavailability.")
    A("* Outliers (Tukey far-out fence) are flagged but kept: they are real demand peaks that inventory must cover.")
    A("")
    A("> **Limitation that applies to every result below: observed sales may underestimate true demand during stockout periods.** The data contain no inventory or lost-sales records, so zero demand and "
      "stockouts cannot be separated; forecasts and simulated service levels are conditional on *observed* (possibly censored) sales.")
    A("")

    # ====================== 6. EDA ======================
    A("## 6. EDA")
    A("")
    e = f.eda
    A(f"(`reports/eda.md`, figures `outputs/figures/eda_*.png`; TRAIN data only.) Daily means range {e['mean'].min():.2f}-{e['mean'].max():.2f} units, zero-day share {e.zero_pct.min():.0f}%-{e.zero_pct.max():.0f}% "
      f"(median {e.zero_pct.median():.0f}%), CV {e.cv.min():.2f}-{e.cv.max():.2f}. Weekly seasonality is **weak** (robust-STL seasonal strength median {e.seasonal_strength_Fs.median():.2f}, max {e.seasonal_strength_Fs.max():.2f}; lag-7 autocorrelation "
      f"above the noise bound in {int((e.acf_lag7 > 2 / np.sqrt(sp.train_end)).sum())} of 18 series). {int(((e.trend_p_value < 0.05) & (e.trend_pct_of_mean_per_year < 0)).sum())} of 18 series show a significantly negative "
      f"trend (median {e.trend_pct_of_mean_per_year.median():.1f}% of the mean per year) - the sample consists of items listed since 2011. ADF/KPSS: "
      + ", ".join(f"{k} = {v}" for k, v in f.stat['level_conclusion'].value_counts().items()) + ".")
    A("")
    A(f"![daily demand]({FIG}eda_01_daily_rolling.png)")
    A(f"![ACF PACF]({FIG}eda_05_acf_pacf.png)")
    A(f"![intermittency]({FIG}eda_06_intermittency_map.png)")
    A("")

    # ====================== 7. Forecasting formulation ======================
    A("## 7. Forecasting Formulation")
    A("")
    A("* `y_t` = observed daily unit sales of one item in one store. At forecast origin `t` (end of day t) only information up to `t` (plus the *published* calendar) is used.")
    A("* **Target** (identical for every model): `Target_H(t) = sum(y[t+1], ..., y[t+H])`, H = 7, 14, 28 (primary). H = 3 is trained/evaluated as a *supplementary* horizon only because the inventory layer needs the 3-day lead-time demand.")
    A("* **Direct strategy** for XGBoost (one model per H, no recursion); statistical models forecast the daily path and are summed (negative daily forecasts clipped at 0).")
    A("* Metrics (on the H-day sums, pooled over series and origins): MAE, RMSE, sMAPE (0/0 := 0), WAPE = sum|e|/sum|y|. MAPE is not used (undefined at zero, asymmetric).")
    A("")

    # ====================== 8. Validation strategy ======================
    A("## 8. Validation Strategy")
    A("")
    A("Chronological two-phase protocol with daily rolling origins (`reports/validation_strategy.md`): choices on validation, one-time evaluation on test after refitting on train + validation. "
      f"Automated leakage audit: **{int(f.audit.status.eq('PASS').sum())}/{len(f.audit)} checks pass** (`reports/leakage_audit.md`) - including perturbation tests that replace all post-cut-off data by random values and verify that features, "
      "ETS/SARIMA/XGBoost forecasts, sigma_L and the inventory policy are unchanged, and a negative-control unit test showing that a deliberately leaky feature *is* detected.")
    A("")

    # ====================== 9. Baselines ======================
    A("## 9. Baselines")
    A("")
    msel = f.ma[f.ma.selected][["H", "window", "val_WAPE"]]
    ma_txt = (f"{int(msel.window.iloc[0])} days at every horizon" if msel.window.nunique() == 1 else ", ".join(f"H={int(r.H)}: {int(r.window)} days" for r in msel.itertuples()))
    A(f"Naive (last day), seasonal naive (same weekday last week), moving average (7/14/28 days; **window chosen on validation: {ma_txt}**) and Croston-SBA (alpha = 0.1; added because intermittent demand dominates).")
    A("")
    A("> **Why SeasonalNaive and MA7 have identical scores at 7, 14 and 28 days:** the sum of a seasonal-naive path over complete weeks equals H x (mean of the last 7 days) = the MA7 forecast of the sum. They differ only for H = 3 (a unit test proves the identity).")
    A("")
    A(f"Test WAPE of the validation-selected moving average: {f.wape('MovingAverage', 7):.1f}% / {f.wape('MovingAverage', 14):.1f}% / {f.wape('MovingAverage', 28):.1f}% at 7/14/28 days, versus seasonal naive "
      f"{f.wape('SeasonalNaive', 7):.1f}% / {f.wape('SeasonalNaive', 14):.1f}% / {f.wape('SeasonalNaive', 28):.1f}% and the best model {best[7]['test_WAPE']:.1f}% / {best[14]['test_WAPE']:.1f}% / {best[28]['test_WAPE']:.1f}%.")
    A("")

    # ====================== 10. ETS ======================
    A("## 10. Exponential Smoothing")
    A("")
    x = f.ets[(f.ets.phase == "val") & f.ets.aicc.notna()].pivot(index="series_id", columns="model", values="aicc")
    n_ev = int(f.ets["event"].notna().sum()) if "event" in f.ets else 0
    A(f"SES (level), Holt (additive **damped** trend - an undamped trend extrapolated up to 28 daily steps is unstable on noisy retail data) and Holt-Winters (additive weekly seasonality s = 7; "
      f"multiplicative is impossible with zero sales; without / with damped trend, chosen by AICc: {int((f.ets[(f.ets.phase == 'val') & (f.ets.model == 'HoltWinters')].estimated_kind == 'HW_N').sum())} / {int((f.ets[(f.ets.phase == 'val') & (f.ets.model == 'HoltWinters')].estimated_kind == 'HW_D').sum())} series). "
      f"Fallback hierarchy Holt-Winters -> Holt -> SES was implemented; **{n_ev} fallback events** occurred. "
      f"Weekly seasonality is {'statistically supported' if int((x.HoltWinters < x.SES).sum()) > 9 else 'not clearly supported'} in-sample (AICc of Holt-Winters < SES for {int((x.HoltWinters < x.SES).sum())} of 18 series), but because additive seasonal indices cancel over complete weeks the "
      f"gain is small for 7/14/28-day sums (test WAPE SES {f.wape('SES', 7):.1f}% vs Holt-Winters {f.wape('HoltWinters', 7):.1f}% at 7 days). Smoothing parameters are small "
      f"(median alpha {f.ets[(f.ets.phase == 'val') & (f.ets.model == 'SES')].alpha.median():.2f}): the level adapts slowly.")
    A("")

    # ====================== 11. SARIMA ======================
    A("## 11. SARIMA")
    A("")
    st = f.stat
    fmt_c = lambda d: "; ".join(f"{k.replace('S1:', '').replace('S2:', '').replace('S3:', '').replace('S4:', '').replace('S5:', '')} x{v}" for k, v in d.items())
    A(f"Stationarity (TRAIN): {', '.join(f'{k}: {v} series' for k, v in st.level_conclusion.value_counts().items())}. The ADF/KPSS conflict (mean moves slowly) and the ACF/PACF of the seasonally differenced series "
      f"(median lag-7 autocorrelation {f.eda.d7_acf_lag7.median():.2f}, range {f.eda.d7_acf_lag7.min():.2f} to {f.eda.d7_acf_lag7.max():.2f}; the PACF at the seasonal lags 7/14/21/28 is {f.eda.d7_pacf_lag7.median():.2f}/{f.eda.d7_pacf_lag14.median():.2f}/{f.eda.d7_pacf_lag21.median():.2f}/{f.eda.d7_pacf_lag28.median():.2f}; `eda_09_acf_pacf_seasonally_differenced.png`) motivate five small candidates with s = 7: "
      "`(1,0,1)(1,0,1)7+c`, `(1,0,0)(1,0,0)7+c`, `(1,0,1)(0,1,1)7`, `(0,1,1)(0,1,1)7`, `(1,1,1)(0,1,1)7`. "
      f"Selection per series by validation error (AIC/BIC logged; AIC is not comparable across differencing orders). Chosen: {fmt_c(f.sar_s.chosen.value_counts().to_dict())}. "
      f"Share of candidate fits that converged: {fmt_c(f.sar_c.groupby('candidate').converged.mean().round(2).to_dict()).replace(' x', ' ')}; fallbacks to Holt-Winters: "
      f"{int((f.sar_s.fallback_val.fillna('') != '').sum())} (validation) / {int((f.sar_s.fallback_test.fillna('') != '').sum())} (test).")
    A("")
    rs = f.res_stat.set_index("family")
    A(f"Residuals (TRAIN, one-step): mean ~ 0 (SARIMA {rs.loc['SARIMA', 'mean_resid']:+.3f}, Holt-Winters {rs.loc['HoltWinters', 'mean_resid']:+.3f}); Ljung-Box (lag 14) rejects white noise for "
      f"{100 * rs.loc['SARIMA', 'share_ljung_box_lag14_p_lt_0_05']:.0f}% (SARIMA) / {100 * rs.loc['HoltWinters', 'share_ljung_box_lag14_p_lt_0_05']:.0f}% (Holt-Winters) of series; "
      f"Jarque-Bera rejects normality for {100 * rs.loc['SARIMA', 'share_jarque_bera_p_lt_0_05']:.0f}% of series (median skew {rs.loc['SARIMA', 'median_skew']:.1f}, excess kurtosis {rs.loc['SARIMA', 'median_excess_kurtosis']:.1f}): "
      + ("daily residuals of count data are right-skewed and heavy-tailed, so some autocorrelation and non-Gaussianity remain" if rs.loc['SARIMA', 'median_skew'] > 0.5 and rs.loc['SARIMA', 'median_excess_kurtosis'] > 1
         else "see `resid_01_statistical_models.png` for the residual diagnostics") + " (`resid_01_statistical_models.png`).")
    A("")

    # ====================== 12. XGBoost ======================
    A("## 12. XGBoost")
    A("")
    lg = f.xgb_log
    gain = lg.groupby("H").apply(lambda g: 100 * (1 - g.val_MAE.min() / g[g.is_default].val_MAE.iloc[0]), include_groups=False)
    A(f"One *global* (pooled over the 18 series) squared-error model per horizon H in {{3, 7, 14, 28}}, raw units, direct target `sum(y[t+1..t+H])`. {len(FEATURES)} features in three groups - "
      "**history** (lag_1/7/14/21/28 measured from the first forecast day, rolling mean 7/14/28 and std 7/28 ending at the origin, price, week-over-week price change, price relative to its 91-day mean), "
      "**calendar_origin** (day of week/month, ISO week, month, quarter, year, weekend, event and SNAP flag of the origin day) and **known_ahead** (counts of SNAP days, event days and Christmas inside the forecast window - "
      "deterministic functions of the published calendar, tested to be independent of sales).")
    A("")
    A("Tuning (modest, validation only): default + 23 random draws from a 5-parameter space (max_depth, learning_rate, subsample, colsample_bytree, min_child_weight; n_estimators by early stopping on validation MAE) per horizon. "
      "Validation MAE improvement of the best vs the default configuration: " + ", ".join(f"H={H}: {g:.1f}%" for H, g in gain.items()) + ". Chosen settings:")
    A("")
    xch = f.xgb_ch.copy()
    for c in ("n_estimators", "train_rows", "final_fit_rows"):
        xch[c] = xch[c].astype(int)
    A(md(xch, floatfmt=".3g"))
    A("")

    # ====================== 13. Model comparison ======================
    A("## 13. Model Comparison")
    A("")
    A("**Test period, pooled over 18 series (lower is better)** - horizons 7/14/28 are primary, 3 is supplementary:")
    A("")
    for met in ("WAPE", "MAE", "RMSE", "sMAPE"):
        A(f"*{met}*")
        A("")
        A(md(_pivot(f.test, met)))
        A("")
    A("**Best model per horizon**")
    A("")
    bt = f.best[["H", "primary_horizon", "best_by_validation_WAPE", "val_WAPE", "best_by_test_WAPE", "test_WAPE", "runner_up_test_WAPE", "runner_up_WAPE", "best_by_test_RMSE", "validation_choice_equals_test_best"]]
    A(md(bt))
    A("")
    A(f"* **Short horizon (7 days):** {best[7]['best_by_test_WAPE']} ({best[7]['test_WAPE']:.1f}%), runner-up {best[7]['runner_up_test_WAPE']} ({best[7]['runner_up_WAPE']:.1f}%); XGBoost {f.wape('XGBoost', 7):.1f}%, SARIMA {f.wape('SARIMA', 7):.1f}%.")
    A(f"* **Medium (14 days):** {best[14]['best_by_test_WAPE']} ({best[14]['test_WAPE']:.1f}%), runner-up {best[14]['runner_up_test_WAPE']} ({best[14]['runner_up_WAPE']:.1f}%).")
    A(f"* **Long (28 days):** {best[28]['best_by_test_WAPE']} ({best[28]['test_WAPE']:.1f}%), runner-up {best[28]['runner_up_test_WAPE']} ({best[28]['runner_up_WAPE']:.1f}%); best simple benchmark (moving average) {f.wape('MovingAverage', 28):.1f}%.")
    A("* No single overall winner is declared. By RMSE the ranking " + ("differs" if any(best[H]['best_by_test_RMSE'] != best[H]['best_by_test_WAPE'] for H in PRIMARY) else "is the same")
      + f" (in the test year RMSE is dominated by a small set of large misses: at 7 days the worst 5% of windows produce {100 * conc7['test']['worst5'][0]:.0f}-{100 * conc7['test']['worst5'][1]:.0f}% of the squared error): "
      + ", ".join(f"H={H}: {best[H]['best_by_test_RMSE']}" for H in PRIMARY)
      + f". sMAPE is shown only for completeness: it scores a zero forecast of a zero actual as perfect, and the best 28-day sMAPE ({sz['smape']:.1f}%) belongs to {sz['model']}, which ranks {sz['wape_rank']} of {sz['n_models']} on WAPE and forecasts exactly zero in "
      f"{100 * sz['share_zero_forecast']:.1f}% of the 28-day windows (the actual is zero in {100 * sz['share_actual_zero']:.1f}%), whereas {', '.join(sz['never_zero']) or 'no model'} never forecast exactly zero.")
    A("")
    A("**Is a difference real? Diebold-Mariano (test, panel, absolute scaled loss, HAC lag H-1, Harvey-Leybourne-Newbold correction, Holm-adjusted within each horizon)** - XGBoost versus each model:")
    A("")
    A(_dm_table(f))
    A("")
    sig_worse = {H: cnt[H]["worse"] for H in PRIMARY}
    A(f"Reading: XGBoost is significantly better than another non-naive model in {sum(len(cnt[H]['better']) for H in PRIMARY)} of {sum(cnt[H]['n'] for H in PRIMARY)} comparisons and significantly worse in "
      f"{sum(len(v) for v in sig_worse.values())} ({'; '.join(f'H={H}: ' + ', '.join(v) for H, v in sig_worse.items() if v) or 'none'}). With daily origins the effective number of independent windows is about n/H "
      f"(about {int(f.dm[(f.dm.H == 28)].n.iloc[0] / 28)} at H = 28), so power is limited and non-significance is not proof of equality. Per-series DM tests (unadjusted p < 0.05, XGBoost vs moving average / Holt-Winters / SARIMA, 3 horizons x 18 series = {f.per_series_tally()[2]} tests): XGBoost significantly better in {f.per_series_tally()[0]}, significantly worse in {f.per_series_tally()[1]} (`dm_tests_per_series.csv`).")
    A("")
    tb28 = {g: f.best_in_group(g, 28) for g in ("tier=high", "tier=medium", "tier=low")}
    A("**Volume-weighted vs equal-weighted view.** Pooled WAPE/MAE are dominated by high-volume series; the DM panel test averages *scaled* losses over series with equal weight, so low-volume series count as much as busy ones. "
      f"The best 28-day model by group is {tb28['tier=high']} for the high-volume tier, {tb28['tier=medium']} for the medium tier and {tb28['tier=low']} for the low-volume tier (`robustness_best_model_by_group.csv`), "
      "so a model that does well on busy series can lead on pooled WAPE while not leading on the equal-weighted test.")
    A("")
    A(f"![model comparison]({FIG}fc_02_model_comparison_wape.png)")
    A(f"![forecast vs actual]({FIG}fc_01_forecast_vs_actual.png)")
    A(f"![error comparison]({FIG}fc_03_error_comparison_skill.png)")
    A("")
    rx = f.res_xgb.set_index("H")
    A(f"**Residual analysis of XGBoost (test).** Mean error (forecast - actual) {rx.loc[7, 'bias(mean f-y)']:+.2f} / {rx.loc[14, 'bias(mean f-y)']:+.2f} / {rx.loc[28, 'bias(mean f-y)']:+.2f} units "
      f"({rx.loc[7, 'bias_pct_of_mean_actual']:+.1f}% / {rx.loc[14, 'bias_pct_of_mean_actual']:+.1f}% / {rx.loc[28, 'bias_pct_of_mean_actual']:+.1f}% of mean demand) at 7/14/28 days"
      + (": a systematic **under-forecast** " if all(rx.loc[H, 'bias(mean f-y)'] < 0 for H in PRIMARY) else " ")
      + (f"(all {f.under_forecast_cells()[1]} model-horizon cells under-forecast on average in the test year; " if f.under_forecast_cells()[0] == f.under_forecast_cells()[1]
       else f"({f.under_forecast_cells()[0]} of {f.under_forecast_cells()[1]} model-horizon cells under-forecast on average in the test year; ")
      + "see `resid_02_xgboost_H7.png` for bias by demand level, tier, month and weekday).")
    A("")

    # ====================== 14. SHAP ======================
    A("## 14. SHAP")
    A("")
    s7 = f.shap[f.shap.H == 7].reset_index(drop=True)
    s28 = f.shap[f.shap.H == 28].reset_index(drop=True)
    A(f"SHAP `TreeExplainer` on the final (train + validation) models, {int(f.shap_chk.rows_used.iloc[0]):,} of {int(f.shap_chk.rows_available.iloc[0]):,} test-period origins (seeded random sample); additivity "
      f"`sum(SHAP) + base = prediction` verified (max error {f.shap_chk.additivity_max_abs_error.max():.1e}). A **positive** SHAP value pushes that prediction *above* the average prediction (base value), a **negative** one pushes it *below*; "
      "SHAP describes the fitted model, **not** causal effects in the world.")
    A("")
    A("Global importance (mean |SHAP|, % of total):")
    A("")
    t7 = s7.head(8)[["feature", "group", "mean_abs_shap", "share_of_total", "corr(feature value, shap)"]].rename(columns={"corr(feature value, shap)": "corr(value, SHAP)"})
    t28 = s28.head(8)[["feature", "group", "mean_abs_shap", "share_of_total", "corr(feature value, shap)"]].rename(columns={"corr(feature value, shap)": "corr(value, SHAP)"})
    A("*7-day model*")
    A("")
    A(md(t7, floatfmt=".3f"))
    A("")
    A("*28-day model*")
    A("")
    A(md(t28, floatfmt=".3f"))
    A("")
    grp = f.shap.groupby(["H", "group"]).share_of_total.sum().unstack()
    A(f"By group: history features {100 * grp.loc[7, 'history']:.0f}% (7-day) / {100 * grp.loc[28, 'history']:.0f}% (28-day), calendar-of-origin {100 * grp.loc[7, 'calendar_origin']:.0f}% / {100 * grp.loc[28, 'calendar_origin']:.0f}%, "
      f"known-ahead window counts {100 * grp.loc[7, 'known_ahead']:.0f}% / {100 * grp.loc[28, 'known_ahead']:.0f}%.")
    pr = s7[s7.feature == "price"].iloc[0]
    A("")
    rho_pl = float(stats.spearmanr(sel.train_mean_price, sel.train_mean).statistic)
    c_snap = float(s7[s7.feature == 'snap_days_window']['corr(feature value, shap)'].iloc[0])
    A("Interpretation (descriptive): higher recent average demand (`rolling_mean_*`) pushes the forecast up (correlation of value and SHAP "
      f"{s7[s7.feature == 'rolling_mean_28']['corr(feature value, shap)'].iloc[0]:.2f}); more SNAP days in the forecast window {'raise' if c_snap > 0 else 'lower'} it (corr {c_snap:+.2f}). "
      f"`price` has a {'negative' if pr['corr(feature value, shap)'] < 0 else 'positive'} relation (corr {pr['corr(feature value, shap)']:+.2f}), but because weekly prices of a given item rarely change (median {f.eda.n_price_changes.median():.0f} changes in TRAIN) this "
      + (f"most likely reflects *between-item* differences (across the 18 series the Spearman correlation between mean price and mean sales is {rho_pl:+.2f}: cheaper items sell more) rather than a price elasticity"
         if rho_pl < -0.3 else f"may reflect *between-item* differences (Spearman correlation between mean price and mean sales across the 18 series {rho_pl:+.2f}) rather than a price elasticity")
      + " - a hypothesis consistent with the data, not a causal finding. "
      f"`year` is a minor step-function feature ({100 * float(s7[s7.feature == 'year'].share_of_total.iloc[0]):.1f}% / {100 * float(s28[s28.feature == 'year'].share_of_total.iloc[0]):.1f}% of the attribution at 7 / 28 days; trees cannot extrapolate beyond the training years).")
    A("")
    A("Three individual explanations (selected by rule, not by outcome) are in `shap_individual_H7_1..3.png` (largest prediction, median prediction, Christmas in the forecast window); case table:")
    A("")
    A(md(f.shap_cases[f.shap_cases.H == 7][["case", "series_id", "origin_date", "prediction", "actual"]]))
    A("")
    A(f"![SHAP summary]({FIG}shap_summary_H7.png)")
    A(f"![SHAP importance]({FIG}shap_importance_H7.png)")
    A(f"![SHAP waterfall]({FIG}shap_individual_H7_1.png)")
    A("")

    # ====================== 15. Uncertainty ======================
    A("## 15. Forecast Uncertainty")
    A("")
    nm = f.norm[f.norm.model.isin(["XGBoost", "SARIMA", "MovingAverage"])]
    cch = f.calib_choice.set_index("method")
    chosen_method = f.calib_choice.sort_values("abs_coverage_error").iloc[0]["method"]
    A(f"Uncertainty comes from **validation residuals** of the H-day sums (models trained on TRAIN). sigma_L = RMSE of those errors per series, model and horizon (RMSE so that bias inflates the safety stock). "
      f"Normality is *not* assumed - pooled standardised validation residuals of the XGBoost/SARIMA/MA sums have skewness {nm.pooled_skew.min():.2f} to {nm.pooled_skew.max():.2f} and excess kurtosis "
      f"{nm.pooled_excess_kurtosis.min():.2f} to {nm.pooled_excess_kurtosis.max():.2f} (mild; the *daily* one-step residuals of the statistical models have median excess kurtosis {f.res_stat.median_excess_kurtosis.max():.1f}, so summing over H days moves the errors much closer to normal), although Shapiro-Wilk rejects exact normality; "
      f"consecutive-origin residuals are strongly autocorrelated (lag-1 {nm['mean_residual_lag1_autocorr(overlapping windows)'].min():.2f}-{nm['mean_residual_lag1_autocorr(overlapping windows)'].max():.2f}) because windows overlap.")
    A("")
    A(f"**Interval method chosen on validation** by a split-half calibration (quantiles/sigma from the first half of validation origins, coverage measured on the second half): "
      f"mean absolute coverage error Gaussian {cch.loc['gaussian', 'abs_coverage_error']:.3f} vs empirical {cch.loc['empirical', 'abs_coverage_error']:.3f} -> **{chosen_method}**.")
    A("")
    cv = f.cov[f.cov.model.isin(["XGBoost", "SARIMA", "MovingAverage"]) & f.cov.H.isin([7, 28]) & (f.cov.method == chosen_method)]
    cvt = cv.pivot_table(index=["model", "H"], columns="level", values="coverage").reset_index()
    cvt.columns = ["model", "H"] + [f"{int(100 * c)}% interval coverage" for c in cvt.columns[2:]]
    A(f"Honest **test** coverage of the {chosen_method} intervals (nominal 80% / 90%):")
    A("")
    A(md(cvt, floatfmt=".3f"))
    A("")
    sr = ratio[ratio.model.isin(["XGBoost", "SARIMA", "MovingAverage"])].pivot(index="model", columns="H", values="r")
    A(f"**The key warning:** test-period RMSE is {ratio.r.min():.2f}-{ratio.r.max():.2f}x the validation RMSE across models and horizons (XGBoost {sr.loc['XGBoost', 7]:.2f}x at 7 days, SARIMA {sr.loc['SARIMA', 28]:.2f}x at 28 days). "
      "A sigma estimated on validation therefore *understates* test-period error, a likely reason why the realised service level of the validation-based safety stocks falls short of the nominal target (section 17).")
    A("")
    A(f"![prediction interval]({FIG}unc_01_prediction_interval_H7.png)")
    A(f"![coverage]({FIG}unc_02_coverage_calibration.png)")
    A("")

    # ====================== 16. Inventory optimisation ======================
    A("## 16. Inventory Optimization")
    A("")
    A("**Policy: reorder point + order-up-to level (s, S) with EOQ.** At the end of each day t (after observing that day's demand), with `IP_t = on-hand + on-order`:")
    A("")
    A("```\nmu_L(t)  = forecast of demand over days t+1 ... t+L          sigma_L = std of the L-day forecast error\nSS       = z * sigma_L          (z = 1.282 / 1.645 / 2.054 for 90 / 95 / 98 % target service)\nROP_t    = mu_L(t) + SS\nQ_t      = ceil( sqrt(2 * D * S_o / H) ),  D = 365 * (28-day forecast)/28,  S_o = order cost,  H = annual holding cost per unit\nif IP_t <= ROP_t:  order  ceil(ROP_t + Q_t - IP_t)  units   (order-up-to level S_t = ROP_t + Q_t)\n```")
    A("")
    A("Timing: an order placed at the end of day t arrives at the **start of day t+L+1**, so the protection interval is exactly the L days t+1..t+L that mu_L and sigma_L describe. Unmet demand is lost (retail). "
      "Lead times 3/7/14 days are **hypothetical** (no supplier data exist). Assumptions: constant lead time, no minimum order sizes, no perishability, one product at a time.")
    A("")
    cp = C.COSTS
    A("**Hypothetical cost assumptions (not observed business costs):**")
    A("")
    A(f"* unit cost = {cp.unit_cost_ratio:.0%} of the last selling price before the test period; annual holding cost = {cp.holding_rate:.0%} of unit cost; order cost = ${cp.order_cost:.2f} per replenishment order (EOQ cycle length on TRAIN data: {f.eoq.eoq_cycle_days.min():.0f}-{f.eoq.eoq_cycle_days.max():.0f} days, median {f.eoq[f.eoq.tier == 'high'].eoq_cycle_days.median():.0f} / {f.eoq[f.eoq.tier == 'medium'].eoq_cycle_days.median():.0f} / {f.eoq[f.eoq.tier == 'low'].eoq_cycle_days.median():.0f} days for high / medium / low tiers - sanity-checked on TRAIN data only, before any validation or test result existed; `eoq_cycle_sanity_train.csv`);")
    A(f"* stockout penalty per lost unit = multiplier x unit margin: LOW = {cp.multipliers()['LOW']}x (part of the sale is recovered), MEDIUM = {cp.multipliers()['MEDIUM']}x (lost margin), HIGH = {cp.multipliers()['HIGH']}x (lost margin plus goodwill). "
      f"A newsvendor critical-ratio argument on TRAIN data (not any simulation result) was used to choose them: the median per-cycle critical ratio is {100 * f.eoq.critical_ratio_LOW.median():.0f}% (LOW), {100 * f.eoq.critical_ratio_MEDIUM.median():.0f}% (MEDIUM) and {100 * f.eoq.critical_ratio_HIGH.median():.0f}% (HIGH), " + ("so the 90/95/98% grid brackets the region of interest." if f.critical_ratio_text()[1] else
                                             "so the 90/95/98% grid does not bracket all three: at least one critical ratio lies outside the simulated service-level range (see section 19).") + "")
    A("")
    A("**Policy A - historical baseline (no forecasting model):** mu_L = L x mean of the last 28 days; sigma_L = sqrt(L) x std of the last 28 days; D = 365 x trailing mean.")
    A(f"**Policy B - forecast-driven:** mu_L and D from a forecasting model, sigma_L from that model's VALIDATION residuals of the L-day sum. Headline B model per lead time = lowest validation WAPE at H = L: "
      f"{', '.join(f'L={L_}: {m}' for L_, m in f.headB.items())}. A and B share lead time, costs, simulation period (the TEST period), initial inventory "
      "(cover of L + 14 days at the pre-test 28-day mean) and decision timing.")
    A("")

    # ====================== 17. Simulation ======================
    A("## 17. Simulation")
    A("")
    A("Daily ledger (`outputs/tables/inventory_ledger_headline_L7_SL95.csv`): receive arrivals -> begin inventory -> observe demand -> fulfil -> record stockout -> update inventory -> inventory position -> check ROP -> order -> schedule arrival -> costs. "
      "Unit tests verify inventory balance, order timing, stockouts, costs, conservation of units and independence of decisions from future demand.")
    A("")
    A("**Metric definitions.** *Fill rate* = fulfilled units / demanded units. *Cycle service level* = share of replenishment cycles (order placed -> arrival) without any stockout in the protection interval - the quantity "
      "the z-based safety stock targets. *In-stock day rate* = share of days without a stockout. *Stockouts* = number of series-days with unmet demand; *stockout units* = lost units. *Average inventory* = mean ending inventory (units, summed over series). "
      "*Turnover* = annualised fulfilled units / average inventory. *Holding cost* = h_day x ending inventory; *ordering cost* = S_o x orders; *stockout cost* = penalty x lost units; *total* = sum.")
    A("")
    A("**Headline results (MEDIUM penalty; pooled over 18 series; TEST period).** `final_inventory_results.csv` holds all three penalty scenarios.")
    A("")
    hv = f.inv_final[f.inv_final.Cost_Scenario == "MEDIUM"][["Policy", "Forecast_Model", "Lead_Time_days", "Service_Level", "Stockouts", "Stockout_Units", "Fill_Rate", "Cycle_Service_Level",
                                                              "Average_Inventory", "Holding_Cost", "Ordering_Cost", "Stockout_Cost", "Total_Cost"]].copy()
    hv["Forecast_Model"] = hv["Forecast_Model"].replace({"none (28-day trailing mean/std)": "none (trailing)"})
    A(md(hv, floatfmt=".3f"))
    A("")
    A(f"At the 95% target and L = 7, B ({hB[7]}) has {int(b95.Stockouts)} stockout series-days / {b95.Stockout_Units:.0f} lost units vs {int(a95.Stockouts)} / {a95.Stockout_Units:.0f} for A, "
      f"with average inventory {b95.Average_Inventory:.0f} vs {a95.Average_Inventory:.0f} units"
      + (": B serves more demand with *less* stock." if (b95.Fill_Rate > a95.Fill_Rate and b95.Average_Inventory < a95.Average_Inventory) else ".")
      + f" The **realised cycle service level is {'far below' if b95.Cycle_Service_Level < 0.9 else 'below'} the 95% design target** "
      f"({100 * b95.Cycle_Service_Level:.0f}% for B, {100 * a95.Cycle_Service_Level:.0f}% for A) while the in-stock day rate is {100 * b95.In_Stock_Day_Rate:.1f}% / {100 * a95.In_Stock_Day_Rate:.1f}%. "
      f"Evidence in the data: (i) test-year forecast errors are larger than the validation errors used to set sigma (RMSE ratio {ratio[ratio.H == 7].r.min():.2f}-{ratio[ratio.H == 7].r.max():.2f}x at 7 days) and RMSE grows much faster than MAE "
      f"(MAE ratio {mae_ratio[mae_ratio.H == 7].r.min():.2f}-{mae_ratio[mae_ratio.H == 7].r.max():.2f}x), the error distribution became heavier-tailed: the worst 5% of windows produce {100 * conc7['test']['worst5'][0]:.0f}-{100 * conc7['test']['worst5'][1]:.0f}% of the squared error in the test period versus {100 * conc7['val']['worst5'][0]:.0f}-{100 * conc7['val']['worst5'][1]:.0f}% in validation (RMSE/MAE {conc7['test']['rmse_mae'][0]:.2f}-{conc7['test']['rmse_mae'][1]:.2f} vs {conc7['val']['rmse_mae'][0]:.2f}-{conc7['val']['rmse_mae'][1]:.2f}; `error_concentration.csv`), so a small set of large misses dominates, although mean demand barely changed ({100 * (f.test[(f.test.model == 'XGBoost') & (f.test.H == 7)].mean_actual.iloc[0] / f.val[(f.val.model == 'XGBoost') & (f.val.H == 7)].mean_actual.iloc[0] - 1):+.0f}%); "
      f"(ii) forecasts {'under-predict' if f.metric('XGBoost', 7, 'bias') < 0 else 'over-predict'} on average (XGBoost bias {f.metric('XGBoost', 7, 'bias'):+.2f} units at 7 days). "
      "A nominal service level is therefore a *design input*, not a guarantee.")
    A("")
    A(f"![trajectory]({FIG}inv_01_trajectory_FOODS_2_347_TX_2.png)")
    A(f"![stockouts]({FIG}inv_02_stockout_periods.png)")
    A("")

    # ====================== 18. Cost analysis ======================
    A("## 18. Cost Analysis")
    A("")
    A(f"![cost components]({FIG}inv_05_total_cost_components.png)")
    A("")
    A("**Paired comparison B vs A (bootstrap over the 18 series, 5,000 resamples; relative change of pooled total cost, negative = B cheaper), MEDIUM penalty:**")
    A("")
    pt = f.pair[f.pair.scenario == "MEDIUM"][["L", "SL", "policy_B", "sum_A", "sum_B", "rel_diff_pooled", "boot_ci_low", "boot_ci_high", "series_B_cheaper", "sign_test_p"]].copy()
    for c in ("rel_diff_pooled", "boot_ci_low", "boot_ci_high"):
        pt[c] = 100 * pt[c]
    pt = pt.rename(columns={"policy_B": "B model", "rel_diff_pooled": "change %", "boot_ci_low": "CI low %", "boot_ci_high": "CI high %", "series_B_cheaper": "series B cheaper (of 18)"})
    A(md(pt, floatfmt=".2f"))
    A("")
    exc = f.pair[f.pair.rel_diff_pooled >= 0]
    exc_txt = "; ".join(f"{r.scenario} L={int(r.L)} {int(100 * r.SL)}% ({100 * r.rel_diff_pooled:+.1f}%)" for r in exc.itertuples()) or "none"
    A(f"B is cheaper in {n_cheaper} of {n_cfg} configurations (3 penalty scenarios x 3 lead times x 3 service levels); the cells where it is not: {exc_txt}. "
      f"The 95% bootstrap interval excludes zero in {n_ci_excl0} of {n_cfg} cells; series are not independent (shared calendar shocks, 3 stores), so even these intervals are optimistic.")
    A("")
    A("**Decomposition of the gain (L = 7, 95%, MEDIUM; same cost model, pooled total cost):**")
    A("")
    dec = []
    ref = f.agg("A_historical", 7, 0.95)["total_cost"]
    for label, pol in (("A: historical (trailing mean, sqrt(L) x trailing std)", "A_historical"),
                       ("B with the SAME point forecast (28-day MA) but validation-residual sigma" if f.ma_same_point_forecast_as_A(7)
                        else f"B with a moving-average point forecast (window {f.ma_windows()[7]} days) and validation-residual sigma", "MovingAverage"),
                       (f"B headline ({hB[7]}, chosen by validation WAPE)", hB[7]), ("B XGBoost", "XGBoost"),
                       ("B SeasonalNaive", "SeasonalNaive"), ("B Holt", "Holt"), (f"B {f.cheapest_ex_post()} (lowest cost ex post)", f.cheapest_ex_post())):
        r = f.agg(pol, 7, 0.95)
        dec.append({"policy": label, "total cost": r["total_cost"], "vs A %": 100 * (r["total_cost"] / ref - 1), "fill rate": r["fill_rate"], "cycle service level": r["cycle_service_level"],
                    "avg inventory": r["avg_inventory_units"]})
    A(md(pd.DataFrame(dec), floatfmt=".3f"))
    A("")
    share_sigma = 100 * ma_gain7 / g7 if g7 != 0 else float("nan")
    cheap7 = f.cheapest_ex_post()
    rb7, rp7 = f.bias_vs_lost(7)
    A(f"Lessons: (i) {share_sigma:.0f}% of the headline A -> B cost change at L = 7 ({sgn(ma_gain7)} of {sgn(g7)}) appears " + ("even with an unchanged point forecast" if f.ma_same_point_forecast_as_A(7) else "with a moving-average point forecast close to A's")
      + ", i.e. it comes from *measuring the lead-time error distribution* instead of assuming i.i.d. daily demand; "
      f"(ii) the model that wins on cost ({'a simple exponential smoother, ' if cheap7 in ('SES', 'Holt', 'HoltWinters') else ''}{cheap7}, found *ex post* on the test data) was "
      + ("not the one that validation accuracy selected" if cheap7 != hB[7] else "also the one that validation accuracy selected") + ", so it is not an achievable benchmark; "
      f"(iii) forecast **bias** goes with stockouts: across the {f.n_nonnaive()} non-naive models the Spearman correlation between test bias at H = L and lost units is {f.bias_vs_lost(3)[0]:+.2f} (L=3), {rb7:+.2f} (L=7, p = {rp7:.3f}) and "
      f"{f.bias_vs_lost(14)[0]:+.2f} (L=14, p = {f.bias_vs_lost(14)[1]:.3f})" + (": the more a model under-forecasts, the more sales it loses (an association across few models, not proof)." if rb7 < 0 else "."))
    A("")

    # ====================== 19. Sensitivity ======================
    A("## 19. Sensitivity Analysis")
    A("")
    A("Grid: 3 stockout-penalty scenarios x 3 target service levels x 3 lead times, for A and the validation-selected B (`sensitivity_headline_policies.csv`; all models in `inventory_all_policies_aggregated.csv`).")
    A("")
    A(f"![sensitivity heatmaps]({FIG}inv_06_sensitivity_heatmaps.png)")
    A(f"![service vs cost]({FIG}inv_03_service_level_vs_cost.png)")
    A("")
    o = f.opt[f.opt.policy_group == "B_forecast_driven"].pivot(index="L", columns="scenario", values="cost_minimising_SL")
    oa = f.opt[f.opt.policy_group == "A_historical"].pivot(index="L", columns="scenario", values="cost_minimising_SL")
    A("**Cost-minimising target service level among {90, 95, 98}%** (policy B / policy A):")
    A("")
    A(md(pd.DataFrame({sc: [f"{100 * o.loc[Lx, sc]:.0f}% / {100 * oa.loc[Lx, sc]:.0f}%" for Lx in o.index] for sc in ("LOW", "MEDIUM", "HIGH")}, index=[f"L={Lx}" for Lx in o.index]), index=True))
    A("")
    top_sl, bot_sl = max(C.SERVICE_LEVELS), min(C.SERVICE_LEVELS)
    edge_hi = [sc for sc in C.SCENARIOS if set(f.opt[(f.opt.policy_group == "B_forecast_driven") & (f.opt.scenario == sc)].cost_minimising_SL) == {top_sl}]
    edge_lo = [sc for sc in C.SCENARIOS if set(f.opt[(f.opt.policy_group == "B_forecast_driven") & (f.opt.scenario == sc)].cost_minimising_SL) == {bot_sl}]
    cr_txt, cr_inside = f.critical_ratio_text()
    edge_txt = ((f"For {' and '.join(edge_hi)} the optimum sits on the upper edge of the grid ({100 * top_sl:.0f}%), so the true cost-optimal level may be even higher. " if edge_hi else "")
                + (f"For {' and '.join(edge_lo)} it sits on the lower edge ({100 * bot_sl:.0f}%), so the true cost-optimal level may be lower. " if edge_lo else ""))
    A(edge_txt + f"The critical-ratio logic computed from the cost parameters alone suggests about {cr_txt} per cycle for {' / '.join(C.SCENARIOS)}"
      + ("; " if cr_inside else " (at least one value lies outside the simulated grid); ")
      + "the simulated optimum and the critical ratio need not coincide because the simulation includes lead-time demand uncertainty that is under-estimated out of sample (section 17).")
    A("")
    A("**Which assumption matters most?** Relative range of the mean total cost when one factor moves over its range (others averaged):")
    A("")
    A(md(f.factor[["policy_group", "factor", "relative_range_of_mean_total_cost", "levels"]], floatfmt=".3f"))
    A("")
    fr = f.factor_ranges()
    order = sorted(fr.items(), key=lambda kv: -kv[1])
    A("Ranking for B by the range of mean total cost: " + ", then ".join(f"{k} ({v:.0f}%)" for k, v in order) + ". "
      + (f"Among the factors a planner can control, **shortening lead time is worth {'far ' if fr['lead time'] > 2 * fr['service level'] else ''}more than raising the service level** (range {fr['lead time']:.0f}% versus {fr['service level']:.0f}%)."
         if fr['lead time'] > fr['service level'] else "Raising the service level moved cost at least as much as the lead time did."))
    A("")
    ge = f.inv_all[(f.inv_all.scenario == "MEDIUM") & (f.inv_all.SL == 0.95) & f.inv_all.policy.isin(["SARIMA", "XGBoost", "SES", "MovingAverage"])]
    gp = ge.pivot_table(index=["policy", "L"], columns="method", values="total_cost")
    eg = f.emp_vs_gauss()
    A(f"*Safety-stock rule:* replacing the Gaussian z x sigma by the empirical quantile of validation residuals changes pooled total cost by {100 * ((gp.empirical / gp.gaussian) - 1).abs().median():.1f}% (median absolute change over {len(gp)} cells; range "
      f"{100 * ((gp.empirical / gp.gaussian) - 1).min():+.1f}% to {100 * ((gp.empirical / gp.gaussian) - 1).max():+.1f}%), compared with the {abs(g7):.0f}% headline A -> B difference at L = 7. Over all {len(eg)} model x lead time x service-level cells it changes "
      f"the realised cycle service level by a mean of {eg.cycle_pp.mean():+.1f} points (range {eg.cycle_pp.min():+.1f} to {eg.cycle_pp.max():+.1f}) and the fill rate by at most {eg.fill_pp.abs().max():.1f} points, whereas the shortfall of the realised cycle service level "
      f"against the 95% target for B at L = 7 is {95 - 100 * b95.Cycle_Service_Level:.0f} points"
      + (": the rule does not repair the shortfall." if abs(eg.cycle_pp.mean()) < 0.5 * max(95 - 100 * b95.Cycle_Service_Level, 0) else ": the rule changes the shortfall materially."))
    A("")

    # ====================== 20. Robustness ======================
    A("## 20. Robustness")
    A("")
    A(f"Groups use TRAIN information only (selection tiers; ADI >= {C.ADI_CUTOFF} = intermittent). With {int(f.rob_best.n_series.min())}-{int(f.rob_best.n_series.max())} series per group these are descriptive.")
    A("")
    rb = f.rob_best[["group", "H", "n_series", "best_model", "WAPE", "SeasonalNaive_WAPE", "MovingAverage_WAPE", "SARIMA_WAPE", "XGBoost_WAPE"]]
    A(md(rb))
    A("")
    ri = f.rob_i
    A("Inventory, B vs A at L = 7, 95%, MEDIUM:")
    A("")
    A(md(ri[["group_type", "group", "n_series", "B_model", "A_total_cost", "B_total_cost", "B_vs_A_total_cost_pct", "A_fill_rate", "B_fill_rate"]]))
    A("")
    intr = ri[ri.group == "intermittent"].iloc[0]
    reg = ri[ri.group == "regular"].iloc[0]
    A(f"* **Stable vs intermittent.** For the {int(intr.n_series)} intermittent/lumpy series B changes cost by {intr.B_vs_A_total_cost_pct:+.1f}% (fill rate {intr.A_fill_rate:.3f} -> {intr.B_fill_rate:.3f}); for the {int(reg.n_series)} regular series by "
      f"{reg.B_vs_A_total_cost_pct:+.1f}% (fill rate {reg.A_fill_rate:.3f} -> {reg.B_fill_rate:.3f}). "
      + ("The gain is concentrated in the intermittent series, which is consistent with (though no proof of) classical i.i.d. sigma assumptions being worst for intermittent demand; " if intr.B_vs_A_total_cost_pct < reg.B_vs_A_total_cost_pct - 2
         else "No clear difference between the two groups is visible; ")
      + f"no general conclusion is drawn from the {int(reg.n_series)} regular series.")
    rbf = f.rob_best
    xg_wins = rbf[(rbf.best_model == "XGBoost")][["group", "H"]].apply(lambda r: f"{r.group}@{int(r.H)}", axis=1).tolist()
    low_groups = rbf[rbf.group.isin(["tier=low", "tier=medium", "intermittent"])]
    simple_wins = int(low_groups.best_model.isin(["SES", "Holt", "HoltWinters", "MovingAverage"]).sum())
    A(f"* **Forecast error by group.** XGBoost has the lowest WAPE in: {', '.join(xg_wins) if xg_wins else 'no group'}; exponential smoothing or the moving average win in {simple_wins} of {len(low_groups)} (group x horizon) cells of the low-demand, medium-demand and intermittent groups - consistent with a pooled model dominated by high-volume series.")
    A("")
    A(f"![robustness]({FIG}robust_01_tiers_and_regularity.png)")
    A("")

    # ====================== 21. Results ======================
    A("## 21. Results")
    A("")
    A("### 21.1 Final tables")
    A("")
    A("`outputs/tables/final_forecasting_results.csv` (test, pooled; primary horizons):")
    A("")
    ff = rd("final_forecasting_results.csv")
    A(md(ff.sort_values(["Horizon", "WAPE"])))
    A("")
    A("`outputs/tables/final_inventory_results.csv` - see section 17 (MEDIUM) and the file for LOW / HIGH.")
    A("")
    A("### 21.2 Does the lowest forecasting error give the lowest inventory cost? (Phase 30)")
    A("")
    sm_ = f.avc_s[(f.avc_s.SL == 0.95)][["scenario", "L", "best_by_val_WAPE", "best_by_test_WAPE", "best_by_total_cost", "spearman_testWAPE_vs_cost", "spearman_testWAPE_vs_cost_excl_Naive", "regret_of_val_choice_pct"]]
    A(md(sm_, floatfmt=".2f"))
    A("")
    same = int(f.avc_s.same_best_test_accuracy_and_cost.sum())
    A(f"**Answer: {'usually no' if same < len(f.avc_s) / 2 else 'often yes'}.** The model with the lowest test WAPE is also the cheapest in {same} of {len(f.avc_s)} (scenario x L x service level) cells. Mean Spearman rho between error rank and cost rank "
      f"(excluding Naive, n = {f.n_nonnaive()} models): LOW {rho_low:+.2f}, MEDIUM {rho_med:+.2f}, HIGH {rho_high:+.2f}. `regret_of_val_choice_pct` is the cost excess of the validation-accuracy choice over the (ex post) cheapest model.")
    A("")
    A("**Why accuracy and cost differ**")
    A("")
    polB = hB[7]
    nd_b, nd_a, nd_days = f.lost_nov_dec(7)
    inv7 = f.inv_all[(f.inv_all.scenario == "MEDIUM") & (f.inv_all.method == "gaussian") & (f.inv_all.SL == 0.95) & (f.inv_all.L == 7)].set_index("policy")
    sig7 = f.sigma[f.sigma.H == 7].groupby("model").sigma_rmse.mean()
    bias7 = f.test[f.test.H == 7].set_index("model")["bias"].drop(index="Naive").sort_values()
    lowcost = f.inv_final[(f.inv_final.Policy == "B_forecast_driven") & (f.inv_final.Lead_Time_days == 7) & (f.inv_final.Service_Level == 0.95)].set_index("Cost_Scenario")
    share_hold = {k: 100 * (v.Holding_Cost + v.Ordering_Cost) / v.Total_Cost for k, v in lowcost.iterrows()}
    r7 = ratio[ratio.H == 7].set_index("model")["r"]
    n_sn, n_cells = f.n_cheapest("SeasonalNaive")
    wc_ok, wc_models = f.winners_curse(7)
    hi_share = 100 * float(f.rob_i[(f.rob_i.group == 'tier=high') | (f.rob_i.group == 'high')].B_total_cost.iloc[0]) / float(f.rob_i[f.rob_i.group_type == 'tier'].B_total_cost.sum())
    n_hi = int(f.rob_i[(f.rob_i.group == 'tier=high') | (f.rob_i.group == 'high')].n_series.iloc[0])
    all_under = f.under_forecast_cells()[0] == f.under_forecast_cells()[1]
    A(f"1. *Asymmetric loss and honest uncertainty.* WAPE/RMSE penalise over- and under-forecasts equally; inventory cost does not (a lost unit costs penalty x margin, an unneeded unit only a small holding cost per day), and the best forecast for a newsvendor-type cost is a *quantile*, not the mean. "
      f"A noisy forecaster also gets a large validation sigma and therefore a large safety stock: at L = 7 the mean sigma is {sig7['SeasonalNaive']:.1f} (seasonal naive) and {sig7['Naive']:.1f} (naive) versus {sig7[['XGBoost', 'SARIMA', 'SES', 'Holt']].min():.1f}-{sig7[['XGBoost', 'SARIMA', 'SES', 'Holt']].max():.1f} for the leading models, "
      f"and the average inventory is {inv7.loc['SeasonalNaive', 'avg_inventory_units']:.0f} units (seasonal naive) versus {inv7.loc[['SARIMA', 'XGBoost', 'SES', 'Holt'], 'avg_inventory_units'].min():.0f}-{inv7.loc[['SARIMA', 'XGBoost', 'SES', 'Holt'], 'avg_inventory_units'].max():.0f}. "
      f"The extra stock buys protection ({int(inv7.loc['SeasonalNaive', 'stockout_units'])} lost units for seasonal naive vs {int(inv7.loc['SARIMA', 'stockout_units'])} for SARIMA at L = 7 / 95%)"
      + (f", and with the lost-sale penalties assumed here this can pay: seasonal naive is the cheapest model in {n_sn} of {n_cells} cells although its 7-day WAPE is {f.wape('SeasonalNaive', 7):.1f}% against {best[7]['test_WAPE']:.1f}% for the best model - an *inaccurate but honest* forecaster can win on cost." if n_sn > 0 else "."))
    A("2. *Bias versus scatter.* Safety stock is designed to absorb random scatter, whereas a systematic under-forecast shifts the whole lead-time demand distribution. "
      + ("Every model under-forecasts in the test year" if all_under else f"{f.under_forecast_cells()[0]} of {f.under_forecast_cells()[1]} model-horizon cells under-forecast in the test year")
      + f" (most at 7 days: {bias7.index[0]} {bias7.iloc[0]:+.2f}, {bias7.index[1]} {bias7.iloc[1]:+.2f}), and across models the Spearman correlation between bias and lost units is {f.bias_vs_lost(7)[0]:+.2f} at L = 7.")
    A("3. *Sigma estimated on validation is optimistic for the models that were selected or tuned on validation.* "
      + (f"The models with the smallest validation sigma ({', '.join(wc_models)}) are also the three with the largest test/validation RMSE ratio at 7 days " if wc_ok else "At 7 days the test/validation RMSE ratios were ")
      + f"(XGBoost {r7['XGBoost']:.2f}x, MovingAverage {r7['MovingAverage']:.2f}x, SARIMA {r7['SARIMA']:.2f}x vs exponential smoothing {r7['SES']:.2f}x, seasonal naive {r7['SeasonalNaive']:.2f}x)"
      + (" - a winner's-curse pattern that is consistent with, though not proven to cause, their under-protection." if wc_ok else "."))
    A("4. *Timing and aggregation.* "
      + (f"Lost units concentrate in the holiday season: November-December hold {nd_b:.0f}% of the lost units of B ({polB}) and {nd_a:.0f}% of A's while being {nd_days:.0f}% of the test days, whereas WAPE averages over all windows" if nd_b > 1.25 * nd_days
         else f"November-December hold {nd_b:.0f}% of the lost units of B ({polB}) and {nd_a:.0f}% of A's against {nd_days:.0f}% of the test days (no strong seasonal concentration)")
      + f"; and the {n_hi} high-demand series account for {hi_share:.0f}% of the pooled total cost of B (MEDIUM, L = 7, 95%).")
    A(f"5. *The cost structure decides what matters.* For B at L = 7 / 95%, holding + ordering make up {share_hold['LOW']:.0f}% of total cost with the LOW penalty but {share_hold['MEDIUM']:.0f}% (MEDIUM) and only {share_hold['HIGH']:.0f}% (HIGH); "
      + (f"where holding dominates (LOW, mean rho = {rho_low:+.2f}) a symmetric accuracy metric is a reasonable proxy for cost, where lost sales dominate it is not (MEDIUM {rho_med:+.2f}, HIGH {rho_high:+.2f})." if rho_low > max(rho_med, rho_high)
         else f"mean rho is {rho_low:+.2f} (LOW), {rho_med:+.2f} (MEDIUM) and {rho_high:+.2f} (HIGH) - no clear link between the cost structure and the usefulness of accuracy as a proxy."))
    A("")
    A(f"![accuracy vs cost]({FIG}inv_07_accuracy_vs_cost.png)")
    A("")

    # ====================== 22. Recommendations ======================
    A("## 22. Business Recommendations")
    A("")
    sensB = f.sens[f.sens.policy_group == "B_forecast_driven"]
    by_L = sensB.groupby("L").total_cost.mean()
    by_SL = sensB.groupby("SL").total_cost.mean()
    cnt_s = f.dm_counts("SARIMA")
    same_best_n = int(f.avc_s.same_best_test_accuracy_and_cost.sum())
    x_better_all = sum(len(cnt[H]["better"]) for H in PRIMARY)
    s_better_all = sum(len(cnt_s[H]["better"]) for H in PRIMARY)
    hi28 = f.rob_best[(f.rob_best.group == "tier=high") & (f.rob_best.H == 28)].iloc[0]
    hi14 = f.rob_best[(f.rob_best.group == "tier=high") & (f.rob_best.H == 14)].iloc[0]
    A("**Which model, for which horizon?**")
    A("")
    ets_gap, xgb_never_beats_ets = f.ets_vs_best()
    lowg = [(g, H) for g in ("tier=low", "intermittent") for H in PRIMARY]
    simple = {"SES", "Holt", "HoltWinters", "MovingAverage"}
    n_simple_low = sum(f.best_in_group(g, H) in simple for g, H in lowg)
    xg_hi = [H for H in (14, 28) if f.best_in_group("tier=high", H) == "XGBoost"]
    n_sn, n_cells = f.n_cheapest("SeasonalNaive")
    A(f"* **Default - all items at 7 days, and low-volume / intermittent items:** exponential smoothing (SES / Holt). {best[7]['best_by_test_WAPE']} had the lowest 7-day test WAPE ({best[7]['test_WAPE']:.1f}%); the best ETS-family model is within {ets_gap:.1f} points of the best model at every primary horizon, and "
      f"exponential smoothing or the moving average has the lowest WAPE in {n_simple_low} of {len(lowg)} (low-volume tier or intermittent group) x horizon cells. Neither XGBoost (significantly better in {x_better_all} of {n_x_cmp} DM comparisons) nor SARIMA (significantly better in {s_better_all} of {sum(cnt_s[H]['n'] for H in PRIMARY)}) "
      + ("significantly beat the ETS family / moving average at any primary horizon. " if x_better_all == 0 and s_better_all == 0 else "was never significantly better than every simple model; see the DM table. ")
      + "These models are cheap, transparent and easy to monitor.")
    A((f"* **High-volume items at {' and '.join(str(H) for H in xg_hi)} days:** XGBoost had the lowest WAPE in the high-volume tier ({hi14['XGBoost_WAPE']:.1f}% at 14 days and {hi28['XGBoost_WAPE']:.1f}% at 28 days, vs SARIMA {hi14['SARIMA_WAPE']:.1f}% / {hi28['SARIMA_WAPE']:.1f}%) "
       f"and the lowest pooled WAPE at {', '.join(str(H) for H in PRIMARY if best[H]['best_by_test_WAPE'] == 'XGBoost') or 'no primary horizon'} days, but the pooled DM tests show {'no significant advantage' if x_better_all == 0 else 'only a partial advantage'}, "
       "so adopt it only where its explanations (SHAP) or extra covariates (promotions, richer prices) add value.") if xg_hi else
      "* **High-volume items:** XGBoost did not have the lowest WAPE in the high-volume tier at 14 or 28 days; no case for it was found on accuracy alone.")
    A(f"* **Statistical vs ML:** prefer statistical forecasting as the production default and use ML where it earns its complexity. Naive and seasonal-naive forecasts are clearly less accurate (28-day test WAPE {f.wape('SeasonalNaive', 28):.1f}% vs {best[28]['test_WAPE']:.1f}% for the best model; significantly worse in the DM tests) "
      + (f"but seasonal naive was the cheapest policy in {n_sn} of {n_cells} cost cells, because its large validation sigma builds a large safety stock - a cost benchmark that rewards insurance rather than forecasting skill, so cost alone is not a reason to choose a forecaster." if n_sn > 0 else "and never the cheapest policy."))
    A("")
    A("**What service level under each cost scenario?** Cost-minimising target on the simulated {90, 95, 98}% grid (section 19): "
      + "; ".join(f"{sc}: {', '.join(sorted({f'{100 * v:.0f}%' for v in f.opt[(f.opt.policy_group == 'B_forecast_driven') & (f.opt.scenario == sc)].cost_minimising_SL}))}" for sc in C.SCENARIOS)
      + ". Where the optimum is the top of the grid (98%) plan for 98% or more. Because the realised cycle service level falls short of the nominal target, monitor the *realised* service level and raise the nominal level or inflate sigma until the realised level meets the business target.")
    A("")
    A(f"**Which lead-time assumption matters most?** Mean total cost of policy B (averaged over the other factors) is {by_L.loc[3]:,.0f} / {by_L.loc[7]:,.0f} / {by_L.loc[14]:,.0f} for L = 3 / 7 / 14 days, "
      f"but only {by_SL.loc[0.9]:,.0f} / {by_SL.loc[0.95]:,.0f} / {by_SL.loc[0.98]:,.0f} for 90 / 95 / 98% service: going from 14 to 7 days of lead time saves {100 * (1 - by_L.loc[7] / by_L.loc[14]):.0f}%, "
      f"going from 90% to 98% service saves {100 * (1 - by_SL.loc[0.98] / by_SL.loc[0.9]):.0f}%" + (", and the stockout-penalty assumption moves cost by more than either" if f.factor_ranges()['stockout-penalty scenario'] > max(f.factor_ranges()['lead time'], f.factor_ranges()['service level']) else "")
      + ". The first practical step is therefore to measure the real cost of a lost sale and the real supplier lead-time distribution.")
    A("")
    reasons = []
    if x_better_all == 0 and s_better_all == 0:
        reasons.append("the leading forecasting families are statistically hard to separate (neither XGBoost nor SARIMA is significantly better than the simple models in any DM comparison)")
    else:
        reasons.append(f"XGBoost is significantly better than another non-naive model in {x_better_all} of {n_x_cmp} DM comparisons and SARIMA in {s_better_all}, so the accuracy gain is at most partial")
    reasons.append(f"a large part of the cost change comes from out-of-sample uncertainty estimation, which adds no model complexity ({sgn(ma_gain7)} at L=7" + (" with the same point forecast and a validation-based sigma" if f.ma_same_point_forecast_as_A(7) else " with a moving-average point forecast and a validation-based sigma") + f" versus {sgn(g7)} for the headline model)"
                   if abs(ma_gain7) > 0.5 * abs(g7) else f"the sigma method alone explains only part of the cost change ({sgn(ma_gain7)} vs {sgn(g7)} at L=7)")
    reasons.append(f"the accuracy ranking {'did not identify' if same_best_n < len(f.avc_s) / 2 else 'often identified'} the cheapest inventory policy (the lowest-WAPE model was the cheapest in {same_best_n} of {len(f.avc_s)} cells; section 21.2)")
    A("**Does advanced forecasting justify its complexity?** On this evidence **not by itself**: " + "; ".join(f"({i}) {r}" for i, r in enumerate(reasons, 1)) + ". A complex forecaster is worth deploying only if it improves the *decision* "
      "(e.g. through quantile / cost-aware training or richer drivers) rather than WAPE.")
    A("")
    A("**Major risks:** hypothetical costs (no absolute savings can be claimed); censored demand; test-year errors "
      f"{ratio.r.min():.1f}-{ratio.r.max():.1f}x validation errors (sigma under-estimation: re-estimate on a rolling basis); only 18 series from {len(f.log['stores'])} stores and one {sp.n_days - sp.val_end}-day test period; validation-based model selection is optimistic; assortment survivorship "
      f"(items listed since 2011; {int(((f.eda.trend_p_value < 0.05) & (f.eda.trend_pct_of_mean_per_year < 0)).sum())} of 18 series have a significantly negative trend); weak price/promotion information; unmodelled supply constraints (pack sizes, shelf life, variable lead times).")
    A("")

    # ====================== 23. Limitations ======================
    A("## 23. Limitations")
    A("")
    long_zero = int((f.zero_s.longest_zero_run_open_days >= 240).sum())
    for t in [
        f"**Censored demand.** Observed sales may underestimate true demand during stockout periods; zero demand and stockouts are indistinguishable in M5 ({long_zero} of the 18 series have TRAIN zero spells of >= 240 store-open days while the item stays listed).",
        "**Hypothetical economics.** Lead times, unit costs, holding rate, order cost and stockout penalties are assumptions, not data; the *direction* of findings was checked across three penalty scenarios, absolute dollar figures carry no business meaning.",
        "**Small, specific sample.** 18 item-store series from 3 stores, all listed continuously since 2011 (eligibility rule); results do not generalise automatically to new items, other categories or other retailers. High-volume extremes (> P97) are not represented.",
        f"**One test period.** {sp.n_days - sp.val_end} days ({sp.date_range('test')[0].date()} to {sp.date_range('test')[1].date()}) that proved harder than validation for {'every' if ratio.r.min() > 1 else 'most'} model (RMSE ratio {ratio.r.min():.2f}-{ratio.r.max():.2f}x); overlapping windows give far fewer independent observations than origins; bootstrap/DM intervals are wide and the series are cross-sectionally dependent.",
        f"**Model selection optimism.** Validation-selected quantities (SARIMA structure per series, MA window, XGBoost hyper-parameters) look better on validation than on test; the test table is the unbiased one. The headline Policy-B model was chosen by validation accuracy and was the cost-cheapest model on test in {f.val_choice_cheapest()[0]} of {f.val_choice_cheapest()[1]} cells.",
        "**Features.** Only history, calendar, SNAP and price at the origin are used; no promotions, weather or competitor data. The known-ahead calendar features are legitimate only because the calendar is published in advance.",
        "**Statistical models are univariate** and cannot use the Christmas closure / SNAP calendar that XGBoost sees (a potential advantage of the ML model on those days that this study did not isolate).",
        "**Supply side.** Constant lead time, no minimum order quantity or case packs, no perishability, no budget/space constraints, independent items; sigma_L is constant over the test period (no rolling re-estimation).",
        f"**Service-level definitions.** Nominal z-based targets are *cycle* service levels under normality and a stationary sigma; the realised cycle service level at the 95% target and L = 7 was {100 * b95.Cycle_Service_Level:.0f}% (B) / {100 * a95.Cycle_Service_Level:.0f}% (A) (section 17).",
        "**Integrity evidence.** The official Kaggle checksums could not be accessed; integrity rests on agreement between two independent mirrors and structural checks.",
    ]:
        A(f"* {t}")
    A("")

    # ====================== 24. Conclusion ======================
    A("## 24. Conclusion")
    A("")
    gi, gr, ni, nr = f.intermittent_vs_regular()
    concl_a = (f"Forecasts *can* be turned into better inventory decisions in this sample: a forecast-driven reorder-point policy lowered pooled total cost versus a historical-demand baseline in {n_cheaper} of {n_cfg} cells "
               f"({sgn(g7)} at L=7 / 95% / MEDIUM with the validation-selected model)") if n_cheaper > n_cfg / 2 else (
               f"Forecast-driven inventory decisions were cheaper than the historical baseline in only {n_cheaper} of {n_cfg} cells ({sgn(g7)} at L=7 / 95% / MEDIUM)")
    concl_b = (", with the benefit concentrated in intermittent items" if gi < gr - 2 else "") + (" and in the way uncertainty is estimated" if abs(ma_gain7) > 0.5 * abs(g7) else "")
    concl_c = (f", but the evidence is suggestive rather than conclusive (bootstrap intervals include zero in {n_cfg - n_ci_excl0} of {n_cfg} cells, hypothetical costs, one test year). " if n_ci_excl0 < n_cfg / 2
               else f"; the bootstrap intervals exclude zero in {n_ci_excl0} of {n_cfg} cells, but costs are hypothetical and there is one test year. ")
    concl_d = ((f"The *most accurate* model was usually not the *cheapest* one ({same_best_n} of {len(f.avc_s)} cells agree), " if same_best_n < len(f.avc_s) / 2 else "")
               + ("the leading forecasting families were statistically hard to separate, " if x_better_all == 0 else "")
               + (f"and nominal service levels were not realised out of sample (cycle service level {100 * b95.Cycle_Service_Level:.0f}% against a 95% target at L=7). " if b95.Cycle_Service_Level < 0.95 else ""))
    A(concl_a + concl_b + concl_c + concl_d +
      "The practical message is to optimise the decision (calibrated lead-time uncertainty, bias control, cost-aware service level, shorter lead times) before optimising model complexity. "
      "Everything is reproducible with `python run_project.py`.")
    A("")
    A("---")
    A("*Appendix: file map in `README.md`; figures in `outputs/figures`; all tables in `outputs/tables`; tests in `tests/` (see `reports/test_results.txt`).*")
    (C.REPORTS / "final_report.md").write_text("\n".join(L) + "\n")
