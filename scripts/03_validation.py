"""PD model validation: discrimination tests, recalibration, backtesting,
stability (PSI/CSI) and vintage analysis.

Protocol: every choice (champion, recalibration method, stacking weights) is
made on the 2016 OOT test year; 2017 is only scored once, at the end.
"""
import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from crm import calibration as cal, config, metrics, models
from crm.plots import SERIES, plt, save

F, T = config.FIG_DIR, config.TAB_DIR
Y = "default_12m"
P = pd.read_parquet(config.DATA_DIR / "pd_predictions.parquet")
te, va = P[P.split == "test_2016"].copy(), P[P.split == "valid_2017"].copy()
names = [c[2:] for c in P.columns if c.startswith("p_")]
out = {}

# 1) Choose champion on 2016 AUC; stack on 2016 --------------------------------
auc16 = {n: metrics.auc_ci(te[Y], te[f"p_{n}"])[0] for n in names}
lr_name = max([n for n in names if n.startswith("LogReg")], key=auc16.get)
out["lr_selected"] = lr_name
stack_in = [lr_name, "LightGBM", "XGBoost"]
stk = cal.Stack().fit(te[[f"p_{n}" for n in stack_in]].to_numpy(), te[Y].to_numpy())
out["stack_coefficients"] = dict(zip(["const"] + stack_in, map(float, stk.m.params)))
out["stack_pvalues"] = dict(zip(["const"] + stack_in, map(float, stk.m.pvalues)))
# Stacking is fitted on 2016, so its 2016 AUC is in-sample; it is judged on 2017.
for d in (te, va):
    d["p_Stacked"] = stk.predict(d[[f"p_{n}" for n in stack_in]].to_numpy())
names.append("Stacked")
champion = max(auc16, key=auc16.get)
out["auc_2016"] = auc16
out["champion"] = champion
print("2016 AUC:", {k: round(v, 4) for k, v in auc16.items()}, "-> champion", champion)

# 2) Discrimination on 2017 with DeLong tests ---------------------------------
rows = []
for n in names:
    a, lo, hi = metrics.auc_ci(va[Y], va[f"p_{n}"])
    rows.append({"model": n, "auc": a, "auc_lo": lo, "auc_hi": hi, "gini": 2 * a - 1,
                 "ks": metrics.ks_stat(va[Y], va[f"p_{n}"])})
disc = pd.DataFrame(rows).set_index("model")
pairs = {}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        pairs[f"{a} vs {b}"] = metrics.delong_test(va[Y], va[f"p_{a}"], va[f"p_{b}"])
dl = pd.DataFrame(pairs).T
dl["p_holm"] = metrics.holm(dl["p_value"])
dl.to_csv(T / "03_delong_pairwise_2017.csv")
disc.to_csv(T / "03_discrimination_2017.csv")
print(disc.round(4).to_string()); print(dl[["diff", "z", "p_value", "p_holm"]].round(5).to_string())

fig, ax = plt.subplots(figsize=(5.5, 5))
for k, n in enumerate(names):
    fpr, tpr, _ = roc_curve(va[Y], va[f"p_{n}"])
    ax.plot(fpr, tpr, color=SERIES[k], lw=1.6, label=f"{n} (AUC {disc.loc[n,'auc']:.3f})")
ax.plot([0, 1], [0, 1], color="#b5b3ad", lw=1, ls="--")
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC - 2017 validation cohort")
ax.legend(loc="lower right", fontsize=8)
save(fig, F / "03_roc_2017.png")

# 3) Recalibration: fit on 2016, evaluate on 2017 ------------------------------
cal_rows = []
recal = {}
for n in names:
    p16, p17 = te[f"p_{n}"].to_numpy(), va[f"p_{n}"].to_numpy()
    y16, y17 = te[Y].to_numpy(), va[Y].to_numpy()
    s16 = metrics.calibration_slope_intercept(y16, p16)
    meths = {"raw": lambda p: p}
    if n != "Stacked":  # stacking already includes a logistic calibration on 2016
        ish, pl, iso = cal.InterceptShift().fit(p16, y16), cal.Platt().fit(p16, y16), cal.Isotonic().fit(p16, y16)
        meths.update({"intercept_shift": ish.predict, "platt": pl.predict, "isotonic": iso.predict})
        recal[n] = {"intercept_shift": ish, "platt": pl, "isotonic": iso}
    for mname, f in meths.items():
        s = metrics.summary(y17, f(p17))
        s.update(model=n, method=mname, slope_2016=s16["slope"], slope_2016_p_vs1=s16["slope_p_vs1"],
                 citl_2016=s16["citl"])
        cal_rows.append(s)
calt = pd.DataFrame(cal_rows)
calt.to_csv(T / "03_calibration_2017.csv", index=False)
show = ["model", "method", "pd_mean", "dr", "citl", "citl_p", "slope", "slope_p_vs1", "hl_chi2",
        "hl_p", "spiegelhalter_p", "brier", "ece", "auc"]
print(calt[show].round(4).to_string(index=False))

# Pre-specified selection rule (2016 evidence only): Platt if the 2016
# calibration slope differs from 1 at 1%, otherwise the intercept shift.
s16c = metrics.calibration_slope_intercept(te[Y], te[f"p_{champion}"])
method = "platt" if s16c["slope_p_vs1"] < 0.01 else "intercept_shift"
out["recalibration_method"] = method
out["champion_2016_calibration"] = s16c
p17_final = recal[champion][method].predict(va[f"p_{champion}"].to_numpy())
p16_final = recal[champion][method].predict(te[f"p_{champion}"].to_numpy())
# Segment check (2016 only): is miscalibration left by grade after Platt?
seg = cal.PlattSegment().fit(te[f"p_{champion}"].to_numpy(), te[Y].to_numpy(), te["grade"])
out["grade_recalibration_lr_test_2016"] = seg.lr_test
if seg.lr_test["p"] < 0.01:
    method = method + "+grade"
    out["recalibration_method"] = method
    p17_final = seg.predict(va[f"p_{champion}"].to_numpy(), va["grade"])
    p16_final = seg.predict(te[f"p_{champion}"].to_numpy(), te["grade"])
    recal[champion][method] = seg
    out["final_2017_platt_only"] = metrics.summary(va[Y], recal[champion]["platt"].predict(va[f"p_{champion}"].to_numpy()))
va["pd_final"], te["pd_final"] = p17_final, p16_final
lr_m = "platt" if metrics.calibration_slope_intercept(te[Y], te[f"p_{lr_name}"])["slope_p_vs1"] < 0.01 else "intercept_shift"
va["pd_lr_final"] = recal[lr_name][lr_m].predict(va[f"p_{lr_name}"].to_numpy())
out["final_2017"] = metrics.summary(va[Y], p17_final)
out["final_2017_logreg"] = metrics.summary(va[Y], va["pd_lr_final"])
out["final_2017_logreg"]["method"] = lr_m

# Reliability diagram before / after recalibration
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, (lab, p) in zip(axs, [("raw", va[f"p_{champion}"]), (method, p17_final)]):
    d = pd.DataFrame({"y": va[Y], "p": p})
    d["b"] = pd.qcut(d.p.rank(method="first"), 20, labels=False)
    g = d.groupby("b").agg(o=("y", "mean"), e=("p", "mean"), n=("y", "size"))
    se = np.sqrt(g.o * (1 - g.o) / g.n)
    ax.plot([0, g.e.max() * 1.1], [0, g.e.max() * 1.1], color="#b5b3ad", lw=1, ls="--")
    ax.errorbar(g.e, g.o, yerr=1.96 * se, fmt="o", ms=5, color=SERIES[0], ecolor=SERIES[0], elinewidth=1)
    c = metrics.calibration_slope_intercept(va[Y], p)
    ax.set(title=f"{champion} - {lab}\nCITL {c['citl']:+.3f}, slope {c['slope']:.3f}",
           xlabel="Mean predicted PD (ventile)")
axs[0].set_ylabel("Observed 12m default rate (95% CI)")
save(fig, F / "03_calibration_2017.png")

# 4) Backtesting (2017): Jeffreys / binomial by grade and PD decile -------------
bt = va.assign(y=va[Y], p=va["pd_final"])
bt["pd_decile"] = pd.qcut(bt.p.rank(method="first"), 10, labels=range(1, 11))
bg = metrics.binomial_backtest(bt, "grade"); bg.to_csv(T / "03_backtest_grade_2017.csv", index=False)
bd = metrics.binomial_backtest(bt, "pd_decile"); bd.to_csv(T / "03_backtest_decile_2017.csv", index=False)
print(bg.round(4).to_string(index=False)); print(bd.round(4).to_string(index=False))
# Quarterly backtest 2016Q1-2017Q4 of the final (2016-calibrated) model
both = pd.concat([te.assign(y=te[Y], p=te["pd_final"]), bt])
bq = metrics.binomial_backtest(both, "issue_q"); bq.to_csv(T / "03_backtest_quarter.csv", index=False)
out["backtest_grade_traffic_lights"] = bg.traffic_light.value_counts().to_dict()
out["backtest_decile_traffic_lights"] = bd.traffic_light.value_counts().to_dict()

# 5) Stability: PSI of score and CSI of features --------------------------------
with open(config.MODEL_DIR / "pd_models.pkl", "rb") as f:
    M = pickle.load(f)
df = pd.read_parquet(config.DATA_DIR / "loans.parquet")
df = df[df.obs_12m == 1]
dev = df[df.issue_year.isin(config.TRAIN_YEARS)]
score_fn = {"LightGBM": lambda X: M["lgb"].predict(X), "XGBoost": lambda X: models.predict_xgb(M["xgb"], X),
            "LogReg_WOE": lambda X: M["lr"].predict(X)}
if "lr_v2" in M:
    score_fn["LogReg_WOE_v2"] = lambda X: M["lr_v2"].predict(X)
psi_model = champion if champion in score_fn else "LightGBM"
dev_score = score_fn[psi_model](dev[models.FEATURES])
psi_rows = []
for yr in range(2012, 2018):
    d = df[df.issue_year == yr]
    s = score_fn[psi_model](d[models.FEATURES])
    psi_rows.append({"year": yr, "psi_score": metrics.psi(dev_score, s)})
psi_t = pd.DataFrame(psi_rows); psi_t["status"] = psi_t.psi_score.map(metrics.psi_label)
psi_t.to_csv(T / "03_psi_score_by_year.csv", index=False)
print(psi_t.round(4).to_string(index=False))
imp = pd.read_csv(T / "02_lgb_gain_importance.csv", index_col=0).iloc[:, 0]
top = [c for c in imp.index[:20] if c in models.NUMERIC]
# Bureau fields are structurally missing before 2012, so CSI vs the full dev
# window mostly measures the missing bin; the 2013-2015 reference avoids that.
ref = df[df.issue_year.between(2013, 2015)]
csi = pd.DataFrame({**{f"vs_dev_{yr}": {c: metrics.psi(dev[c], df.loc[df.issue_year == yr, c]) for c in top}
                       for yr in (2016, 2017)},
                    **{f"vs_2013_15_{yr}": {c: metrics.psi(ref[c], df.loc[df.issue_year == yr, c]) for c in top}
                       for yr in (2016, 2017)}})
csi["dev_missing_share"] = dev[top].isna().mean()
csi.to_csv(T / "03_csi_top_features.csv")
print(csi.round(3).to_string())
out["psi_2017"] = float(psi_t.set_index("year").loc[2017, "psi_score"])

# 6) Vintage analysis: cumulative default rate by months on book ----------------
snap = pd.Timestamp(config.SNAPSHOT)
all_loans = pd.read_parquet(config.DATA_DIR / "loans.parquet",
                            columns=["issue_dt", "issue_year", "term_m", "default_mob", "bad_lifetime", "grade"])
rows = []
for yr in range(2009, 2019):
    d = all_loans[(all_loans.issue_year == yr) & (all_loans.term_m == 36)]
    max_mob = (snap.year - yr) * 12 + snap.month - 12 - 3  # last fully observed MOB for Dec issues
    for mob in range(3, min(36, max_mob) + 1, 3):
        dr = ((d.bad_lifetime == 1) & (d.default_mob <= mob)).mean()
        rows.append({"vintage": yr, "mob": mob, "cum_default_rate": dr})
vin = pd.DataFrame(rows)
vin.pivot(index="mob", columns="vintage", values="cum_default_rate").to_csv(T / "03_vintage_36m.csv")
fig, ax = plt.subplots(figsize=(7, 4.4))
for k, (yr, g) in enumerate(vin.groupby("vintage")):
    ax.plot(g.mob, g.cum_default_rate * 100, color=SERIES[k % 8] if k < 8 else "#8a8984",
            lw=1.6, ls="-" if k < 8 else "--", label=str(yr))
ax.set(xlabel="Months on book", ylabel="Cumulative default rate (%)",
       title="Vintage curves - 36-month loans")
ax.legend(ncol=2, fontsize=8, title="Issue year")
save(fig, F / "03_vintage_36m.png")

va[["id", "issue_q", "grade", "term_m", "funded_amnt", Y, "pd_final", "pd_lr_final"]].to_parquet(
    config.DATA_DIR / "pd_final_2017.parquet")
with open(config.MODEL_DIR / "recalibrators.pkl", "wb") as f:
    pickle.dump({"champion": champion, "method": method, "recal": recal[champion][method],
                 "by_grade": method.endswith("+grade"), "stack": stk}, f)
json.dump(out, open(T / "03_validation_summary.json", "w"), indent=2, default=float)
print(json.dumps({k: out[k] for k in ("champion", "recalibration_method", "psi_2017")}, default=float))
