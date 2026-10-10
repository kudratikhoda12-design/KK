"""Write LaTeX table fragments (booktabs) from reports/tables into reports/latex/tables."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from crm import config

T = config.TAB_DIR
OUT = config.REPORT_DIR / "latex" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("$", r"\$")


def num(x, d=4):
    if isinstance(x, str):
        return esc(x)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:,.{d}f}"


def pval(p):
    if p is None or np.isnan(p):
        return "--"
    if p < 1e-4:
        e = int(np.floor(np.log10(max(p, 1e-300))))
        return rf"$<10^{{{e + 1}}}$"
    return f"{p:.4f}"


def write(name, header, rows, align, caption=None):
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


# Cohorts
c = pd.read_csv(T / "01_cohort_summary.csv")
write("cohort", ["Issue year", "Loans", "Funded (\\$bn)", "Terminal loans", "Lifetime DR (terminal)", "12m DR (all loans)"],
      [[str(r.issue_year), num(int(r.loans)), num(r["funded_$bn"], 3), num(int(r.terminal_loans)),
        f"{r.dr_lifetime_terminal*100:.2f}\\%", "--" if np.isnan(r.dr_12m_all_loans) else f"{r.dr_12m_all_loans*100:.2f}\\%"]
       for _, r in c.iterrows()], "lrrrrr")

# Window experiment
w = pd.read_csv(T / "02a_train_window.csv")
write("window", ["Train window", "$n$ train", "AUC 2015", "95\\% CI", "DeLong $p$ vs 2007--2014"],
      [[r.train_window, num(int(r.n_train)), num(r.auc_2015), f"[{r.auc_lo:.4f}, {r.auc_hi:.4f}]",
        pval(r.delong_p_vs_2007) if not np.isnan(r.delong_p_vs_2007) else "ref."] for _, r in w.iterrows()], "lrrcr")

# IV table with status
iv = pd.read_csv(T / "02_information_value.csv", index_col=0)["iv"]
v2 = json.load(open(T / "02b_lr_improvement.json"))
v1set, v2set = set(v2["v1_vars"]), set(v2["v2_vars"])
sel = pd.read_csv(T / "02b_lr_v2_selection_log.csv")
rows = []
for var, val in iv[iv >= 0.02].items():
    status = "v1, v2" if var in v1set else ("v2" if var in v2set else "")
    rows.append([esc(var), f"{val:.4f}", status])
write("iv", ["Variable", "IV", "Selected in"], rows, "lrl")

# Stepwise log (accepted only) for v2
acc = sel[sel.accepted]
write("stepwise", ["Step", "Variable added", "Holdout AUC", "AUC gain", "Wald $p$"],
      [[str(int(r.step)), esc(r["var"]), f"{r.holdout_auc:.4f}", f"{r.gain:+.4f}", pval(r.wald_p)] for _, r in acc.iterrows()],
      "rlrrr")

# Coefficients
for nm, f in [("coef_v1", "02_lr_coefficients.csv"), ("coef_v2", "02b_lr_v2_coefficients.csv")]:
    t = pd.read_csv(T / f, index_col=0)
    write(nm, ["Term", r"$\hat\beta$", "SE", "$z$", "$p$", "VIF"],
          [[esc(i), f"{r.coef:.4f}", f"{r.se:.4f}", f"{r.z:.1f}", pval(r.p_value), "--" if np.isnan(r.vif) else f"{r.vif:.2f}"]
           for i, r in t.iterrows()], "lrrrrr")

# Hyper-parameters
hp = json.load(open(T / "02_final_hyperparameters.json"))
rows = [["LightGBM", esc(", ".join(f"{k}={v}" for k, v in hp["lgb"].items()))],
        ["XGBoost", esc(", ".join(f"{k}={v}" for k, v in hp["xgb"].items()))],
        ["Random Forest", esc(", ".join(f"{k}={v}" for k, v in hp["rf"].items())) + r", n\_estimators=500, max\_samples=0.3"]]
write("hyper", ["Model", "Selected hyper-parameters (refit on 2007--2015)"], rows, "lp{12cm}")

# Discrimination 2016 & 2017
m = pd.read_csv(T / "02_model_metrics_raw.csv")
d17 = pd.read_csv(T / "03_discrimination_2017.csv", index_col=0)
rows = []
for _, r in m.iterrows():
    rows.append([r.split.replace("_", " "), esc(r.model), f"{r.auc:.4f}", f"[{r.auc_lo:.4f}, {r.auc_hi:.4f}]",
                 f"{r.gini:.4f}", f"{r.ks:.4f}", f"{r.brier:.4f}", f"{r.log_loss:.4f}"])
write("disc", ["Sample", "Model", "AUC", "95\\% DeLong CI", "Gini", "KS", "Brier", "Log-loss"], rows, "llrcrrrr")
write("disc17", ["Model (2017)", "AUC", "95\\% CI", "Gini", "KS"],
      [[esc(i), f"{r.auc:.4f}", f"[{r.auc_lo:.4f}, {r.auc_hi:.4f}]", f"{r.gini:.4f}", f"{r.ks:.4f}"] for i, r in d17.iterrows()],
      "lrcrr")

# DeLong pairwise
dl = pd.read_csv(T / "03_delong_pairwise_2017.csv", index_col=0)
write("delong", ["Comparison (A vs B)", r"$\Delta$AUC", "$z$", "$p$", "$p$ (Holm)"],
      [[esc(i), f"{r['diff']:+.4f}", f"{r.z:.2f}", pval(r.p_value), pval(r.p_holm)] for i, r in dl.iterrows()], "lrrrr")

# Calibration table (2017)
cal = pd.read_csv(T / "03_calibration_2017.csv")
rows = []
for _, r in cal.iterrows():
    rows.append([esc(r.model), esc(r.method), f"{r.pd_mean*100:.2f}", f"{r.citl:+.3f}", pval(r.citl_p),
                 f"{r.slope:.3f}", pval(r.slope_p_vs1), f"{r.hl_chi2:.1f}", pval(r.spiegelhalter_p),
                 f"{r.ece*100:.2f}", f"{r.brier:.4f}"])
write("calib", ["Model", "Method", "PD\\%", "CITL", "$p$", "Slope", "$p$(=1)", "HL $\\chi^2_8$", "Spieg. $p$", "ECE pp", "Brier"],
      rows, "llrrrrrrrrr")

vs = json.load(open(T / "03_validation_summary.json"))
rows = []
for k, lab in [("final_2017_platt_only", "LightGBM + Platt"), ("final_2017", "LightGBM + Platt + grade (final)"),
               ("final_2017_logreg", "LogReg v2 + Platt")]:
    r = vs[k]
    rows.append([lab, f"{r['pd_mean']*100:.2f}\\%", f"{r['citl']:+.4f} [{r['citl_lo']:+.3f}, {r['citl_hi']:+.3f}]",
                 f"{r['slope']:.4f} [{r['slope_lo']:.3f}, {r['slope_hi']:.3f}]", f"{r['hl_chi2']:.1f}",
                 f"{r['spiegelhalter_z']:.2f} ({r['spiegelhalter_p']:.3f})", f"{r['ece']*100:.2f}"])
write("calib_final", ["Model (2017, DR 5.80\\%)", "Mean PD", "CITL [95\\% CI]", "Slope [95\\% CI]", "HL $\\chi^2_8$",
                      "Spiegelhalter $z$ ($p$)", "ECE pp"], rows, "lrccrcr")

# Backtests
for nm, f, key in [("bt_grade", "03_backtest_grade_2017.csv", "grade"), ("bt_decile", "03_backtest_decile_2017.csv", "pd_decile"),
                   ("bt_quarter", "03_backtest_quarter.csv", "issue_q")]:
    t = pd.read_csv(T / f)
    write(nm, [key.replace("_", " ").title(), "$N$", "$D$", "Mean PD", "ODR", "Binomial $p$", "Jeffreys $p$", "Light"],
          [[str(r[key]), num(int(r.n)), num(int(r.defaults)), f"{r.pd_mean*100:.2f}\\%", f"{r.odr*100:.2f}\\%",
            pval(r.binom_p_two_sided), f"{r.jeffreys_p:.4f}", esc(r.traffic_light)] for _, r in t.iterrows()],
          "lrrrrrrl")

# PSI / CSI
p = pd.read_csv(T / "03_psi_score_by_year.csv")
write("psi", ["Year", "Score PSI", "Status"], [[str(r.year), f"{r.psi_score:.4f}", r.status] for _, r in p.iterrows()], "lrl")
cs = pd.read_csv(T / "03_csi_top_features.csv", index_col=0)
write("csi", ["Feature", "vs dev 2016", "vs dev 2017", "vs 2013--15: 2016", "vs 2013--15: 2017", "Dev missing"],
      [[esc(i), f"{r.vs_dev_2016:.3f}", f"{r.vs_dev_2017:.3f}", f"{r.vs_2013_15_2016:.3f}", f"{r.vs_2013_15_2017:.3f}",
        f"{r.dev_missing_share*100:.1f}\\%"] for i, r in cs.iterrows()], "lrrrrr")

# LGD / EAD comparisons
for nm, f in [("lgd", "04_lgd_model_comparison.csv"), ("ead", "04_ead_model_comparison.csv")]:
    t = pd.read_csv(T / f, index_col=0)
    write(nm, ["Model", "Mean obs.", "Mean pred.", "RMSE", "MAE", "$R^2_{oos}$", "Spearman", "DM $t$ vs mean", "$p$ (1-sided)"],
          [[esc(i), f"{r.mean_obs:.4f}", f"{r.mean_pred:.4f}", f"{r.rmse:.4f}", f"{r.mae:.4f}", f"{r.r2_oos:.4f}",
            "--" if np.isnan(r.spearman) else f"{r.spearman:.3f}",
            "--" if np.isnan(r.get("vs_mean_t", np.nan)) else f"{r.vs_mean_t:.2f}",
            "--" if np.isnan(r.get("vs_mean_p_value_one_sided", np.nan)) else pval(r.vs_mean_p_value_one_sided)]
           for i, r in t.iterrows()], "lrrrrrrrr")
ly = pd.read_csv(T / "04_lgd_by_default_year.csv")
ly = ly[ly["size"] >= 40]
write("lgd_year", ["Default year", "Mean LGD", "$n$"], [[str(int(r.default_dt)), f"{r['mean']:.4f}", num(int(r['size']))]
                                                       for _, r in ly.iterrows()], "lrr")
eg = pd.read_csv(T / "04_ead_glm_coefficients.csv", index_col=0)
write("ead_coef", ["Term", r"$\hat\beta$", "Robust SE", "$z$", "$p$"],
      [[esc(i), f"{r['Coef.']:.4f}", f"{r['Std.Err.']:.4f}", f"{r['z']:.2f}", pval(r['P>|z|'])] for i, r in eg.iterrows()], "lrrrr")

# Term structure
ts = pd.read_csv(T / "05_pd_term_structure.csv")
write("term", ["Grade", "Term", "$n$", "CDR$_{12}$", "CDR$_{life}$", "$k$", "EAD ratio (life)"],
      [[r.grade, f"{int(r.term_m)}", num(int(r.n)), f"{r.cdr_12m*100:.2f}\\%", f"{r.cdr_life*100:.2f}\\%", f"{r.k:.3f}",
        f"{r.ead_ratio_life:.3f}"] for _, r in ts.iterrows()], "lrrrrrr")

# ECL by grade
g = pd.read_csv(T / "05_ecl_by_grade_2017.csv")
tot = {"grade": "Total", "loans": g.loans.sum(), "funded": g.funded.sum(), "ecl_12m": g.ecl_12m.sum(), "ecl_life": g.ecl_life.sum()}
e = json.load(open(T / "05_ecl_summary.json"))
rows = [[r.grade, num(int(r.loans)), f"{r.funded/1e6:,.1f}", f"{r.pd_12m*100:.2f}\\%", f"{r.odr_12m*100:.2f}\\%",
         f"{r.pd_life*100:.2f}\\%", f"{r.ecl_12m/1e6:,.1f}", f"{r.ecl_12m_pct*100:.2f}\\%", f"{r.ecl_life/1e6:,.1f}",
         f"{r.ecl_life_pct*100:.2f}\\%"] for _, r in g.iterrows()]
rows.append([r"\textbf{Total}", num(int(tot["loans"])), f"{tot['funded']/1e6:,.1f}", f"{e['mean_pd_12m']*100:.2f}\\%",
             f"{e['observed_dr_12m']*100:.2f}\\%", f"{e['mean_pd_lifetime']*100:.2f}\\%", f"{tot['ecl_12m']/1e6:,.1f}",
             f"{e['ecl_12m_pct']*100:.2f}\\%", f"{tot['ecl_life']/1e6:,.1f}", f"{e['ecl_lifetime_pct']*100:.2f}\\%"])
write("ecl_grade", ["Grade", "Loans", "Funded \\$M", "PD$_{12}$", "ODR$_{12}$", "PD$_{life}$", "ECL$_{12}$ \\$M", "\\%",
                    "ECL$_{life}$ \\$M", "\\%"], rows, "lrrrrrrrrr")

# Stress
st = pd.read_csv(T / "06_stress_results.csv")
rows = []
for _, r in st.iterrows():
    f6 = lambda x: "--" if pd.isna(x) else f"{x/1e6:,.1f}"
    rows.append([esc(r.scenario), f6(r.EL_mc), f6(r.VaR99), f6(r.ES99), f6(r["VaR99.9"]), f6(r["ES99.9"])])
write("stress", ["Scenario", "EL", "VaR$_{99\\%}$", "ES$_{99\\%}$", "VaR$_{99.9\\%}$", "ES$_{99.9\\%}$"], rows, "p{7.2cm}rrrrr")

# MC convergence
cv = pd.read_csv(T / "06_mc_convergence.csv")
write("mc_conv", ["Scenarios", "VaR$_{99\\%}$ \\$M", "95\\% CI \\$M", "CI width"],
      [[num(int(r.n_scen)), f"{r.VaR99/1e6:,.1f}", f"[{r.VaR99_lo/1e6:,.1f}, {r.VaR99_hi/1e6:,.1f}]",
        f"{(r.VaR99_hi-r.VaR99_lo)/r.VaR99*100:.1f}\\%"] for _, r in cv.iterrows()], "rrcr")

# Quarterly cohorts
q = pd.read_csv(T / "06_quarterly_cohorts_rho.csv")
half = int(np.ceil(len(q) / 2))
rows = []
for i in range(half):
    a = q.iloc[i]
    b = q.iloc[i + half] if i + half < len(q) else None
    row = [a.issue_q, num(int(a.n)), f"{a.dr*100:.2f}", f"{a.pd*100:.2f}", f"{a.z_t:+.2f}"]
    row += ([b.issue_q, num(int(b.n)), f"{b.dr*100:.2f}", f"{b.pd*100:.2f}", f"{b.z_t:+.2f}"] if b is not None else [""] * 5)
    rows.append(row)
write("quarters", ["Quarter", "$n$", "DR\\%", "PD\\%", "$\\hat z_t$"] * 2, rows, "lrrrr|lrrrr")
print("tables written:", len(list(OUT.glob("*.tex"))))
