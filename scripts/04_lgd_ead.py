"""LGD and EAD estimation with out-of-time validation.

LGD sample: charged-off loans whose recovery process is mature (default
>= 18 months before the 2019-03 snapshot; recoveries are flat after ~18m).
OOT split by default date: train defaults <= 2015-06, test 2015-07..2017-09.
EAD (conditional on 12m default) = outstanding principal at default / funded;
train issue years 2007-2015, test 2016.
"""
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from scipy import stats
from crm import config, lgd_ead as le

df = pd.read_parquet(config.DATA_DIR / "loans.parquet")
res = {}

# Maturity check: is LGD still drifting after 18 months since default? -------
co = df[df.loan_status.isin(["Charged Off", "Default"]) & df.lgd_realised.notna()]
early = co[co.months_since_default.between(18, 36)].lgd_realised
late = co[co.months_since_default.between(37, 72)].lgd_realised
res["maturity_check"] = {"lgd_18_36m": early.mean(), "lgd_37_72m": late.mean(),
                         "diff": early.mean() - late.mean(),
                         "welch_p": stats.ttest_ind(early, late, equal_var=False).pvalue}
print(res["maturity_check"])

lg = le.prep(co[co.months_since_default >= 18])
lg["y"] = lg["lgd_realised"]
cut = pd.Timestamp("2015-07-01")
tr, te = lg[lg.default_dt < cut], lg[lg.default_dt >= cut]
print("LGD train", len(tr), "test", len(te))

glm = le.fit_fractional_logit(tr, le.LGD_FORMULA)
seg = tr.groupby(["grade", "term_m"], observed=True).y.mean()
Xcols = ["int_rate", "term_m", "funded_amnt", "fico", "dti", "revol_util_f", "log_inc_f",
         "default_mob", "ead_ratio"]
gbm = le.fit_lgb_fraction(tr[Xcols], tr.y)
preds = {
    "mean": np.full(len(te), tr.y.mean()),
    "segment_grade_term": te.set_index(["grade", "term_m"]).index.map(seg).to_numpy(float),
    "frac_logit_glm": glm.predict(te).to_numpy(),
    "lightgbm_xentropy": gbm.predict(te[Xcols]),
}
rows = []
for k, p in preds.items():
    r = le.regression_metrics(te.y, p); r["model"] = k
    if k != "mean":
        r.update({f"vs_mean_{a}": b for a, b in le.paired_loss_test(te.y, p, preds["mean"]).items()})
    rows.append(r)
lgd_tab = pd.DataFrame(rows).set_index("model")
print(lgd_tab.round(4).T.to_string())
lgd_tab.to_csv(config.TAB_DIR / "04_lgd_model_comparison.csv")
glm.summary2().tables[1].to_csv(config.TAB_DIR / "04_lgd_glm_coefficients.csv")

# Downturn LGD: highest annual mean LGD across mature default years (n>=1000).
by_year = lg.groupby(lg.default_dt.dt.year).y.agg(["mean", "size"])
by_year.to_csv(config.TAB_DIR / "04_lgd_by_default_year.csv")
dt = by_year[by_year["size"] >= 1000]
res["lgd_long_run_mean"] = float(lg.y.mean())
res["lgd_downturn"] = float(dt["mean"].max())
res["lgd_downturn_year"] = int(dt["mean"].idxmax())

# EAD conditional on default within 12 months ----------------------------------
ed = le.prep(df[(df.default_12m == 1) & (df.obs_12m == 1)])
ed["y"] = ed["ead_ratio"]
etr, ete = ed[ed.issue_year <= 2015], ed[ed.issue_year == 2016]
eglm = le.fit_fractional_logit(etr, le.EAD_FORMULA)
eseg = etr.groupby("term_m").y.mean()
ep = {"mean": np.full(len(ete), etr.y.mean()),
      "segment_term": ete.term_m.map(eseg).to_numpy(float),
      "frac_logit_glm": eglm.predict(ete).to_numpy()}
rows = []
for k, p in ep.items():
    r = le.regression_metrics(ete.y, p); r["model"] = k
    if k != "mean":
        r.update({f"vs_mean_{a}": b for a, b in le.paired_loss_test(ete.y, p, ep["mean"]).items()})
    rows.append(r)
ead_tab = pd.DataFrame(rows).set_index("model")
print(ead_tab.round(4).T.to_string())
ead_tab.to_csv(config.TAB_DIR / "04_ead_model_comparison.csv")
eglm.summary2().tables[1].to_csv(config.TAB_DIR / "04_ead_glm_coefficients.csv")

# Refit chosen specifications on all available data for production use.
glm_all = le.fit_fractional_logit(lg, le.LGD_FORMULA)
eglm_all = le.fit_fractional_logit(ed[ed.issue_year <= 2016], le.EAD_FORMULA)
with open(config.MODEL_DIR / "lgd_ead.pkl", "wb") as f:
    pickle.dump({"lgd_glm": glm_all, "ead_glm": eglm_all, "lgd_mean": float(lg.y.mean()),
                 "ead_mean": float(ed.y.mean())}, f)
json.dump(res, open(config.TAB_DIR / "04_lgd_summary.json", "w"), indent=2, default=float)
print(res)
