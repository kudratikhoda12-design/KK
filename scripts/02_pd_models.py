"""Fit and compare PD models on the 12-month default target.

Split (out-of-time): train 2007-2015 | test 2016 | validation 2017.
Hyper-parameters: tuned on 2007-2014 -> 2015 (time-based), then refit on
2007-2015 with the tuned number of boosting rounds scaled by sample size.
"""
import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from crm import config, metrics, models
from crm.models import FEATURES

LOG = open(config.REPORT_DIR / "02_pd_models.log", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

df = pd.read_parquet(config.DATA_DIR / "loans.parquet")
df = df[df.obs_12m == 1].reset_index(drop=True)
Y = "default_12m"
tr = df[df.issue_year.isin(config.TRAIN_YEARS)]
te = df[df.issue_year == config.TEST_YEAR]
va = df[df.issue_year == config.VALID_YEAR]
tr_a, tr_b = tr[tr.issue_year <= 2014], tr[tr.issue_year == 2015]   # tuning split
log(f"train {len(tr)} dr={tr[Y].mean():.4f} | test {len(te)} dr={te[Y].mean():.4f} | valid {len(va)} dr={va[Y].mean():.4f}")
scale = len(tr) / len(tr_a)
preds = {}

# 1) WOE logistic scorecard ------------------------------------------------
lr = models.WOELogit().fit(tr_a[FEATURES], tr_a[Y], tr_b[FEATURES], tr_b[Y], log=log)
sel = lr.vars
pd.DataFrame(lr.selection_log).to_csv(config.TAB_DIR / "02_lr_selection_log.csv", index=False)
lr.woe.iv.rename("iv").to_csv(config.TAB_DIR / "02_information_value.csv")
lr_final = models.WOELogit()
lr_final.woe = models.WOETransformer(models.NUMERIC, models.CATEGORICAL).fit(tr[FEATURES], tr[Y])
lr_final.vars = sel
import statsmodels.api as sm
W = lr_final.woe.transform(tr[FEATURES], sel)
lr_final.model = sm.Logit(tr[Y].to_numpy(), sm.add_constant(W)).fit(disp=0)
lr_final.vif = models.vif_table(W)
ct = lr_final.coef_table()
ct.to_csv(config.TAB_DIR / "02_lr_coefficients.csv")
log("LR selected", len(sel), "vars\n", ct.round(4).to_string())
log(f"LR LR-test chi2={2*(lr_final.model.llf-lr_final.model.llnull):.0f} p={lr_final.model.llr_pvalue:.2e} pseudoR2={lr_final.model.prsquared:.4f}")
models.scorecard_points(lr_final).to_csv(config.TAB_DIR / "02_scorecard_points.csv", index=False)
preds["LogReg_WOE"] = (lr_final.predict(te[FEATURES]), lr_final.predict(va[FEATURES]))

# 2) LightGBM ---------------------------------------------------------------
t_lgb = models.tune_lgb(tr_a[FEATURES], tr_a[Y], tr_b[FEATURES], tr_b[Y], n_iter=20, log=log)
t_lgb.to_csv(config.TAB_DIR / "02_tuning_lgb.csv", index=False)
best = t_lgb.iloc[0].to_dict()
n_it = int(best.pop("best_iter") * scale); best.pop("auc")
best = {k: (int(v) if k in ("num_leaves", "min_child_samples") else v) for k, v in best.items()}
m_lgb = models.fit_lgb(tr[FEATURES], tr[Y], best, n_it)
preds["LightGBM"] = (m_lgb.predict(te[FEATURES]), m_lgb.predict(va[FEATURES]))
imp = pd.Series(m_lgb.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
imp.to_csv(config.TAB_DIR / "02_lgb_gain_importance.csv")

# 3) XGBoost ----------------------------------------------------------------
t_xgb = models.tune_xgb(tr_a[FEATURES], tr_a[Y], tr_b[FEATURES], tr_b[Y], n_iter=12, log=log)
t_xgb.to_csv(config.TAB_DIR / "02_tuning_xgb.csv", index=False)
bx = t_xgb.iloc[0].to_dict()
nx = int(bx.pop("best_iter") * scale); bx.pop("auc")
bx = {k: (int(v) if k in ("max_depth",) else v) for k, v in bx.items()}
m_xgb = models.fit_xgb(tr[FEATURES], tr[Y], bx, nx)
preds["XGBoost"] = (models.predict_xgb(m_xgb, te[FEATURES]), models.predict_xgb(m_xgb, va[FEATURES]))

# 4) Random forest ----------------------------------------------------------
Xa, med = models.rf_matrix(tr_a[FEATURES]); Xb, _ = models.rf_matrix(tr_b[FEATURES], med)
t_rf = models.tune_rf(Xa, tr_a[Y], Xb, tr_b[Y], log=log)
t_rf.to_csv(config.TAB_DIR / "02_tuning_rf.csv", index=False)
br = t_rf.iloc[0].drop("auc").to_dict(); br["min_samples_leaf"] = int(br["min_samples_leaf"])
Xtr, med = models.rf_matrix(tr[FEATURES])
m_rf = models.fit_rf(Xtr, tr[Y], br)
preds["RandomForest"] = (m_rf.predict_proba(models.rf_matrix(te[FEATURES], med)[0])[:, 1],
                         m_rf.predict_proba(models.rf_matrix(va[FEATURES], med)[0])[:, 1])
del m_rf

# Save ----------------------------------------------------------------------
out = []
for split, d, k in [("test_2016", te, 0), ("valid_2017", va, 1)]:
    o = d[["id", "issue_year", "issue_q", Y, "grade", "term_m", "funded_amnt"]].copy()
    for name, p in preds.items():
        o[f"p_{name}"] = p[k]
    o["split"] = split
    out.append(o)
pd.concat(out).to_parquet(config.DATA_DIR / "pd_predictions.parquet")
with open(config.MODEL_DIR / "pd_models.pkl", "wb") as f:
    pickle.dump({"lr": lr_final, "lgb": m_lgb, "xgb": m_xgb, "lgb_params": best, "lgb_iter": n_it,
                 "xgb_params": bx, "xgb_iter": nx, "rf_params": br}, f)
json.dump({"lgb": {**best, "n_iter": n_it}, "xgb": {**bx, "n_iter": nx}, "rf": br, "lr_vars": sel},
          open(config.TAB_DIR / "02_final_hyperparameters.json", "w"), indent=2)

rows = []
for split, d, k in [("test_2016", te, 0), ("valid_2017", va, 1)]:
    for name, p in preds.items():
        s = metrics.summary(d[Y].to_numpy(), p[k]); s.update(model=name, split=split); rows.append(s)
res = pd.DataFrame(rows)
res.to_csv(config.TAB_DIR / "02_model_metrics_raw.csv", index=False)
log(res[["split", "model", "auc", "auc_lo", "auc_hi", "gini", "ks", "brier", "pd_mean", "dr", "citl", "slope"]].round(4).to_string(index=False))
