"""Can the logistic scorecard close the gap to the GBMs?

Variant v2 relaxes the forward-selection gain threshold (5e-4 -> 1e-4) and
allows up to 25 variables, keeping every other gate (Wald p < 0.001,
positive WOE signs, VIF < 5). The two LRs are nested, so we test with:
  * a likelihood-ratio test on the development sample, and
  * a paired DeLong test on the 2016 OOT test year (2017 untouched).
The better LR is added to the prediction file as "LogReg_WOE_v2".
"""
import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import statsmodels.api as sm
from scipy import stats

from crm import config, metrics, models
from crm.models import FEATURES

df = pd.read_parquet(config.DATA_DIR / "loans.parquet")
df = df[df.obs_12m == 1]
Y = "default_12m"
tr = df[df.issue_year.isin(config.TRAIN_YEARS)]
tr_a, tr_b = tr[tr.issue_year <= 2014], tr[tr.issue_year == 2015]
te, va = df[df.issue_year == 2016], df[df.issue_year == 2017]

with open(config.MODEL_DIR / "pd_models.pkl", "rb") as f:
    M = pickle.load(f)
v1 = M["lr"]

sel = models.WOELogit(min_gain=1e-4, max_vars=25).fit(tr_a[FEATURES], tr_a[Y], tr_b[FEATURES], tr_b[Y])
pd.DataFrame(sel.selection_log).to_csv(config.TAB_DIR / "02b_lr_v2_selection_log.csv", index=False)
v2 = models.WOELogit()
v2.woe, v2.vars = v1.woe, sel.vars           # same WOE binning (fit on full train)
W = v2.woe.transform(tr[FEATURES], v2.vars)
v2.model = sm.Logit(tr[Y].to_numpy(), sm.add_constant(W)).fit(disp=0)
v2.vif = models.vif_table(W)
v2.coef_table().to_csv(config.TAB_DIR / "02b_lr_v2_coefficients.csv")

out = {"v1_vars": v1.vars, "v2_vars": v2.vars}
nested = set(v1.vars) <= set(v2.vars)
out["nested"] = nested
if nested:
    lr_stat = 2 * (v2.model.llf - v1.model.llf)
    dfree = len(v2.vars) - len(v1.vars)
    out["lr_test"] = {"chi2": lr_stat, "df": dfree, "p": float(stats.chi2.sf(lr_stat, dfree))}
out["aic"] = {"v1": v1.model.aic, "v2": v2.model.aic}
out["bic"] = {"v1": v1.model.bic, "v2": v2.model.bic}
p1, p2 = v1.predict(te[FEATURES]), v2.predict(te[FEATURES])
out["delong_2016_v2_vs_v1"] = metrics.delong_test(te[Y], p2, p1)
print(v2.coef_table().round(4).to_string())
print(json.dumps(out, indent=2, default=float))

P = pd.read_parquet(config.DATA_DIR / "pd_predictions.parquet")
P.loc[P.split == "test_2016", "p_LogReg_WOE_v2"] = v2.predict(te.set_index("id").loc[P.loc[P.split == "test_2016", "id"], FEATURES].reset_index(drop=True)).to_numpy()
P.loc[P.split == "valid_2017", "p_LogReg_WOE_v2"] = v2.predict(va.set_index("id").loc[P.loc[P.split == "valid_2017", "id"], FEATURES].reset_index(drop=True)).to_numpy()
P.to_parquet(config.DATA_DIR / "pd_predictions.parquet")
M["lr_v2"] = v2
with open(config.MODEL_DIR / "pd_models.pkl", "wb") as f:
    pickle.dump(M, f)
json.dump(out, open(config.TAB_DIR / "02b_lr_improvement.json", "w"), indent=2, default=float)
