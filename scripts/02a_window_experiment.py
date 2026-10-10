"""Does including pre-2012 vintages help? Train windows ending 2014, score 2015.

Fixed LightGBM params; AUC differences tested with paired DeLong.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from crm import config, metrics
from crm.models import FEATURES, fit_lgb

df = pd.read_parquet(config.DATA_DIR / "loans.parquet")
df = df[df.obs_12m == 1]
va = df[df.issue_year == 2015]
prm = dict(num_leaves=31, min_child_samples=500, learning_rate=0.05,
           feature_fraction=0.7, bagging_fraction=0.8)
preds, rows = {}, []
for start in [2007, 2010, 2012, 2013]:
    tr = df[(df.issue_year >= start) & (df.issue_year <= 2014)]
    m = fit_lgb(tr[FEATURES], tr.default_12m, prm, 400)
    preds[start] = m.predict(va[FEATURES])
    a, lo, hi = metrics.auc_ci(va.default_12m, preds[start])
    rows.append({"train_window": f"{start}-2014", "n_train": len(tr), "auc_2015": a,
                 "auc_lo": lo, "auc_hi": hi})
res = pd.DataFrame(rows)
for start in [2010, 2012, 2013]:
    t = metrics.delong_test(va.default_12m, preds[start], preds[2007])
    res.loc[res.train_window == f"{start}-2014", "delong_p_vs_2007"] = t["p_value"]
print(res.round(4).to_string(index=False))
res.to_csv(config.TAB_DIR / "02a_train_window.csv", index=False)
