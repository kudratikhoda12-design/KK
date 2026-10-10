"""Replicate the original project design and quantify its bias.

Original design: terminal loans only (Fully Paid / Charged Off as of the
2019-03 snapshot), lifetime default target, train <=2015, validate 2017.
For recent cohorts only loans that already ended are included, so the 2017
"cohort" is dominated by early charge-offs and early prepayments.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from crm import config, metrics, models
from crm.models import FEATURES

T = config.TAB_DIR
df = pd.read_parquet(config.DATA_DIR / "loans.parquet")
term = df[df.terminal == 1]
Y = "bad_lifetime"
tr = term[term.issue_year <= 2015]
tr_a, tr_b = tr[tr.issue_year <= 2014], tr[tr.issue_year == 2015]
va = term[term.issue_year == 2017]
res = {"n_terminal": int(len(term)), "n_valid_2017": int(len(va)),
       "funded_valid_2017": float(va.funded_amnt.sum()), "dr_valid_2017": float(va[Y].mean())}

lr = models.WOELogit().fit(tr_a[FEATURES], tr_a[Y], tr_b[FEATURES], tr_b[Y])
# refit coefficients on full train with the same variables
import statsmodels.api as sm
lr.woe = models.WOETransformer(models.NUMERIC, models.CATEGORICAL).fit(tr[FEATURES], tr[Y])
lr.model = sm.Logit(tr[Y].to_numpy(), sm.add_constant(lr.woe.transform(tr[FEATURES], lr.vars))).fit(disp=0)
p_lr = lr.predict(va[FEATURES])
hp = json.load(open(T / "02_final_hyperparameters.json"))["lgb"]
n_it = hp.pop("n_iter")
gbm = models.fit_lgb(tr[FEATURES], tr[Y], hp, n_it)
p_gb = gbm.predict(va[FEATURES])
for name, p in [("LogReg_WOE", p_lr), ("LightGBM", p_gb)]:
    s = metrics.summary(va[Y], p)
    res[name] = {k: s[k] for k in ("auc", "auc_lo", "auc_hi", "gini", "ks", "pd_mean", "dr", "citl")}
res["delong_lgb_vs_lr"] = metrics.delong_test(va[Y], p_gb, p_lr)

# ECL as in the original: lifetime PD x LGD x funded amount on terminal 2017 loans.
lgd = json.load(open(T / "04_lgd_summary.json"))["lgd_long_run_mean"]
res["ecl_replication_lr"] = float(np.sum(p_lr * lgd * va.funded_amnt))

# ---- Bias diagnostics -------------------------------------------------------
all17 = df[df.issue_year == 2017]
res["bias"] = {
    "share_of_2017_loans_terminal": float(len(va) / len(all17)),
    "share_of_2017_funded_terminal": float(va.funded_amnt.sum() / all17.funded_amnt.sum()),
    # Early-event enrichment: 12m default rate in the terminal subset vs the whole cohort.
    "dr12m_terminal_subset_2017": float(va.default_12m.mean()),
    "dr12m_all_2017": float(all17.default_12m.mean()),
    "prepaid_within_12m_share_of_terminal_good": float(
        ((va[Y] == 0) & (va.total_rec_prncp >= va.funded_amnt - 1)).mean()),
}
ct = np.array([[va.default_12m.sum(), len(va) - va.default_12m.sum()],
               [all17.default_12m.sum() - va.default_12m.sum(),
                (len(all17) - len(va)) - (all17.default_12m.sum() - va.default_12m.sum())]])
res["bias"]["chi2_terminal_vs_nonterminal_12m_dr_p"] = float(stats.chi2_contingency(ct)[1])
# Mature 36m vintages: true lifetime DR vs terminal-only DR (identical once fully matured)
yr = df[(df.term_m == 36)].groupby("issue_year").agg(
    share_terminal=("terminal", "mean"),
    dr_terminal=(Y, lambda s: s[df.loc[s.index, "terminal"] == 1].mean()),
    dr_all_loans_incl_open=(Y, "mean"))
yr.to_csv(T / "07_terminal_bias_by_year.csv")
print(yr.round(4).to_string())
json.dump(res, open(T / "07_replication_summary.json", "w"), indent=2, default=float)
print(json.dumps(res, indent=2, default=float))
