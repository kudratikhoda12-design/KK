"""Expected Credit Loss for the 2017 cohort: 12-month (IFRS 9 stage 1) and
lifetime, plus an out-of-time backtest of the 12-month ECL.

PD   : champion 12m PD, recalibrated on 2016 (script 03).
LGD  : pooled long-run mean (no borrower-level model beat it OOT, script 04).
EAD  : fractional-logit EAD-ratio GLM x funded amount (beat the mean OOT).
Lifetime PD: PD_L = 1 - (1 - PD_12)^k, with k = ln(1-CDR_life)/ln(1-CDR_12)
estimated per grade x term on fully matured vintages (cumulative-hazard scaling).
"""
import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from crm import config, lgd_ead as le

T = config.TAB_DIR
with open(config.MODEL_DIR / "lgd_ead.pkl", "rb") as f:
    LE = pickle.load(f)
LGD = LE["lgd_mean"]

loans = pd.read_parquet(config.DATA_DIR / "loans.parquet")
pdf = pd.read_parquet(config.DATA_DIR / "pd_final_2017.parquet")[["id", "pd_final"]]
v = loans[loans.issue_year == 2017].merge(pdf, on="id", how="inner")
assert len(v) == (loans.issue_year == 2017).sum()

v_le = le.prep(v)
v["ead_ratio_hat"] = LE["ead_glm"].predict(v_le).to_numpy()
v["ead_12m"] = v.funded_amnt * v.ead_ratio_hat
v["lgd"] = LGD
v["ecl_12m"] = v.pd_final * v.lgd * v.ead_12m

# --- Lifetime PD term structure from matured vintages ---------------------------
mature = loans[((loans.term_m == 36) & loans.issue_year.between(2012, 2015)) |
               ((loans.term_m == 60) & loans.issue_year.between(2012, 2013))]
ts = mature.groupby(["grade", "term_m"], observed=True).agg(
    n=("id", "size"), cdr_12m=("default_12m", "mean"), cdr_life=("bad_lifetime", "mean"))
ts["k"] = np.log(1 - ts.cdr_life) / np.log(1 - ts.cdr_12m)
dl = mature[mature.bad_lifetime == 1]
ts["ead_ratio_life"] = (dl.ead_default / dl.funded_amnt).groupby([dl.grade, dl.term_m], observed=True).mean()
ts.to_csv(T / "05_pd_term_structure.csv")
print(ts.round(4).to_string())
key = pd.MultiIndex.from_arrays([v.grade, v.term_m])
v["k"] = ts["k"].reindex(key).to_numpy()
v["pd_life"] = 1 - (1 - v.pd_final) ** v.k
v["ead_life"] = v.funded_amnt * ts["ead_ratio_life"].reindex(key).to_numpy()
v["ecl_life"] = v.pd_life * v.lgd * v.ead_life

# --- Backtest of 12m ECL vs realised 12m credit loss --------------------------------
d = v[v.default_12m == 1]
realised_loss_lgdmean = float((d.ead_default * LGD).sum())
realised_ead = float(d.ead_default.sum())
# Independence-only variance (lower bound; correlation handled by the MC in script 06)
var_ind = float(((v.lgd * v.ead_12m) ** 2 * v.pd_final * (1 - v.pd_final)).sum())
z = (realised_loss_lgdmean - v.ecl_12m.sum()) / np.sqrt(var_ind)
res = {
    "n_loans": len(v), "funded_exposure": float(v.funded_amnt.sum()),
    "lgd_used": LGD, "mean_pd_12m": float(v.pd_final.mean()), "observed_dr_12m": float(v.default_12m.mean()),
    "ead_12m_total": float(v.ead_12m.sum()),
    "ecl_12m": float(v.ecl_12m.sum()), "ecl_12m_pct": float(v.ecl_12m.sum() / v.funded_amnt.sum()),
    "ecl_lifetime": float(v.ecl_life.sum()), "ecl_lifetime_pct": float(v.ecl_life.sum() / v.funded_amnt.sum()),
    "mean_pd_lifetime": float(v.pd_life.mean()),
    "backtest": {"predicted_ecl_12m": float(v.ecl_12m.sum()),
                 "realised_12m_ead_of_defaults": realised_ead,
                 "predicted_ead_of_defaults": float((v.pd_final * v.ead_12m).sum()),
                 "realised_12m_loss_at_mean_lgd": realised_loss_lgdmean,
                 "ratio_realised_to_predicted": realised_loss_lgdmean / float(v.ecl_12m.sum()),
                 "z_independence": float(z), "p_independence": float(2 * stats.norm.sf(abs(z)))},
}
by_grade = v.groupby("grade", observed=True).agg(
    loans=("id", "size"), funded=("funded_amnt", "sum"), pd_12m=("pd_final", "mean"),
    odr_12m=("default_12m", "mean"), pd_life=("pd_life", "mean"),
    ecl_12m=("ecl_12m", "sum"), ecl_life=("ecl_life", "sum"))
by_grade["ecl_12m_pct"] = by_grade.ecl_12m / by_grade.funded
by_grade["ecl_life_pct"] = by_grade.ecl_life / by_grade.funded
by_grade.to_csv(T / "05_ecl_by_grade_2017.csv")
print(by_grade.round(4).to_string())
v[["id", "grade", "term_m", "funded_amnt", "default_12m", "ead_default", "pd_final", "pd_life",
   "lgd", "ead_12m", "ead_life", "ecl_12m", "ecl_life"]].to_parquet(config.DATA_DIR / "portfolio_2017.parquet")
json.dump(res, open(T / "05_ecl_summary.json", "w"), indent=2, default=float)
print(json.dumps(res, indent=2, default=float))
