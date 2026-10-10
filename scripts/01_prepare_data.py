"""Load raw LendingClub data, build targets and features, save parquet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from crm import config, data

df = data.prepare()
df.to_parquet(config.DATA_DIR / "loans.parquet")

obs = df[df.obs_12m == 1]
summary = pd.DataFrame({
    "loans": df.groupby("issue_year").size(),
    "funded_$bn": df.groupby("issue_year").funded_amnt.sum() / 1e9,
    "terminal_loans": df.groupby("issue_year").terminal.sum(),
    "dr_lifetime_terminal": df[df.terminal == 1].groupby("issue_year").bad_lifetime.mean(),
    "dr_12m_all_loans": obs.groupby("issue_year").default_12m.mean(),
})
summary.to_csv(config.TAB_DIR / "01_cohort_summary.csv")
print(summary.round(4))
print("rows", len(df), "terminal", int(df.terminal.sum()), "12m-observable", len(obs))
