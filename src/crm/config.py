"""Central configuration: paths, dates, and sample definitions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = Path("/home/user/data_raw/accepted_2007_to_2018Q4.csv.gz")
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
TAB_DIR = REPORT_DIR / "tables"
MODEL_DIR = ROOT / "models"

for _d in (DATA_DIR, FIG_DIR, TAB_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Data snapshot: the latest last_pymnt_d in the file is 2019-03.
SNAPSHOT = "2019-03-01"
# 12-month PD horizon (Basel / IFRS 9 stage 1). A loan needs the full
# 12-month window plus >=3 months for a late payment to reach 90+ dpd.
HORIZON_M = 12
LAST_OBSERVABLE_ISSUE = "2017-12-01"

# Out-of-time design.
TRAIN_YEARS = range(2007, 2016)   # development sample 2007-2015
TEST_YEAR = 2016                  # OOT test: model selection + recalibration fit
VALID_YEAR = 2017                 # OOT validation cohort: untouched until the end

SEED = 42
