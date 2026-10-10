"""Step 1 - SQL data pipeline.

    python scripts/01_extract_and_engineer.py [--raw path/to/accepted_2007_to_2018Q4.csv]

--raw is only required to (re)create data/lendingclub_2016_sample.parquet from the
public LendingClub file. The 50k-loan sample is shipped with the repository, so by
default this script just runs the survival-table + feature-engineering SQL on it.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coxkan_churn import config as C, data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", help="path to accepted_2007_to_2018Q4.csv (optional)")
args = ap.parse_args()

if args.raw:
    s = data.extract_cohort(args.raw)
    print(f"extracted cohort sample: {s.shape} -> {C.SAMPLE_PATH}")
df = data.build_model_table()
print(f"model table: {df.shape} -> {C.MODEL_TABLE_PATH}")
print(df["exit_reason"].value_counts())
print("event rate:", df["event"].mean().round(4), "| median duration:", df["duration_months"].median())
