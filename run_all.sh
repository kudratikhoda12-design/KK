#!/usr/bin/env bash
# Reproduce the entire project end to end (~1 minute on CPU).
set -euo pipefail
cd "$(dirname "$0")"

python3 scripts/00_sanity_check.py
python3 scripts/01_generate_data.py
python3 scripts/02_train_denoise.py
python3 scripts/03_train_pretrain.py
python3 scripts/04_fewshot_eval.py
python3 scripts/05_failure_analysis.py
python3 scripts/06_bandit.py
python3 scripts/07_build_report.py

echo "Done. See README.md, reports/figures/, reports/results/, and reports/final_report.pdf"
