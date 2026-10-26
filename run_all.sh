#!/usr/bin/env bash
# Reproduce the entire project end to end.
# Synthetic study: ~1 minute on CPU.
# Real-data study (PmatData): pass --with-real to also download (~107MB,
# no login needed) and run it; takes a few more minutes on CPU.
set -euo pipefail
cd "$(dirname "$0")"

python3 scripts/00_sanity_check.py
python3 scripts/01_generate_data.py
python3 scripts/02_train_denoise.py
python3 scripts/03_train_pretrain.py
python3 scripts/04_fewshot_eval.py
python3 scripts/05_failure_analysis.py
python3 scripts/06_bandit.py

if [[ "${1:-}" == "--with-real" ]]; then
  bash scripts/08_download_real_data.sh
  python3 scripts/09_prepare_real_data.py
  python3 scripts/10_real_denoise.py
  python3 scripts/11_real_pretrain.py
  python3 scripts/12_real_fewshot_eval.py
  python3 scripts/13_real_failure_analysis.py
  python3 scripts/14_real_bandit.py
fi

python3 scripts/07_build_report.py

echo "Done. See README.md, reports/figures/, reports/results/, and reports/final_report.pdf"
