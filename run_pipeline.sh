#!/usr/bin/env bash
# End-to-end pipeline. Step 1 uses the shipped 50k sample unless RAW_CSV is set.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${RAW_CSV:-}" ]]; then
  python scripts/01_extract_and_engineer.py --raw "$RAW_CSV"
else
  python scripts/01_extract_and_engineer.py
fi
if [[ "${TUNE:-0}" == "1" ]]; then python scripts/02_tune.py; fi   # ~25 min, optional
python scripts/03_eda.py
python scripts/04_train_evaluate.py
python scripts/05_interpret.py
