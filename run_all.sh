#!/usr/bin/env bash
# End-to-end pipeline. Raw data: Kaggle "wordsforthewise/lending-club"
# (accepted_2007_to_2018Q4.csv.gz) at the path set in src/crm/config.py.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pytest -q tests
python3 scripts/01_prepare_data.py          # targets & features
python3 scripts/02a_window_experiment.py    # training-window test (DeLong)
python3 scripts/02_pd_models.py             # LR / LightGBM / XGBoost / RF
python3 scripts/02b_lr_improvement.py       # scorecard v2 (LR test + DeLong)
python3 scripts/03_validation.py            # recalibration, backtests, PSI, vintages
python3 scripts/04_lgd_ead.py               # LGD / EAD models (OOT)
python3 scripts/05_ecl.py                   # 12m & lifetime ECL + backtest
python3 scripts/06_portfolio_mc.py          # rho MLE, Monte Carlo VaR/ES, stress tests
python3 scripts/07_replication_original.py  # original design + bias diagnostics
