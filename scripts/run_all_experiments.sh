#!/usr/bin/env bash
# Runs the unblocked experiments sequentially (single writer to reports/experiment_results.jsonl).
set -u
cd "$(dirname "$0")/.."
scripts/run_logged.sh exp_synthetic_baselines_v2 .venv/bin/python scripts/run_synthetic.py --reps 5 --n 1000
scripts/run_logged.sh exp_cmapss_baselines .venv/bin/python scripts/run_cmapss.py --splits 5
