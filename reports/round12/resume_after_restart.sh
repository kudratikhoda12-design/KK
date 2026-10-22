#!/usr/bin/env bash
# D127: resume after the container restart at about 04:27 UTC on 2026-10-22 (all workers killed). Each slot first
# completes one interrupted job (completed cells are skipped: the runner is resumable per lane file), then continues as
# a normal queue worker with the ORIGINAL launch time, so the D113 budget deadline is unchanged.
set -u
cd "$(dirname "$0")/../.."
slot=$1
case $slot in
  1) R12_OUT=reports/round12/results_lane4.jsonl .venv/bin/python scripts/run_round12.py --scenario S-Sparse --reps 18-18 --only prune_symbolic ;;
  2) KANFAE_RESULTS=reports/round12/p0_lane1.jsonl .venv/bin/python scripts/run_round12_p0.py --reps 19-19 --cells kan:S3 ;;
  3) R12_OUT=reports/round12/results_lane2.jsonl .venv/bin/python scripts/run_round12.py --scenario S-Int --reps 19-19 ;;
  4) R12_OUT=reports/round12/results_lane3.jsonl .venv/bin/python scripts/run_round12.py --scenario S-Sparse --reps 19-19 ;;
esac
.venv/bin/python scripts/run_round12_lanes.py --lane $slot --start-utc 2026-10-21T19:12:34Z
