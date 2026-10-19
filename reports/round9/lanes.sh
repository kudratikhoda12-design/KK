#!/usr/bin/env bash
# D108 fairness lanes (4 parallel single-threaded processes); each lane writes its own results file.
cd "$(dirname "$0")/../.."
r() { KANFAE_RESULTS=reports/round9/$1.jsonl scripts/run_logged.sh r9_$1 .venv/bin/python scripts/run_kan_fairness.py --cells $2; }
case "$1" in
  1) r lane1 mlp:S2,mlp:S3,mlp_wide:S2 ;;
  2) r lane2 kan:S2,kan:S3 ;;
  3) r lane3 kan_shared:S2,kan_shared:S3 ;;
  4) r lane4 kan_unb:S2,kan_unb:S3,mlp_wide:S3 ;;
esac
