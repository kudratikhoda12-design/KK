#!/usr/bin/env bash
# D104 Step-3 lanes (4 parallel single-threaded processes); each group writes its own results file.
cd "$(dirname "$0")/../.."
g() { KANFAE_RESULTS=reports/step3/$1.jsonl scripts/run_logged.sh step3_$1 .venv/bin/python scripts/run_neural_synthetic.py --analyses $2; }
p() { scripts/run_logged.sh step3_prune_$1 .venv/bin/python scripts/kanfae_prune_symbolic.py --variant $1; }
case "$1" in
  A) g H1 H1; g SECONDARY SECONDARY ;;
  B) g H2_CONTROLS H2,CONTROLS; g AB1_AB3_AB4 AB1,AB3,AB4 ;;
  C) g H3 H3; g AB5_AB8 AB5,AB6,AB7,AB8 ;;
  D) p plain; p reg ;;
esac
