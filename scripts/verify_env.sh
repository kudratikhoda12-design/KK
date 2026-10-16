#!/usr/bin/env bash
# Evidence for criteria E-1 (global env unchanged) and E-5 (lock-file reproducibility).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OUT=reports/logs/env_evidence; mkdir -p "$OUT"
python3 -m pip list 2>/dev/null > "$OUT/global_pip_list_after.txt"
if diff reports/logs/global_pip_list_before.txt "$OUT/global_pip_list_after.txt" > "$OUT/global_pip_diff.txt"; then
  echo "E-1 global pip list unchanged: YES"; echo PASS > "$OUT/E1_status.txt"
else echo "E-1 global pip list unchanged: NO"; cat "$OUT/global_pip_diff.txt"; echo FAIL > "$OUT/E1_status.txt"; fi
SCR="${KANFAE_SCRATCH:-/tmp}/kanfae_lockcheck_venv"; rm -rf "$SCR"
python3 -m venv "$SCR"
"$SCR/bin/pip" install -q --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
RC=$?
"$SCR/bin/pip" freeze --exclude-editable > "$OUT/lockcheck_freeze.txt"
if [ $RC -eq 0 ] && diff requirements-lock.txt "$OUT/lockcheck_freeze.txt" > "$OUT/lockcheck_diff.txt"; then
  echo "E-5 reinstall from lock reproduces lock exactly: YES"; echo PASS > "$OUT/E5_status.txt"
else echo "E-5 reinstall from lock: FAILED (rc=$RC)"; cat "$OUT/lockcheck_diff.txt"; echo FAIL > "$OUT/E5_status.txt"; fi
rm -rf "$SCR"
