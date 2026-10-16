#!/usr/bin/env bash
# Usage: scripts/run_logged.sh <log-name> <command> [args...]
# Runs the command from the repo root, saves full stdout+stderr to reports/logs/<log-name>.log
# and appends a JSON record (command, exit code, timings, git HEAD) to reports/logs/commands.jsonl.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="$1"; shift
LOG="$ROOT/reports/logs/$NAME.log"
mkdir -p "$ROOT/reports/logs"
START=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s.%N)
( cd "$ROOT" && "$@" ) > "$LOG" 2>&1
CODE=$?
T1=$(date +%s.%N); END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
HEAD=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo none)
python3 - "$ROOT/reports/logs/commands.jsonl" "$NAME" "$CODE" "$START" "$END" "$T0" "$T1" "$HEAD" "$@" <<'PY'
import json, sys
out, name, code, start, end, t0, t1, head, *cmd = sys.argv[1:]
rec = dict(name=name, command=" ".join(cmd), exit_code=int(code), start_utc=start, end_utc=end,
           elapsed_s=round(float(t1) - float(t0), 3), git_head=head, log=f"reports/logs/{name}.log")
open(out, "a").write(json.dumps(rec) + "\n")
PY
echo "[run_logged] $NAME exit=$CODE elapsed=$(python3 -c "print(round($T1-$T0,1))")s log=reports/logs/$NAME.log"
exit $CODE
