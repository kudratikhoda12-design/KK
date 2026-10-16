#!/usr/bin/env bash
# Creates the project-local environment (D10). Never touches the global Python environment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CPU_INDEX=https://download.pytorch.org/whl/cpu
python3 -m venv .venv
.venv/bin/python -m pip --version
if [ -f requirements-lock.txt ] && [ "${KANFAE_FROM_LOCK:-0}" = "1" ]; then
  # Exact reproduction from the lock file.
  .venv/bin/pip install --extra-index-url "$CPU_INDEX" -r requirements-lock.txt
  .venv/bin/pip install --no-deps -e .
else
  .venv/bin/pip install --index-url "$CPU_INDEX" "torch==2.14.1+cpu"
  .venv/bin/pip install -e ".[dev]"
  .venv/bin/pip freeze --exclude-editable > requirements-lock.txt
fi
.venv/bin/pip check
