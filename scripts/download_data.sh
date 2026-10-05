#!/usr/bin/env bash
# Shell-only alternative to `python -m src.data download` (Linux / macOS / Git-Bash).
# Downloads Binance Vision monthly 1-minute klines and verifies each file's SHA-256.
#
#   bash scripts/download_data.sh BTCUSDT 2017-08 2026-09
#
# Files land in data/raw/spot/<SYMBOL>/1m/, which is where src/data.py expects them.
set -euo pipefail
SYMBOL="${1:-BTCUSDT}"; START="${2:-2017-08}"; END="${3:-2026-09}"
OUT="data/raw/spot/${SYMBOL}/1m"; mkdir -p "$OUT"; cd "$OUT"
BASE="https://data.binance.vision/data/spot/monthly/klines/${SYMBOL}/1m"
SHA=$(command -v sha256sum >/dev/null && echo "sha256sum" || echo "shasum -a 256")

m="$START"
while [[ "$m" < "$END" || "$m" == "$END" ]]; do
  f="${SYMBOL}-1m-${m}.zip"
  if [[ ! -f "$f" ]]; then
    curl -fsS -O "${BASE}/${f}" && curl -fsS -O "${BASE}/${f}.CHECKSUM" \
      && $SHA -c "${f}.CHECKSUM" || echo "!! ${f}: missing or checksum failed"
    sleep 0.2
  fi
  # next month
  y=${m%-*}; mo=${m#*-}; mo=$((10#$mo + 1)); if (( mo > 12 )); then mo=1; y=$((y + 1)); fi
  m=$(printf "%04d-%02d" "$y" "$mo")
done
echo "Done. Now run: python -m src.data build --symbol ${SYMBOL}"
