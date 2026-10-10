#!/usr/bin/env bash
# Render reports/REPORT.md -> reports/REPORT.html (self-contained) -> reports/REPORT.pdf
set -euo pipefail
cd "$(dirname "$0")"
pandoc REPORT.md -o REPORT.html --standalone --embed-resources --toc --toc-depth=2 \
  --mathml --css report.css --metadata pagetitle="CoxKAN Churn Report"
CHROME="${CHROME:-/opt/pw-browsers/chromium-1194/chrome-linux/chrome}"
"$CHROME" --headless --no-sandbox --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=REPORT.pdf "file://$PWD/REPORT.html" 2>/dev/null
echo "built REPORT.html and REPORT.pdf"
