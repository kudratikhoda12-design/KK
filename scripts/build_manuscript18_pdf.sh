#!/usr/bin/env bash
# Round 18 (D146): docs/manuscript_round18.md -> reports/manuscript_round18.html (pandoc) -> reports/manuscript_round18.pdf
# (headless Chromium), the same toolchain as build_pdf.sh. Figures are referenced relative to docs/.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -c "import json, sys; sys.path.insert(0, 'scripts'); import manuscript_round18 as M
open('reports/logs/manuscript18_meta.json', 'w', encoding='utf-8').write(json.dumps({'title': M.TITLE}, ensure_ascii=False))"
pandoc docs/manuscript_round18.md -f markdown+pipe_tables+lists_without_preceding_blankline -t html5 --standalone \
  --toc --toc-depth=2 --resource-path=docs --metadata-file=reports/logs/manuscript18_meta.json --css reports/report.css \
  --embed-resources \
  -o reports/manuscript_round18.html
CHROME=/opt/pw-browsers/chromium-1194/chrome-linux/chrome
"$CHROME" --headless --no-sandbox --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=reports/manuscript_round18.pdf reports/manuscript_round18.html 2>/dev/null
pdftotext -layout reports/manuscript_round18.pdf reports/logs/manuscript18_pdf_text.txt
rm -rf reports/logs/manuscript18_pages && mkdir -p reports/logs/manuscript18_pages
pdftoppm -r 60 -png reports/manuscript_round18.pdf reports/logs/manuscript18_pages/page
ls -la reports/manuscript_round18.pdf; pdfinfo reports/manuscript_round18.pdf | grep -E "Pages|Title|Page size"
