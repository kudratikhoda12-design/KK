#!/usr/bin/env bash
# Round 24 (D165): docs/manuscript_journal_main_round24.md -> reports/manuscript_journal_main_round24.html (pandoc) -> reports/manuscript_journal_main_round24.pdf
# (headless Chromium), the same toolchain as build_pdf.sh. Figures are referenced relative to docs/.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -c "import json, sys; sys.path.insert(0, 'scripts'); import manuscript_journal_main_round24 as M
open('reports/logs/manuscript_journal24_meta.json', 'w', encoding='utf-8').write(json.dumps({'title': M.TITLE}, ensure_ascii=False))"
pandoc docs/manuscript_journal_main_round24.md -f markdown+pipe_tables+lists_without_preceding_blankline -t html5 --standalone \
  --resource-path=docs --metadata-file=reports/logs/manuscript_journal24_meta.json --css reports/report.css \
  --embed-resources \
  -o reports/manuscript_journal_main_round24.html
CHROME=/opt/pw-browsers/chromium-1194/chrome-linux/chrome
"$CHROME" --headless --no-sandbox --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=reports/manuscript_journal_main_round24.pdf reports/manuscript_journal_main_round24.html 2>/dev/null
pdftotext -layout reports/manuscript_journal_main_round24.pdf reports/logs/manuscript_journal24_pdf_text.txt
rm -rf reports/logs/manuscript_journal24_pages && mkdir -p reports/logs/manuscript_journal24_pages
pdftoppm -r 60 -png reports/manuscript_journal_main_round24.pdf reports/logs/manuscript_journal24_pages/page
ls -la reports/manuscript_journal_main_round24.pdf; pdfinfo reports/manuscript_journal_main_round24.pdf | grep -E "Pages|Title|Page size"
