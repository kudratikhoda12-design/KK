#!/usr/bin/env bash
# Markdown -> standalone HTML (pandoc) -> PDF (headless Chromium). Then text extraction and page renders for checks.
set -euo pipefail
cd "$(dirname "$0")/../reports"
pandoc final_research_prototype_report.md -f markdown+pipe_tables+lists_without_preceding_blankline -t html5 --standalone --toc --toc-depth=2 \
  --metadata title="KAN-FAE research prototype: final report" --css report.css --embed-resources \
  -o final_research_prototype_report.html
CHROME=/opt/pw-browsers/chromium-1194/chrome-linux/chrome
"$CHROME" --headless --no-sandbox --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=final_research_prototype_report.pdf final_research_prototype_report.html 2>/dev/null
pdftotext -layout final_research_prototype_report.pdf logs/report_pdf_text.txt
rm -rf logs/pdf_pages && mkdir -p logs/pdf_pages
pdftoppm -r 60 -png final_research_prototype_report.pdf logs/pdf_pages/page
ls -la final_research_prototype_report.pdf; pdfinfo final_research_prototype_report.pdf | grep -E "Pages|Title|Page size"
