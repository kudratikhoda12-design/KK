"""Verify the final deliverables (owner requirement 7). Writes reports/logs/deliverable_verification.json.

Checks: PDF exists/opens/has selectable text and no unrendered placeholders; page renders exist and are not blank;
every rendered value and every table number in the Markdown appears in the PDF text; summary-table numbers match
the CSV; every test in test_results_latest.csv appears in a pytest log; every logged command has its log file.
"""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
res = {}


def norm(s):
    return re.sub(r"\s+", " ", s)


pdf = REP / "final_research_prototype_report.pdf"
res["pdf_exists"] = pdf.exists()
info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
res["pdf_opens"] = info.returncode == 0
res["pdf_pages"] = int(re.search(r"Pages:\s+(\d+)", info.stdout).group(1))
text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
res["pdf_text_chars"] = len(text)
T = norm(text)
res["pdf_no_placeholders"] = "{{" not in text
res["pdf_has_all_17_sections"] = all(re.search(rf"\b{k}\.\s", T) for k in range(1, 18))

pages = sorted((REP / "logs/pdf_pages").glob("page-*.png"))
blank = []
for p in pages:
    a = np.asarray(Image.open(p).convert("L"))
    if (a < 200).mean() < 0.002:
        blank.append(p.name)
res["page_renders"] = len(pages)
res["blank_pages"] = blank

md = (REP / "final_research_prototype_report.md").read_text()
nums_md = set(re.findall(r"(?<![\w.])\d+\.\d{3}(?![\d])", md))
missing = sorted(n for n in nums_md if n not in T)
res["md_3dp_numbers"] = len(nums_md)
res["md_numbers_missing_in_pdf"] = missing[:50]
vals = json.loads((REP / "report_values.json").read_text())
res["report_values_missing_in_pdf"] = [k for k, v in vals.items() if str(v) not in T and k not in ("git_head",)]

S = pd.read_csv(REP / "tables/summary_by_model.csv")
p0 = S[(S.setting == "P0")]
chk = []
for m in p0.model:
    for metric in ("curve_error", "ibs", "uno_c"):
        v = float(p0[p0.model == m][f"{metric}__mean"].iloc[0])
        chk.append(f"{v:.3f}" in T)
res["P0_summary_values_in_pdf"] = f"{sum(chk)}/{len(chk)}"

TL = pd.read_csv(REP / "test_results_latest.csv")
logs = "".join(p.read_text(errors="ignore") for p in (REP / "logs").glob("*.log"))
not_run = [n for n in TL.nodeid if n.split("::")[-1].split("[")[0] not in logs]
res["tests_reported"] = len(TL)
res["tests_without_log_evidence"] = not_run
recs = [json.loads(l) for l in (REP / "logs/test_records.jsonl").read_text().splitlines() if l.strip()]
final = [r for r in recs if r["run_id"] == "final_full_suite"]
res["final_suite_counts"] = {o: sum(r["outcome"] == o for r in final) for o in ("passed", "failed", "skipped")}

cmds = [json.loads(l) for l in (REP / "logs/commands.jsonl").read_text().splitlines() if l.strip()]
res["commands_logged"] = len(cmds)
res["commands_missing_log_file"] = [c["name"] for c in cmds if not (ROOT / c["log"]).exists()]
csv = pd.read_csv(REP / "experiment_results.csv", low_memory=False)
jl = sum(1 for l in (REP / "experiment_results.jsonl").read_text().splitlines() if l.strip())
res["experiment_csv_rows_equal_jsonl"] = len(csv) == jl
res["experiment_records"] = jl

ok = (res["pdf_exists"] and res["pdf_opens"] and res["pdf_text_chars"] > 20000 and res["pdf_no_placeholders"]
      and res["pdf_has_all_17_sections"] and not blank and not missing and not res["report_values_missing_in_pdf"]
      and all(chk) and not not_run and not res["commands_missing_log_file"] and res["experiment_csv_rows_equal_jsonl"])
res["ALL_CHECKS_PASS"] = bool(ok)
(REP / "logs/deliverable_verification.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
