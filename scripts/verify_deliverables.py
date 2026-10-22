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


FINAL_RUN = "final_full_suite_round12c_rerun"


def norm(s):
    return re.sub(r"\s+", " ", s)


pdf = REP / "final_research_prototype_report.pdf"
res["pdf_exists"] = pdf.exists()
info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
res["pdf_opens"] = info.returncode == 0
res["pdf_pages"] = int(re.search(r"Pages:\s+(\d+)", info.stdout).group(1))
text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
res["pdf_text_chars"] = len(text)
raw = subprocess.run(["pdftotext", "-raw", str(pdf), "-"], capture_output=True, text=True).stdout
T = norm(text) + " || " + norm(raw)          # layout and raw extraction (raw keeps wrapped table cells contiguous)
res["pdf_no_placeholders"] = "{{" not in text
# round 3 (2026-10-17): Part A sections 0.1-0.3 and R3.1-R3.11; round 2 renumbered R2.1-R2.15; round 1 R1.1-R1.17
res["pdf_has_part1_sections"] = all(f"0.{k} " in T for k in range(1, 4)) and all(
    re.search(rf"\bR12c\.{k}\s", T) for k in range(1, 12)) and all(
    re.search(rf"\bR12b\.{k}\s", T) for k in range(1, 8)) and all(
    re.search(rf"\bR12\.{k}\s", T) for k in range(1, 10)) and all(
    re.search(rf"\bR11\.{k}\s", T) for k in range(1, 10)) and all(
    re.search(rf"\bR10\.{k}\s", T) for k in range(1, 9)) and all(
    re.search(rf"\bR9\.{k}\s", T) for k in range(1, 9)) and all(
    re.search(rf"\bR8\.{k}\s", T) for k in range(1, 15)) and all(
    re.search(rf"\bR7\.{k}\s", T) for k in range(1, 13)) and all(
    re.search(rf"\bR6\.{k}\s", T) for k in range(1, 11)) and all(
    re.search(rf"\bR5\.{k}\s", T) for k in range(1, 12)) and all(
    re.search(rf"\bR4\.{k}\s", T) for k in range(1, 12)) and all(
    re.search(rf"\bR3\.{k}\s", T) for k in range(1, 12)) and all(re.search(rf"\bR2\.{k}\s", T) for k in range(1, 16))
res["pdf_has_all_17_sections"] = all(re.search(rf"\bR1\.{k}\s", T) for k in range(1, 18))

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
used = set(re.findall(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", (REP / "report_template.md").read_text()))
res["report_values_used"] = len(used)
res["report_values_missing_in_pdf"] = [k for k in sorted(used) if norm(str(vals[k])) not in T]
res["round2_values_missing_in_pdf"] = []
for tpl, vf in (("report_template_round2.md", "report_values_round2.json"),
                ("report_template_round3.md", "report_values_round3.json"),
                ("report_template_round4.md", "report_values_round4.json"),
                ("report_template_round5.md", "report_values_round5.json"),
                ("report_template_round6.md", "report_values_round6.json"),
                ("report_template_round7.md", "report_values_round7.json"),
                ("report_template_round8.md", "report_values_round8.json"),
                ("report_template_round9.md", "report_values_round9.json"),
                ("report_template_round10.md", "report_values_round10.json"),
                ("report_template_round11.md", "report_values_round11.json"),
                ("report_template_round12.md", "report_values_round12.json"),
                ("report_template_round12b.md", "report_values_round12b.json"),
                ("report_template_round12c.md", "report_values_round12c.json")):
    vals2 = json.loads((REP / vf).read_text())
    used2 = set(re.findall(r"\{\{V:([A-Za-z0-9_\-]+)\}\}", (REP / tpl).read_text()))
    res[f"{tpl}_values_used"] = len(used2)
    res["round2_values_missing_in_pdf"] += [f"{tpl}:{k}" for k in sorted(used2) if norm(str(vals2[k])) not in T]

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
res["final_suite_counts_round1"] = {o: sum(r["outcome"] == o for r in final) for o in ("passed", "failed", "skipped")}
final2 = [r for r in recs if r["run_id"] == FINAL_RUN]
res["final_suite_run"] = FINAL_RUN
res["final_suite_counts"] = {o: sum(r["outcome"] == o for r in final2) for o in ("passed", "failed", "skipped", "xfailed")}
# round 3: the failures allowed are exactly the revised gate's (reported as R2/R1 FAILED); everything else must pass
res["final_suite_failures"] = sorted(r["nodeid"] for r in final2 if r["outcome"] == "failed")
res["final_suite_only_failure_is_R2"] = set(res["final_suite_failures"]) <= {
    "tests/test_learnability_fa4f.py::test_R1_oracle_features_head",
    # D125/D126: CK-6b is a strict xfail (D123); CK-6c stays a visible failure (D124, D126)
    "tests/test_round12b_ck.py::test_CK6c_repaired_afcm_identifiable_rise"} and len(final2) > 0   # round 8: the 4f gate's
                                                                                         # original R1 (not gating, D103)
                                                                                         # (the 4d gate file skips: D89)

cmds = [json.loads(l) for l in (REP / "logs/commands.jsonl").read_text().splitlines() if l.strip()]
res["commands_logged"] = len(cmds)
res["commands_missing_log_file"] = [c["name"] for c in cmds if not (ROOT / c["log"]).exists()]
csv = pd.read_csv(REP / "experiment_results.csv", low_memory=False)
jl = sum(1 for l in (REP / "experiment_results.jsonl").read_text().splitlines() if l.strip())
res["experiment_csv_rows_equal_jsonl"] = len(csv) == jl
res["experiment_records"] = jl

ok = (res["pdf_exists"] and res["pdf_opens"] and res["pdf_text_chars"] > 20000 and res["pdf_no_placeholders"]
      and res["pdf_has_all_17_sections"] and res["pdf_has_part1_sections"] and not res["round2_values_missing_in_pdf"]
      and res["final_suite_only_failure_is_R2"] and not blank and not missing and not res["report_values_missing_in_pdf"]
      and all(chk) and not not_run and not res["commands_missing_log_file"] and res["experiment_csv_rows_equal_jsonl"])
res["ALL_CHECKS_PASS"] = bool(ok)
(REP / "logs/deliverable_verification.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
