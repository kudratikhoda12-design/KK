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


FINAL_RUN = "final_full_suite_round22"


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
MS_SECTIONS = ["Abstract", "Keywords", "1. Introduction", "2. Models and estimation", "3. Simulation study",
               "4. Real-data applications", "5. Discussion", "6. Conclusions", "Declarations",
               "Appendix A. Number provenance and verification"]
MS16_SECTIONS = ["5. Phase 2: pre-registered methodological revisions (Rounds 15–16)", "6. Discussion", "7. Conclusions",
                 "5. Phase 2: pre-registered methodological revisions (Rounds 15–17)",
                 "5. Phase 2: pre-registered methodological revisions (Rounds 15–18)",
                 "5.5 Anchoring decided on a confirmation split (H-R18a, H-R18b)",
                 "5. Phase 2: pre-registered methodological revisions (Rounds 15–19)",
                 "5.6 Refitting on the whole validation set (development check)",
                 "5. Phase 2: pre-registered methodological revisions (Rounds 15–20)",
                 "5.7 Semiparametric KAN-AFT and cross-confirmation (H-R20a, H-R20b)",
                 "5. Phase 2: pre-registered methodological revisions (Rounds 15–21)",
                 "5.9 Round-20 methods on real data (H-R21a, H-R21b, H-R21c)",
                 # Round 22 (D159): the journal main manuscript is the embedded one (the Supplementary Material is a
                 # separate PDF, verified below)
                 "5.2 Where simpler landmark models suffice", "Appendix. Number provenance",
                 "5.4 Two-way gated anchoring and a log-normal KAN-AFT (H-R17a, H-R17b)",
                 "5.2 Symbolic summaries: functional distillation (H-R15a, H-R16a)",
                 "5.3 A landmark-anchored functional model (H-R15b, H-R16b)"]
res["pdf_has_manuscript_sections"] = all(s in T for s in MS_SECTIONS + MS16_SECTIONS)
res["pdf_has_part1_sections"] = all(f"0.{k} " in T for k in range(1, 3)) and all(
    re.search(rf"\bR22\.{k}\s", T) for k in range(1, 12)) and all(
    re.search(rf"\bR21\.{k}\s", T) for k in range(1, 11)) and all(
    re.search(rf"\bR20\.{k}\s", T) for k in range(1, 12)) and all(
    re.search(rf"\bR19\.{k}\s", T) for k in range(1, 9)) and all(
    re.search(rf"\bR18\.{k}\s", T) for k in range(1, 11)) and all(
    re.search(rf"\bR17\.{k}\s", T) for k in range(1, 12)) and all(
    re.search(rf"\bR16\.{k}\s", T) for k in range(1, 12)) and all(
    re.search(rf"\bR15\.{k}\s", T) for k in range(1, 14)) and all(
    re.search(rf"\bR14\.{k}\s", T) for k in range(1, 6)) and all(
    re.search(rf"\bR13\.{k}\s", T) for k in range(1, 12)) and all(
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
                ("report_template_round12c.md", "report_values_round12c.json"),
                ("report_template_round13.md", "report_values_round13.json"),
                ("report_template_round14.md", "report_values_round14.json"),
                ("report_template_round15.md", "report_values_round15.json"),
                ("report_template_round16.md", "report_values_round16.json"),
                ("report_template_round17.md", "report_values_round17.json"),
                ("report_template_round18.md", "report_values_round18.json"),
                ("report_template_round19.md", "report_values_round19.json"),
                ("report_template_round20.md", "report_values_round20.json"),
                ("report_template_round21.md", "report_values_round21.json"),
                ("report_template_round22.md", "report_values_round22.json")):
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
    "tests/test_round12b_ck.py::test_CK6c_repaired_afcm_identifiable_rise",
    # D157: CK-6d failed on the held-out seed 24 and is kept a visible failure
    "tests/test_round22_ck.py::test_CK6d_per_channel_reml_afcm_identifiable_rise"} and len(final2) > 0   # round 8: the 4f gate's
                                                                                         # original R1 (not gating, D103)
                                                                                         # (the 4d gate file skips: D89)

# Round 14 (D131): the manuscript's own verification (ledger, re-render, number audit, citations, sections, its PDF),
# its page renders, every logged manuscript value present in the report PDF, and the revision-16 copy of the report
import hashlib  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import verify_manuscript_round14 as VM  # noqa: E402
vm = dict(ledger=VM.check_ledger(), rerender=VM.check_rerender(), numbers=VM.check_numbers(),
          citations=VM.check_citations(), sections=VM.check_sections(), pdf=VM.check_pdf())
vm["pdf"]["passed"] = vm["pdf"]["passed"] and vm["pdf"].get("images", 0) >= len(VM.FIGURES)
res["manuscript_checks"] = {k: v["passed"] for k, v in vm.items()}
res["manuscript_pdf_pages"] = vm["pdf"].get("pages")
ms_pages = sorted((REP / "logs/manuscript_pages").glob("page-*.png"))
res["manuscript_page_renders"] = len(ms_pages)
res["manuscript_blank_pages"] = [p.name for p in ms_pages if (np.asarray(Image.open(p).convert("L")) < 200).mean() < 0.002]
led = json.loads((REP / "manuscript_values_round14.json").read_text())
texts = {e["text"] for e in led["values"] if re.search(r"\d", e["text"]) and e["spec"] != "s"}
res["manuscript_values_checked_in_report_pdf"] = len(texts)
res["manuscript_values_missing_in_report_pdf"] = sorted(x for x in texts if norm(x) not in T)[:50]
# Round 16 (D138/D139): the updated manuscript's verification (scripts/verify_manuscript.py, the same checks on
# docs/manuscript_round16.md), its page renders, its logged values in the report PDF, and the revision-18 copy
import verify_manuscript as VM16  # noqa: E402
vm16 = dict(ledger=VM16.W.check_ledger(), rerender=VM16.check_rerender(), numbers=VM16.W.check_numbers(),
            citations=VM16.W.check_citations(), sections=VM16.W.check_sections(),
            pdf=VM16.W.check_pdf(pdf=VM16.W.PDF, md=VM16.W.MD, sections=VM16.W.REQUIRED_SECTIONS, n_figures=len(VM16.FIGURES)))
vm16["pdf"]["passed"] = vm16["pdf"]["passed"] and vm16["pdf"].get("images", 0) >= len(VM16.FIGURES)
res["manuscript16_checks"] = {k: v["passed"] for k, v in vm16.items()}
res["manuscript16_pdf_pages"] = vm16["pdf"].get("pages")
ms16_pages = sorted((REP / "logs/manuscript16_pages").glob("page-*.png"))
res["manuscript16_page_renders"] = len(ms16_pages)
res["manuscript16_blank_pages"] = [p.name for p in ms16_pages if (np.asarray(Image.open(p).convert("L")) < 200).mean() < 0.002]
led16 = json.loads((REP / "manuscript_values_round16.json").read_text())
texts16 = {e["text"] for e in led16["values"] if re.search(r"\d", e["text"]) and e["spec"] != "s"}
res["manuscript16_values_checked_in_report_pdf"] = len(texts16)
res["manuscript16_values_missing_in_report_pdf"] = sorted(x for x in texts16 if norm(x) not in T)[:50]
# Round 17 (D143): the Round-17 manuscript's verification (scripts/verify_manuscript_round17.py), its page renders and
# its logged values in the report PDF; the revision-19 copy
import verify_manuscript_round17 as VM17  # noqa: E402
vm17 = dict(ledger=VM17.W.check_ledger(), rerender=VM17.check_rerender(), numbers=VM17.W.check_numbers(),
            citations=VM17.W.check_citations(), sections=VM17.W.check_sections(),
            pdf=VM17.W.check_pdf(pdf=VM17.W.PDF, md=VM17.W.MD, sections=VM17.W.REQUIRED_SECTIONS, n_figures=len(VM17.FIGURES)))
vm17["pdf"]["passed"] = vm17["pdf"]["passed"] and vm17["pdf"].get("images", 0) >= len(VM17.FIGURES)
res["manuscript17_checks"] = {k: v["passed"] for k, v in vm17.items()}
res["manuscript17_pdf_pages"] = vm17["pdf"].get("pages")
ms17_pages = sorted((REP / "logs/manuscript17_pages").glob("page-*.png"))
res["manuscript17_page_renders"] = len(ms17_pages)
res["manuscript17_blank_pages"] = [p.name for p in ms17_pages if (np.asarray(Image.open(p).convert("L")) < 200).mean() < 0.002]
led17 = json.loads((REP / "manuscript_values_round17.json").read_text())
texts17 = {e["text"] for e in led17["values"] if re.search(r"\d", e["text"]) and e["spec"] != "s"}
res["manuscript17_values_checked_in_report_pdf"] = len(texts17)
res["manuscript17_values_missing_in_report_pdf"] = sorted(x for x in texts17 if norm(x) not in T)[:50]
# Round 18 (D146): the Round-18 manuscript's verification (scripts/verify_manuscript_round18.py), its page renders and
# its logged values in the report PDF; the revision-20 copy
import verify_manuscript_round18 as VM18  # noqa: E402
vm18 = dict(ledger=VM18.W.check_ledger(), rerender=VM18.check_rerender(), numbers=VM18.W.check_numbers(),
            citations=VM18.W.check_citations(), sections=VM18.W.check_sections(),
            pdf=VM18.W.check_pdf(pdf=VM18.W.PDF, md=VM18.W.MD, sections=VM18.W.REQUIRED_SECTIONS, n_figures=len(VM18.FIGURES)))
vm18["pdf"]["passed"] = vm18["pdf"]["passed"] and vm18["pdf"].get("images", 0) >= len(VM18.FIGURES)
res["manuscript18_checks"] = {k: v["passed"] for k, v in vm18.items()}
res["manuscript18_pdf_pages"] = vm18["pdf"].get("pages")
ms18_pages = sorted((REP / "logs/manuscript18_pages").glob("page-*.png"))
res["manuscript18_page_renders"] = len(ms18_pages)
res["manuscript18_blank_pages"] = [p.name for p in ms18_pages if (np.asarray(Image.open(p).convert("L")) < 200).mean() < 0.002]
led18 = json.loads((REP / "manuscript_values_round18.json").read_text())
texts18 = {e["text"] for e in led18["values"] if re.search(r"\d", e["text"]) and e["spec"] != "s"}
res["manuscript18_values_checked_in_report_pdf"] = len(texts18)
res["manuscript18_values_missing_in_report_pdf"] = sorted(x for x in texts18 if norm(x) not in T)[:50]
# Round 19 (D149): the Round-19 manuscript's verification (scripts/verify_manuscript_round19.py), its page renders and
# its logged values in the report PDF; the revision-21 copy
import verify_manuscript_round19 as VM19  # noqa: E402
vm19 = dict(ledger=VM19.W.check_ledger(), rerender=VM19.check_rerender(), numbers=VM19.W.check_numbers(),
            citations=VM19.W.check_citations(), sections=VM19.W.check_sections(),
            pdf=VM19.W.check_pdf(pdf=VM19.W.PDF, md=VM19.W.MD, sections=VM19.W.REQUIRED_SECTIONS, n_figures=len(VM19.FIGURES)))
vm19["pdf"]["passed"] = vm19["pdf"]["passed"] and vm19["pdf"].get("images", 0) >= len(VM19.FIGURES)
res["manuscript19_checks"] = {k: v["passed"] for k, v in vm19.items()}
res["manuscript19_pdf_pages"] = vm19["pdf"].get("pages")
ms19_pages = sorted((REP / "logs/manuscript19_pages").glob("page-*.png"))
res["manuscript19_page_renders"] = len(ms19_pages)
res["manuscript19_blank_pages"] = [p.name for p in ms19_pages if (np.asarray(Image.open(p).convert("L")) < 200).mean() < 0.002]
led19 = json.loads((REP / "manuscript_values_round19.json").read_text())
texts19 = {e["text"] for e in led19["values"] if re.search(r"\d", e["text"]) and e["spec"] != "s"}
res["manuscript19_values_checked_in_report_pdf"] = len(texts19)
res["manuscript19_values_missing_in_report_pdf"] = sorted(x for x in texts19 if norm(x) not in T)[:50]
# Round 20 (D152): the Round-20 manuscript's verification (scripts/verify_manuscript_round20.py), its page renders and
# its logged values in the report PDF; the revision-22 copy (revision 21 kept)
import verify_manuscript_round20 as VM20  # noqa: E402
vm20 = dict(ledger=VM20.W.check_ledger(), rerender=VM20.check_rerender(), numbers=VM20.W.check_numbers(),
            citations=VM20.W.check_citations(), sections=VM20.W.check_sections(),
            pdf=VM20.W.check_pdf(pdf=VM20.W.PDF, md=VM20.W.MD, sections=VM20.W.REQUIRED_SECTIONS, n_figures=len(VM20.FIGURES)))
vm20["pdf"]["passed"] = vm20["pdf"]["passed"] and vm20["pdf"].get("images", 0) >= len(VM20.FIGURES)
res["manuscript20_checks"] = {k: v["passed"] for k, v in vm20.items()}
res["manuscript20_pdf_pages"] = vm20["pdf"].get("pages")
ms20_pages = sorted((REP / "logs/manuscript20_pages").glob("page-*.png"))
res["manuscript20_page_renders"] = len(ms20_pages)
res["manuscript20_blank_pages"] = [p.name for p in ms20_pages if (np.asarray(Image.open(p).convert("L")) < 200).mean() < 0.002]
led20 = json.loads((REP / "manuscript_values_round20.json").read_text())
texts20 = {e["text"] for e in led20["values"] if re.search(r"\d", e["text"]) and e["spec"] != "s"}
res["manuscript20_values_checked_in_report_pdf"] = len(texts20)
res["manuscript20_values_missing_in_report_pdf"] = sorted(x for x in texts20 if norm(x) not in T)[:50]
# Round 21 (D155): the Round-21 manuscript's verification, page renders and logged values; the revision-23 copy
import verify_manuscript_round21 as VM21  # noqa: E402
vm21 = dict(ledger=VM21.W.check_ledger(), rerender=VM21.check_rerender(), numbers=VM21.W.check_numbers(),
            citations=VM21.W.check_citations(), sections=VM21.W.check_sections(),
            pdf=VM21.W.check_pdf(pdf=VM21.W.PDF, md=VM21.W.MD, sections=VM21.W.REQUIRED_SECTIONS, n_figures=len(VM21.FIGURES)))
vm21["pdf"]["passed"] = vm21["pdf"]["passed"] and vm21["pdf"].get("images", 0) >= len(VM21.FIGURES)
res["manuscript21_checks"] = {k: v["passed"] for k, v in vm21.items()}
res["manuscript21_pdf_pages"] = vm21["pdf"].get("pages")
ms21_pages = sorted((REP / "logs/manuscript21_pages").glob("page-*.png"))
res["manuscript21_page_renders"] = len(ms21_pages)
res["manuscript21_blank_pages"] = [p.name for p in ms21_pages if (np.asarray(Image.open(p).convert("L")) < 200).mean() < 0.002]
led21 = json.loads((REP / "manuscript_values_round21.json").read_text())
texts21 = {e["text"] for e in led21["values"] if re.search(r"\d", e["text"]) and e["spec"] != "s"}
res["manuscript21_values_checked_in_report_pdf"] = len(texts21)
res["manuscript21_values_missing_in_report_pdf"] = sorted(x for x in texts21 if norm(x) not in T)[:50]
# Round 22 (D159): the Supplementary Material (docs/manuscript_round22.md, scripts/verify_manuscript_round22.py) and the
# journal main manuscript (scripts/verify_manuscript_journal_main.py: also every value a verified Supplementary value,
# reference rows identical, 18-22 pages); page renders of both; the journal's logged values in the report PDF (it is the
# embedded manuscript); the revision-24 copy (revision 23 kept)
import verify_manuscript_round22 as VM22  # noqa: E402
vm22 = dict(ledger=VM22.W.check_ledger(), rerender=VM22.check_rerender(), numbers=VM22.W.check_numbers(),
            citations=VM22.W.check_citations(), sections=VM22.W.check_sections(),
            pdf=VM22.W.check_pdf(pdf=VM22.W.PDF, md=VM22.W.MD, sections=VM22.W.REQUIRED_SECTIONS, n_figures=len(VM22.FIGURES)))
vm22["pdf"]["passed"] = vm22["pdf"]["passed"] and vm22["pdf"].get("images", 0) >= len(VM22.FIGURES)
res["manuscript22_checks"] = {k: v["passed"] for k, v in vm22.items()}
res["manuscript22_pdf_pages"] = vm22["pdf"].get("pages")
ms22_pages = sorted((REP / "logs/manuscript22_pages").glob("page-*.png"))
res["manuscript22_page_renders"] = len(ms22_pages)
res["manuscript22_blank_pages"] = [p.name for p in ms22_pages if (np.asarray(Image.open(p).convert("L")) < 200).mean() < 0.002]
import verify_manuscript_journal_main as VMJ  # noqa: E402
vmj = dict(ledger=VMJ.W.check_ledger(), rerender=VMJ.check_rerender(), numbers=VMJ.W.check_numbers(),
           citations=VMJ.W.check_citations(), sections=VMJ.W.check_sections(), supplement=VMJ.check_supplement(),
           references=VMJ.check_references(),
           pdf=VMJ.W.check_pdf(pdf=VMJ.W.PDF, md=VMJ.W.MD, sections=VMJ.W.REQUIRED_SECTIONS, n_figures=len(VMJ.FIGURES)))
vmj["pdf"]["passed"] = vmj["pdf"]["passed"] and vmj["pdf"].get("images", 0) >= len(VMJ.FIGURES) and \
    VMJ.PAGES[0] <= vmj["pdf"].get("pages", 0) <= VMJ.PAGES[1]
res["journal_checks"] = {k: v["passed"] for k, v in vmj.items()}
res["journal_pdf_pages"] = vmj["pdf"].get("pages")
msj_pages = sorted((REP / "logs/manuscript_journal_pages").glob("page-*.png"))
res["journal_page_renders"] = len(msj_pages)
res["journal_blank_pages"] = [p.name for p in msj_pages if (np.asarray(Image.open(p).convert("L")) < 200).mean() < 0.002]
ledj = json.loads((REP / "manuscript_values_journal_main.json").read_text())
textsj = {e["text"] for e in ledj["values"] if re.search(r"\d", e["text"]) and e["spec"] != "s"}
res["journal_values_checked_in_report_pdf"] = len(textsj)
res["journal_values_missing_in_report_pdf"] = sorted(x for x in textsj if norm(x) not in T)[:50]
rev24 = REP / "final_research_prototype_report_rev24.pdf"          # round 22: the current revision's copy
res["rev24_copy_identical"] = rev24.exists() and hashlib.sha256(rev24.read_bytes()).hexdigest() == \
    hashlib.sha256(pdf.read_bytes()).hexdigest()
res["rev23_kept"] = (REP / "final_research_prototype_report_rev23.pdf").exists()
res["rev22_kept"] = (REP / "final_research_prototype_report_rev22.pdf").exists()
res["rev21_kept"] = (REP / "final_research_prototype_report_rev21.pdf").exists()
res["rev20_kept"] = (REP / "final_research_prototype_report_rev20.pdf").exists()
res["rev19_kept"] = (REP / "final_research_prototype_report_rev19.pdf").exists()
res["rev18_kept"] = (REP / "final_research_prototype_report_rev18.pdf").exists()
res["rev17_kept"] = (REP / "final_research_prototype_report_rev17.pdf").exists()
res["rev16_kept"] = (REP / "final_research_prototype_report_rev16.pdf").exists()
res["report_blank_pages_by_text"] = VM.check_pdf(pdf, REP / "final_research_prototype_report.md", [], 0)["blank_pages"]

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
      and all(chk) and not not_run and not res["commands_missing_log_file"] and res["experiment_csv_rows_equal_jsonl"]
      and res["pdf_has_manuscript_sections"] and all(res["manuscript_checks"].values())
      and len(ms_pages) == res["manuscript_pdf_pages"] and not res["manuscript_blank_pages"]
      and not res["manuscript_values_missing_in_report_pdf"] and res["rev24_copy_identical"] and res["rev23_kept"]
      and res["rev22_kept"] and all(res["manuscript22_checks"].values())
      and len(ms22_pages) == res["manuscript22_pdf_pages"] and not res["manuscript22_blank_pages"]
      and all(res["journal_checks"].values()) and len(msj_pages) == res["journal_pdf_pages"]
      and not res["journal_blank_pages"] and not res["journal_values_missing_in_report_pdf"]
      and res["rev21_kept"] and all(res["manuscript21_checks"].values())
      and len(ms21_pages) == res["manuscript21_pdf_pages"] and not res["manuscript21_blank_pages"]
      and not res["manuscript21_values_missing_in_report_pdf"]
      and all(res["manuscript20_checks"].values())
      and len(ms20_pages) == res["manuscript20_pdf_pages"] and not res["manuscript20_blank_pages"]
      and not res["manuscript20_values_missing_in_report_pdf"] and res["rev20_kept"]
      and res["rev19_kept"] and all(res["manuscript19_checks"].values())
      and len(ms19_pages) == res["manuscript19_pdf_pages"] and not res["manuscript19_blank_pages"]
      and not res["manuscript19_values_missing_in_report_pdf"]
      and res["rev18_kept"] and all(res["manuscript18_checks"].values())
      and len(ms18_pages) == res["manuscript18_pdf_pages"] and not res["manuscript18_blank_pages"]
      and not res["manuscript18_values_missing_in_report_pdf"]
      and res["rev17_kept"] and all(res["manuscript17_checks"].values())
      and len(ms17_pages) == res["manuscript17_pdf_pages"] and not res["manuscript17_blank_pages"]
      and not res["manuscript17_values_missing_in_report_pdf"]
      and res["rev16_kept"] and all(res["manuscript16_checks"].values())
      and len(ms16_pages) == res["manuscript16_pdf_pages"] and not res["manuscript16_blank_pages"]
      and not res["manuscript16_values_missing_in_report_pdf"]
      and not res["report_blank_pages_by_text"])
res["ALL_CHECKS_PASS"] = bool(ok)
(REP / "logs/deliverable_verification.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
