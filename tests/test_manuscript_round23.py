"""Round 23 (D160-D162): the cumulative manuscript (the Supplementary Material) takes its numbers from the saved records,
and the verifier catches a deviation (the Round-14 checks applied to docs/manuscript_round23.md through
scripts/verify_manuscript_round23.py).

R23-6   every value logged by the builder re-reads and re-formats identically from its source (independent reader)
R23-7   rendering the template again from the records reproduces docs/manuscript_round23.md and its ledger exactly
R23-8   every number in the manuscript is a logged value or a registered literal in a registered context
R23-9   negative controls: a hand-typed number in the template and a changed logged Round-23 value are both detected
R23-10  citations, required sections (including Section 5.13) and figure files; the compiled PDF has no blank page and
        embeds every figure; the Round-22 manuscript and its ledger are untouched by the Round-23 builder
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_manuscript_round23 as VM23  # noqa: E402

W = VM23.W
pytestmark = [pytest.mark.stage("R23"), pytest.mark.kind("DET")]


@pytest.mark.criterion("R23-6")
def test_R23_6_logged_values_match_records():
    r = W.check_ledger()
    assert r["passed"], r["mismatches"]
    assert r["n_values"] > 1000


@pytest.mark.criterion("R23-7")
def test_R23_7_manuscript_rerenders_identically():
    r = VM23.check_rerender()
    assert r["passed"], r


@pytest.mark.criterion("R23-8")
def test_R23_8_every_number_is_accounted_for():
    r = W.check_numbers()
    assert r["passed"], {k: v for k, v in r.items() if k != "passed"}


@pytest.mark.criterion("R23-9")
def test_R23_9_verifier_detects_hand_typed_numbers_and_changed_values(tmp_path, monkeypatch):
    tpl = tmp_path / "template.md"
    md = tmp_path / "manuscript.md"
    tpl.write_text(W.TPL.read_text().replace("## 7. Conclusions\n", "## 7. Conclusions\n\nThe effect was 0.4321.\n", 1))
    md.write_text(W.MD.read_text().replace("## 7. Conclusions\n", "## 7. Conclusions\n\nThe effect was 0.4321.\n", 1))
    monkeypatch.setattr(W, "TPL", tpl)
    monkeypatch.setattr(W, "MD", md)
    r = W.check_numbers()
    assert not r["passed"] and "0.4321" in r["unregistered_in_manuscript"]
    assert any(x.startswith("0.4321:") for x in r["template_literals_without_registered_context"])
    monkeypatch.undo()

    led = json.loads(W.LEDGER.read_text())
    i = next(k for k, e in enumerate(led["values"]) if e["src"].endswith("round23/analysis_summary.json/H_R23b/mean_diff"))
    led["values"][i]["text"] = "+0.0099"
    bad = tmp_path / "ledger.json"
    bad.write_text(json.dumps(led, ensure_ascii=False))
    monkeypatch.setattr(W, "LEDGER", bad)
    r = W.check_ledger()
    assert not r["passed"] and any(e["src"].endswith("round23/analysis_summary.json/H_R23b/mean_diff") for e in r["mismatches"])


@pytest.mark.criterion("R23-10")
def test_R23_10_citations_sections_figures_pdf_and_earlier_manuscripts_untouched():
    assert W.check_citations()["passed"]
    s = W.check_sections()
    assert s["passed"], s
    p = W.check_pdf(pdf=W.PDF, md=W.MD, sections=W.REQUIRED_SECTIONS, n_figures=len(VM23.FIGURES))
    assert p["passed"] and not p["blank_pages"] and p["images"] >= len(VM23.FIGURES), p
    assert shutil.which("pdftotext") is not None
    import verify_manuscript_round22 as V22
    assert V22.check_rerender()["passed"] and V22.W.check_ledger()["passed"]
    assert json.loads((ROOT / "reports/manuscript_values_round22.json").read_text())["output"] == "docs/manuscript_round22.md"
