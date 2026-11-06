"""Round 22 (D156-D159): the cumulative manuscript (the Supplementary Material) takes its numbers from the saved records,
and the verifier catches a deviation (the Round-14 checks applied to docs/manuscript_round22.md through
scripts/verify_manuscript_round22.py).

R22-5  every value logged by the builder re-reads and re-formats identically from its source (independent reader)
R22-6  rendering the template again from the records reproduces docs/manuscript_round22.md and its ledger exactly
R22-7  every number in the manuscript is a logged value or a registered literal in a registered context
R22-8  negative controls: a hand-typed number in the template and a changed logged Round-22 value are both detected
R22-9  citations, required sections (including the Phase-2 section) and figure files; the compiled PDF has no blank
       page and embeds every figure; the Round-14 and Round-16 to Round-21 manuscripts and their ledgers are untouched by the Round-22 builder
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_manuscript_round22 as VM22  # noqa: E402

W = VM22.W
pytestmark = [pytest.mark.stage("R22"), pytest.mark.kind("DET")]


@pytest.mark.criterion("R22-5")
def test_R22_5_logged_values_match_records():
    r = W.check_ledger()
    assert r["passed"], r["mismatches"]
    assert r["n_values"] > 1000


@pytest.mark.criterion("R22-6")
def test_R22_6_manuscript_rerenders_identically():
    r = VM22.check_rerender()
    assert r["passed"], r


@pytest.mark.criterion("R22-7")
def test_R22_7_every_number_is_accounted_for():
    r = W.check_numbers()
    assert r["passed"], {k: v for k, v in r.items() if k != "passed"}


@pytest.mark.criterion("R22-8")
def test_R22_8_verifier_detects_hand_typed_numbers_and_changed_values(tmp_path, monkeypatch):
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
    i = next(k for k, e in enumerate(led["values"]) if e["src"].endswith("ck6d.json/shapes/phi1/rise_id"))
    led["values"][i]["text"] = "+0.0099"
    bad = tmp_path / "ledger.json"
    bad.write_text(json.dumps(led, ensure_ascii=False))
    monkeypatch.setattr(W, "LEDGER", bad)
    r = W.check_ledger()
    assert not r["passed"] and any(e["src"].endswith("ck6d.json/shapes/phi1/rise_id") for e in r["mismatches"])


@pytest.mark.criterion("R22-9")
def test_R22_9_citations_sections_figures_pdf_and_earlier_manuscripts_untouched():
    assert W.check_citations()["passed"]
    s = W.check_sections()
    assert s["passed"], s
    p = W.check_pdf(pdf=W.PDF, md=W.MD, sections=W.REQUIRED_SECTIONS, n_figures=len(VM22.FIGURES))
    assert p["passed"] and not p["blank_pages"] and p["images"] >= len(VM22.FIGURES), p
    assert shutil.which("pdftotext") is not None
    import verify_manuscript as V16
    import verify_manuscript_round14 as VM14
    assert VM14.check_rerender()["passed"] and VM14.check_ledger()["passed"]
    assert V16.check_rerender()["passed"] and V16.W.check_ledger()["passed"]
    assert json.loads((ROOT / "reports/manuscript_values_round14.json").read_text())["output"] == "docs/manuscript_round14.md"
    assert json.loads((ROOT / "reports/manuscript_values_round16.json").read_text())["output"] == "docs/manuscript_round16.md"
    import verify_manuscript_round17 as V17
    assert V17.check_rerender()["passed"] and V17.W.check_ledger()["passed"]
    assert json.loads((ROOT / "reports/manuscript_values_round17.json").read_text())["output"] == "docs/manuscript_round17.md"
    import verify_manuscript_round18 as V18
    assert V18.check_rerender()["passed"] and V18.W.check_ledger()["passed"]
    assert json.loads((ROOT / "reports/manuscript_values_round18.json").read_text())["output"] == "docs/manuscript_round18.md"
    import verify_manuscript_round19 as V19
    assert V19.check_rerender()["passed"] and V19.W.check_ledger()["passed"]
    assert json.loads((ROOT / "reports/manuscript_values_round19.json").read_text())["output"] == "docs/manuscript_round19.md"
    import verify_manuscript_round20 as V20
    assert V20.check_rerender()["passed"] and V20.W.check_ledger()["passed"]
    assert json.loads((ROOT / "reports/manuscript_values_round20.json").read_text())["output"] == "docs/manuscript_round20.md"
    import verify_manuscript_round21 as V21
    assert V21.check_rerender()["passed"] and V21.W.check_ledger()["passed"]
    assert json.loads((ROOT / "reports/manuscript_values_round21.json").read_text())["output"] == "docs/manuscript_round21.md"
