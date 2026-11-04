"""Round 21 (D153-D155): the updated standalone manuscript's numbers come from the saved records, and the verifier
catches a deviation (the Round-14 checks applied to docs/manuscript_round21.md through scripts/verify_manuscript_round21.py).

R21-5  every value logged by the builder re-reads and re-formats identically from its source (independent reader)
R21-6  rendering the template again from the records reproduces docs/manuscript_round21.md and its ledger exactly
R21-7  every number in the manuscript is a logged value or a registered literal in a registered context
R21-8  negative controls: a hand-typed number in the template and a changed logged Round-21 value are both detected
R21-9  citations, required sections (including the Phase-2 section) and figure files; the compiled PDF has no blank
       page and embeds every figure; the Round-14, -16, -17, -18, -19 and -20 manuscripts and their ledgers are untouched by the Round-21 builder
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_manuscript_round21 as VM21  # noqa: E402

W = VM21.W
pytestmark = [pytest.mark.stage("R21"), pytest.mark.kind("DET")]


@pytest.mark.criterion("R21-5")
def test_R21_5_logged_values_match_records():
    r = W.check_ledger()
    assert r["passed"], r["mismatches"]
    assert r["n_values"] > 1000


@pytest.mark.criterion("R21-6")
def test_R21_6_manuscript_rerenders_identically():
    r = VM21.check_rerender()
    assert r["passed"], r


@pytest.mark.criterion("R21-7")
def test_R21_7_every_number_is_accounted_for():
    r = W.check_numbers()
    assert r["passed"], {k: v for k, v in r.items() if k != "passed"}


@pytest.mark.criterion("R21-8")
def test_R21_8_verifier_detects_hand_typed_numbers_and_changed_values(tmp_path, monkeypatch):
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
    i = next(k for k, e in enumerate(led["values"]) if e["src"].endswith("H_R21a/part2_mean_vs_waft"))
    led["values"][i]["text"] = "+0.0099"
    bad = tmp_path / "ledger.json"
    bad.write_text(json.dumps(led, ensure_ascii=False))
    monkeypatch.setattr(W, "LEDGER", bad)
    r = W.check_ledger()
    assert not r["passed"] and any(e["src"].endswith("H_R21a/part2_mean_vs_waft") for e in r["mismatches"])


@pytest.mark.criterion("R21-9")
def test_R21_9_citations_sections_figures_pdf_and_earlier_manuscripts_untouched():
    assert W.check_citations()["passed"]
    s = W.check_sections()
    assert s["passed"], s
    p = W.check_pdf(pdf=W.PDF, md=W.MD, sections=W.REQUIRED_SECTIONS, n_figures=len(VM21.FIGURES))
    assert p["passed"] and not p["blank_pages"] and p["images"] >= len(VM21.FIGURES), p
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
