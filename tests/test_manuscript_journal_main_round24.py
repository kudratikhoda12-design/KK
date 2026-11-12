"""Round 24 (D165): the Round-24 journal main manuscript takes its numbers from the saved records, every one of them is
also a verified number of the Round-24 Supplementary Material, and the verifier catches a deviation
(scripts/verify_manuscript_journal_main_round24.py).

R24-10  every value logged by the journal builder re-reads and re-formats identically from its source, and rendering
        again reproduces docs/manuscript_journal_main_round24.md and its ledger exactly
R24-11  every number is a logged value or a registered literal; every logged value occurs in the Supplementary
        Material's ledger; every reference row is identical to its Supplementary Material row
R24-12  negative controls: a hand-typed number, a value absent from the Supplementary Material and an altered
        reference row are each detected
R24-13  citations, required sections (including Sections 4.3 and 4.4) and figures; the compiled PDF has no blank page, embeds
        every figure and has 18 to 22 pages; the Supplementary Material and the Round-23 journal manuscript and their
        ledgers are untouched by the journal builder
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_manuscript_journal_main_round24 as VJ  # noqa: E402

W = VJ.W
pytestmark = [pytest.mark.stage("R24"), pytest.mark.kind("DET")]


@pytest.mark.criterion("R24-10")
def test_R24_10_journal_values_match_records_and_rerender():
    r = W.check_ledger()
    assert r["passed"], r["mismatches"]
    assert r["n_values"] > 100
    r = VJ.check_rerender()
    assert r["passed"], r


@pytest.mark.criterion("R24-11")
def test_R24_11_numbers_supplement_and_references():
    r = W.check_numbers()
    assert r["passed"], {k: v for k, v in r.items() if k != "passed"}
    s = VJ.check_supplement()
    assert s["passed"], s
    ref = VJ.check_references()
    assert ref["passed"], ref


@pytest.mark.criterion("R24-12")
def test_R24_12_verifier_detects_typed_numbers_foreign_values_and_altered_references(tmp_path, monkeypatch):
    tpl, md = tmp_path / "template.md", tmp_path / "manuscript.md"
    ins = "### 5.5 Conclusions\n\nThe effect was 0.4321.\n"
    tpl.write_text(W.TPL.read_text().replace("### 5.5 Conclusions\n", ins, 1))
    md.write_text(W.MD.read_text().replace("### 5.5 Conclusions\n", ins, 1))
    monkeypatch.setattr(W, "TPL", tpl)
    monkeypatch.setattr(W, "MD", md)
    r = W.check_numbers()
    assert not r["passed"] and "0.4321" in r["unregistered_in_manuscript"]
    monkeypatch.undo()

    led = json.loads(W.LEDGER.read_text())
    led["values"][0]["text"] = led["values"][0]["text"] + "9"
    bad = tmp_path / "ledger.json"
    bad.write_text(json.dumps(led, ensure_ascii=False))
    monkeypatch.setattr(W, "LEDGER", bad)
    s = VJ.check_supplement()
    assert not s["passed"] and s["not_in_supplement"]
    monkeypatch.undo()

    refs = tmp_path / "refs.md"
    lines = W.REFS.read_text().splitlines()
    k = next(i for i, l in enumerate(lines) if l.startswith("|") and "---" not in l and not l.startswith("| Key"))
    lines[k] = lines[k].replace("|", "| altered", 2).replace("| altered", "|", 1)
    refs.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(W, "REFS", refs)
    ref = VJ.check_references()
    assert not ref["passed"] and ref["differing_or_unknown"]


@pytest.mark.criterion("R24-13")
def test_R24_13_citations_sections_pdf_length_and_supplement_untouched():
    assert W.check_citations()["passed"]
    s = W.check_sections()
    assert s["passed"], s
    p = W.check_pdf(pdf=W.PDF, md=W.MD, sections=W.REQUIRED_SECTIONS, n_figures=len(VJ.FIGURES))
    assert p["passed"] and not p["blank_pages"] and p["images"] >= len(VJ.FIGURES), p
    assert VJ.PAGES[0] <= p["pages"] <= VJ.PAGES[1], p["pages"]
    import verify_manuscript_round24 as VM24
    assert VM24.check_rerender()["passed"] and VM24.W.check_ledger()["passed"]
    assert json.loads((ROOT / "reports/manuscript_values_round24.json").read_text())["output"] == "docs/manuscript_round24.md"
    import verify_manuscript_journal_main_round23 as VJ23
    assert VJ23.check_rerender()["passed"] and VJ23.W.check_ledger()["passed"]
    assert json.loads((ROOT / "reports/manuscript_values_journal_main_round23.json").read_text())["output"] == \
        "docs/manuscript_journal_main_round23.md"
