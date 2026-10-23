"""Round 14 (D131): the manuscript's numbers come from the saved records, and the verifier catches a deviation.

R14-1  every value logged by the builder re-reads and re-formats identically from its source (independent reader)
R14-2  rendering the template again from the records reproduces docs/manuscript_round14.md and the ledger exactly
R14-3  every number in the manuscript is a logged value or a registered literal in a registered context
R14-4  negative controls: a hand-typed number in the template and a changed logged value are both detected
R14-5  citations, required sections and figure files; the compiled PDF has no blank page and embeds every figure
R14-6  the descriptive cohort record agrees with the counts registered in D129
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_manuscript_round14 as VM  # noqa: E402

pytestmark = [pytest.mark.stage("R14"), pytest.mark.kind("DET")]


@pytest.mark.criterion("R14-1")
def test_R14_1_logged_values_match_records():
    r = VM.check_ledger()
    assert r["passed"], r["mismatches"]
    assert r["n_values"] > 1000


@pytest.mark.criterion("R14-2")
def test_R14_2_manuscript_rerenders_identically():
    r = VM.check_rerender()
    assert r["passed"], r


@pytest.mark.criterion("R14-3")
def test_R14_3_every_number_is_accounted_for():
    r = VM.check_numbers()
    assert r["passed"], {k: v for k, v in r.items() if k != "passed"}


@pytest.mark.criterion("R14-4")
def test_R14_4_verifier_detects_hand_typed_numbers_and_changed_values(tmp_path, monkeypatch):
    tpl = tmp_path / "template.md"
    md = tmp_path / "manuscript.md"
    tpl.write_text(VM.TPL.read_text().replace("## 6. Conclusions\n", "## 6. Conclusions\n\nThe effect was 0.4321.\n", 1))
    md.write_text(VM.MD.read_text().replace("## 6. Conclusions\n", "## 6. Conclusions\n\nThe effect was 0.4321.\n", 1))
    monkeypatch.setattr(VM, "TPL", tpl)
    monkeypatch.setattr(VM, "MD", md)
    r = VM.check_numbers()
    assert not r["passed"] and "0.4321" in r["unregistered_in_manuscript"]
    assert any(x.startswith("0.4321:") for x in r["template_literals_without_registered_context"])
    monkeypatch.undo()

    led = json.loads(VM.LEDGER.read_text())
    i = next(k for k, e in enumerate(led["values"]) if e["src"].endswith("H_R12/mean_diff"))
    led["values"][i]["text"] = "+0.0099"
    bad = tmp_path / "ledger.json"
    bad.write_text(json.dumps(led, ensure_ascii=False))
    monkeypatch.setattr(VM, "LEDGER", bad)
    r = VM.check_ledger()
    assert not r["passed"] and any(e["src"].endswith("H_R12/mean_diff") for e in r["mismatches"])


@pytest.mark.criterion("R14-5")
def test_R14_5_citations_sections_figures_and_pdf():
    assert VM.check_citations()["passed"]
    s = VM.check_sections()
    assert s["passed"], s
    p = VM.check_pdf()
    assert p["passed"] and not p["blank_pages"] and p["images"] >= len(VM.FIGURES), p
    assert shutil.which("pdftotext") is not None


@pytest.mark.criterion("R14-6")
def test_R14_6_cohort_record_matches_D129():
    c = json.loads((ROOT / "reports/round14/cohort_summary.json").read_text())
    d129 = next(l for l in (ROOT / "docs/decisions.md").read_text().splitlines() if l.startswith("| D129 |"))
    assert "278 patients (E2 135 events primary; E1 107 sensitivity)" in d129
    assert "`prothro` 1 y, 332 patients, 183 deaths" in d129 and "`aids` 6 months, 403 patients, 126 deaths" in d129
    got = {k: (c[k]["n_cohort"], c[k]["n_events"]) for k in ("pbc_E2", "pbc_E1", "prothro", "aids")}
    assert got == {"pbc_E2": (278, 135), "pbc_E1": (278, 107), "prothro": (332, 183), "aids": (403, 126)}
    assert all(c[k]["visits_median"] == 3 for k in got)                                 # "median 3 visits" (D129)
