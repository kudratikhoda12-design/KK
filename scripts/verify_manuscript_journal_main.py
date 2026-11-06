"""Round 22 (D159): verification of the journal main manuscript (docs/manuscript_journal_main.md) and its PDF.

The Round-14 verifier's checks, through scripts/verify_manuscript_round22.py loaded as a separate module instance and
pointed at the journal files (ledger, rerender, numbers, citations, sections, PDF), plus three checks of its own:

  supplement   every logged (source, format, text) triple of the main manuscript occurs in the verified ledger of the
               Supplementary Material (reports/manuscript_values_round22.json), so every number in the paper is a
               number of the Supplementary Material
  references   every row of docs/references_journal_main.md is identical to its row in docs/references_round22.md
  length       (with --pdf) the compiled PDF has 18 to 22 pages, the length the owner asked for

The literal registry is the subset of the Round-22 registry whose contexts occur in the journal template, plus the
journal's own entries. Result: reports/manuscript_verification_journal_main.json.
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("verify_manuscript_journal_v22", ROOT / "scripts/verify_manuscript_round22.py")
V22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V22)
W = V22.W

SUPP_LEDGER = ROOT / "reports/manuscript_values_round22.json"
SUPP_REFS = ROOT / "docs/references_round22.md"
PAGES = (18, 22)
W.MD = ROOT / "docs/manuscript_journal_main.md"
W.TPL = ROOT / "docs/manuscript_journal_main_template.md"
W.LEDGER = ROOT / "reports/manuscript_values_journal_main.json"
W.PDF = ROOT / "reports/manuscript_journal_main.pdf"
W.OUT = ROOT / "reports/manuscript_verification_journal_main.json"
W.REFS = ROOT / "docs/references_journal_main.md"
W.REQUIRED_SECTIONS = ["Abstract", "Keywords", "1. Introduction", "2. Methods", "3. Simulation results",
                       "4. Real-data results", "5. Discussion", "5.2 Where simpler landmark models suffice",
                       "5.3 Relation to prior work", "5.4 Limitations", "5.5 Conclusions", "Declarations", "References", "Appendix. Number provenance"]
FIGURES = ["ms14_r13_effects_pbc_E2_b2s_cox_a.png"]
W.FIGURES = FIGURES
D, CONV = W.D, W.CONV
_tpl = re.sub(r"\s+", " ", W._strip(W.TPL.read_text())) if W.TPL.exists() else ""
W.LITERALS = [e for e in W.LITERALS if e[1] in _tpl]      # the inherited entries this template uses
W.LITERALS += [
    # journal-only contexts (design constants of D156 and conventions)
]
W.BUILDER_LITERALS = [e for e in W.BUILDER_LITERALS if e[1] in re.sub(r"\s+", " ", W.MD.read_text())] if W.MD.exists() else []


def check_rerender():
    sys.path.insert(0, str(ROOT / "scripts"))
    import manuscript_journal_main as M
    M.B._cache.clear()
    text = M.render(W.TPL.read_text())
    led = json.loads(W.LEDGER.read_text())
    same_md = text == W.MD.read_text()
    same_led = [dict(src=e["src"], spec=e["spec"], text=e["text"]) for e in M.LEDGER] == \
               [dict(src=e["src"], spec=e["spec"], text=e["text"]) for e in led["values"]]
    return dict(passed=same_md and same_led, manuscript_identical=same_md, ledger_identical=same_led)


def check_supplement():
    supp = {(e["src"], e["spec"], e["text"]) for e in json.loads(SUPP_LEDGER.read_text())["values"]}
    vals = json.loads(W.LEDGER.read_text())["values"]
    missing = [dict(src=e["src"], spec=e["spec"], text=e["text"]) for e in vals if (e["src"], e["spec"], e["text"]) not in supp]
    return dict(passed=not missing and len(vals) > 0, n_values=len(vals), not_in_supplement=missing[:20])


def _rows(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(c) > 1 and c[0] not in ("Key", "") and not set(c[0]) <= set("-"):
            out[c[0]] = line.strip()
    return out


def check_references():
    j, s = _rows(W.REFS), _rows(SUPP_REFS)
    bad = sorted(k for k, v in j.items() if s.get(k) != v)
    return dict(passed=bool(j) and not bad, n_rows=len(j), differing_or_unknown=bad)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args(argv)
    res = dict(ledger=W.check_ledger(), rerender=check_rerender(), numbers=W.check_numbers(), citations=W.check_citations(),
               sections=W.check_sections(), supplement=check_supplement(), references=check_references())
    if a.pdf:
        p = W.check_pdf(pdf=W.PDF, md=W.MD, sections=W.REQUIRED_SECTIONS, n_figures=len(FIGURES))
        p["page_range"] = list(PAGES)
        p["passed"] = p["passed"] and p.get("images", 0) >= len(FIGURES) and PAGES[0] <= p.get("pages", 0) <= PAGES[1]
        res["pdf"] = p
    res["ALL_PASS"] = all(v["passed"] for v in res.values() if isinstance(v, dict))
    W.OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0 if res["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
