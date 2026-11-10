"""Round 23 (D162): verification of docs/manuscript_round23.md (the Supplementary Material) and its PDF.

The checks are those of scripts/verify_manuscript_round22.py (itself the Round-21 .. Round-14 checks), loaded here as a
separate module instance and pointed at the Round-23 files, so the earlier verifications are unchanged. The reference
table is docs/references_round22.md (unchanged). The literal registry is the Round-22 one plus the design constants the
Round-23 text types (each found in docs/round23_preregistration.md, D160).
Result: reports/manuscript_verification_round23.json.
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("verify_manuscript_r23_v22", ROOT / "scripts/verify_manuscript_round22.py")
V22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V22)
W = V22.W

W.MD = ROOT / "docs/manuscript_round23.md"
W.TPL = ROOT / "docs/manuscript_round23_template.md"
W.LEDGER = ROOT / "reports/manuscript_values_round23.json"
W.PDF = ROOT / "reports/manuscript_round23.pdf"
W.OUT = ROOT / "reports/manuscript_verification_round23.json"
W.REFS = ROOT / "docs/references_round22.md"
W.REQUIRED_SECTIONS = [s.replace("(Rounds 15–22)", "(Rounds 15–23)") for s in W.REQUIRED_SECTIONS] + \
    ["5.13 Where the functional model helps"]
D, CONV = W.D, W.CONV
_tpl = re.sub(r"\s+", " ", W._strip(W.TPL.read_text()))
W.LITERALS = [e for e in W.LITERALS if e[1] in _tpl]      # inherited entries whose context this template still uses
P23 = "docs/round23_preregistration.md"
GRID = "w ∈ {0, 0.25, 0.5, 0.75, 1}"
W.LITERALS += [
    # Round-23 design constants (D160)
    ("0", "w in {0, 0.25, 0.5, 0.75, 1} chosen", P23, GRID),
    ("0.25", "w in {0, 0.25, 0.5, 0.75, 1} chosen", P23, GRID),
    ("0.5", "w in {0, 0.25, 0.5, 0.75, 1} chosen", P23, GRID),
    ("0.75", "w in {0, 0.25, 0.5, 0.75, 1} chosen", P23, GRID),
    ("1", "w in {0, 0.25, 0.5, 0.75, 1} chosen", P23, GRID),
    ("1", "+ (1 − w) S_NP", P23, "(1 − w)·S_npaft"),
    ("0", "w = 0 (NP-KAN-AFT alone) or 0.25", P23, GRID),
    ("0.25", "w = 0 (NP-KAN-AFT alone) or 0.25", P23, GRID),
    ("0.005", "above the registered +0.005", P23, "≤ +0.005 (the Round-20/21 anchoring margin)"),
    ("3e-3", "learning rate 3e-3, weight decay 1e-4", P23, "lr 3e-3, wd 1e-4"),
    ("1e-4", "learning rate 3e-3, weight decay 1e-4", P23, "lr 3e-3, wd 1e-4"),
    ("20", "every 20 cycles to bound", P23, "every **20** cycles"),
    ("20", "windows every 20 cycles", P23, "every **20** cycles"),
    ("30", "(30-cycle windows", P23, "30-cycle windows"),
    ("20", "(partitions out of 20)", P23, "J = 20 (clinical)"),
    ("5", "(k = 5 to 24)", D, "new partitions 5–24"),
    ("24", "(k = 5 to 24)", D, "new partitions 5–24"),
    ("0", "upper bounds below 0; H-R23a", CONV, None),
    ("95", "Nadeau–Bengio 95% intervals", CONV, None),
]
W.BUILDER_LITERALS += [
    # blend-weight column header typed by the builder (Table 65; D160 grid)
    ("0.75", "w = 0.75", P23, GRID),
]
FIGURES = V22.FIGURES


def check_rerender():
    sys.path.insert(0, str(ROOT / "scripts"))
    import manuscript_round23 as M
    M.B._cache.clear()
    text = M.render(W.TPL.read_text())
    led = json.loads(W.LEDGER.read_text())
    same_md = text == W.MD.read_text()
    same_led = [dict(src=e["src"], spec=e["spec"], text=e["text"]) for e in M.LEDGER] == \
               [dict(src=e["src"], spec=e["spec"], text=e["text"]) for e in led["values"]]
    return dict(passed=same_md and same_led, manuscript_identical=same_md, ledger_identical=same_led)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args(argv)
    res = dict(ledger=W.check_ledger(), rerender=check_rerender(), numbers=W.check_numbers(),
               citations=W.check_citations(), sections=W.check_sections())
    if a.pdf:
        p = W.check_pdf(pdf=W.PDF, md=W.MD, sections=W.REQUIRED_SECTIONS, n_figures=len(FIGURES))
        p["passed"] = p["passed"] and p.get("images", 0) >= len(FIGURES)
        res["pdf"] = p
    res["ALL_PASS"] = all(v["passed"] for v in res.values() if isinstance(v, dict))
    W.OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0 if res["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
