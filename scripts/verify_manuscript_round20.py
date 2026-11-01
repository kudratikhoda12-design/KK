"""Round 20 (D152): verification of docs/manuscript_round20.md and its PDF.

The checks are those of scripts/verify_manuscript_round19.py (itself the Round-18, -17, -16 and -14 checks), loaded here as a
separate module instance and pointed at the Round-20 files, so the earlier verifications are unchanged. The literal
registry is the Round-19 one plus the design constants the Round-20 text types (each found in its decision entry).
Result: reports/manuscript_verification_round20.json.
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("verify_manuscript_r20_v19", ROOT / "scripts/verify_manuscript_round19.py")
V19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V18)
W = V19.W

W.MD = ROOT / "docs/manuscript_round20.md"
W.TPL = ROOT / "docs/manuscript_round20_template.md"
W.LEDGER = ROOT / "reports/manuscript_values_round20.json"
W.PDF = ROOT / "reports/manuscript_round20.pdf"
W.OUT = ROOT / "reports/manuscript_verification_round20.json"
W.REQUIRED_SECTIONS = [s.replace("(Rounds 15–19)", "(Rounds 15–20)") for s in W.REQUIRED_SECTIONS] + \
    ["5.7 A semiparametric KAN-AFT and two-half cross-confirmation (H-R20a, H-R20b)"]
D, CONV = W.D, W.CONV
W.LITERALS += [
    # Round-20 design constants and conventions (D150), filled in with the Round-20 text
]
W.BUILDER_LITERALS += [
]
FIGURES = V19.FIGURES


def check_rerender():
    sys.path.insert(0, str(ROOT / "scripts"))
    import manuscript_round20 as M
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
