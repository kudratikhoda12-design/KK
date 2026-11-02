"""Round 21 (D155): verification of docs/manuscript_round21.md and its PDF.

The checks are those of scripts/verify_manuscript_round20.py (itself the Round-19, -18, -17, -16 and -14 checks), loaded here as a
separate module instance and pointed at the Round-21 files, so the earlier verifications are unchanged. The literal
registry is the Round-20 one plus the design constants the Round-21 text types (each found in its decision entry).
Result: reports/manuscript_verification_round21.json.
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("verify_manuscript_r21_v20", ROOT / "scripts/verify_manuscript_round20.py")
V20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V20)
W = V20.W

W.MD = ROOT / "docs/manuscript_round21.md"
W.TPL = ROOT / "docs/manuscript_round21_template.md"
W.LEDGER = ROOT / "reports/manuscript_values_round21.json"
W.PDF = ROOT / "reports/manuscript_round21.pdf"
W.OUT = ROOT / "reports/manuscript_verification_round21.json"
W.REQUIRED_SECTIONS = [s.replace("(Rounds 15–20)", "(Rounds 15–21)") for s in W.REQUIRED_SECTIONS] + \
    ["5.9 Round-20 methods on real data (H-R21a, H-R21b, H-R21c)"]
D, CONV = W.D, W.CONV
W.LITERALS += [
    # Round-21 design constants and conventions (D153), filled in with the Round-21 text
]
W.BUILDER_LITERALS += [
]
FIGURES = V20.FIGURES


def check_rerender():
    sys.path.insert(0, str(ROOT / "scripts"))
    import manuscript_round21 as M
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
