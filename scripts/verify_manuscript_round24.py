"""Round 24 (D165): verification of docs/manuscript_round24.md (the Supplementary Material) and its PDF.

The checks are those of scripts/verify_manuscript_round23.py (itself the Round-22 .. Round-14 checks), loaded here as a
separate module instance and pointed at the Round-24 files, so the earlier verifications are unchanged. The reference
table is docs/references_round22.md (unchanged). The literal registry is the Round-23 one plus the design constants the
Round-24 text types (each found in docs/round24_preregistration.md, D163, or the Round-24 results, D164).
Result: reports/manuscript_verification_round24.json.
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("verify_manuscript_r24_v23", ROOT / "scripts/verify_manuscript_round23.py")
V23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V23)
W = V23.W

W.MD = ROOT / "docs/manuscript_round24.md"
W.TPL = ROOT / "docs/manuscript_round24_template.md"
W.LEDGER = ROOT / "reports/manuscript_values_round24.json"
W.PDF = ROOT / "reports/manuscript_round24.pdf"
W.OUT = ROOT / "reports/manuscript_verification_round24.json"
W.REFS = ROOT / "docs/references_round22.md"
W.REQUIRED_SECTIONS = [s.replace("(Rounds 15–23)", "(Rounds 15–24)") for s in W.REQUIRED_SECTIONS] + \
    ["5.14 A Bayesian joint model as comparator"]
D, CONV = W.D, W.CONV
_tpl = re.sub(r"\s+", " ", W._strip(W.TPL.read_text()))
W.LITERALS = [e for e in W.LITERALS if e[1] in _tpl]      # inherited entries whose context this template still uses
P24 = "docs/round24_preregistration.md"
RES24 = "docs/round24_results.md"
W.LITERALS += [
    # Round-24 design constants (D163) and the convergence rule reported in the results (D164)
    ("6,500", "three chains of 6,500 iterations with 1,500 burn-in", P24, "3 chains, 6,500 iterations, 1,500 burn-in"),
    ("1,500", "three chains of 6,500 iterations with 1,500 burn-in", P24, "3 chains, 6,500 iterations, 1,500 burn-in"),
    ("0.005", "above the registered +0.005, so non-inferiority was shown only on renal", P24, "≤ +0.005 (the Round-20/21/23 margin)"),
    ("1.1", "association parameters was above 1.1 on", RES24, "Partitions with Rhat > 1.1"),
    ("1.1", "(max Rhat above 1.1)", RES24, "Partitions with Rhat > 1.1"),
    ("0", "partitions and the interval excluded 0 (", CONV, None),
    ("0", "Landmark Cox minus the joint model also had every interval including 0 outside renal", CONV, None),
    ("0", "every interval included 0, but", CONV, None),
    ("0", "Harrell C interval of NP-KAN-AFT minus the joint model excluded 0", CONV, None),
    ("0", "with an interval that included 0 (", CONV, None),
]
W.BUILDER_LITERALS += [
    # Round-24 tables typed by the builder (D163 margin; D164 convergence rule)
    ("0.005", "conditional IBS: upper bound ≤ +0.005", P24, "≤ +0.005 (the Round-20/21/23 margin)"),
    ("0", "mean < 0 and Nadeau–Bengio upper bound < 0", CONV, None),
    ("1.1", "Partitions with Rhat above 1.1", RES24, "Partitions with Rhat > 1.1"),
    ("95", "joint model (NB 95% CI)", CONV, None),
]
FIGURES = V23.FIGURES


def check_rerender():
    sys.path.insert(0, str(ROOT / "scripts"))
    import manuscript_round24 as M
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
