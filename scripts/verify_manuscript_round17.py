"""Round 17 (D143): verification of docs/manuscript_round17.md and its PDF.

The checks are those of scripts/verify_manuscript.py (Round 16; itself the Round-14 checks), loaded here as a separate
module instance and pointed at the Round-17 files, so the Round-14 and Round-16 verifications are unchanged. The
literal registry is the Round-16 one plus the design constants the Round-17 text types (each found in its decision
entry). Result: reports/manuscript_verification_round17.json.
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("verify_manuscript_r17_v16", ROOT / "scripts/verify_manuscript.py")
V16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V16)
W = V16.W

W.MD = ROOT / "docs/manuscript_round17.md"
W.TPL = ROOT / "docs/manuscript_round17_template.md"
W.LEDGER = ROOT / "reports/manuscript_values_round17.json"
W.PDF = ROOT / "reports/manuscript_round17.pdf"
W.OUT = ROOT / "reports/manuscript_verification_round17.json"
W.REQUIRED_SECTIONS = [s.replace("(Rounds 15–16)", "(Rounds 15–17)") for s in W.REQUIRED_SECTIONS] + \
    ["5.4 Two-way gated anchoring and a log-normal KAN-AFT (H-R17a, H-R17b)"]
D, CONV = W.D, W.CONV
W.LITERALS += [
    # Round-17 design constants and conventions (D141)
    ("0", "difference of exactly 0", CONV, None), ("0", "at last_tol 0 and", D, "last_tol 0"),
    ("20", "seed-0 member, 20 replicates", D, "N = 20 replicates per scenario"),
    ("20", "all 20 distillation refits", D, "20/20 AFT distillation refits"),
    ("95", "and a 95% CI entirely", CONV, None),
]
for _ph in ("S-Last **4100–4119**", "S-Sparse **4200–4219**", "S-Misspec-lognormal **4300–4319**"):
    _ctx = _ph.replace("**", "")
    for _tok in W.tokens(" " + _ctx + " "):
        W.LITERALS.append((_tok, _ctx, D, _ph))
W.SKIP_LABEL = re.compile(W.SKIP_LABEL.pattern.replace("(?:Phases?|", "(?:Phases?|parts?|"))
W.BUILDER_LITERALS += [
    ("0", "(rule: upper < 0)", CONV, None), ("0.005", "(rule: upper < +0.005)", D, "< +0.005"),
    ("0.05", "(rule: < +0.05)", D, "< +0.05"), ("10", "μ = 10, NaN-safe", D, "mu = 10"),
    ("1", "(channels 1, 2, 3)", CONV, None), ("2", "(channels 1, 2, 3)", CONV, None), ("3", "(channels 1, 2, 3)", CONV, None),
    ("3900", "S-Sparse 3900–3919", D, "S-Sparse **3900–3919**"), ("3919", "S-Sparse 3900–3919", D, "S-Sparse **3900–3919**"),
    ("4000", "S-Last 4000–4019", D, "S-Last **4000–4019**"), ("4019", "S-Last 4000–4019", D, "S-Last **4000–4019**"),
    ("961", "S-Misspec 961–969", D, "S-Misspec 961–969"), ("969", "S-Misspec 961–969", D, "S-Misspec 961–969"),
]
FIGURES = V16.FIGURES


def check_rerender():
    sys.path.insert(0, str(ROOT / "scripts"))
    import manuscript_round17 as M
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
