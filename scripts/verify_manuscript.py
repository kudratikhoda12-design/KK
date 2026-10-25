"""Round 16 (D138/D139): verification of the standalone manuscript docs/manuscript_round16.md and its PDF.

The checks are those of scripts/verify_manuscript_round14.py (ledger re-resolution with an independent reader,
byte-identical re-rendering, number audit with a context-specific literal registry, citations, sections and figures,
compiled PDF), loaded here as a separate module instance and pointed at the Round-16 files, so the Round-14
verification (and its record) is unchanged. The literal registry is the Round-14 one plus the design constants that
the Round-15/16 text types (each found in its cited decision entry). Result: reports/manuscript_verification_round16.json.
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("verify_manuscript_r16_base", ROOT / "scripts/verify_manuscript_round14.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)

W.MD = ROOT / "docs/manuscript_round16.md"
W.TPL = ROOT / "docs/manuscript_round16_template.md"
W.LEDGER = ROOT / "reports/manuscript_values_round16.json"
W.PDF = ROOT / "reports/manuscript_round16.pdf"
W.OUT = ROOT / "reports/manuscript_verification_round16.json"
W.REQUIRED_SECTIONS = ["Abstract", "Keywords", "1. Introduction", "2. Models and estimation", "3. Simulation study",
                       "3.2 Confirmatory test H-R12", "4. Real-data applications", "4.4 Stability and clinical alignment",
                       "5. Phase 2: pre-registered methodological revisions (Rounds 15–16)",
                       "5.2 Symbolic summaries: functional distillation (H-R15a, H-R16a)",
                       "5.3 A landmark-anchored functional model (H-R15b, H-R16b)",
                       "6. Discussion", "6.3 Limitations", "6.4 Reproducibility", "7. Conclusions", "Declarations",
                       "References", "Appendix A. Number provenance and verification"]
D, CONV = W.D, W.CONV
W.LITERALS += [
    # Round-15/16 design constants (decision log D133-D138)
    ("10", "μ = 10", D, "mu = 10"), ("50", "|a x + b| exceeds 50", D, "exceeds 50"),
    ("200", "200 L-BFGS iterations", D, "200 L-BFGS iterations"),
    ("1e-2", "μ ∈ {1e-2, 1e-1, 1, 10}", D, "{1e-2, 1e-1, 1, 10}"), ("1e-1", "μ ∈ {1e-2, 1e-1, 1, 10}", D, "{1e-2, 1e-1, 1, 10}"),
    ("1", "μ ∈ {1e-2, 1e-1, 1, 10}", D, "{1e-2, 1e-1, 1, 10}"), ("10", "μ ∈ {1e-2, 1e-1, 1, 10}", D, "{1e-2, 1e-1, 1, 10}"),
    ("0.05", "below +0.05", D, "< +0.05"), ("0.005", "below +0.005", D, "< +0.005"), ("0", "entirely below 0", CONV, None),
    ("0.01", "tolerance 0.01", D, "tol 0.01"), ("0.15", "φ₁ below 0.15", D, "phi_1 < 0.15"), ("0.10", "φ₂ below 0.10", D, "phi_2 < 0.10"),
    ("10", "smooth cap 10 above", D, "smooth cap 10 above"),
    ("3", "3–8 visits", D, "3–8 visits"), ("8", "3–8 visits", D, "3–8 visits"),
    ("40", "all 40 distillation refits", D, "all 40 distillation refits"),
    ("5", "5 folds", D, "5 folds"),
]
for _ph in ("S-Add 3500–3519", "S-Sparse 3600–3619", "S-Last 3700–3719", "S-Add **3800–3819**", "S-Sparse **3900–3919**",
            "S-Last **4000–4019**", "seeds 22, 914, 966", "seed 24"):
    _ctx = _ph.replace("**", "")
    for _tok in W.tokens(" " + _ctx + " "):
        W.LITERALS.append((_tok, _ctx, D, _ph))
W.BUILDER_LITERALS += [
    ("0.05", "(rule: < +0.05)", D, "< +0.05"), ("0.005", "(rule: < +0.005)", D, "< +0.005"),
    ("0", "(rule: below 0)", CONV, None), ("95", "95% bound", CONV, None), ("95", "95% CI, (rule", CONV, None),
    ("3", "3-resample bag", D, "3-resample"), ("10", "μ = 10", D, "mu = 10"),
    ("0.15", "(rule: φ₁ < 0.15, φ₂ < 0.10)", D, "phi_1 < 0.15"), ("0.10", "(rule: φ₁ < 0.15, φ₂ < 0.10)", D, "phi_2 < 0.10"),
]
FIGURES = W.FIGURES


def check_rerender():
    sys.path.insert(0, str(ROOT / "scripts"))
    import manuscript_round16 as M
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
