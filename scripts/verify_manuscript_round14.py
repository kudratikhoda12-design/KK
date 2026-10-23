"""Round 14 (D131): verification of the manuscript's numbers and of its compiled PDF.

Checks (each recorded in reports/manuscript_verification_round14.json; ALL_PASS only if every check passes):
  ledger      every value logged by the builder is re-read from its source with this script's own reader, compared
              with the logged value, re-formatted with this script's own formatter and compared with the logged text
  rerender    rendering the template again from the records reproduces docs/manuscript_round14.md byte for byte
  numbers     every numeric token in the manuscript text is either the text of a logged value or a registered design
              constant / documented literal (LITERALS below) whose phrase is found in its cited source document;
              literal numbers typed into the template itself must all be registered (stricter: logged values do not
              excuse them)
  citations   every reference in docs/references_round14.md is cited, and every cited key is in it
  sections    the required sections and all eight effect-figure files are present (figure files exist)
  pdf         (with --pdf) reports/manuscript_round14.pdf exists, is newer than the manuscript, has no blank page
              (a page without text and without an image), contains every required section title and embeds the figures
Excluded from the number audit: code spans, HTML tags, link/image targets, headings, the reference list, and numbers
that are labels (Table 3, Figure 1, Section 2, Round 12, Rounds 1–9, Appendix A, identifiers such as B1-FPCA-1, CK-6c,
D131, φ₁, S3, P0, 4f).
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs/manuscript_round14.md"
TPL = ROOT / "docs/manuscript_round14_template.md"
LEDGER = ROOT / "reports/manuscript_values_round14.json"
REFS = ROOT / "docs/references_round14.md"
PDF = ROOT / "reports/manuscript_round14.pdf"
OUT = ROOT / "reports/manuscript_verification_round14.json"

REQUIRED_SECTIONS = ["Abstract", "Keywords", "1. Introduction", "2. Models and estimation", "3. Simulation study",
                     "3.2 Confirmatory test H-R12", "4. Real-data applications", "4.4 Stability and clinical alignment",
                     "5. Discussion", "5.3 Limitations", "5.4 Reproducibility", "6. Conclusions", "Declarations",
                     "References", "Appendix A. Number provenance and verification"]
FIGURES = ["ms14_r13_effects_pbc_E2_b2s_cox_a.png", "ms14_r13_effects_pbc_E2_b2s_cox_b.png",
           "ms14_r13_effects_pbc_E2_b1_fpca1_cox_a.png", "ms14_r13_effects_pbc_E2_b1_fpca1_cox_b.png",
           "ms14_r13_effects_prothro_b2s_cox.png", "ms14_r13_effects_prothro_b1_fpca1_cox.png",
           "ms14_r13_effects_aids_b2s_cox.png", "ms14_r13_effects_aids_b1_fpca1_cox.png"]

D = "docs/decisions.md"
DN = "docs/design_note_round10.md"
CONV = None          # a statistical convention (e.g. "95% CI", "includes 0"), not a registered constant
# Every literal number typed into the template must be covered by an entry (token, template context, source, phrase):
# the context is a substring of the template (after stripping) containing that occurrence of the token, and the phrase
# must occur in the source document and contain the token. A token typed anywhere else fails the audit.
LITERALS = [
    # statistical conventions
    ("0", "95% CI lies below 0", CONV, None), ("95", "95% CI lies below 0", CONV, None),
    ("0", "includes 0", CONV, None), ("0", "included 0", CONV, None), ("0", "excluded 0", CONV, None),
    ("0", "excluding 0", CONV, None), ("0", "interval includes 0", CONV, None),
    ("2", "(t)/2)dt", DN, "sin(πx/2)"), ("0", "[0, 1]", CONV, None), ("1", "[0, 1]", CONV, None), ("1", "R² equals 1", CONV, None),
    ("95", "(95% CI", CONV, None), ("95", "95% t-interval", CONV, None), ("95", "corrected 95% intervals", CONV, None),
    ("90", "(90% CI", CONV, None), ("0.05", "p < 0.05", CONV, None),
    # registered design constants (decision log, design note)
    ("0.05", "within ±0.05", D, "within ±0.05"), ("0.005", "within ±0.005", D, "margin ±0.005"),
    ("0", "seed 0, then seeds 0–2", D, "seeds 0–2"), ("2", "seed 0, then seeds 0–2", D, "seeds 0–2"),
    ("0", "split 0.", D, "split 0"), ("2", "rank 2", D, "B2-S rank 2"),
    ("1", "(1 + t)", DN, "(1 + t)"), ("4", "[p → 4 → 1]", DN, "[p → 4 → 1]"), ("1", "[p → 4 → 1]", DN, "[p → 4 → 1]"),
    ("4", "p → 4 → 1]", DN, "[D·M + p → 4 → 1]"), ("1", "p → 4 → 1]", DN, "[D·M + p → 4 → 1]"),
    ("1", "log T = 1 − η/1.5", D, "log T = 1 − η/1.5"),
    ("1.5", "shape 1.5", DN, "shape 1.5"), ("0.8", "0.8 times", DN, "0.8·[∫φ₁(X₁)dt]"),
    ("1", "landmark 1 year", D, "landmark 1 year"), ("6", "landmark 6 months", D, "landmark 6 months"),
    ("730", "730 days", D, "730-day"), ("2", "2·entropy", DN, "2·S(Φ)"),
    ("3", "3–8", DN, "3–8 visits"), ("8", "3–8", DN, "3–8 visits"), ("3", "Channel 3 is a null channel", DN, "channel 3 is a null"),
    ("5", "grid 5, order 3", D, "grid 5, order 3"), ("3", "grid 5, order 3", D, "grid 5, order 3"),
    ("5", "at most 5 components", DN, "max 5"), ("95", "(95% variance explained", DN, "95% fraction of variance explained"),
    ("5", "at least 5%", DN, "at least 5% of"), ("10", "at most 10 per model", DN, "at most 10 per"),
    ("5", "5 splits × 20 bootstrap resamples", D, "5 splits × 20 bootstrap resamples"),
    ("20", "5 splits × 20 bootstrap resamples", D, "5 splits × 20 bootstrap resamples"),
    ("5", "5 outer splits by subject", D, "5 outer splits by subject"),
    ("10", "10 × 10", D, "10 × 10"), ("10", "≥ 10 cases and ≥ 10 controls", D, "≥ 10 cases and ≥ 10 controls"),
    ("10", "at least 10 cases and 10 controls", D, "≥ 10 cases and ≥ 10 controls"),
    ("12", "(12 basis functions", D, "12 functions"), ("60", "60/20/20", D, "60/20/20"), ("20", "60/20/20", D, "60/20/20"),
    ("30", "about 30%", DN, "calibrated per scenario to about 30%"), ("40", "40 dense times", DN, "Dense: 40 equispaced times"),
    ("50", "grid of 50 points", DN, "G is a fixed grid of 50 equispaced points"), ("50", "common 50-point", D, "50 points"),
    ("60", "(60 s per edge", D, "60 s"), ("80", "at least 80%", D, "≥ 80% of fits"),
    ("80", "threshold 80%", D, "≥ 80% of fits"), ("90", "τ the 90% quantile", DN, "τ = 90% quantile"),
    ("90", "90% band", D, "90% bootstrap bands"), ("100", "patience 100", DN, "patience 100"),
    ("3,000", "at most 3,000 epochs", DN, "maximum 3,000 epochs"), ("200", "200 epochs", D, "200-epoch masked fine-tuning"),
    ("1,000", "n = 1,000", D, "n 1,000"), ("1,408", "the 1,408 its documentation", D, "1,408 rows"),
    ("488", "both 488 and 467", D, "488 and 467"), ("467", "both 488 and 467", D, "488 and 467"),
    ("1.5", "JM 1.5-2", D, "JM 1.5-2"), ("3.8", "3.8-12", D, "survival 3.8-12"),
    ("0.99", "exceeds 0.99", DN, "validation R² > 0.99"), ("0.01", "within 0.01", D, "within 0.01"),
    ("3e-8", "below 3e-8", D, "≤ 3e-8"), ("1e-8", "below 1e-8", D, "< 1e-8"), ("1e-6", "below 1e-6", D, "< 1e-6"),
]
# literal numbers typed into the builder's table code (labels and headers); the context must occur in the manuscript
BUILDER_LITERALS = [
    ("21", "(A3, seed 21)", D, "seed 21 with the selected configuration"), ("512", "(of 512)", D, "0–1 of 512 risk-encoder edges"),
    ("0.05", "TOST p (±0.05)", D, "within ±0.05"), ("0.05", "D-cal p < 0.05", CONV, None),
    ("80", "aligned if ≥ 80%", D, "≥ 80% of fits"), ("95", "95% CI", CONV, None), ("95", "95% NB interval", CONV, None),
    ("20", "20 replicates)", D, "N = 20 replicates per scenario"), ("5", "over 5 splits", D, "5 outer splits by subject"),
]
_GRID_L = ("λ {0, 1e-4, 1e-3, 1e-2, 3e-2}", D, "λ {0, 1e-4, 1e-3, 1e-2, 3e-2}")
_GRID_T = ("θ_rel {0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25}", D, "θ_rel {0, .01, .02, .05, .10, .15, .20, .25}")
_GRID_E = ("{1e-3, 3e-3} × weight decay {1e-4, 1e-3}", D, "lr {1e-3, 3e-3} × wd {1e-4, 1e-3}")
_GRID_A = ("{1e-7, …, 1e1} × λ_t {1e-4, 1e-2, 1, 1e2, 1e4}", D, "λ_x {1e-7, …, 1e1} × λ_t {1e-4, 1e-2, 1, 1e2, 1e4}")
_SEEDS = ("S-Add 3000–3019, S-Int 3100–3119, S-Sparse 3200–3219, S-Misspec 3300–3319, S-Int/P0 3400–3419", D,
          "S-Add 3000–3019, S-Int 3100–3119, S-Sparse 3200–3219, S-Misspec 3300–3319, S-Int/P0 3400–3419")
for _ctx, _src, _ph in (_GRID_L, _GRID_T, _GRID_E, _GRID_A, _SEEDS):
    for _tok in sorted(set(re.findall(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?(?:e[-−+]?\d+)?)",
                                      re.sub(r"(?<![\d.])\.(\d)", r"0.\1", _ph)))):
        LITERALS.append((_tok, _ctx, _src, _ph))
SKIP_LABEL = re.compile(r"(?:Tables?|Figures?|Sections?|Rounds?|rounds?|Appendix|Revision|revision|Parts?|CK|seed-)"
                        r"[\s-]+\d+(?:\.\d+)?[a-z]?(?:\s*(?:–|-|/|to|and)\s*\d+(?:\.\d+)?[a-z]?)*")
TOKEN = re.compile(r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?(?:e[-−+]?\d+)?")


# ----------------------------------------------------------------------------------------------------------------------
# own reader and formatter (deliberately not imported from the builder)
def _read_table(path):
    rows = {}
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("| ") and len(cells) == 5 and cells[0] not in ("Key", "---"):
            rows[cells[0]] = {"Key": cells[0], "Cite": cells[1], "Year": cells[2], "Reference": cells[3], "Verified": cells[4]}
    return rows


_files = {}


def read_source(src):
    segs = src.split("/")
    k = next(i for i, s in enumerate(segs) if re.search(r"\.(json|jsonl|csv|md)$", s))
    rel, rest = "/".join(segs[:k + 1]), segs[k + 1:]
    if rel not in _files:
        p = ROOT / rel
        if rel.endswith(".jsonl"):
            _files[rel] = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        elif rel.endswith(".json"):
            _files[rel] = json.loads(p.read_text())
        elif rel.endswith(".csv"):
            import csv
            with open(p) as fh:
                _files[rel] = list(csv.DictReader(fh))
        else:
            _files[rel] = _read_table(p)
    obj = _files[rel]
    if rel.endswith(".csv"):
        conds = [c.split("=", 1) for c in rest[0].split(",")]
        hit = [r for r in obj if all((r[a] == b) for a, b in conds)]
        assert len(hit) == 1, f"{src}: {len(hit)} rows"
        raw = hit[0][rest[1]]
        if raw == "":
            return float("nan")
        try:
            f = float(raw)
            return int(f) if re.fullmatch(r"-?\d+", raw) else f
        except ValueError:
            return raw
    if rel.endswith(".jsonl"):
        conds = [c.split("=", 1) for c in rest[0].split(",")]
        hit = [r for r in obj if all(str(r.get(a)) == b for a, b in conds)]
        assert len(hit) == 1, f"{src}: {len(hit)} records"
        obj, rest = hit[0], rest[1:]
    for s in rest:
        s = s.replace("%2F", "/")
        obj = obj[int(s)] if isinstance(obj, list) else obj[s]
    return obj


def format_value(v, spec):
    if isinstance(v, list):
        return "[" + ", ".join(format_value(x, spec) for x in v) + "]"
    if spec in ("s", "key"):
        return str(v)
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    if spec == "d":
        n = int(round(float(v)))
        assert float(v) == n
        text = f"{n:,}" if abs(n) >= 10000 else str(n)
    elif spec == "pct":
        text = "%.0f%%" % (100 * v)
    else:
        text = format(v, spec)
    text = re.sub(r"e([+-])0*(\d)", lambda m: "e" + ("-" if m.group(1) == "-" else "") + m.group(2), text)
    return text.replace("-", "−")


def same(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-15)
    return a == b


def check_ledger():
    led = json.loads(LEDGER.read_text())
    bad = []
    for e in led["values"]:
        if e["spec"] == "key":
            ok = e["value"] in read_source(e["src"]) and e["text"] == e["value"]
        else:
            v = read_source(e["src"])
            ok = same(v, e["value"]) and format_value(v, e["spec"]) == e["text"]
        if not ok:
            bad.append(e)
    return dict(passed=not bad, n_values=len(led["values"]), mismatches=bad[:20])


def check_rerender():
    sys.path.insert(0, str(ROOT / "scripts"))
    import manuscript_round14 as B
    B._cache.clear()
    text = B.render(TPL.read_text())
    led = json.loads(LEDGER.read_text())
    same_md = text == MD.read_text()
    same_led = [dict(src=e["src"], spec=e["spec"], text=e["text"]) for e in B.LEDGER] == \
               [dict(src=e["src"], spec=e["spec"], text=e["text"]) for e in led["values"]]
    return dict(passed=same_md and same_led, manuscript_identical=same_md, ledger_identical=same_led)


def _strip(text, drop_refs=True):
    out = []
    in_refs = False
    for line in text.splitlines():
        if line.startswith("## "):
            in_refs = drop_refs and line.strip() == "## References"
            continue
        if in_refs or line.startswith("#") or re.match(r"\|(-+\|)+$", line.strip()):
            continue
        line = re.sub(r"`[^`]*`", " ", line)
        line = re.sub(r"<[^>]+>", " ", line)
        line = re.sub(r"\]\([^)]*\)(\{[^}]*\})?", "] ", line)
        line = re.sub(r"\{\{[^}]*\}\}", " ", line)
        line = re.sub(r"^(\s*)\d+\.\s", r"\1", line)                       # ordered-list markers
        out.append(line)
    return SKIP_LABEL.sub(" ", "\n".join(out))


def tokens(text, with_pos=False):
    found = []
    for m in TOKEN.finditer(text):
        a, b = m.start(), m.end()
        prev = text[a - 1] if a else " "
        nxt = text[b] if b < len(text) else " "
        if prev.isalnum() or prev in "_.₀₁₂₃₄₅₆₇₈₉" or nxt.isalpha() or nxt in "_₀₁₂₃₄₅₆₇₈₉":
            continue                                                  # identifiers, subscripts, 0.3z₁, 4f, 5th
        if prev == "-" and a >= 2 and text[a - 2].isalnum():          # identifier hyphen (B1-FPCA-1, 3.8-12)
            continue
        if prev == "/" and a >= 2 and text[a - 2].isalpha():          # S-Int/P0
            continue
        found.append((m.group(0), a) if with_pos else m.group(0))
    return found


def _norm(s):
    return re.sub(r"(?<![\d.])\.(\d)", r"0.\1", s)                    # ".25" in a registered grid = 0.25


def _in_phrase(tok, phrase):
    return re.search(r"(?<![\d.,])" + re.escape(tok) + r"(?![\d,]\d|\.\d)", _norm(phrase)) is not None


def check_numbers():
    led = json.loads(LEDGER.read_text())
    from_values = set()
    for e in led["values"]:
        from_values.update(tokens(" " + e["text"] + " "))
    bad_lit = [dict(token=tok, context=ctx, source=src) for tok, ctx, src, phrase in LITERALS
               if src is not None and (phrase not in (ROOT / src).read_text() or not _in_phrase(tok, phrase))]
    bad_lit += [dict(token=tok, context=ctx, source="context lacks token") for tok, ctx, src, phrase in LITERALS
                if tok not in tokens(" " + ctx + " ")]
    tpl = re.sub(r"\s+", " ", _strip(TPL.read_text()))
    covered = set()
    for tok, ctx, _, _ in LITERALS:
        for m in re.finditer(re.escape(ctx), tpl):
            for t2, pos in tokens(" " + ctx + " ", with_pos=True):
                if t2 == tok:
                    covered.add((tok, m.start() + pos - 1))
    tpl_tokens = tokens(tpl, with_pos=True)
    uncovered = sorted({(tok, tpl[max(0, pos - 30):pos + 30]) for tok, pos in tpl_tokens if (tok, pos) not in covered})
    unused = sorted({(tok, ctx) for tok, ctx, _, _ in LITERALS
                     if not any(c[0] == tok for c in covered) or ctx not in tpl})
    md_flat = re.sub(r"\s+", " ", MD.read_text())
    bad_lit += [dict(token=tok, context=ctx, source=src or "convention") for tok, ctx, src, phrase in BUILDER_LITERALS
                if ctx not in md_flat or tok not in tokens(" " + ctx + " ")
                or (src is not None and (phrase not in (ROOT / src).read_text() or not _in_phrase(tok, phrase)))]
    allowed = from_values | {tok for tok, _, _, _ in LITERALS + BUILDER_LITERALS}
    md_tokens = tokens(_strip(MD.read_text()))
    unknown = sorted({tk for tk in md_tokens if tk not in allowed})
    return dict(passed=not unknown and not uncovered and not bad_lit and not unused, n_tokens=len(md_tokens),
                n_template_literals=len(tpl_tokens), unregistered_in_manuscript=unknown,
                template_literals_without_registered_context=[f"{a}: ...{b}..." for a, b in uncovered],
                registry_entries_not_found_in_source=bad_lit, registry_entries_unused=[f"{a}: {b}" for a, b in unused])


def check_citations():
    refs = set(_read_table(REFS))
    cited = set(json.loads(LEDGER.read_text())["cited"])
    keys_in_tpl = set(k.strip() for grp in re.findall(r"\{\{CP?:([^}]+)\}\}", TPL.read_text()) for k in grp.split(";"))
    return dict(passed=refs == cited == keys_in_tpl, uncited=sorted(refs - cited), unknown=sorted(keys_in_tpl - refs))


def check_sections():
    md = MD.read_text()
    heads = [l.lstrip("#").strip() for l in md.splitlines() if l.startswith("## ") or l.startswith("### ")]
    missing = [s for s in REQUIRED_SECTIONS if not any(h.startswith(s) for h in heads)]
    figs = re.findall(r"!\[[^]]*\]\(([^)]+)\)", md)
    fig_missing = [f for f in FIGURES if not any(x.endswith(f) for x in figs)]
    fig_files = [f for f in figs if not (MD.parent / f).resolve().exists()]
    return dict(passed=not missing and not fig_missing and not fig_files, missing_sections=missing,
                missing_figures=fig_missing, missing_figure_files=fig_files, n_figures=len(figs))


def pdf_pages(pdf):
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    pages = txt.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    imgs = subprocess.run(["pdfimages", "-list", str(pdf)], capture_output=True, text=True, check=True).stdout
    img_pages = {int(l.split()[0]) for l in imgs.splitlines()[2:] if l.split() and l.split()[0].isdigit()}
    return pages, img_pages


def check_pdf(pdf=PDF, md=MD, sections=REQUIRED_SECTIONS, n_figures=len(FIGURES)):
    if not pdf.exists():
        return dict(passed=False, reason="missing")
    pages, img_pages = pdf_pages(pdf)
    blank = [i + 1 for i, p in enumerate(pages) if not p.strip() and (i + 1) not in img_pages]
    flat = re.sub(r"\s+", " ", " ".join(pages))
    missing = [s for s in sections if s not in flat]
    return dict(passed=not blank and not missing and len(img_pages) >= 1 and pdf.stat().st_mtime >= md.stat().st_mtime
                and sum(1 for _ in img_pages) >= 1, pages=len(pages), blank_pages=blank, missing_sections=missing,
                pages_with_images=len(img_pages), newer_than_source=pdf.stat().st_mtime >= md.stat().st_mtime,
                images=_count_images(pdf), images_expected_at_least=n_figures)


def _count_images(pdf):
    imgs = subprocess.run(["pdfimages", "-list", str(pdf)], capture_output=True, text=True, check=True).stdout
    return sum(1 for l in imgs.splitlines()[2:] if l.split() and l.split()[2] == "image")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args(argv)
    res = dict(ledger=check_ledger(), rerender=check_rerender(), numbers=check_numbers(), citations=check_citations(),
               sections=check_sections())
    if a.pdf:
        p = check_pdf()
        p["passed"] = p["passed"] and p.get("images", 0) >= len(FIGURES)
        res["pdf"] = p
    res["ALL_PASS"] = all(v["passed"] for v in res.values() if isinstance(v, dict))
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0 if res["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
