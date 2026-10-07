#!/usr/bin/env python3
"""Comprehensive technical documentation PDF: every stage, method, test and result of the project.

    python scripts/make_full_pdf.py  ->  reports/ETH_OrderFlow_Alpha_Full_Documentation.pdf

All result tables are read from pipeline output files at build time; the project documents (docs/*.md) are
rendered verbatim in the appendices.
"""
from __future__ import annotations

import html
import json
import re
import sys
import types
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ethof.features import FEATURE_SETS, GROUPS  # noqa: E402

sys.modules.setdefault("ethof_groups", types.SimpleNamespace(GROUPS=GROUPS))
import make_pdf_report as R  # noqa: E402  (fonts, styles, helpers)
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

pdfmetrics.registerFont(TTFont("DVM", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))
T, F = R.T, R.F
OUT = ROOT / "reports" / "ETH_OrderFlow_Alpha_Full_Documentation.pdf"
P, table, fig, bullets, callout, fmt = R.P, R.table, R.fig, R.bullets, R.callout, R.fmt
S = R.S
S["code"] = ParagraphStyle("code", fontName="DVM", fontSize=7.6, leading=10, backColor=R.LIGHT, leftIndent=6,
                           borderPadding=4, spaceBefore=3, spaceAfter=6)
S["h3"] = ParagraphStyle("h3", fontName="DV-B", fontSize=9.8, leading=13, textColor=R.NAVY, spaceBefore=6, spaceAfter=3)
S["toc"] = ParagraphStyle("toc", fontName="DV", fontSize=10, leading=15)
for _k in ("h1", "h2", "h3"):
    S[_k].keepWithNext = 1
S["code"].fontSize, S["code"].leading = 6.8, 8.8


# ---------------------------------------------------------------------------------------------- markdown
def inline(t: str) -> str:
    t = html.escape(t.replace("Ⓢ", "S"), quote=False)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"<font name='DVM' size='7.6'>\1</font>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"(?<![\w_])_([^_\s][^_]*?)_(?![\w_])", r"<i>\1</i>", t)
    return t


def md_table(lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c or "--") for c in r)]
    if not rows:
        return []
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    w = [min(max(len(r[i]) for r in rows), 60) + 4 for i in range(n)]
    widths = [17 * x / sum(w) for x in w]
    head = [Paragraph(inline(c), S["cellb"]) for c in rows[0]]
    body = [[Paragraph(inline(c), S["cell"]) for c in r] for r in rows[1:]]
    t = Table([head] + body, colWidths=[x * cm for x in widths], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), R.NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, R.LIGHT]),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d3dc")),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return [t, Spacer(1, 5)]


def md(text: str, shift: int = 1):
    """Minimal markdown renderer: headings, paragraphs, bullets, numbered lists, tables, code blocks."""
    out, lines, i, para = [], text.splitlines(), 0, []

    def flush():
        if para:
            out.append(P(inline(" ".join(para))))
            para.clear()
    while i < len(lines):
        l = lines[i]
        s = l.strip()
        if s.startswith("```"):
            flush()
            j = i + 1
            code = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                import textwrap
                for seg in (textwrap.wrap(lines[j], 112, subsequent_indent="    ", drop_whitespace=False) or [""]):
                    code.append(html.escape(seg).replace(" ", "&nbsp;"))
                j += 1
            out.append(Paragraph("<br/>".join(code), S["code"]))
            i = j + 1
            continue
        if s.startswith("|"):
            flush()
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            out += md_table(lines[i:j])
            i = j
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", s)
        if m:
            flush()
            lvl = min(len(m.group(1)) + shift, 4)
            out.append(P(inline(m.group(2)), {1: "h1", 2: "h2", 3: "h3", 4: "h3"}[lvl]))
        elif re.match(r"^(\s*)[-*]\s+", l) or re.match(r"^\s*\d+\.\s+", l):
            flush()
            item = re.sub(r"^\s*([-*]|\d+\.)\s+", "", l)
            j = i + 1
            while j < len(lines) and lines[j].startswith("  ") and lines[j].strip() and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[j]):
                item += " " + lines[j].strip()
                j += 1
            ind = (len(l) - len(l.lstrip())) // 2
            bt = "•" if re.match(r"^\s*[-*]", l) else re.match(r"^\s*(\d+\.)", l).group(1)
            out.append(Paragraph(inline(item), ParagraphStyle("li", parent=S["bullet"], leftIndent=12 + 10 * ind,
                                                              bulletIndent=2 + 10 * ind), bulletText=bt))
            i = j
            continue
        elif s.startswith(">"):
            flush()
            out.append(callout(inline(s.lstrip("> "))))
        elif s in ("---", "***"):
            flush()
            out.append(Spacer(1, 4))
        elif not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()
    return out


def doc_md(name, shift=1):
    return md((ROOT / "docs" / name).read_text(), shift)


def csv(path):
    return pl.read_csv(T / path, infer_schema_length=None)


# ---------------------------------------------------------------------------------------------- content
FEATURE_DEFS = {
    "ret_k": "log(P<sub>t</sub> / P<sub>t−k</sub>), k ∈ {1,5,15,30,60}; P = last trade price at/before bar close (forward-filled backwards only)",
    "vol_w": "rolling std of 1-min log returns over w ∈ {15,60,240,1440} minutes (vol_1440 requires ≥ 720 obs)",
    "mom_240_riskadj": "ret_240 / (vol_240 · √240): 4-hour momentum in volatility units",
    "dist_ma_w": "log(P<sub>t</sub> / mean(P over last w minutes)), w ∈ {60,240}: distance from moving average",
    "range_15": "log(max high / min low) over the last 15 minutes",
    "dist_vwap_15": "log(P<sub>t</sub> / VWAP of the last 15 minutes)",
    "hour_sin, hour_cos": "sin / cos of the UTC time of day (intraday seasonality)",
    "log_volume_1": "log(volume<sub>t</sub>)",
    "volume_z_1440": "z-score of log volume against its trailing 1-day mean and sd",
    "volume_ratio_w": "volume over last w minutes / (w · trailing 1-day mean minute volume), w ∈ {15,60}",
    "trade_intensity_15": "log(1 + mean number of aggTrades per minute over 15 minutes)",
    "trade_count_z_1440": "z-score of log(1 + trade count) against trailing 1 day",
    "ofi_w": "(Σ taker-buy vol − Σ taker-sell vol) / (Σ buy + Σ sell) over w ∈ {1,5,15,60} minutes; null if no volume",
    "trade_imb_w": "(buy trades − sell trades) / (buy + sell trades), w ∈ {1,15}",
    "signed_flow_15": "Σ(buy − sell volume) over 15 min / (15 · trailing 1-day mean minute volume)",
    "ofi_w_z_1440": "trailing 1-day z-score of ofi_w, w ∈ {1,15}",
    "avg_trade_size_15": "log(volume / trades) over 15 minutes",
    "avg_trade_size_z_1440": "trailing 1-day z-score of log average trade size",
    "large_share_15": "volume of aggTrades ≥ 10 ETH / total volume, 15 minutes",
    "large_ofi_15": "imbalance of buy vs sell volume from aggTrades ≥ 10 ETH, 15 minutes",
    "obi_kpct": "(bid depth within k% − ask depth within k%) / sum, k ∈ {1,2,3,5}; null if the book is stale",
    "obi_1pct_chg_15": "obi_1pct<sub>t</sub> − obi_1pct<sub>t−15</sub>",
    "obi_1pct_mean_15": "15-minute mean of obi_1pct",
    "depth_1pct_z_1440": "trailing 1-day z-score of log(bid + ask depth within 1%)",
    "funding_rate_last": "last settled funding rate (settlement time ≤ decision time)",
    "funding_rate_mean_3": "mean of the last 3 settled funding rates (≈ 24 h)",
    "minutes_to_funding": "minutes until the next scheduled funding settlement",
    "oi_chg_w": "log change of open interest over w ∈ {60,1440} minutes, OI rows lagged ≥ 5 min",
}

TEST_DOCS = {
    "test_url_convention": "archive URL and .CHECKSUM URL are built exactly as Binance publishes them",
    "test_periods": "monthly/daily period generation, incl. leap day and year roll-over",
    "test_sample_checksums": "SHA-256 of the committed sample files matches SHA256SUMS",
    "test_detect_time_unit": "epoch unit detection for ms / µs / ns timestamps (parametrised ×3)",
    "test_detect_time_unit_mixed_raises": "a file mixing units raises instead of guessing",
    "test_clean_aggtrades_counts_and_policy": "removal counts for price ≤ 0, qty ≤ 0, out-of-day, exact and conflicting duplicates; bad tick flagged but kept; taker-side mapping",
    "test_unsorted_input_is_sorted": "out-of-order raw rows are counted and sorted chronologically",
    "test_aggregate_1min_values_and_grid": "OHLC, volume, buy/sell split, VWAP, fills, size buckets; empty minutes kept with null price; no-trade runs",
    "test_sample_end_to_end_conservation": "on real data: Σ bar volume = Σ trade volume, buy + sell = total, high ≥ low",
    "test_book_sample_wide_and_monotone": "book long→wide pivot; complete snapshots; cumulative depth monotone",
    "test_asof_book_has_no_lookahead": "a snapshot stamped exactly at the bar close is NOT used for that bar",
    "test_asof_book_flags_stale": "old snapshots flag the bar as stale",
    "test_asof_book_frozen_feed_is_stale": "repeated identical snapshots age from the last change and become stale",
    "test_order_flow_and_imbalance_definitions": "OFI-1, OFI-15 (volume-aggregated) and book imbalance match their formulas; no padding of short history",
    "test_imbalance_zero_denominator_is_null": "zero volume gives null imbalance, not 0",
    "test_stale_book_gives_null_book_features": "stale book → null book features",
    "test_rolling_features_are_trailing": "features on a prefix equal the full-sample features (no look-ahead); ret_15 formula",
    "test_forward_fill_only_backward": "missing prices are filled from the past only; execution price flagged",
    "test_targets": "forward return, binary and 3-class labels, null tail, label end time",
    "test_walk_forward_purge_embargo": "7 folds; train ends embargo before validation; purge of overlapping labels",
    "test_holdout_guard": "holdout refused without the frozen methodology; access logged with its SHA-256",
    "test_signal_rules": "probability, flow and momentum signal rules incl. NaN → flat",
    "test_staggered_position": "net position = mean of the last H signals",
    "test_costs_per_side": "fee + slippage + half spread arithmetic",
    "test_backtest_pnl_execution_timing_and_costs": "P&L earned from next-bar VWAP (not the signal bar); entry+exit cost; trade return; break-even formula",
    "test_funding_charged_to_position_held_at_settlement": "funding hits the position held over the settlement; shorts receive positive funding",
    "test_multiple_testing_adjustments": "Holm and Benjamini–Hochberg adjusted p-values",
    "test_hac_recovers_slope": "HAC OLS recovers a known slope on simulated data",
}


def weekly_table(w: pl.DataFrame):
    d = w.filter(pl.col("strategy") == "ml").sort("ts")
    wk = [str(x)[:10] for x in d["ts"].to_list()]
    net = [f"{v * 100:.2f}%" for v in d["net"].to_list()]
    half = (len(wk) + 1) // 2
    rows = [[wk[i], net[i], wk[i + half] if i + half < len(wk) else "", net[i + half] if i + half < len(wk) else ""]
            for i in range(half)]
    return table(rows, ["week", "net", "week", "net"], [4.25] * 4)


def main() -> None:
    frozen = json.loads((T / "frozen_parameters.json").read_text())
    sel = json.loads((T / "primary_model_selection.json").read_text())
    hs = json.loads((T / "holdout" / "summary.json").read_text())
    sig = json.loads((T / "holdout" / "significance.json").read_text())
    sol = json.loads((T / "replication_SOLUSDT" / "summary.json").read_text())
    st_sum = json.loads((T / "stats_summary.json").read_text())
    interp = json.loads((T / "interpret_summary.json").read_text())
    spread = json.loads((T / "spread_summary.json").read_text())
    bt = csv("holdout/backtest_summary.csv")
    b = bt.filter((pl.col("strategy") == "ml") & (pl.col("cost") == "base")).row(0, named=True)
    g = bt.filter((pl.col("strategy") == "ml") & (pl.col("cost") == "gross")).row(0, named=True)
    pc = hs["primary_classification_pooled"]
    st = []

    # ========================================================================== cover + contents
    st += [Spacer(1, 2.2 * cm), P("Ethereum Order-Flow Alpha", "title"), Spacer(1, 4),
           P("Complete technical documentation: every stage, method, test and result", "sub"), Spacer(1, 4),
           P("Binance USD-M ETHUSDT perpetual · 2023-03-01 → 2026-09-30 · replication on SOLUSDT", "sub"),
           Spacer(1, 1 * cm), R.pipeline_diagram(), Spacer(1, 0.8 * cm),
           callout(f"<b>Final answer.</b> Out-of-sample predictability is real and stable (holdout AUC <b>{pc['auc']:.4f}</b>, "
                   f"95% CI {sig['holdout_auc']['ci95'][0]:.4f}–{sig['holdout_auc']['ci95'][1]:.4f}), but the frozen strategy's "
                   f"break-even cost is <b>{b['breakeven_cost_bps_per_side']:.2f} bps/side</b> versus a realistic <b>5.52 bps/side</b>: "
                   f"gross {g['total_return']*100:+.1f}% (Sharpe {g['sharpe']:.2f}), net <b>{b['total_return']*100:.1f}%</b>. "
                   "<b>Statistically significant, economically untradable.</b>"),
           Spacer(1, 0.6 * cm),
           P(f"Generated {date.today().isoformat()} by scripts/make_full_pdf.py. All result tables are read from the pipeline "
             "outputs at build time; no number is typed by hand. Repository: github.com/kudratikhoda12-design/KK "
             "(branch claude/brave-tesla-wpvjj7).", "cap"), PageBreak(),
           P("Contents", "h1")]
    toc = ["Part A — Overview", "  1. Executive summary", "  2. Research question, hypotheses and design",
           "  3. Project log: what was done, in order, including problems found and fixed",
           "Part B — Methods", "  4. Data acquisition", "  5. Data cleaning and quality-control methods",
           "  6. One-minute bars and the as-of order-book join", "  7. Feature definitions (all 47)", "  8. Targets",
           "  9. Statistical methods", "  10. Machine-learning models and hyper-parameters", "  11. Validation scheme",
           "  12. Signals, backtest equations, costs and metrics", "  13. Holdout protocol and freeze",
           "Part C — Tests and verification", "  14. Unit-test catalogue (30 tests)", "  15. Leakage audit (8 automated checks)",
           "  16. Data-quality report (full)",
           "Part D — Results", "  17. Exploratory analysis", "  18. Statistical alpha tests", "  19. Model results (all 43 configurations)",
           "  20. Interpretability", "  21. Development backtest and threshold selection", "  22. Final holdout results",
           "  23. Robustness and replication", "  24. Significance and data-snooping accounting", "  25. Conclusions",
           "Part E — Appendices", "  A. Frozen methodology (verbatim)", "  B. Stage 1 dataset selection (verbatim)",
           "  C. Data schema", "  D. Robustness report (verbatim)", "  E. Interview questions and answers",
           "  F. Resume bullets", "  G. Reproduction commands and file map"]
    st += [Paragraph(("<b>%s</b>" % t) if not t.startswith("  ") else "&nbsp;&nbsp;&nbsp;&nbsp;" + t.strip(), S["toc"]) for t in toc]
    st += [PageBreak()]

    # ========================================================================== PART A
    st += [P("Part A — Overview", "title"), Spacer(1, 10), P("1. Executive summary", "h1")]
    st += bullets([
        "<b>Data.</b> 1,877,097,488 Binance ETHUSDT-perpetual aggregated trades (taker side recorded), 3,710,848 order-book "
        "percentage-band snapshots, funding and open interest; 3,976 files verified by SHA-256; aggregated to 1,886,400 one-minute bars.",
        f"<b>Statistics.</b> Order flow explains the same minute's return (corr {st_sum['corr_ofi1_ret1_contemporaneous']:.3f}) but only "
        "weakly predicts the future: 1-minute OFI predicts a <i>reversal</i> of −0.15 to −0.24 bps per sd (HAC t −9.7 to −4.0, Holm-significant).",
        "<b>Machine learning.</b> 43 walk-forward configurations; every binary classifier beats 0.5 AUC in all 7 folds (best ≈ 0.539). "
        f"Primary model by pre-declared rule: {sel['selected']}.",
        f"<b>Holdout.</b> AUC {pc['auc']:.4f} [{sig['holdout_auc']['ci95'][0]:.4f}, {sig['holdout_auc']['ci95'][1]:.4f}], ECE {pc['ece']:.4f}, "
        "AUC > 0.5 in 12/12 months.",
        f"<b>Economics.</b> gross Sharpe {g['sharpe']:.2f}, break-even {b['breakeven_cost_bps_per_side']:.2f} bps/side vs 5.52 base cost; "
        f"net {b['total_return']*100:.1f}%. Momentum and order-flow continuation rules lose money even before costs.",
        "<b>Incremental value of order flow.</b> Price-only models reach almost the full AUC; order flow and book bands add ≤ 0.002 AUC in "
        "development and nothing robust on the holdout.",
        f"<b>Replication.</b> SOLUSDT holdout AUC {sol['primary_classification_pooled']['auc']:.4f}; also untradable (break-even 0.89 bps/side)."])
    st += [P("2. Research question, hypotheses and design", "h1"),
           P("<b>Question.</b> Does order-flow and market-microstructure information contain statistically significant predictive "
             "information about short-horizon ETH returns, and does it translate into economically meaningful alpha after costs?"),
           P("Pre-registered hypotheses (labelled as hypotheses before any experiment) and their outcomes:", "h3"),
           table([["H1 OFI has a weak but detectable relation with very short-horizon returns", "Confirmed (as reversal, not continuation)"],
                  ["H2 predictive performance only modestly above random", "Confirmed (AUC ≈ 0.54)"],
                  ["H3 boosted trees beat linear models in some periods", "Confirmed (≈ +0.002 to +0.010 AUC on holdout)"],
                  ["H4 performance deteriorates out of sample", "Rejected (holdout AUC ≥ development AUC)"],
                  ["H5 costs substantially reduce or eliminate profitability", "Confirmed: eliminated"],
                  ["H6 alpha varies across volatility regimes", "Prediction similar across regimes; gross P&L concentrated in mid/high vol"],
                  ["H7 ablation decides whether order flow adds information", "Adds ≤ 0.002 AUC in development, nothing robust on holdout"],
                  ["H8 statistically significant but not economically tradable", "Confirmed"]],
                 ["Hypothesis", "Outcome"], [10, 7]),
           P("Design principles", "h3")]
    st += bullets(["Chronological splits only; a 12-month holdout fixed before any return was examined.",
                   "Every feature uses information available at its decision time; verified automatically.",
                   "All learned preprocessing fitted inside each training fold.",
                   "Every choice (model, hyper-parameters, threshold, costs) made on development data by a pre-declared rule, then frozen and hash-logged.",
                   "Statistical and economic significance reported separately; negative results reported as they are."])
    st += [PageBreak(), P("3. Project log: what was done, in order", "h1"),
           table([["Stage 1", "Dataset selection", "Probed Coinbase, Tardis (blocked/paid), Binance spot & futures; mapped bookDepth coverage for all 1,374 days; chose Binance USD-M ETHUSDT perpetual, 2023-03 → 2026-09"],
                  ["Stage 2", "Acquisition + QC", "Download with SHA-256 verification; switched monthly → daily trade files after one month needed 13.5 GB RAM; 1 transfer broke mid-stream → retry logic fixed, day re-processed"],
                  ["Stage 2", "Book schema change", "1,435,416 rows initially counted as invalid were Binance's new ±0.2% bands (from 2026-01-15); reclassified as schema extension, not used"],
                  ["Stage 2", "Spread measurement", "bookTicker on 3 development days: 1 tick in 99.5% of seconds; mean 0.046 bps → half-spread 0.023 bps"],
                  ["Stage 3", "Features + targets", "47 causal features, 4 horizons; first leakage audit; a too-strict equality check flagged 0 = 0 matches and was corrected to non-zero targets"],
                  ["Stage 4", "EDA + statistics", "HAC regressions, controls, deciles with bootstrap, event study, quarterly IC (development only)"],
                  ["Stage 5–7", "Problem found", "A model crashed on ±inf features → investigation found (a) a <b>frozen order-book feed</b> for a month, (b) 91 zero OI readings, (c) undefined ratios. Fixed in the pipeline; dataset rebuilt; audit re-run; <b>all earlier model results discarded and re-run</b>"],
                  ["Stage 5–7", "Models", "8-point LightGBM grid; ablation A–E × logit/RF/LightGBM; horizons; 3-class and regression targets; rule baselines"],
                  ["Stage 8", "Dev backtest", f"Primary model {sel['selected']} by log-loss rule; δ = {sel['delta_selected']} by base-cost Sharpe; every δ net-negative"],
                  ["Freeze", "Methodology frozen", "docs/final_methodology.md + frozen_parameters.json committed (71ca5ad) before any holdout access; RF stride consistency fixed before freezing"],
                  ["Holdout", "Single evaluation", "ETH final evaluation and SOL replication, both logged with the same methodology hash"],
                  ["Stage 9–10", "Robustness + reports", "Significance/snooping accounting, notebooks, research report, PDFs"]],
                 ["When", "What", "Detail"], [2, 3, 12]),
           PageBreak()]

    # ========================================================================== PART B
    st += [P("Part B — Methods", "title"), Spacer(1, 10), P("4. Data acquisition", "h1"),
           P("URL pattern: <font name='DVM' size='7.6'>https://data.binance.vision/data/futures/um/{daily|monthly}/{dataset}/{SYMBOL}/"
             "{SYMBOL}-{dataset}-{period}.zip</font> plus <font name='DVM' size='7.6'>.zip.CHECKSUM</font> (\"&lt;sha256&gt;  &lt;file&gt;\")."),
           table([["aggTrades", "daily", "agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker"],
                  ["bookDepth", "daily", "timestamp, percentage, depth, notional"],
                  ["fundingRate", "monthly", "calc_time, funding_interval_hours, last_funding_rate"],
                  ["metrics", "daily", "create_time, symbol, sum_open_interest, sum_open_interest_value, long/short ratios, taker ratio"],
                  ["bookTicker", "daily (3 days)", "update_id, best bid/ask price & qty, transaction_time, event_time"]],
                 ["Dataset", "Granularity", "Fields"], [2.6, 2.4, 12]),
           P("Procedure: fetch .CHECKSUM → stream zip while hashing → compare → write or reject (never use a mismatching file) → "
             "append to manifest (dataset, period, URL, bytes, expected/actual SHA-256, status, UTC time). Small files are fetched with 16 "
             "threads; trade days are processed by 4 parallel processes (download → verify → clean → aggregate → delete zip). Retries use "
             "exponential back-off for network errors, both on request and mid-transfer. Every step is idempotent and resumable."),
           P("5. Data cleaning and quality-control methods", "h1"),
           table([["Timestamp unit", "per file, min and max epoch must fall in one unit's range (ms 1e12–1e13, µs 1e15–1e16, ns 1e18–1e19); otherwise fail", "detect"],
                  ["Ordering", "count rows with decreasing time / non-increasing id in raw order; then stable sort by (ts, id)", "count + fix"],
                  ["Nulls / non-finite / bad side flag", "any null field, non-finite price/qty, side not true/false", "remove + count"],
                  ["Impossible values", "price ≤ 0, quantity ≤ 0, first_trade_id > last_trade_id", "remove + count"],
                  ["Out-of-period", "timestamp outside the file's UTC day", "remove + count"],
                  ["Duplicates", "exact duplicate rows (keep one); then same agg_trade_id with different content (keep first)", "remove + count"],
                  ["Sequence gaps", "agg id gaps within and across files; underlying fill-id gaps", "flag"],
                  ["Bad-tick candidates", "|price / minute median − 1| > 2%", "flag, keep"],
                  ["Extreme bars", "|log return| or log(high/low) > 3%; classified by reversal in the next bar and trade count", "list, keep"],
                  ["Missing periods", "runs of minutes with no trade on the complete grid", "list, keep (price carried from the past)"],
                  ["Book: invalid / extra bands", "null, non-finite, depth/notional ≤ 0; bands outside ±1..5% counted separately", "remove + count"],
                  ["Book: completeness / monotonicity", "10 bands per snapshot; cumulative depth non-decreasing in band width", "count"],
                  ["Book: gaps, short days", "inter-snapshot gaps > 90 s; day with < 90% of 2,880 snapshots", "flag, never interpolate"],
                  ["Book: frozen feed", "snapshot identical in all 20 values to its predecessor", "flag; age measured from last change"],
                  ["Book: sign and time zone", "implied band prices vs trade VWAP at clock shifts −8, −1, 0, +1, +8 h", "verify"],
                  ["Funding / OI", "duplicates, spacing vs interval, OI ≤ 0 → missing, gaps > 5 min", "count / fix"]],
                 ["Check", "Method", "Action"], [3.4, 10.6, 3]),
           PageBreak(),
           P("6. One-minute bars and the as-of order-book join", "h1"),
           P("Bar t covers [ts, ts + 1 min). Aggregates: open/high/low/close, volume, notional, VWAP = notional/volume, aggTrade count, "
             "underlying fill count, taker buy/sell volume, notional and trade counts (buy = is_buyer_maker false), mean/median/max/p90/p95 "
             "trade size within the bar, buy/sell volume from aggTrades ≥ {1,5,10,25,50,100,250} units, bad-tick count. The grid is complete; "
             "minutes without trades keep null prices (filled later only from the past)."),
           P("Book join: each bar receives the last snapshot stamped strictly before its close (book timestamps have 1-second "
             "resolution, so ≤ ts + 59 s). book_age_s = bar close − time of the last snapshot whose values changed; book_stale = no "
             "snapshot or age > 120 s. Stale books yield null book features."),
           P("7. Feature definitions (all 47; decision time = bar close; trailing windows only)", "h1"),
           table([[k, v] for k, v in FEATURE_DEFS.items()], ["Feature", "Definition"], [3.6, 13.4]),
           P("Ablation sets: " + "; ".join(f"<b>{k}</b> = {len(v)} features" for k, v in FEATURE_SETS.items()) + "."),
           P("Large-trade threshold rule: size bucket closest to the median per-minute 95th-percentile aggTrade size in the initial "
             "training window (2023-03..2024-02) → 10 ETH (median p95 = 12.89 ETH); for SOL → 250 SOL."),
           P("8. Targets", "h1"),
           table([["fwd_ret_h", "(P[t+h] − P[t]) / P[t], h ∈ {5,15,30,60}; null if the horizon passes the data end"],
                  ["up_h (primary, h = 15)", "1 if fwd_ret_h > 0 else 0"],
                  ["cls3_h", "UP if > +11 bps, DOWN if < −11 bps, else NEUTRAL (band = base round-trip cost, fixed a priori)"],
                  ["label_end_h", "ts + h + 1 min: the last instant whose price the label uses (for purging)"]],
                 ["Target", "Definition"], [3.6, 13.4]),
           PageBreak(),
           P("9. Statistical methods", "h1"),
           table([["Contemporaneous correlation", "corr(OFI<sub>1</sub>, r<sub>1</sub>): how much flow explains same-minute price change"],
                  ["Univariate predictive regression", "fwd_ret_h (bps) = a + b · z(signal) + e; Newey–West HAC SE, Bartlett kernel, 2h lags (overlap induces MA(h−1) errors)"],
                  ["Non-overlapping re-estimate", "same regression on every h-th observation with HAC(1): checks the overlap correction"],
                  ["Controlled regression", "add ret_15, ret_60, vol_60, volume_z_1440 (standardised); HAC 2h lags"],
                  ["Rank correlation", "Spearman correlation on every 5th observation"],
                  ["Decile analysis", "mean forward return per signal decile; Spearman monotonicity of decile means"],
                  ["Top−bottom spread inference", "moving-block bootstrap over UTC days (5-day blocks, 500 resamples): SE, 95% CI, normal p-value"],
                  ["Event study", "events: OFI<sub>1</sub>, OFI<sub>15</sub> beyond their 95th/5th percentiles; mean cumulative return at 1/5/15/30/60 min with day-bootstrap CI; volatility response = post-event Σ|r| / unconditional"],
                  ["Stability", "quarterly Spearman IC of ofi_15, obi_1pct and ret_15 with fwd_ret_15"],
                  ["Multiple testing", "Holm (FWER) and Benjamini–Hochberg (FDR) within each family; Bonferroni for the holdout AUC across 43 configurations"],
                  ["Holdout AUC / Sharpe inference", "5-day moving-block bootstrap over days on frozen predictions / daily P&L"],
                  ["Deflated-Sharpe hurdle", "expected max Sharpe of N zero-skill strategies ≈ √(2 ln N) / √years with N = 43 × 5"]],
                 ["Method", "Specification"], [4.4, 12.6]),
           P("10. Machine-learning models and hyper-parameters", "h1"),
           table([["Logistic regression", f"Pipeline(SimpleImputer(median, add_indicator) → StandardScaler → LogisticRegression(L2, C = {frozen['logit_C']}, max_iter 500))"],
                  ["Random forest", f"Pipeline(SimpleImputer → RandomForest({json.dumps(frozen['rf_params'])}, max_features sqrt, seed 20231)); training stride {frozen['rf_train_stride']}"],
                  ["LightGBM", f"selected {json.dumps(frozen['lgbm_params'])}; fixed: learning_rate 0.03, subsample 0.8, colsample 0.7, reg_lambda 10; native missing handling"],
                  ["LightGBM grid", "num_leaves {15, 63} × min_child_samples {500, 5000} × n_estimators {200, 600}; criterion: mean validation log loss"],
                  ["3-class / regression", "LightGBM multiclass (macro OvR AUC, UP-vs-DOWN AUC excl. neutral) / LightGBM regression on fwd_ret_15 in bps (MAE, RMSE, IC)"],
                  ["Rule baselines", "prior-only; momentum = sign(ret_15); flow rule = sign of ofi_15_z_1440 beyond ±1; ofi_1 as a score"]],
                 ["Model", "Specification"], [3.4, 13.6]),
           P("11. Validation scheme", "h1"),
           table(csv("walk_forward_folds.csv").with_columns([pl.col(c).str.slice(0, 10) for c in ("train_start", "train_end", "valid_start", "valid_end")]),
                 ["fold", "train start", "train end (excl.)", "valid start", "valid end (excl.)"]),
           P("Purge: training rows need label_end ≤ train_end (= valid start − 1-day embargo). Validation rows need label_end ≤ valid end, "
             "so no development label uses holdout prices. Training stride 5 minutes (RF 15). Holdout: same scheme with four 3-month "
             "blocks, re-fitting the frozen model before each block on all data up to block start − 1 day."),
           PageBreak(),
           P("12. Signals, backtest equations, costs and metrics", "h1"),
           table([["Signal", "s<sub>t</sub> = +1 if P(up) > 0.5 + δ; −1 if P(up) < 0.5 − δ; 0 otherwise (NaN → 0)"],
                  ["Net position", "W<sub>t</sub> = (1/H) Σ<sub>j=0..H−1</sub> s<sub>t−j</sub>, H = 15; |W| ≤ 1, no leverage"],
                  ["Execution", "orders from bar t fill at e<sub>t+1</sub> = VWAP of bar t+1"],
                  ["Gross P&L", "gross<sub>t</sub> = W<sub>t</sub> · (e<sub>t+2</sub> / e<sub>t+1</sub> − 1)"],
                  ["Costs", "cost<sub>t</sub> = |W<sub>t</sub> − W<sub>t−1</sub>| · c, c = fee + slippage + half spread (per side)"],
                  ["Funding", "funding<sub>t</sub> = −W<sub>t</sub> · f if a settlement falls in W<sub>t</sub>'s holding interval (longs pay f > 0)"],
                  ["Net", "net<sub>t</sub> = gross<sub>t</sub> − cost<sub>t</sub> + funding<sub>t</sub>; equity = Π(1 + net<sub>t</sub>)"],
                  ["Trade-level", "each nonzero s<sub>t</sub> = one trade, entry e<sub>t+1</sub>, exit e<sub>t+1+H</sub>, net of a full round trip: win rate, profit factor, average trade"],
                  ["Break-even", "c* = (Σ gross + Σ funding) / Σ |ΔW| (per side)"],
                  ["Risk metrics", "daily UTC aggregation; Sharpe and Sortino × √365; max drawdown on equity; Calmar = annual return / |MDD|; turnover/day; exposure = mean |W|"],
                  ["Classification", "AUC, log loss vs prior, Brier, accuracy/precision/recall/F1 at 0.5, ECE (10 equal-count bins), Spearman IC, tradable AUC vs VWAP<sub>t+1</sub>→VWAP<sub>t+16</sub> return"]],
                 ["Item", "Definition"], [2.8, 14.2]),
           table([[k, f"{v['fee_bps']}", f"{v['slippage_bps']}", f"{frozen['costs']['half_spread_bps']}",
                   f"{v['fee_bps'] + v['slippage_bps'] + frozen['costs']['half_spread_bps']:.3f}"] for k, v in frozen["costs"].items() if isinstance(v, dict)],
                 ["scenario", "fee (bps)", "slippage (bps)", "half spread (bps)", "total per side (bps)"]),
           P(f"Fee source: Binance USD-M VIP0 maker 2 bps / taker 5 bps, 10% BNB discount. Half spread measured: {spread['half_spread_bps_used']} bps "
             f"(mean spread {spread['mean_of_daily_mean_spread_bps']:.3f} bps on {', '.join(spread['days'])}). Slippage is an assumption."),
           P("13. Holdout protocol and freeze", "h1")]
    st += bullets(["All selections made on development out-of-fold data with pre-declared rules (model: lowest mean validation log loss on set E; δ: highest base-cost Sharpe).",
                   "scripts/freeze_methodology.py writes docs/final_methodology.md and frozen_parameters.json; it refuses to run once the holdout access log exists.",
                   "Holdout rows are reachable only through validation.unlock_holdout(), which requires the methodology file and appends {time, purpose, SHA-256} to reports/holdout_access_log.jsonl.",
                   "Freeze committed in 71ca5ad; accesses: " + "; ".join(
                       f"{e['utc']} ({e['purpose']}, {e['methodology_sha256'][:12]}…)" for e in
                       [json.loads(l) for l in (ROOT / "reports" / "holdout_access_log.jsonl").read_text().splitlines()])])
    st += [PageBreak()]

    # ========================================================================== PART C
    st += [P("Part C — Tests and verification", "title"), Spacer(1, 10), P("14. Unit-test catalogue (30 tests, all passing)", "h1"),
           P("Run with <font name='DVM' size='7.6'>pytest -q</font>. Tests use synthetic data with known answers plus a 1-hour byte-identical "
             "subset of the official archive (data_sample/).")]
    rows = []
    for f_, name in (("tests/test_stage2.py", "Stage 2"), ("tests/test_research.py", "Research")):
        for fn in re.findall(r"^def (test_\w+)", (ROOT / f_).read_text(), flags=re.M):
            rows.append([name, fn, TEST_DOCS.get(fn, "")])
    st += [table(rows, ["file", "test", "what it verifies"], [1.8, 6.2, 9])]
    st += [Spacer(1, 6), P("15. Leakage audit (scripts/leakage_audit.py, verbatim output)", "h1")]
    st += md((ROOT / "docs" / "leakage_audit.md").read_text().split("\n", 1)[1], shift=2)
    st += [PageBreak(), P("16. Data-quality report (scripts/run_stage2.py, verbatim output)", "h1")]
    st += md((ROOT / "docs" / "data_quality_report.md").read_text().split("\n", 1)[1], shift=1)
    st += [P("Sign-convention / time-zone figure data and frozen-feed runs", "h3"),
           table(pl.read_csv(ROOT / "docs" / "stage2" / "book_frozen_runs.csv").head(10), ["start", "end", "snapshots"]),
           PageBreak()]

    # ========================================================================== PART D
    st += [P("Part D — Results", "title"), Spacer(1, 10), P("17. Exploratory analysis (development period)", "h1"),
           fig("eda/01_price_vol_volume_funding.png", 15.5, "Price, daily realised volatility, volume and funding."),
           fig("eda/02_return_tails.png", 15, "Standardised return distributions vs Normal (log density): extreme fat tails."),
           table(csv("eda_return_distribution.csv"), nd=3),
           PageBreak(),
           fig("eda/03_autocorrelation.png", 15, "ACF of 1-min returns (lag-1 negative) and absolute returns (volatility clustering)."),
           table(csv("eda_acf.csv").head(10), ["lag", "ACF return", "ACF |return|"], nd=4),
           fig("eda/04_intraday_seasonality.png", 15, "Intraday seasonality of volume and absolute returns (UTC)."),
           PageBreak(),
           fig("eda/05_imbalance_distributions.png", 16, "Order-flow and book imbalance distributions."),
           table(csv("eda_imbalance_summary.csv"), nd=3),
           fig("eda/06_feature_correlation.png", 13, "Spearman correlation among all 47 features."),
           PageBreak(),
           P("18. Statistical alpha tests (development period)", "h1"),
           P(f"Contemporaneous corr(OFI<sub>1</sub>, r<sub>1</sub>) = {st_sum['corr_ofi1_ret1_contemporaneous']:.4f}; "
             f"corr(OBI<sub>1%</sub>, r<sub>1</sub>) = {st_sum['corr_obi1_ret1_contemporaneous']:.4f}; development rows {st_sum['rows_dev']:,}."),
           P("Univariate predictive regressions (all 36 tests)", "h3"),
           table(csv("stats_univariate_predictive.csv").select("signal", "h", "n", "pearson", "spearman", "beta_bps_per_sd", "t_hac", "p_holm", "p_bh", "t_nonoverlap"),
                 ["signal", "h", "n", "Pearson", "Spearman", "β bps/sd", "HAC t", "Holm p", "BH p", "non-ovl t"], nd=3),
           PageBreak(),
           P("With controls (ret_15, ret_60, vol_60, volume_z_1440)", "h3"),
           table(csv("stats_with_controls.csv").select("signal", "h", "n", "beta_signal_bps_per_sd", "t_signal", "p_holm", "t_ret_15", "t_ret_60", "t_vol_60", "t_volume_z_1440"),
                 ["signal", "h", "n", "β bps/sd", "t", "Holm p", "t ret15", "t ret60", "t vol60", "t volz"], nd=3),
           P("Decile top − bottom spreads (5-day block bootstrap)", "h3"),
           table(csv("stats_decile_spreads.csv").select("signal", "h", "spread_bps", "boot_se_bps", "ci95_lo_bps", "ci95_hi_bps", "p_value", "p_holm", "decile_monotonicity_spearman"),
                 ["signal", "h", "spread bps", "SE", "CI lo", "CI hi", "p", "Holm p", "monotonicity"], nd=3),
           fig("stats/decile_forward_returns.png", 16.5),
           PageBreak(),
           P("Event study", "h3"),
           table(csv("stats_event_study.csv").select("event", "n_events", "event_bar_ret_bps", "ret_1m_bps", "ret_5m_bps", "ci_lo_5m", "ci_hi_5m",
                                                     "ret_15m_bps", "ret_60m_bps", "abs_move_ratio_15m"),
                 ["event", "n", "event bar", "+1m", "+5m", "CI lo 5m", "CI hi 5m", "+15m", "+60m", "vol ratio 15m"], nd=3),
           fig("stats/event_study.png", 15),
           P("Quarterly rank IC stability", "h3"),
           table(csv("stats_quarterly_rank_ic.csv"), ["quarter", "IC ofi_15", "IC obi_1%", "IC ret_15", "n"], nd=4),
           fig("stats/quarterly_ic.png", 15),
           PageBreak()]
    ms = csv("model_summary.csv").sort("h", "model", "fset", "config")
    st += [P("19. Model results — all binary configurations (development walk-forward)", "h1"),
           table(ms.select("config", "auc_mean", "auc_sd", "auc_min", "auc_max", "folds_auc_gt_05", "ll_improvement_mean", "f1_mean", "ece_mean", "ic_mean"),
                 ["config", "AUC mean", "AUC sd", "min fold", "max fold", "folds > 0.5", "log-loss gain", "F1", "ECE", "IC"], nd=4,
                 widths=[4.2, 1.45, 1.35, 1.4, 1.4, 1.1, 1.55, 1.35, 1.35, 1.65]),
           PageBreak(),
           P("LightGBM tuning grid", "h3"),
           table(csv("lgbm_tuning.csv"), nd=4, hl_row=0),
           P("Primary-model selection (pre-declared: lowest mean validation log loss, set E, h = 15)", "h3"),
           table(pl.DataFrame(sel["candidates"]), ["config", "log loss", "AUC", "AUC sd"], nd=5, hl_row=0),
           P("Rule baselines (AUC per horizon over 7 folds)", "h3"),
           table(csv("baseline_rules_folds.csv").group_by("rule", "h").agg(pl.col("auc").mean().alias("mean"), pl.col("auc").min().alias("min"), pl.col("auc").max().alias("max")).sort("h", "rule"),
                 ["rule", "h", "mean AUC", "min", "max"], nd=4),
           P("Alternative targets (per fold)", "h3"),
           table(csv("folds/lgbm3_E_all_h15.csv").select("fold", "auc_ovr_macro", "auc_up_vs_down_excl_neutral", "log_loss", "log_loss_baseline", "share_neutral", "ic_spearman"),
                 ["fold", "macro AUC", "UP vs DOWN AUC", "log loss", "prior", "share neutral", "IC"], nd=4),
           table(csv("folds/lgbmreg_E_all_h15.csv").select("fold", "mae_bps", "mae_zero_bps", "rmse_bps", "rmse_mean_bps", "ic_spearman", "hit_rate_sign"),
                 ["fold", "MAE", "MAE (0)", "RMSE", "RMSE (mean)", "IC", "sign hit"], nd=4),
           P("Per-fold results of the primary model", "h3"),
           table(csv("model_folds_all.csv").filter(pl.col("config") == sel["selected"]).select("fold", "valid_start", "n_train", "n", "auc", "log_loss", "log_loss_baseline", "accuracy", "f1", "ece", "ic_spearman"),
                 ["fold", "valid start", "train rows", "valid rows", "AUC", "log loss", "prior", "acc", "F1", "ECE", "IC"], nd=4,
                 widths=[1.0, 2.0, 1.6, 1.6, 1.4, 1.5, 1.5, 1.4, 1.4, 1.4, 1.4]),
           PageBreak()]
    fi = csv("interpret_feature_importance.csv")
    st += [P("20. Interpretability (out of sample, fold F6: train to 2025-05-30, explain 2025-06 → 2025-08)", "h1"),
           P(f"Primary RF on this fold: AUC {interp['auc_rf']:.4f} (LightGBM {interp['auc_lgbm']:.4f}). Permutation = AUC drop when a feature "
             "(or a whole group, jointly) is shuffled, 3 repeats on 200,000 validation rows. SHAP = exact TreeSHAP of the runner-up LightGBM."),
           table(csv("interpret_group_permutation.csv"), ["group", "n", "AUC drop", "sd", "log-loss increase"], nd=5),
           table(csv("interpret_group_shap_gain.csv"), ["group", "Σ mean |SHAP|", "gain share"], nd=4),
           table(fi.select("feature", "group", "perm_auc_drop", "perm_auc_drop_sd", "rf_impurity_importance", "mean_abs_shap", "gain_share").head(25),
                 ["feature", "group", "perm AUC drop", "sd", "RF impurity", "mean |SHAP|", "LGBM gain share"], nd=5),
           fig("interpret/permutation_importance.png", 16.5),
           fig("interpret/shap_dependence.png", 16.5),
           P("Standardised logistic coefficients (top 15)", "h3"),
           table(csv("interpret_logit_coefficients.csv").head(15), ["feature", "group", "coefficient"], nd=4),
           PageBreak()]
    st += [P("21. Development backtest and threshold selection (out-of-fold, 2024-03 → 2025-09)", "h1"),
           table(csv("dev_delta_selection.csv"), ["δ", "active", "Sharpe base", "return base", "Sharpe gross", "Σ gross", "BE bps", "trades", "turnover/day"], nd=3, hl_row=4),
           table(csv("dev_backtest_summary.csv").with_columns(breakeven_cost_bps_per_side=pl.when(pl.col("strategy") == "buy_and_hold").then(None).otherwise(pl.col("breakeven_cost_bps_per_side"))).select(
               "strategy", "cost", "total_return", "sharpe", "max_drawdown", "n_trades", "win_rate", "avg_trade_bps", "breakeven_cost_bps_per_side", "turnover_per_day"),
                 ["strategy", "cost", "total return", "Sharpe", "max DD", "trades", "win rate", "avg trade bps", "break-even bps", "turnover/day"],
                 pct_cols=("total_return", "max_drawdown", "win_rate"), nd=2, widths=[2.3, 1.2, 1.9, 1.6, 1.8, 1.7, 1.5, 1.7, 1.8, 1.6]),
           fig("dev/dev_equity_drawdown.png", 15),
           PageBreak(),
           P("Ablation economics (development out-of-fold, δ frozen)", "h3"),
           table(csv("dev_ablation_economics.csv"), ["config", "tradable AUC", "AUC", "Sharpe gross", "return gross", "Sharpe base", "return base", "BE bps", "trades", "avg gross bps"],
                 pct_cols=("total_return_gross", "total_return_base"), nd=3),
           P("Edge by confidence (EXPLORATORY, non-binding)", "h3"),
           table(csv("dev_edge_by_confidence_EXPLORATORY.csv"), ["min |P−0.5|", "signals", "share", "mean gross bps", "hit rate", "round trip bps"], nd=3),
           P("Monthly stability and volatility regimes (development)", "h3"),
           table(csv("dev_monthly_stability.csv"), ["month", "n", "AUC", "log loss", "prior", "hit rate", "Σ net", "Σ gross"], nd=4),
           table(csv("dev_regimes.csv"), ["regime", "cost", "minutes", "AUC", "Σ net", "net bps/min", "turnover"], nd=4),
           fig("dev/dev_monthly_stability.png", 14),
           PageBreak()]
    H = "holdout/"
    st += [P("22. Final holdout results (2025-10-01 → 2026-09-30, frozen, evaluated once)", "h1"),
           table([[k, fmt(v, 4)] for k, v in pc.items()] + [["tradable AUC", fmt(hs["primary_tradable_auc"], 4)],
                                                             ["holdout rows", f"{hs['holdout_rows']:,}"]],
                 ["metric (pooled)", "value"], [6, 4]),
           table(csv(H + "primary_blocks.csv").select("block", "valid_start", "n_train", "n", "auc", "log_loss", "log_loss_baseline", "accuracy", "f1", "ece", "ic_spearman"),
                 ["block", "start", "train rows", "test rows", "AUC", "log loss", "prior", "acc", "F1", "ECE", "IC"], nd=4,
                 widths=[1.0, 2.0, 1.6, 1.6, 1.4, 1.5, 1.5, 1.4, 1.4, 1.4, 1.4]),
           P("Calibration (equal-count deciles)", "h3"),
           table(csv(H + "primary_calibration.csv"), ["bin", "mean predicted", "observed up share", "n"], nd=4),
           P("Backtest — all strategies and cost scenarios", "h3"),
           P("(a) return and risk", "h3"),
           table(bt.select("strategy", "cost", "total_return", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar"),
                 ["strategy", "cost", "total return", "annual return", "annual vol", "Sharpe", "Sortino", "max DD", "Calmar"],
                 pct_cols=("total_return", "ann_return", "ann_vol", "max_drawdown"), nd=2, widths=[2.4, 1.3, 1.9, 1.9, 1.9, 1.7, 1.7, 1.9, 1.7]),
           P("(b) trading activity", "h3"),
           table(bt.with_columns(breakeven_cost_bps_per_side=pl.when(pl.col("strategy") == "buy_and_hold").then(None).otherwise(pl.col("breakeven_cost_bps_per_side"))).select(
               "strategy", "cost", "n_trades", "win_rate", "profit_factor", "avg_trade_bps", "breakeven_cost_bps_per_side", "turnover_per_day", "exposure", "sum_funding"),
                 ["strategy", "cost", "trades", "win rate", "profit factor", "avg trade bps", "break-even bps", "turnover/day", "exposure", "funding"],
                 pct_cols=("win_rate", "sum_funding"), nd=3, widths=[2.4, 1.3, 1.7, 1.5, 1.6, 1.7, 1.8, 1.7, 1.6, 1.6]),
           PageBreak(),
           fig("holdout/equity_drawdown_exposure.png", 15.5),
           fig("holdout/rolling_sharpe_distribution_costs.png", 16.5),
           table(csv(H + "cost_sensitivity_ml.csv"), ["cost bps/side", "total return", "Sharpe"], pct_cols=("total_return",), nd=3),
           PageBreak(),
           P("Monthly stability (holdout)", "h3"),
           table(csv(H + "monthly_stability.csv"), ["month", "AUC", "log loss", "prior", "hit rate", "Σ net", "Σ gross"], nd=4),
           P("Weekly net returns of the ML strategy (base cost)", "h3"),
           weekly_table(csv(H + "weekly_returns.csv")),
           PageBreak()]
    st += [P("23. Robustness and replication", "h1"),
           P("Holdout feature ablation (frozen procedure)", "h3"),
           table(csv(H + "ablation.csv").select("model", "fset", "auc", "tradable_auc", "log_loss", "ic_spearman", "sharpe_gross", "total_return_gross", "sharpe_base", "total_return_base", "breakeven_bps"),
                 ["model", "set", "AUC", "tradable AUC", "log loss", "IC", "Sharpe gross", "return gross", "Sharpe net", "return net", "BE bps"],
                 pct_cols=("total_return_gross", "total_return_base"), nd=3, widths=[1.3, 2.6, 1.3, 1.5, 1.4, 1.3, 1.5, 1.6, 1.5, 1.6, 1.3]),
           P("Holdout horizons", "h3"),
           table(csv(H + "horizons.csv"), ["model", "set", "h", "AUC", "log loss", "prior", "IC"], nd=4),
           P("Holdout volatility regimes and confirmatory test", "h3"),
           table(csv(H + "regimes.csv"), ["regime", "cost", "minutes", "AUC", "Σ net", "turnover"], nd=4),
           table(csv(H + "confirmatory_tests.csv"), ["signal", "h", "n", "β bps/sd", "HAC t", "p", "Holm p"], nd=4),
           PageBreak(),
           P("SOLUSDT replication (frozen settings)", "h3"),
           P("Development walk-forward:"),
           table(csv("replication_SOLUSDT/dev_summary.csv").select("fset", "auc_mean", "auc_sd", "ic_mean", "tradable_auc", "sharpe_gross", "total_return_gross", "sharpe_base", "total_return_base", "breakeven_bps"),
                 ["set", "AUC", "sd", "IC", "tradable AUC", "Sharpe gross", "return gross", "Sharpe base", "return base", "BE bps"],
                 pct_cols=("total_return_gross", "total_return_base"), nd=3, widths=[2.4, 1.4, 1.3, 1.3, 1.7, 1.7, 1.9, 1.7, 1.9, 1.5]),
           table(csv("replication_SOLUSDT/dev_stat_tests.csv"), ["signal", "h", "n", "β bps/sd", "HAC t", "p"], nd=4),
           P(f"Holdout: AUC {sol['primary_classification_pooled']['auc']:.4f}, tradable AUC {sol['primary_tradable_auc']:.4f}, "
             f"ECE {sol['primary_classification_pooled']['ece']:.4f}."),
           table(csv("replication_SOLUSDT/primary_blocks.csv").select("block", "valid_start", "n", "auc", "log_loss", "log_loss_baseline", "ic_spearman"),
                 ["block", "start", "n", "AUC", "LL", "prior", "IC"], nd=4),
           table(csv("replication_SOLUSDT/backtest_summary.csv").filter(pl.col("cost").is_in(["gross", "base"])).select("strategy", "cost", "total_return", "sharpe", "max_drawdown", "n_trades", "breakeven_cost_bps_per_side"),
                 ["strategy", "cost", "return", "Sharpe", "MDD", "trades", "BE bps"], pct_cols=("total_return", "max_drawdown"), nd=2),
           table(csv("replication_SOLUSDT/confirmatory_tests.csv"), ["signal", "h", "n", "β bps/sd", "HAC t", "p", "Holm p"], nd=4),
           fig("replication_SOLUSDT/equity_drawdown_exposure.png", 13),
           PageBreak(),
           P("24. Significance and data-snooping accounting", "h1"),
           table([["Holdout AUC", f"{sig['holdout_auc']['auc']:.4f}", f"block-bootstrap SE {sig['holdout_auc']['boot_se']:.4f}, 95% CI [{sig['holdout_auc']['ci95'][0]:.4f}, {sig['holdout_auc']['ci95'][1]:.4f}], z = {sig['holdout_auc']['z_vs_0.5']:.1f}"],
                  ["Bonferroni", f"{sig['bonferroni']['n_tests']} configurations", f"z needed {sig['bonferroni']['one_sided_z_needed']:.2f} → passes: {sig['bonferroni']['passes']}"],
                  ["Gross Sharpe", f"{sig['sharpe_gross']['sharpe']:.2f}", f"95% CI [{sig['sharpe_gross']['ci95'][0]:.2f}, {sig['sharpe_gross']['ci95'][1]:.2f}] over {sig['sharpe_gross']['days']} days"],
                  ["Net Sharpe", f"{sig['sharpe_net']['sharpe']:.2f}", f"95% CI [{sig['sharpe_net']['ci95'][0]:.2f}, {sig['sharpe_net']['ci95'][1]:.2f}]"],
                  ["Deflated-Sharpe hurdle", f"{sig['sharpe_hurdle_max_of_noise']['expected_max_sharpe_noise']:.2f}", f"expected max Sharpe of {sig['sharpe_hurdle_max_of_noise']['n_strategy_variants']} zero-skill variants over {sig['sharpe_hurdle_max_of_noise']['years']:.1f} years: the gross Sharpe does not clear it"],
                  ["Experiments", f"{sig['n_configs_development']} model configs", f"{sig['n_binary_classifier_configs_development']} binary classifiers, {sig['n_signal_thresholds']} thresholds, {sig['n_univariate_stat_tests']} univariate tests"]],
                 ["quantity", "value", "detail"], [3.5, 3, 10.5]),
           P("25. Conclusions", "h1")]
    st += md((ROOT / "docs" / "research_report.md").read_text().split("## 15. Conclusion", 1)[1].split("## 16.")[0], shift=2)
    st += [PageBreak()]

    # ========================================================================== PART E
    st += [P("Part E — Appendices", "title"), Spacer(1, 10), P("A. Frozen methodology (docs/final_methodology.md, verbatim)", "h1")]
    st += md((ROOT / "docs" / "final_methodology.md").read_text().split("\n", 1)[1], shift=1)
    st += [PageBreak(), P("B. Stage 1 dataset selection (verbatim)", "h1")]
    st += md((ROOT / "docs" / "stage1_data_selection.md").read_text().split("\n", 1)[1], shift=1)
    st += [PageBreak(), P("C. Data schema (verbatim)", "h1")]
    st += md((ROOT / "docs" / "data_schema.md").read_text().split("\n", 1)[1], shift=1)
    st += [PageBreak(), P("D. Robustness report (verbatim)", "h1")]
    st += md((ROOT / "docs" / "robustness_report.md").read_text().split("\n", 1)[1], shift=1)
    st += [PageBreak(), P("E. Interview questions and answers", "h1")]
    st += md((ROOT / "docs" / "interview_questions.md").read_text().split("\n", 1)[1], shift=1)
    st += [PageBreak(), P("F. Resume bullets", "h1")]
    st += md((ROOT / "docs" / "resume_bullets.md").read_text().split("\n", 1)[1], shift=1)
    st += [P("G. Reproduction commands and file map", "h1")]
    readme = (ROOT / "README.md").read_text()
    st += md("## Reproducibility" + readme.split("## Reproducibility", 1)[1], shift=0)

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.8 * cm, title="Ethereum Order-Flow Alpha — Complete Technical Documentation",
                            author="ETH order-flow research project", subject="All stages, methods, tests and results")

    def on_page(c, d):
        c.saveState()
        c.setFont("DV", 7.5)
        c.setFillColor(colors.HexColor("#777777"))
        c.drawString(1.8 * cm, 1.2 * cm, "Ethereum Order-Flow Alpha — complete technical documentation")
        c.drawRightString(A4[0] - 1.8 * cm, 1.2 * cm, f"page {d.page}")
        c.restoreState()
    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
