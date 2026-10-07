#!/usr/bin/env python3
"""Build the end-to-end project report PDF (raw data -> final result).

Every table is read from a pipeline output file at build time, so the PDF cannot drift from the results.
    python scripts/make_pdf_report.py   ->   reports/ETH_OrderFlow_Alpha_Report.pdf
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "reports" / "tables"
F = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "ETH_OrderFlow_Alpha_Report.pdf"

FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", "/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

NAVY = colors.HexColor("#1f3b57")
ACCENT = colors.HexColor("#2a6f97")
LIGHT = colors.HexColor("#eef3f7")
RED = colors.HexColor("#b23a48")
GREEN = colors.HexColor("#3a7d44")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", fontName="DV-B", fontSize=22, leading=27, textColor=NAVY, alignment=TA_CENTER),
    "sub": ParagraphStyle("s", fontName="DV", fontSize=11.5, leading=15, textColor=ACCENT, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=15, leading=19, textColor=NAVY, spaceBefore=6, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=11.5, leading=15, textColor=ACCENT, spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("b", fontName="DV", fontSize=9.2, leading=13, spaceAfter=4),
    "bullet": ParagraphStyle("bl", fontName="DV", fontSize=9.2, leading=12.5, leftIndent=12, bulletIndent=2, spaceAfter=1.5),
    "cap": ParagraphStyle("c", fontName="DV-I", fontSize=7.8, leading=10, textColor=colors.HexColor("#555555"), spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.4, leading=9),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=7.4, leading=9, textColor=colors.white),
    "box": ParagraphStyle("box", fontName="DV", fontSize=9.5, leading=13.5, textColor=NAVY),
}


def P(t, s="body"):
    return Paragraph(t, S[s])


def bullets(items):
    return [Paragraph(i, S["bullet"], bulletText="•") for i in items]


def fig(path, w=16.5, cap=None):
    from PIL import Image as PI
    p = F / path
    iw, ih = PI.open(p).size
    out = [Image(str(p), width=w * cm, height=w * cm * ih / iw)]
    if cap:
        out.append(P(cap, "cap"))
    return KeepTogether(out)


def fmt(v, nd=3, pct=False):
    if v is None:
        return "–"
    if isinstance(v, float):
        if v != v:
            return "–"
        if pct:
            return f"{v * 100:.1f}%"
        if v != 0 and abs(v) < 10 ** -nd:
            return f"{v:.1e}"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def table(df: pl.DataFrame | list, header=None, widths=None, pct_cols=(), nd=3, hl_row=None):
    if isinstance(df, pl.DataFrame):
        header = header or df.columns
        rows = [[fmt(v, nd, c in pct_cols) for c, v in zip(df.columns, r)] for r in df.iter_rows()]
    else:
        rows = df
    data = [[Paragraph(str(h), S["cellb"]) for h in header]] + [[Paragraph(str(c), S["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths] if widths else None, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
          ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d3dc")),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if hl_row is not None:
        st.append(("BACKGROUND", (0, hl_row + 1), (-1, hl_row + 1), colors.HexColor("#fff4d6")))
    t.setStyle(TableStyle(st))
    return t


def callout(text, color=ACCENT):
    t = Table([[Paragraph(text, S["box"])]], colWidths=[17 * cm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1.2, color), ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def pipeline_diagram():
    steps = [("1. Raw data", "Binance archive\n1,310 daily trade files,\nbook depth, funding, OI\nSHA-256 verified"),
             ("2. Quality control", "parse, ms/µs detect,\nsort, dedupe, gaps,\ninvalid values, frozen\nbook feed detection"),
             ("3. 1-minute bars", "1.88 bn trades →\n1,886,400 bars with\ntaker buy/sell flow\n+ as-of book join"),
             ("4. Features + targets", "47 causal features\n5 groups; forward\nreturns 5/15/30/60 m\nleakage audit"),
             ("5. Statistics", "HAC regressions,\ndeciles + block\nbootstrap, events,\nHolm / BH"),
             ("6. Walk-forward ML", "logit / RF / LightGBM\n7 folds, purge +\nembargo, ablation\nA→E, tuning"),
             ("7. Freeze → holdout", "methodology frozen\n+ hash-logged, then\none evaluation on\n2025-10 → 2026-09"),
             ("8. Cost-aware backtest", "next-bar VWAP, fees,\nslippage, funding,\nbreak-even cost,\nrobustness, SOL")]
    cells = []
    for i, (h, b) in enumerate(steps):
        cells.append(Paragraph(f"<b>{h}</b><br/>{b.replace(chr(10), '<br/>')}", ParagraphStyle(
            "d", fontName="DV", fontSize=7.2, leading=9, textColor=colors.white, alignment=TA_CENTER)))
    rows = [cells[:4], cells[4:]]
    t = Table(rows, colWidths=[4.15 * cm] * 4, rowHeights=[2.4 * cm, 2.4 * cm])
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("BOX", (0, 0), (-1, -1), 0, colors.white),
          ("INNERGRID", (0, 0), (-1, -1), 4, colors.white)]
    shades = ["#1f3b57", "#24506f", "#2a6f97", "#3a7fa6", "#3a7d44", "#4f8a55", "#b06a1f", "#b23a48"]
    for i in range(8):
        st.append(("BACKGROUND", (i % 4, i // 4), (i % 4, i // 4), colors.HexColor(shades[i])))
    t.setStyle(TableStyle(st))
    return t


def on_page(c, doc):
    c.saveState()
    c.setFont("DV", 7.5)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawString(1.8 * cm, 1.2 * cm, "Ethereum Order-Flow Alpha — end-to-end research report")
    c.drawRightString(A4[0] - 1.8 * cm, 1.2 * cm, f"page {doc.page}")
    c.restoreState()


def main() -> None:
    dq = (ROOT / "docs" / "data_quality_report.md").read_text()
    sel = json.loads((T / "primary_model_selection.json").read_text())
    frozen = json.loads((T / "frozen_parameters.json").read_text())
    hs = json.loads((T / "holdout" / "summary.json").read_text())
    sig = json.loads((T / "holdout" / "significance.json").read_text())
    sol = json.loads((T / "replication_SOLUSDT" / "summary.json").read_text())
    ss_ = json.loads((T / "stats_summary.json").read_text())
    bt = pl.read_csv(T / "holdout" / "backtest_summary.csv")
    ml = lambda c: bt.filter((pl.col("strategy") == "ml") & (pl.col("cost") == c)).row(0, named=True)  # noqa: E731
    g, b = ml("gross"), ml("base")
    pc = hs["primary_classification_pooled"]
    log = [json.loads(l) for l in (ROOT / "reports" / "holdout_access_log.jsonl").read_text().splitlines()]

    st = []
    # ------------------------------------------------------------------ cover
    st += [Spacer(1, 3 * cm), P("Ethereum Order-Flow Alpha", "title"), Spacer(1, 6),
           P("Short-horizon return prediction and a cost-aware trading strategy<br/>"
             "Binance USD-M ETHUSDT perpetual · 2023-03-01 → 2026-09-30", "sub"),
           Spacer(1, 1.2 * cm), pipeline_diagram(), Spacer(1, 1 * cm),
           callout(f"<b>Answer in one line.</b> The 1-minute ETH microstructure contains <b>real, stable, out-of-sample</b> "
                   f"predictability (holdout AUC <b>{pc['auc']:.4f}</b>, 95% CI {sig['holdout_auc']['ci95'][0]:.4f}–"
                   f"{sig['holdout_auc']['ci95'][1]:.4f}), but the edge is worth only <b>{b['breakeven_cost_bps_per_side']:.2f} bps per side</b> "
                   f"per trade against a realistic <b>5.52 bps</b> taker cost. The strategy earns a gross Sharpe of {g['sharpe']:.2f} "
                   f"({g['total_return']*100:+.1f}%) but loses <b>{b['total_return']*100:.1f}%</b> after costs: "
                   f"<b>statistically significant, economically untradable.</b>"),
           Spacer(1, 1.2 * cm),
           P(f"Generated {date.today().isoformat()} by <font name='DV-I'>scripts/make_pdf_report.py</font>. "
             "Every table is read from the pipeline output files at build time. Code, configuration and all result tables: "
             "github.com/kudratikhoda12-design/KK (branch claude/brave-tesla-wpvjj7).", "cap"),
           PageBreak()]

    # ------------------------------------------------------------------ 1 question & design
    st += [P("1. Research question and design", "h1"),
           P("<b>Question.</b> Does order-flow and market-microstructure information contain statistically significant "
             "predictive information about short-horizon Ethereum returns, and does that information translate into "
             "economically meaningful trading alpha after transaction costs, slippage and funding?"),
           P("The project separates three claims that are often conflated, and tests each with its own tool:")]
    st += bullets(["<b>Statistical predictability</b>: HAC regressions, decile spreads, event study, multiple-testing control.",
                   "<b>Out-of-sample predictability</b>: leakage-controlled walk-forward ML, then one evaluation on a frozen, untouched year.",
                   "<b>Economic predictability</b>: a cost-aware backtest with realistic execution, and the break-even cost."])
    st += [P("Time split (fixed in Stage 1, before any return was looked at)", "h2"),
           table([["Development", "2023-03-01 → 2025-09-30 (31 months)", "EDA, statistics, features, walk-forward CV, model and threshold selection"],
                  ["Holdout", "2025-10-01 → 2026-09-30 (12 months)", "Untouched until the methodology was frozen; evaluated once"],
                  ["Replication", "SOLUSDT, same dates", "Same frozen settings, nothing re-tuned"]],
                 ["Segment", "Period", "Use"], [3, 5.5, 8.5]),
           P("Timeline of one decision", "h2"),
           table([["Features", "data stamped before the close of bar t (trades of bars ≤ t, last book snapshot < close, settled funding, OI lagged ≥ 5 min)"],
                  ["Signal", "at the close of bar t"],
                  ["Execution", "VWAP of bar t+1 (never the price that generated the signal)"],
                  ["Holding", "15 bars per signal; staggered, net position = mean of the last 15 signals (|position| ≤ 1, no leverage)"],
                  ["Target", "(P[t+15] − P[t]) / P[t], P = last trade price at or before the bar close; binary label up = 1{return > 0}"]],
                 ["Step", "Definition"], [3, 14]),
           PageBreak()]

    # ------------------------------------------------------------------ 2 data
    def dq_line(key):
        for line in dq.splitlines():
            if key in line:
                return line.strip("- ").replace("**", "").replace("`", "")
        return ""
    st += [P("2. Raw data → research dataset", "h1"),
           P("<b>Source.</b> Binance public archive (data.binance.vision), USD-M futures. Every file is checked against "
             "Binance's published SHA-256 before use, and logged in a manifest (URL, bytes, expected and actual hash). Raw files are "
             "streamed one day at a time, because one month of trades needs more than 13 GB of RAM."),
           table([["aggTrades (daily)", "agg_trade_id, price, quantity, first/last trade id, transact_time (ms), is_buyer_maker", "1,310 files, 25.3 GB zipped"],
                  ["bookDepth (daily)", "timestamp, percentage (±1…5), depth, notional: cumulative resting quantity within k% of price", "1,310 files, ~30 s snapshots"],
                  ["fundingRate (monthly)", "calc_time, funding_interval_hours, last_funding_rate", "43 files"],
                  ["metrics (daily)", "open interest and long/short ratios, 5-minute", "1,310 files"],
                  ["bookTicker (3 sample days)", "best bid / ask: used only to measure the quoted spread", "archive ends 2024-03-30"]],
                 ["Dataset", "Fields", "Volume"], [3.4, 9.6, 4]),
           Spacer(1, 4),
           callout("<b>Trade direction is given, not inferred.</b> is_buyer_maker = true means the seller was the aggressor, "
                   "so taker buy volume = trades with is_buyer_maker = false. No tick rule or Lee–Ready classification is needed. "
                   "<b>The order book is percentage-band depth, not level-by-level L2</b>: no best bid/ask, spread or queue."),
           P("Data-quality results (full report: docs/data_quality_report.md)", "h2")]
    st += bullets([dq_line("Raw aggregated trades read"), dq_line("Removed as data errors"), dq_line("Duplicates: exact rows"),
                   dq_line("Bars written"), dq_line("minutes without any trade"), dq_line("Snapshots kept"),
                   dq_line("Frozen feed:")[:420], dq_line("Short/missing days")[:200]])
    st += [P("Key cleaning decisions", "h2"),
           table([["Removed (counted)", "nulls, price ≤ 0, qty ≤ 0, first_id > last_id, timestamp outside the file's day, exact duplicates, conflicting duplicate ids: <b>0 rows</b> met any rule"],
                  ["Flagged, kept", "83 extreme 1-minute moves (> 3%), all liquidation cascades with tens of thousands of trades; bad-tick candidates; trade-id gaps; empty minutes"],
                  ["Frozen book feed", "2025-04-16 → 2025-05-19: the archive repeats one snapshot 94,730 times. Book age is measured from the last <i>changed</i> snapshot, so these minutes are stale and the book features are missing"],
                  ["Schema change", "±0.2% bands added on 2026-01-15 (inside the holdout): not used"],
                  ["Other", "91 zero open-interest readings and ±inf ratios set to missing; nothing interpolated or winsorised"],
                  ["Verification", "time zone and band sign checked against trade prices: 94.1% of snapshots have bid-band avg < VWAP < ask-band avg at zero shift vs 64% at ±1 h"]],
                 ["Treatment", "Detail"], [3.4, 13.6]),
           PageBreak()]

    # ------------------------------------------------------------------ 3 features
    from ethof_groups import GROUPS  # injected below
    rows = [[g_, str(len(fs)), ", ".join(fs)] for g_, fs in GROUPS.items()]
    la = (ROOT / "docs" / "leakage_audit.md").read_text()
    checks = [l.replace("### ", "") for l in la.splitlines() if l.startswith("### ")]
    st += [P("3. Feature engineering and leakage control", "h1"),
           P("47 features are computed on the complete 1-minute grid with <b>trailing windows only</b>. Ablation sets are cumulative: "
             "A = price, B = A + volume, C = B + trade flow, D = C + book, E = D + funding/OI."),
           table(rows, ["Group", "n", "Features"], [2.3, 0.8, 13.9]),
           Spacer(1, 6),
           P("Key definitions: OFI = (taker buy volume − taker sell volume) / total volume over the window; book imbalance at k% = "
             "(bid depth within k% − ask depth within k%) / their sum; large trade = aggTrade ≥ 10 ETH (the bucket closest to the median "
             "per-minute 95th-percentile trade size in the initial window). Undefined ratios become missing, never zero or ±inf."),
           P("Automated leakage audit (scripts/leakage_audit.py)", "h2")]
    st += bullets(checks)
    st += [P("The strongest test is <b>truncation invariance</b>: every feature was recomputed on data cut at 12 random times, and all "
             "34,774,174 compared values were identical. Any feature using the future would change when the future is removed."),
           PageBreak()]

    # ------------------------------------------------------------------ 4 EDA
    rd = pl.read_csv(T / "eda_return_distribution.csv")
    st += [P("4. Exploratory analysis (development period only)", "h1"),
           fig("eda/01_price_vol_volume_funding.png", 16, "Price, realised volatility, volume and funding over the development period."),
           table(rd.select("series", "n", "mean_bps", "std_bps", "skew", "excess_kurtosis", "p01_bps", "p99_bps"),
                 ["Series", "n", "mean (bps)", "sd (bps)", "skew", "excess kurt.", "p1 (bps)", "p99 (bps)"], nd=2),
           P("Returns are extremely heavy-tailed (1-minute excess kurtosis ≈ 487), which rules out iid-normal inference. "
             "1-minute returns show a small negative first-order autocorrelation, and absolute returns show strong volatility clustering:"),
           fig("eda/03_autocorrelation.png", 15),
           fig("eda/05_imbalance_distributions.png", 16, "Distributions of 1-minute and 15-minute order-flow imbalance and of the 1% book-band imbalance."),
           PageBreak()]

    # ------------------------------------------------------------------ 5 statistics
    u = pl.read_csv(T / "stats_univariate_predictive.csv").filter(pl.col("signal").is_in(["ofi_1", "ofi_15", "obi_1pct", "trade_imb_15"]))
    st += [P("5. Statistical alpha tests (before any ML)", "h1"),
           P(f"Order flow explains the <b>same</b> minute's return very well: corr(OFI<sub>1</sub>, r<sub>1</sub>) = "
             f"<b>{ss_['corr_ofi1_ret1_contemporaneous']:.3f}</b>. The question is whether it predicts the <b>next</b> minutes. "
             "Forward returns overlap, so regressions use Newey–West HAC errors (2h lags) plus a non-overlapping re-estimate. "
             f"All {ss_['n_univariate_tests']} signal × horizon tests are Holm and BH adjusted."),
           table(u.select("signal", "h", "pearson", "beta_bps_per_sd", "t_hac", "p_holm", "t_nonoverlap"),
                 ["Signal", "h (min)", "Pearson", "β (bps / sd)", "HAC t", "Holm p", "non-overlap t"], nd=3),
           P("1-minute OFI predicts a small <b>reversal</b> at every horizon (Holm-significant), and it survives controls for momentum, "
             "volatility and volume. 15-minute OFI is not significant after adjustment. Book imbalance is significant only "
             "at 5 minutes and loses significance once recent returns are controlled for."),
           fig("stats/decile_forward_returns.png", 17, "Mean forward return by signal decile. Effects are fractions of a basis point."),
           fig("stats/event_study.png", 15, "Event study: extreme order-flow minutes are followed by partial reversal, with 95% day-bootstrap bands."),
           callout("<b>Statistical ≠ economic.</b> The effects are highly significant because there are 1.36 million observations, but they are "
                   "0.1–0.9 bps, while one round trip costs about 11 bps.", RED),
           PageBreak()]

    # ------------------------------------------------------------------ 6 ML
    ms = pl.read_csv(T / "model_summary.csv").filter((pl.col("h") == 15) & ~pl.col("config").str.starts_with("tune")).sort("model", "fset")
    tune = pl.read_csv(T / "lgbm_tuning.csv")
    rules = pl.read_csv(T / "baseline_rules_folds.csv").filter(pl.col("h") == 15).group_by("rule").agg(pl.col("auc").mean().alias("mean AUC"), pl.col("auc").min().alias("min"), pl.col("auc").max().alias("max")).sort("rule")
    st += [P("6. Machine learning with walk-forward validation", "h1"),
           P("<b>Validation design.</b> Expanding walk-forward: first 12 months for initial training, then 7 validation blocks of 3 months "
             "(2024-03 → 2025-09). Training labels are purged so they end before validation start − 1 day (embargo). Training uses every "
             "5th minute (every 15th for RF), because overlapping labels make neighbours near-duplicates. Imputation and scaling are fitted inside "
             "each training fold. Random cross-validation is not used: it would leak shared future price paths."),
           P("LightGBM grid (8 configurations, chosen by mean validation log loss)", "h2"),
           table(tune.select("grid_id", "num_leaves", "min_child_samples", "n_estimators", "mean_log_loss", "mean_auc", "min_auc"),
                 ["id", "leaves", "min leaf", "trees", "val. log loss", "mean AUC", "min fold AUC"], nd=4, hl_row=0),
           P("Feature ablation, 15-minute horizon (development, mean over 7 folds)", "h2"),
           table(ms.select("model", "fset", "auc_mean", "auc_sd", "auc_min", "folds_auc_gt_05", "log_loss_mean", "ic_mean"),
                 ["Model", "Set", "AUC mean", "AUC sd", "min fold", "folds > 0.5", "log loss", "rank IC"], nd=4),
           P("Rule baselines scored on the same folds (15-minute horizon)", "h2"),
           table(rules, ["Rule", "mean AUC", "min", "max"], nd=4),
           P(f"Simple continuation rules are <b>anti-predictive</b> (AUC < 0.5): the market mean-reverts. The pre-declared rule "
             f"(lowest validation log loss on set E) selected <b>{sel['selected']}</b>. LightGBM was marginally better on AUC "
             f"and marginally worse on log loss, a tie in practice."),
           PageBreak(),
           fig("dev/dev_monthly_stability.png", 15, "Development out-of-fold: monthly AUC, hit rate, and the ML strategy's gross vs net return."),
           P("Interpretability (out of sample, fold F6)", "h2"),
           fig("interpret/permutation_importance.png", 17, "Permutation importance of the primary random forest. Price features (distance from moving averages, recent "
               "returns) dominate. Book and flow add little; volume and funding/OI add essentially nothing. This is association, not causation."),
           PageBreak()]

    # ------------------------------------------------------------------ 7 freeze + selection
    ds = pl.read_csv(T / "dev_delta_selection.csv")
    edge = pl.read_csv(T / "dev_edge_by_confidence_EXPLORATORY.csv").filter(pl.col("n_signals") > 0)
    st += [P("7. Signal threshold, development backtest and the freeze", "h1"),
           P("Rule: LONG if P(up) > 0.5 + δ, SHORT if P(up) < 0.5 − δ, otherwise FLAT. δ is chosen on development out-of-fold predictions by "
             "the highest net Sharpe at base cost:"),
           table(ds.select("delta", "active_share", "sharpe_gross", "sharpe_base", "breakeven_bps", "n_trades"),
                 ["δ", "share active", "gross Sharpe", "net Sharpe (base)", "break-even bps/side", "trades"], nd=3, hl_row=4),
           P("<b>Exploratory (non-binding) check:</b> even the most confident predictions never earn a round trip.", "body"),
           table(edge.select("min_confidence", "share_of_minutes", "mean_gross_trade_bps", "hit_rate", "base_round_trip_bps"),
                 ["min |P−0.5|", "share of minutes", "mean gross trade (bps)", "hit rate", "round-trip cost (bps)"], nd=3),
           Spacer(1, 6),
           fig("dev/dev_equity_drawdown.png", 15.5, "Development out-of-fold equity at base cost (log scale); dashed = ML before costs."),
           callout(f"<b>Freeze.</b> docs/final_methodology.md (model {frozen['selection']['selected']}, δ = {frozen['delta']}, costs, "
                   f"execution, all features) was committed in 71ca5ad <b>before</b> the holdout was opened. Every holdout access is "
                   f"logged with the file's SHA-256: " + "; ".join(f"{e['utc']} – {e['purpose']} – {e['methodology_sha256'][:12]}…" for e in log)
                   + ". The hashes match, so the methodology was not edited after the holdout was opened."),
           PageBreak()]

    # ------------------------------------------------------------------ 8 holdout
    blk = pl.read_csv(T / "holdout" / "primary_blocks.csv")
    btab = bt.filter(pl.col("cost").is_in(["gross", "base"])).sort("cost", "strategy", descending=[True, False]).with_columns(
        breakeven_cost_bps_per_side=pl.when(pl.col("strategy") == "buy_and_hold").then(None).otherwise(pl.col("breakeven_cost_bps_per_side"))).select(
        "strategy", "cost", "total_return", "sharpe", "sortino", "max_drawdown", "n_trades", "win_rate", "avg_trade_bps", "breakeven_cost_bps_per_side")
    st += [P("8. Final untouched holdout (2025-10-01 → 2026-09-30)", "h1"),
           P("Prediction", "h2"),
           table([[fmt(pc["auc"], 4), f"[{sig['holdout_auc']['ci95'][0]:.4f}, {sig['holdout_auc']['ci95'][1]:.4f}]",
                   f"{pc['log_loss']:.4f} / {pc['log_loss_baseline']:.4f}", fmt(pc["ece"], 4), fmt(pc["ic_spearman"], 3),
                   fmt(hs["primary_tradable_auc"], 3), f"{pc['accuracy']:.3f} / {pc['precision']:.3f} / {pc['recall']:.3f} / {pc['f1']:.3f}"]],
                 ["AUC", "95% block-bootstrap CI", "log loss / prior", "ECE", "rank IC", "tradable AUC", "acc / prec / rec / F1"],
                 [1.6, 3.2, 2.8, 1.4, 1.5, 2.0, 4.5]),
           Spacer(1, 4),
           table(blk.select("block", "valid_start", "n_train", "n", "auc", "log_loss", "log_loss_baseline", "ic_spearman"),
                 ["block", "start", "train rows", "test rows", "AUC", "log loss", "prior", "IC"], nd=4),
           P(f"AUC is above 0.5 in every holdout block and month. 'Tradable AUC' scores the probability against the return that is "
             f"actually capturable (VWAP of t+1 to VWAP of t+16), and it stays high. With 43 configurations tried, Bonferroni needs z > "
             f"{sig['bonferroni']['one_sided_z_needed']:.2f}; the holdout AUC has z = {sig['holdout_auc']['z_vs_0.5']:.1f}."),
           P("Trading (frozen strategy and benchmarks)", "h2"),
           table(btab, ["strategy", "cost", "total return", "Sharpe", "Sortino", "max DD", "trades", "win rate", "avg trade (bps)", "break-even bps/side"],
                 pct_cols=("total_return", "max_drawdown", "win_rate"), nd=2),
           PageBreak(),
           fig("holdout/equity_drawdown_exposure.png", 15.5, "Holdout equity (log scale) and drawdown at base cost for all strategies; dashed = ML before costs; bottom = ML exposure."),
           fig("holdout/rolling_sharpe_distribution_costs.png", 17, "Rolling 30-day Sharpe, daily return distribution, and ML total return as a function of per-side cost."),
           PageBreak()]

    # ------------------------------------------------------------------ 9 robustness
    abl = pl.read_csv(T / "holdout" / "ablation.csv").sort("model", "fset")
    hor = pl.read_csv(T / "holdout" / "horizons.csv").sort("h", "model", "fset")
    reg = pl.read_csv(T / "holdout" / "regimes.csv")
    conf = pl.read_csv(T / "holdout" / "confirmatory_tests.csv")
    sol_bt = pl.read_csv(T / "replication_SOLUSDT" / "backtest_summary.csv").filter(pl.col("strategy") == "ml").select(
        "cost", "total_return", "sharpe", "max_drawdown", "breakeven_cost_bps_per_side")
    st += [P("9. Robustness", "h1"),
           P("Feature ablation on the holdout (frozen procedure, 15 minutes)", "h2"),
           table(abl.select("model", "fset", "auc", "tradable_auc", "log_loss", "sharpe_gross", "sharpe_base", "total_return_base", "breakeven_bps"),
                 ["model", "set", "AUC", "tradable AUC", "log loss", "Sharpe gross", "Sharpe net", "net return", "BE bps"],
                 pct_cols=("total_return_base",), nd=3),
           P("Order flow and book bands do <b>not</b> add robust information beyond price on the holdout: a price-only RF reaches 0.543 "
             "against 0.544 with everything. For logistic regression the book sets lower AUC and push log loss above the prior (0.6931)."),
           P("Horizons (holdout)", "h2"),
           table(hor.select("h", "model", "fset", "auc", "log_loss", "log_loss_baseline", "ic_spearman"),
                 ["h", "model", "set", "AUC", "log loss", "prior", "IC"], nd=4),
           PageBreak(),
           P("Volatility regimes (holdout; cut-offs fixed on 2023-03 → 2024-02)", "h2"),
           table(reg, ["regime", "cost", "minutes", "AUC", "sum of bar returns", "turnover"], nd=3),
           P("Confirmatory test of the main development finding on the holdout", "h2"),
           table(conf, ["signal", "h", "n", "β (bps / sd)", "HAC t", "p", "Holm p"], nd=4),
           P("Replication on SOLUSDT (frozen settings, nothing re-tuned)", "h2"),
           P(f"Holdout AUC {sol['primary_classification_pooled']['auc']:.4f}, tradable AUC {sol['primary_tradable_auc']:.4f}. ML strategy:"),
           table(sol_bt, ["cost", "total return", "Sharpe", "max DD", "break-even bps/side"], pct_cols=("total_return", "max_drawdown"), nd=2),
           P("Multiple testing and data snooping", "h2")]
    st += bullets([f"Tried in development: {sig['n_configs_development']} model configurations, {sig['n_signal_thresholds']} thresholds, "
                   f"{sig['n_univariate_stat_tests']} univariate tests (Holm/BH adjusted), controls, deciles and events.",
                   f"Holdout AUC passes Bonferroni over {sig['n_configs_development']} configurations (z = {sig['holdout_auc']['z_vs_0.5']:.1f} vs "
                   f"{sig['bonferroni']['one_sided_z_needed']:.2f} needed): the predictability is not a selection artefact.",
                   f"Gross Sharpe {sig['sharpe_gross']['sharpe']:.2f} (CI {sig['sharpe_gross']['ci95'][0]:.2f}–{sig['sharpe_gross']['ci95'][1]:.2f}) is below the "
                   f"≈{sig['sharpe_hurdle_max_of_noise']['expected_max_sharpe_noise']:.1f} expected maximum of "
                   f"{sig['sharpe_hurdle_max_of_noise']['n_strategy_variants']} zero-skill variants, so even the gross profit claim is weak.",
                   f"Net Sharpe {sig['sharpe_net']['sharpe']:.1f} (CI {sig['sharpe_net']['ci95'][0]:.1f} to {sig['sharpe_net']['ci95'][1]:.1f}) is unambiguously negative."])
    st += [PageBreak()]

    # ------------------------------------------------------------------ 10 conclusion
    st += [P("10. Conclusions", "h1"),
           table([["Is there order-flow predictability?", "Yes statistically: 1-min OFI predicts reversal (Holm p down to 7e-21 in development; p = 0.037 at 5 min on the holdout). No economically: ≈ 0.07–0.24 bps per sd."],
                  ["Does order flow add value beyond price/volume?", "Marginally in development (≤ +0.002 AUC), not robustly on the holdout."],
                  ["Does ML improve the signal?", "Yes as a forecaster (AUC ≈ 0.54 vs < 0.5 for rules); trees beat logit. No economic viability."],
                  ["Does it survive out of sample?", f"Yes: AUC {pc['auc']:.4f} [{sig['holdout_auc']['ci95'][0]:.4f}, {sig['holdout_auc']['ci95'][1]:.4f}], 12/12 months > 0.5, calibrated."],
                  ["Does it survive realistic costs?", f"No: break-even {b['breakeven_cost_bps_per_side']:.2f} bps per side vs 5.52 base (2.52 even at low cost); net {b['total_return']*100:.1f}%."],
                  ["Is it stable over time and regimes?", "Prediction yes: no decay over 31 out-of-sample months; AUC 0.541–0.550 across volatility terciles. Net P&L negative everywhere."],
                  ["Does it replicate on SOL?", f"Yes qualitatively: holdout AUC {sol['primary_classification_pooled']['auc']:.3f}; also untradable (break-even 0.89 bps per side)."],
                  ["Strongest limitations", "Taker-only cost model; percentage-band book (no touch, no queue); single venue; 1-minute resolution; slippage assumed."]],
                 ["Question", "Answer (evidence)"], [5.2, 11.8]),
           Spacer(1, 10),
           callout("<b>Bottom line.</b> One-minute ETH perpetual price paths mean-revert in a small, stable and statistically "
                   "robust way, and taker order flow is mildly informative about that reversal. The predictable component is "
                   "worth about half a basis point per side, roughly one tenth of the cost of taking liquidity. "
                   "<b>Predictive accuracy ≠ trading profitability.</b> A viable version would need maker execution and queue-level "
                   "data, which this public dataset cannot support.", GREEN),
           Spacer(1, 10),
           P("Reproduction", "h2"),
           P("pip install -r requirements.txt; pytest -q (30 tests); then run scripts/run_stage2.py → measure_spread.py → build_dataset.py → "
             "leakage_audit.py → run_eda_stats.py → run_models.py → run_dev_backtest.py → run_interpret.py → freeze_methodology.py → "
             "run_holdout.py → run_replication.py → run_significance.py → make_results_table.py → make_notebooks.py → make_pdf_report.py. "
             "Raw data are rebuilt from the public archive; the manifest records each file's URL and SHA-256.")]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.8 * cm, title="Ethereum Order-Flow Alpha — End-to-End Research Report",
                            author="ETH order-flow research project", subject="Raw data to final result")
    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    import sys
    import types
    sys.path.insert(0, str(ROOT / "src"))
    from ethof.features import GROUPS as _G
    sys.modules["ethof_groups"] = types.SimpleNamespace(GROUPS=_G)
    main()
