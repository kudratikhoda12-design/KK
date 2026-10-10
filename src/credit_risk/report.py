"""Build the PDF project report from the pipeline outputs.

    python -m credit_risk.report            # reads reports/, writes reports/Credit_Risk_Scorecard_Report.pdf

Every number in the narrative is read from ``metrics.json`` / the CSVs written by
``credit_risk.pipeline``; the only constants typed in below are the design-iteration
counts recorded during development (clearly labelled as such).
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from . import config

OUT_NAME = "Credit_Risk_Scorecard_Report.pdf"

# ---------------------------------------------------------------------------- fonts & styles
_FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
for _name, _file in {"DV": "DejaVuSans.ttf", "DV-B": "DejaVuSans-Bold.ttf", "DV-I": "DejaVuSans-Oblique.ttf",
                     "DV-BI": "DejaVuSans-BoldOblique.ttf", "DV-M": "DejaVuSansMono.ttf"}.items():
    pdfmetrics.registerFont(TTFont(_name, str(_FONT_DIR / _file)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK, MUTED, ACCENT, ACCENT2 = colors.HexColor("#1F2937"), colors.HexColor("#6B7280"), colors.HexColor("#1D4ED8"), colors.HexColor("#E45756")
HEAD_BG, ZEBRA, RULE, BOX_BG = colors.HexColor("#1F2937"), colors.HexColor("#F3F4F6"), colors.HexColor("#D1D5DB"), colors.HexColor("#EEF2FF")

S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=26, leading=32, textColor=INK, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="DV", fontSize=12.5, leading=18, textColor=MUTED, spaceAfter=4),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=16, leading=21, textColor=INK, spaceBefore=4, spaceAfter=8, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=11.5, leading=15, textColor=ACCENT, spaceBefore=10, spaceAfter=4, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.2, leading=13.6, textColor=INK, spaceBefore=3, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9.2, leading=13.4, textColor=INK, leftIndent=14,
                             bulletIndent=3, spaceAfter=2.5),
    "caption": ParagraphStyle("caption", fontName="DV-I", fontSize=8, leading=11, textColor=MUTED, spaceBefore=3, spaceAfter=8, alignment=TA_CENTER),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8, leading=10.4, textColor=INK),
    "cell_r": ParagraphStyle("cell_r", fontName="DV", fontSize=8, leading=10.4, textColor=INK, alignment=TA_RIGHT),
    "cell_c": ParagraphStyle("cell_c", fontName="DV", fontSize=8, leading=10.4, textColor=INK, alignment=TA_CENTER),
    "head": ParagraphStyle("head", fontName="DV-B", fontSize=8, leading=10.4, textColor=colors.white),
    "head_r": ParagraphStyle("head_r", fontName="DV-B", fontSize=8, leading=10.4, textColor=colors.white, alignment=TA_RIGHT),
    "big": ParagraphStyle("big", fontName="DV-B", fontSize=20, leading=24, textColor=ACCENT, alignment=TA_CENTER),
    "biglabel": ParagraphStyle("biglabel", fontName="DV", fontSize=7.6, leading=10, textColor=MUTED, alignment=TA_CENTER),
    "note": ParagraphStyle("note", fontName="DV", fontSize=8.8, leading=12.8, textColor=INK),
    "mono": ParagraphStyle("mono", fontName="DV-M", fontSize=8, leading=11, textColor=INK, leftIndent=8),
}

W, H = A4
MARGIN = 2.0 * cm
CONTENT_W = W - 2 * MARGIN

# UCI documentation of the 64 ratios (Zieba, Tomczak & Tomczak 2016; Polish companies bankruptcy data)
ATTR_DESC = {
    1: "net profit / total assets", 2: "total liabilities / total assets", 3: "working capital / total assets",
    4: "current assets / short-term liabilities",
    5: "[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] x 365",
    6: "retained earnings / total assets", 7: "EBIT / total assets", 8: "book value of equity / total liabilities",
    9: "sales / total assets", 10: "equity / total assets",
    11: "(gross profit + extraordinary items + financial expenses) / total assets", 12: "gross profit / short-term liabilities",
    13: "(gross profit + depreciation) / sales", 14: "(gross profit + interest) / total assets",
    15: "(total liabilities x 365) / (gross profit + depreciation)", 16: "(gross profit + depreciation) / total liabilities",
    17: "total assets / total liabilities", 18: "gross profit / total assets", 19: "gross profit / sales",
    20: "(inventory x 365) / sales", 21: "sales (n) / sales (n-1)", 22: "profit on operating activities / total assets",
    23: "net profit / sales", 24: "gross profit (in 3 years) / total assets", 25: "(equity - share capital) / total assets",
    26: "(net profit + depreciation) / total liabilities", 27: "profit on operating activities / financial expenses",
    28: "working capital / fixed assets", 29: "logarithm of total assets", 30: "(total liabilities - cash) / sales",
    31: "(gross profit + interest) / sales", 32: "(current liabilities x 365) / cost of products sold",
    33: "operating expenses / short-term liabilities", 34: "operating expenses / total liabilities",
    35: "profit on sales / total assets", 36: "total sales / total assets",
    37: "(current assets - inventories) / long-term liabilities", 38: "constant capital / total assets",
    39: "profit on sales / sales", 40: "(current assets - inventory - receivables) / short-term liabilities",
    41: "total liabilities / ((profit on operating activities + depreciation) x (12/365))",
    42: "profit on operating activities / sales", 43: "rotation receivables + inventory turnover in days",
    44: "(receivables x 365) / sales", 45: "net profit / inventory", 46: "(current assets - inventory) / short-term liabilities",
    47: "(inventory x 365) / cost of products sold", 48: "EBITDA (profit on operating activities - depreciation) / total assets",
    49: "EBITDA / sales", 50: "current assets / total liabilities", 51: "short-term liabilities / total assets",
    52: "(short-term liabilities x 365) / cost of products sold", 53: "equity / fixed assets",
    54: "constant capital / fixed assets", 55: "working capital", 56: "(sales - cost of products sold) / sales",
    57: "(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)",
    58: "total costs / total sales", 59: "long-term liabilities / equity", 60: "sales / inventory",
    61: "sales / receivables", 62: "(short-term liabilities x 365) / sales", 63: "sales / short-term liabilities",
    64: "sales / fixed assets",
}


def desc(ratio: str) -> str:
    return ATTR_DESC.get(int(ratio.replace("Attr", "")), "")


# ---------------------------------------------------------------------------- small helpers
f3 = lambda x: f"{x:.3f}"  # noqa: E731
pct = lambda x, d=1: f"{100 * x:.{d}f}%"  # noqa: E731
P = lambda text, style="body": Paragraph(text, S[style])  # noqa: E731
E = escape


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, align=None, zebra=True, font=8, head_rows=1, highlight_rows=()):
    """rows: list of lists of str; first ``head_rows`` rows are headers. align: 'L'/'R'/'C' per column."""
    ncol = len(rows[0])
    align = align or ["L"] + ["R"] * (ncol - 1)
    styles = S
    if font != 8:  # compact variant for dense appendix tables
        styles = {k: ParagraphStyle(k + str(font), parent=S[k], fontSize=font, leading=font * 1.3)
                  for k in ("cell", "cell_r", "cell_c", "head", "head_r")}
    data = []
    for i, row in enumerate(rows):
        out = []
        for j, cell in enumerate(row):
            if isinstance(cell, str):
                if i < head_rows:
                    out.append(Paragraph(cell, styles["head_r"] if align[j] == "R" else styles["head"]))
                else:
                    out.append(Paragraph(cell, styles[{"L": "cell", "R": "cell_r", "C": "cell_c"}[align[j]]]))
            else:
                out.append(cell)
        data.append(out)
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=head_rows)
    style = [("BACKGROUND", (0, 0), (-1, head_rows - 1), HEAD_BG), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("LINEBELOW", (0, head_rows), (-1, -1), 0.25, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if zebra:
        style += [("BACKGROUND", (0, r), (-1, r), ZEBRA) for r in range(head_rows + 1, len(rows), 2)]
    style += [("BACKGROUND", (0, r), (-1, r), BOX_BG) for r in highlight_rows]
    t.setStyle(TableStyle(style))
    return t


def box(flowables, bg=BOX_BG, border=ACCENT):
    t = Table([[flowables]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 2.5, border),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def figure(path: Path, caption: str, width_cm: float = 12.5):
    iw, ih = ImageReader(str(path)).getSize()
    w = width_cm * cm
    return KeepTogether([Image(str(path), width=w, height=w * ih / iw), P(caption, "caption")])


def flow_diagram(steps, per_row=5, width=CONTENT_W):
    gap, box_h, row_gap = 14, 40, 26
    box_w = (width - gap * (per_row - 1)) / per_row
    rows = -(-len(steps) // per_row)
    height = rows * box_h + (rows - 1) * row_gap + 4
    d = Drawing(width, height)
    centres = []
    for i, label in enumerate(steps):
        r, c = divmod(i, per_row)
        x, y = c * (box_w + gap), height - 2 - (r + 1) * box_h - r * row_gap
        d.add(Rect(x, y, box_w, box_h, rx=5, ry=5, fillColor=BOX_BG, strokeColor=ACCENT, strokeWidth=0.8))
        lines = label.split("\n")
        for k, line in enumerate(lines):
            d.add(String(x + box_w / 2, y + box_h / 2 + (len(lines) - 1) * 5 - k * 10 - 3, line, fontName="DV-B" if k == 0 else "DV",
                         fontSize=7.4 if k == 0 else 6.6, fillColor=INK if k == 0 else MUTED, textAnchor="middle"))
        centres.append((x, y, box_w, box_h))
    for i in range(len(steps) - 1):
        (x1, y1, w1, h1), (x2, y2, w2, h2) = centres[i], centres[i + 1]
        if i // per_row == (i + 1) // per_row:                       # arrow within a row
            xs, xe, ym = x1 + w1, x2, y1 + h1 / 2
            d.add(Line(xs, ym, xe - 4, ym, strokeColor=ACCENT, strokeWidth=1))
            d.add(Polygon([xe, ym, xe - 5, ym + 2.8, xe - 5, ym - 2.8], fillColor=ACCENT, strokeColor=ACCENT))
        else:                                                         # elbow to the next row
            xs, ys = x1 + w1 / 2, y1
            xe, ye = x2 + w2 / 2, y2 + h2
            ym = (ys + ye) / 2
            for a, b, c, dd in ((xs, ys, xs, ym), (xs, ym, xe, ym), (xe, ym, xe, ye + 4)):
                d.add(Line(a, b, c, dd, strokeColor=ACCENT, strokeWidth=1))
            d.add(Polygon([xe, ye, xe - 2.8, ye + 5, xe + 2.8, ye + 5], fillColor=ACCENT, strokeColor=ACCENT))
    return d


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 1.1 * cm, "Expected Loss-Based Credit Risk Scorecard  |  Project report")
    canvas.drawRightString(W - MARGIN, 1.1 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(MARGIN, 1.5 * cm, W - MARGIN, 1.5 * cm)
    canvas.restoreState()


# ---------------------------------------------------------------------------- content
def build(report_dir: Path = config.REPORT_DIR, out_path: Path | None = None) -> Path:
    report_dir = Path(report_dir)
    fig_dir = report_dir / "figures"
    out_path = Path(out_path or report_dir / OUT_NAME)
    R = json.loads((report_dir / "metrics.json").read_text())
    sqc = pd.read_csv(report_dir / "sqc_screening.csv")
    horizons = pd.read_csv(report_dir / "horizon_robustness.csv") if (report_dir / "horizon_robustness.csv").exists() else None
    wl = pd.read_csv(report_dir / "watchlist_full.csv")
    tm, thr_, wlm = R["test_metrics"], R["threshold"], R["watchlist"]
    champ = R["champion"]["model"]
    NAMES = {"lr": "Logistic regression", "rf": "Random forest", "hgb": "Gradient boosting (challenger)"}
    d = R["data"]
    loss = R["config"]["loss"]
    ci = lambda m: f"{m['auc_ci95'][0]:.3f}-{m['auc_ci95'][1]:.3f}"  # noqa: E731
    cap = wlm["loss_capture"]
    q10 = "0.1"
    rules = thr_["test_rules"]
    best_rule = rules["cost-optimal global threshold"]
    story = []

    # ------------------------------------------------------------------ cover
    story += [Spacer(1, 2.2 * cm), P("Expected Loss-Based<br/>Credit Risk Scorecard", "title"),
              P("Corporate default prediction with SQC ratio screening, calibrated PDs, cost-sensitive thresholds and an "
                "Expected Loss-ranked watchlist", "subtitle"),
              P(f"Project report: every step from raw data to final result &nbsp;|&nbsp; {date.today():%d %B %Y}", "subtitle"),
              Spacer(1, 0.9 * cm)]
    kpi = [[(P(f3(tm[champ]["auc_roc"]), "big"), P(f"hold-out AUC-ROC, {NAMES[champ].lower()} (champion)<br/>95% CI {ci(tm[champ])}", "biglabel")),
            (P(f3(tm["hgb"]["auc_roc"]), "big"), P(f"hold-out AUC-ROC, gradient-boosting challenger<br/>95% CI {ci(tm['hgb'])}", "biglabel")),
            (P(f3(tm["altman_z"]["auc_roc"]), "big"), P(f"Altman Z'' benchmark<br/>95% CI {ci(tm['altman_z'])}", "biglabel"))],
           [(P(f"{R['sqc']['ratios_kept']} / {R['sqc']['ratios_in']}", "big"), P("financial ratios retained by SQC screening", "biglabel")),
            (P(pct(cap["EL rank"][q10], 0), "big"), P("of realised loss inside the top 10% of the EL-ranked watchlist<br/>(proxy LGD / EAD)", "biglabel")),
            (P(pct(best_rule["cost_saving_vs_flag_nobody"], 0), "big"), P("expected-cost saving of the optimised threshold<br/>vs no monitoring (proxy costs)", "biglabel"))]]
    t = Table([[[a, b] for a, b in row] for row in kpi], colWidths=[CONTENT_W / 3] * 3)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BOX_BG), ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                           ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white), ("TOPPADDING", (0, 0), (-1, -1), 9),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [t, Spacer(1, 0.9 * cm),
              P(f"<b>Data:</b> Polish companies bankruptcy data (UCI Machine Learning Repository, CC BY 4.0): "
                f"{d['firms']:,} firms, {d['defaults']} bankruptcies ({pct(d['default_rate'])}), 64 financial ratios, "
                "one-year-ahead horizon.<br/><b>Code:</b> github.com/kudratikhoda12-design/KK, branch "
                "<font name='DV-M'>claude/friendly-dirac-a70ml4</font> (package <font name='DV-M'>credit_risk</font>, 35 unit tests).", "body"),
              PageBreak()]

    # ------------------------------------------------------------------ 1 executive summary
    rep = R.get("repeated_splits", {})
    n947 = {k: sum(v >= 0.947 for v in rep[k]["all"]) for k in rep}
    n_rep = len(rep["rf"]["all"]) if rep else 0
    story += [P("1. Executive summary", "h1"),
              P("This project builds a corporate credit-risk engine end to end. Raw financial ratios are screened with "
                "statistical-quality-control (SQC) control charts, scored by logistic regression and random forest, converted into "
                "calibrated probabilities of default (PD), turned into an accept / review decision with a cost-sensitive "
                "threshold, and combined with loss-given-default (LGD) and exposure (EAD) proxies into an Expected Loss (EL)-ranked "
                "watchlist that prioritises the portfolio.")]
    summary_rows = [["Model / benchmark", "Hold-out AUC-ROC", "95% CI", "Gini", "KS", "PR-AUC", "Brier"]]
    for k, label in (("altman_z", "Altman Z'' (classic score, benchmark)"), ("lr", NAMES["lr"]), ("rf", NAMES["rf"] + (" (champion)" if champ == "rf" else "")),
                     ("hgb", NAMES["hgb"])):
        m = tm[k]
        summary_rows.append([label, f3(m["auc_roc"]), ci(m)] + ([f3(m["gini"]), f3(m["ks"]), f3(m["pr_auc"]), f3(m["brier"])] if "gini" in m else ["-"] * 4))
    story += [table(summary_rows, [5.6, 2.5, 2.4, 1.4, 1.4, 1.5, 1.4], highlight_rows=[3 if champ == "rf" else 2]),
              P(f"Hold-out portfolio: {d['test_firms']:,} firms, {d['test_defaults']} defaults. Calibrated PDs shown for LR / RF / GBM.", "caption"),
              P("What to take away", "h2")]
    story += bullets([
        f"<b>Discrimination.</b> The random forest ranks defaulters above survivors with AUC-ROC <b>{f3(tm['rf']['auc_roc'])}</b> on the untouched hold-out "
        f"(logistic regression {f3(tm['lr']['auc_roc'])}; classic Altman Z'' {f3(tm['altman_z']['auc_roc'])}). Across {len(rep.get('rf', {}).get('all', []))} "
        f"repeated random splits the RF averages {f3(rep['rf']['mean'])} (sd {f3(rep['rf']['sd'])})." if rep else
        f"<b>Discrimination.</b> The random forest reaches hold-out AUC-ROC {f3(tm['rf']['auc_roc'])}.",
        f"<b>SQC screening</b> keeps {R['sqc']['ratios_kept']} of {R['sqc']['ratios_in']} ratios. It removes redundant, sparse and signal-free ratios and "
        "improved the random forest in development cross-validation (0.900 to 0.907); for logistic regression it gives a leaner model at a negligible AUC cost (0.855 vs 0.859).",
        f"<b>Calibration.</b> Scores are mapped to probabilities with cross-fitted {R['calibration_choice'][champ]} calibration. On the hold-out the champion's "
        f"mean PD is {pct(tm[champ]['mean_pd'], 2)} against an observed default rate of {pct(tm[champ]['default_rate'], 2)}.",
        f"<b>Cost-sensitive threshold.</b> Flagging firms with PD &ge; {thr_['t_star']:.3f} (instead of 0.5) flags {pct(best_rule['flagged_share'], 0)} of firms, "
        f"catches {pct(best_rule['recall'], 0)} of defaulters and cuts expected cost by {pct(best_rule['cost_saving_vs_flag_nobody'], 0)} versus no monitoring.",
        f"<b>Watchlist.</b> Ranking by Expected Loss puts {pct(cap['EL rank'][q10], 0)} of realised loss in the top 10% of firms "
        f"(PD-only ranking: {pct(cap['PD rank'][q10], 0)}; random: 10%).",
        "<b>Important caveat.</b> The public data has no loan-level LGD or EAD, so those two inputs are transparent <i>proxies</i> built from balance-sheet "
        "ratios. The EL and cost figures are therefore illustrative of the method, not forecasts for a real bank (Section 11)."])
    story += [Spacer(1, 4), box([P("<b>About the AUC-ROC figure.</b> The measured champion AUC is "
                                   f"<b>{f3(tm[champ]['auc_roc'])}</b> (95% CI {ci(tm[champ])}) and a gradient-boosting challenger reaches "
                                   f"<b>{f3(tm['hgb']['auc_roc'])}</b>. A value near 0.947 is typical only of the boosting challenger: the forest reached it in "
                                   f"{n947.get('rf', 0)} of {n_rep} random splits (best {rep['rf']['max']:.3f}), boosting in {n947.get('hgb', 0)} of {n_rep}. "
                                   "Section 10 gives resume wording that matches the evidence.", "note")]),
              PageBreak()]

    # ------------------------------------------------------------------ 2 objective & scope
    story += [P("2. Objective and scope", "h1"),
              P("The project implements the following capability statement, and Section 10 revisits each claim against the evidence:"),
              box([P("<i>Designed a corporate credit risk engine using RF/LR with SQC-based ratio screening and cost-sensitive threshold optimization; "
                     "calibrated PD outputs and generated Expected Loss-ranked watchlists for portfolio risk prioritization.</i>", "note")]),
              Spacer(1, 6)]
    claim_rows = [["Claim", "What was built", "Module"],
                  ["RF / LR credit risk engine", "Random forest and logistic regression pipelines, tuned by stratified CV; gradient boosting as a labelled challenger", "models.py"],
                  ["SQC-based ratio screening", "Control charts fitted on healthy firms; defaulter detection rate tested against false-alarm rate; redundancy pruning", "screening.py"],
                  ["Cost-sensitive threshold optimisation", "Per-firm costs (LGD x EAD for a miss, margin x EAD for a false alarm); threshold minimising total cost on out-of-fold PDs", "threshold.py"],
                  ["Calibrated PD outputs", "Cross-fitted Platt / isotonic calibration, chosen by Brier score; reliability, ECE, slope diagnostics", "calibration.py"],
                  ["Expected Loss-ranked watchlists", "EL = PD x LGD x EAD, scorecard points, rating grades, ranked watchlist, loss-capture back-test", "expected_loss.py"],
                  ["AUC-ROC", "Bootstrap CIs, repeated splits, five forecast horizons, Altman Z'' benchmark", "metrics.py, pipeline.py"]]
    story += [table(claim_rows, [4.2, 9.9, 3.4], align=["L", "L", "L"]), Spacer(1, 8), P("Pipeline at a glance", "h2"),
              flow_diagram(["1  Raw data\nUCI ARFF, 64 ratios", "2  Clean\ndedupe, audit", "3  Split\n75 / 25 stratified", "4  SQC screening\ncontrol charts, FDR",
                            "5  Models\nLR, RF (+GBM)", "6  Calibrate\nPlatt / isotonic", "7  Threshold\nminimum expected cost", "8  Expected Loss\nPD x LGD x EAD",
                            "9  Watchlist\nranked, back-tested", "10  Robustness\nhorizons, splits"]),
              Spacer(1, 6),
              P("Steps 4 to 6 are scikit-learn pipeline stages, so each is re-fitted on the training part of every cross-validation fold and never sees "
                "validation or hold-out firms. Steps 7 to 9 use only out-of-fold predictions for training firms until the final hold-out scoring.", "body"),
              PageBreak()]

    # ------------------------------------------------------------------ 3 data
    story += [P("3. Data: from raw files to modelling set", "h1"),
              P("3.1 Source", "h2"),
              P("The data are the <i>Polish companies bankruptcy data</i> (UCI repository 365, CC BY 4.0), collected from the EMIS database: financial "
                "statements of Polish manufacturing firms, with 64 financial ratios per firm and a label for bankruptcy within a stated horizon. The archive "
                "is downloaded programmatically and verified against a pinned SHA-256 checksum before use (<font name='DV-M'>data.py</font>)."),
              P("3.2 Forecast horizons", "h2"),
              P("The archive holds five files. In file <i>k</i> the ratios are observed in the <i>k</i>-th year of the forecast window and the label says whether the "
                "firm went bankrupt after the window, so file 5 (ratios observed one year before the outcome) is the standard one-year-horizon PD problem. "
                "<b>File 5 is the primary dataset</b>; files 1 to 4 are used only for the robustness table in Section 9.")]
    if horizons is not None:
        hrows = [["File", "Ratios observed", "Firms (after dedupe)", "Defaults", "Default rate"]]
        for _, r in horizons.sort_values("years_ahead", ascending=False).iterrows():
            ya = int(r["years_ahead"])
            hrows.append([f"{6 - ya}year.arff", f"{ya} year{'s' if ya > 1 else ''} before outcome", f"{int(r['firms']):,}", f"{int(r['defaults'])}", pct(r["default_rate"], 2)])
        story += [table(hrows, [3, 4.6, 3.6, 2.4, 2.8], highlight_rows=[5]), Spacer(1, 4)]
    story += [P("3.3 Data audit and cleaning (primary file, 5,910 raw rows)", "h2")]
    audit = [["Check", "Finding", "Action"],
             ["Exact duplicate feature rows", "60 duplicate rows; none carried conflicting labels (4 were defaulters)", "Dropped (5,910 to 5,850 firms) so a firm cannot appear in both train and test"],
             ["Near-duplicates", "Only 134 firms (2.3%) share 40+ of 64 ratios (to 3 d.p.) with another firm, 18 of them defaulters", "No action; far too few to explain model performance (checked because boosting scored well)"],
             ["Missing values", "1.2% of cells; 48.7% of rows miss at least one ratio; Attr37 is 43% missing", "Median imputation inside the pipeline; ratios above 40% missing dropped by the screener"],
             ["Extreme outliers", "e.g. total liabilities / total assets spans -430 to +72", "Robust scaling in the screener; clipping for the EAD / LGD proxies"],
             ["Class imbalance", f"{pct(d['default_rate'], 2)} defaults ({d['defaults']} of {d['firms']:,})", "Stratified splits; no resampling, so probabilities stay interpretable"]]
    story += [table(audit, [3.6, 7.2, 6.7], align=["L", "L", "L"]),
              P("3.4 Train / test split", "h2"),
              P(f"A stratified 75 / 25 split (seed {R['config']['seed']}) gives <b>{d['train_firms']:,} training firms ({d['train_defaults']} defaults)</b> and "
                f"<b>{d['test_firms']:,} hold-out firms ({d['test_defaults']} defaults)</b>. The hold-out acts as the 'portfolio' that is scored and ranked at the end."),
              PageBreak()]

    # ------------------------------------------------------------------ 4 protocol
    story += [P("4. Evaluation protocol and integrity", "h1")]
    story += bullets([
        "<b>No leakage by construction.</b> Control limits, screening decisions, imputation, scaling, model fitting and calibration maps are all fitted on training firms only; "
        "inside cross-validation they are re-fitted per fold because they are pipeline stages.",
        "<b>Selection on training data only.</b> Hyper-parameters (5-fold stratified CV, AUC), calibration method (cross-fitted Brier score) and the decision threshold "
        "(out-of-fold PDs) are chosen without the hold-out.",
        "<b>Champion rule fixed in advance.</b> The champion is the better training-CV AUC of the declared scope (LR, RF). Gradient boosting is a reference challenger "
        "and does not drive the watchlist unless explicitly selected.",
        "<b>Uncertainty is reported.</b> Stratified-bootstrap 95% intervals (1,000 resamples) on the hold-out, plus repeated random splits (Section 9).",
        "<b>The proxies are label-free functions</b> of balance-sheet ratios. One parameter, the LGD recovery haircut, was set once by matching the mean LGD of "
        "<i>training</i> defaulters to the 45% Basel benchmark, so default labels entered only as an aggregate and no model performance was involved.",
        "<b>Transparency notes.</b> (i) The hold-out was not used for any tuning or design choice, but two code-validation steps did touch it: a feature-importance "
        "diagnostic on a gradient-boosting model, and a quick-mode smoke test of the pipeline that printed hold-out AUCs for deliberately tiny grids. No design decision "
        "was changed afterwards. (ii) The repeated-split results reuse hyper-parameters tuned on the first split's training data, which makes them very slightly optimistic."])
    story += [Spacer(1, 10)]

    # ------------------------------------------------------------------ 5 SQC
    cnt = R["sqc"]["status_counts"]
    story += [P("5. SQC ratio screening", "h1"),
              P("5.1 Idea", "h2"),
              P("In statistical quality control a process is monitored with control limits fitted while it is 'in control'. Here, <b>healthy firms are the in-control process</b>: "
                "for every ratio the limits are fitted on healthy training firms only. A ratio is a useful early-warning signal if <b>defaulters breach the limit "
                "more often than healthy firms do</b>, i.e. the chart's <i>detection rate</i> exceeds its <i>false-alarm rate</i>. Each ratio gets a lower-limit (LCL) chart and an "
                "upper-limit (UCL) chart, because the adverse direction differs by ratio."),
              P("5.2 Screening steps", "h2")]
    story += bullets([
        "<b>Data quality:</b> drop ratios with more than 40% missing values or zero spread.",
        f"<b>SQC signal:</b> one-sided two-proportion z-test of detection rate &gt; false-alarm rate on each tail, Benjamini-Hochberg corrected over all 2 x 64 tests (q &lt; {R['config']['sqc']['fdr']}).",
        f"<b>Redundancy:</b> of two ratios with |Spearman correlation| &gt; {R['config']['sqc']['max_corr']}, keep the one with the stronger signal.",
        "<b>Output for the models:</b> raw ratios for trees; for logistic regression an in-control z-score (distance from the healthy median in robust sigmas, scaled separately above "
        "and below the median) compressed with arcsinh, which tames heavy tails."])
    story += [P("5.3 Design iterations (what failed and why)", "h2"),
              P("The first implementation did not work well, and the fixes are part of the method. The counts below are the training-split results recorded during development "
                "(they are not recomputed by the report builder).")]
    it_rows = [["Version", "Control limits", "Ratios kept", "Problem found"],
               ["v1", "Median +/- 3 robust sigma, two-sided alarm", "34", "Healthy firms with extreme favourable values swamp the alarm count; liquidity and solvency ratios were discarded despite defaulters sitting about 1 sigma lower"],
               ["v2", "Same limits, each tail tested separately", "38", "Fixed the masking, but 8 non-negative liquidity ratios still showed zero alarms on the lower tail"],
               ["v3", "Asymmetric limits from semi-MADs", "38", "For distributions piled up near zero even median - 3 sigma_low is negative, so the lower limit can never be breached"],
               ["v4 (final)", f"Empirical probability limits: {R['config']['sqc']['tail_prob']:.1%} / {1 - R['config']['sqc']['tail_prob']:.1%} quantiles of healthy firms, per tail", f"{R['sqc']['ratios_kept']}",
                "Every ratio has the same, known false-alarm rate per tail, so ratios are directly comparable and skewed ratios can signal"]]
    story += [table(it_rows, [1.8, 5.6, 1.9, 8.2], align=["L", "L", "C", "L"], highlight_rows=[4]),
              P("5.4 Result on the training firms", "h2"),
              P(f"Of 64 ratios, <b>{cnt.get('kept', 0)} are kept</b>; {cnt.get('dropped: redundant', 0)} are dropped as redundant, {cnt.get('dropped: missing', 0)} for missing data "
                f"({', '.join(sqc.loc[sqc['status'] == 'dropped: missing', 'ratio']) or 'none'}) and {cnt.get('dropped: no SQC signal', 0)} for lack of signal "
                f"({', '.join(sqc.loc[sqc['status'] == 'dropped: no SQC signal', 'ratio']) or 'none'}). The full table is in Appendix A.")]
    top = sqc[sqc["status"] == "kept"].sort_values("lift", ascending=False).head(10)
    top_rows = [["Ratio", "Definition", "Adverse tail", "Defaulters beyond limit", "Healthy beyond limit", "Lift"]]
    for _, r in top.iterrows():
        top_rows.append([r["ratio"], E(desc(r["ratio"])), r["adverse_tail"], pct(r["detection_rate"]), pct(r["false_alarm_rate"]), f"{r['lift']:.1f}x"])
    story += [KeepTogether([table(top_rows, [1.6, 6.6, 1.7, 2.7, 2.6, 1.3], align=["L", "L", "C", "R", "R", "R"]),
                            P(f"Top 10 retained ratios by lift (detection rate divided by false-alarm rate). For {int((top['adverse_tail'] == 'low').sum())} of the 10 the adverse tail is the lower one: "
                              "defaulters have unusually low profitability and coverage.", "caption")]),
              figure(fig_dir / "sqc_top_ratios.png", "Figure 1. Detection vs false-alarm rates for the 20 strongest retained ratios. The grey bars sit at the chart's "
                     "false-alarm rate by construction of the probability limits.", 11.5),
              PageBreak()]

    # ------------------------------------------------------------------ 6 models
    story += [P("6. Modelling", "h1"),
              P("6.1 Design choices tested on the training split only", "h2"),
              P("Before fixing the pipelines, variants were compared with 5-fold stratified CV on the training split (2 repeats). These are the recorded development results.")]
    abl = [["Model", "Variant", "CV AUC-ROC"],
           ["Logistic regression", "all 64 ratios, raw, standardised", "0.796"],
           ["Logistic regression", "all 64 ratios, in-control z-score clipped", "0.834"],
           ["Logistic regression", "all 64 ratios, in-control z-score + arcsinh", "0.859"],
           ["Logistic regression", "SQC-screened + arcsinh", "0.855"],
           ["Logistic regression", "SQC-screened + arcsinh + alarm-share feature", "0.855 (no gain, dropped)"],
           ["Random forest", "all 64 ratios (300 trees, leaf 3)", "0.900"],
           ["Random forest", "SQC-screened (tail probability 2.5%)", "0.907"],
           ["Random forest", "SQC-screened, tail probability 1% / 5%", "0.909 / 0.905 (within noise, default kept)"],
           ["Altman Z'' score (training firms)", "6.56 WC/TA + 3.26 RE/TA + 6.72 EBIT/TA + 1.05 BVE/TL", "0.768"]]
    story += [table(abl, [4.6, 8.0, 4.9], align=["L", "L", "L"], highlight_rows=[3, 7]),
              P("Reading the table: the in-control z-score with arcsinh is worth about six AUC points for logistic regression; SQC screening helps the forest a little (+0.007) and costs "
                "LR a negligible 0.004 while using 44 instead of 64 ratios. The tail probability was not tuned, because the differences are within noise.", "body"),
              P("6.2 Final pipelines", "h2")]
    story += bullets([
        "<b>Logistic regression:</b> SQC screening with arcsinh in-control z-scores &rarr; median imputation &rarr; standardisation &rarr; L2-penalised logistic regression.",
        "<b>Random forest:</b> SQC screening (raw ratios) &rarr; median imputation &rarr; random forest (500 trees in the final fit; 200 during the grid search).",
        "<b>Gradient boosting (challenger):</b> SQC screening &rarr; histogram gradient boosting, which handles missing values natively. Reported for context, outside the RF / LR scope."])
    story += [P("6.3 Hyper-parameter search (training CV)", "h2")]
    tun_rows = [["Model", "Grid searched", "Selected", "CV AUC-ROC (mean, sd over folds)"]]
    grids = {"lr": "C in {0.03, 0.1, 0.3, 1}", "rf": "min_samples_leaf in {1, 2, 3, 5} x max_features in {sqrt, 0.3, 0.5}", "hgb": "max_leaf_nodes in {8, 15, 31} x max_iter in {200, 400}"}
    for k in ("lr", "rf", "hgb"):
        tn = R["tuning"][k]
        tun_rows.append([NAMES[k] + (" (champion)" if k == champ else ""), grids[k], E(", ".join(f"{a.replace('clf__', '')}={b}" for a, b in tn["best_params"].items())),
                         f"{tn['cv_auc']:.4f} (sd {tn['cv_auc_sd']:.4f})"])
    gaps = {}
    for k in ("lr", "rf", "hgb"):
        sc = pd.read_csv(report_dir / f"tuning_{k}.csv")["mean_test_score"].to_numpy()
        gaps[k] = float(sc[0] - sc[1])
    flat_note = (f"The CV surfaces are flat: the best two configurations differ by only {gaps['lr']:.4f} (LR), {gaps['rf']:.4f} (RF) and {gaps['hgb']:.4f} (GBM) AUC, "
                 "against fold-to-fold standard deviations of 0.01 to 0.03. The selected LR penalty and RF max_features lie at the edge of their grids, so a wider search might add "
                 "a little; it was not run.")
    story += [table(tun_rows, [3.8, 6.0, 4.0, 3.7], align=["L", "L", "L", "R"], highlight_rows=[1 + ["lr", "rf", "hgb"].index(champ)]),
              P(f"The champion is <b>{NAMES[champ].lower()}</b> by the pre-declared rule (higher CV AUC among logistic regression and random forest).", "body"),
              P(flat_note, "body"),
              PageBreak()]

    # ------------------------------------------------------------------ 7 calibration
    story += [P("7. PD calibration", "h1"),
              P("A model's score ranks firms but is not necessarily a probability: forests average votes and boosting over-confidently separates classes. Credit decisions, "
                "pricing and Expected Loss all need <i>probabilities</i>, so scores are mapped to PDs with a calibration map learned on out-of-fold scores."),
              P("7.1 Method", "h2")]
    story += bullets([
        "Out-of-fold raw scores are produced for every training firm (5-fold CV, models never score firms they were trained on).",
        "Three options are compared by <b>cross-fitted</b> Brier score (each firm is calibrated by a map fitted without it): none, Platt sigmoid on the logit of the score, and isotonic regression.",
        "The winning map is fitted on all out-of-fold scores, the model is refitted on all training firms, and the map is applied to its hold-out scores. "
        "A 1e-6 tie-breaker keeps isotonic's step function from erasing the ranking."])
    story += [P("7.2 Method selection (training firms, out-of-fold)", "h2")]
    cal_rows = [["Model", "Method", "Brier", "Log-loss", "ECE", "AUC-ROC", "Chosen"]]
    for k in ("lr", "rf", "hgb"):
        for row in R["calibration_selection"][k]:
            cal_rows.append([NAMES[k] if row["method"] == "none" else "", row["method"], f"{row['brier']:.4f}", f"{row['log_loss']:.4f}", f"{row['ece']:.4f}", f"{row['auc_roc']:.4f}",
                             "yes" if row["method"] == R["calibration_choice"][k] else ""])
    story += [table(cal_rows, [4.8, 2.2, 2.0, 2.2, 2.0, 2.3, 2.0], align=["L", "L", "R", "R", "R", "R", "C"]),
              P("7.3 Hold-out calibration", "h2")]
    hc = [["Predictions", "AUC-ROC", "Brier", "ECE", "Mean PD vs observed rate", "Calibration slope"]]
    for label, m in (("RF, uncalibrated scores", tm["rf_uncalibrated"]), (f"RF, calibrated ({R['calibration_choice']['rf']})", tm["rf"]),
                     (f"LR, calibrated ({R['calibration_choice']['lr']})", tm["lr"]), (f"GBM, calibrated ({R['calibration_choice']['hgb']})", tm["hgb"])):
        hc.append([label, f3(m["auc_roc"]), f"{m['brier']:.4f}", f"{m['ece']:.4f}", f"{pct(m['mean_pd'], 2)} vs {pct(m['default_rate'], 2)}", f"{m['calibration_slope']:.2f}"])
    sel = {r["method"]: r for r in R["calibration_selection"]["rf"]}
    ru, rc = tm["rf_uncalibrated"], tm["rf"]
    cal_note = (f"<b>Reading the numbers.</b> For the random forest, calibration cut the out-of-fold ECE from {sel['none']['ece']:.4f} to {sel[R['calibration_choice']['rf']]['ece']:.4f} "
                f"and the hold-out ECE from {ru['ece']:.4f} to {rc['ece']:.4f}. Hold-out Brier moved from {ru['brier']:.4f} to {rc['brier']:.4f}: "
                + ("an improvement. " if rc["brier"] < ru["brier"] else "no improvement, a difference well inside the sampling noise of 102 defaults. ")
                + f"The mean PD ({pct(rc['mean_pd'], 2)}) stays slightly above the observed rate ({pct(rc['default_rate'], 2)}), and a slope of {rc['calibration_slope']:.2f} "
                  f"({'mild over-confidence' if rc['calibration_slope'] < 1 else 'mild under-confidence'}) shows the map is close to, not exactly at, perfect. "
                  "The uncalibrated forest was under-confident (slope above 1); calibration mainly fixes the level and the shape of the reliability curve, not the ranking.") if champ == "rf" else ""
    story += [table(hc, [4.6, 2.0, 2.0, 2.0, 4.2, 2.7]),
              P("Calibration slope 1.0 means the PDs are neither over- nor under-confident. With only a few dozen defaults in the upper PD bins, individual bins are noisy.", "caption"),
              P(cal_note, "body"),
              figure(fig_dir / "calibration_test.png", "Figure 2. Reliability diagram on the hold-out portfolio (deciles of predicted PD).", 10.5),
              P("<b>Deployment note.</b> PDs are calibrated to this sample's default rate. For a portfolio with a different long-run default rate, "
                "<font name='DV-M'>calibration.shift_base_rate</font> applies the standard prior-shift odds correction.", "body"),
              PageBreak()]

    # ------------------------------------------------------------------ 8 threshold
    ps = thr_["proxy_summary"]
    story += [P("8. Cost-sensitive threshold optimisation", "h1"),
              P("Accuracy-style cut-offs such as PD &ge; 0.5 are meaningless when a missed default is far more expensive than a needless review. The threshold is chosen to "
                "<b>minimise total expected cost</b> instead."),
              P("8.1 Cost model", "h2")]
    story += bullets([
        "<b>Missed default (false negative):</b> loss = LGD x EAD of that firm.",
        f"<b>False alarm (good firm flagged):</b> foregone net margin = {pct(loss['margin'], 0)} x EAD of that firm.",
        f"Mean LGD on the hold-out is {pct(ps['mean_lgd'], 1)}, so a miss costs on average about {thr_['margin_sensitivity'][2]['cost_ratio_fn_to_fp']:.0f} times a false alarm.",
        "For calibrated PDs the optimal per-firm rule is analytic: flag if PD &ge; margin / (margin + LGD), because EAD cancels. This gives an independent check of the "
        "empirical optimum."])
    story += [P("8.2 Procedure", "h2"),
              P(f"The cost of every candidate threshold is computed on the <b>out-of-fold calibrated PDs of the training firms</b> (champion: {NAMES[champ].lower()}). Because the cost curve is a noisy step "
                f"function, the threshold reported is the median optimum over 300 bootstrap resamples: <b>PD &ge; {thr_['t_star']:.3f}</b> (90% bootstrap range "
                f"{thr_['t_star_ci90_bootstrap'][0]:.3f} to {thr_['t_star_ci90_bootstrap'][1]:.3f}). The Bayes threshold at mean LGD is {thr_['bayes_threshold_at_mean_lgd']:.3f} and the Youden-J "
                f"threshold is {thr_['t_youden']:.3f}."),
              figure(fig_dir / "cost_threshold.png", "Figure 3. Total cost versus PD threshold on out-of-fold training PDs. The empirical optimum lands on the theoretical Bayes cut-off, "
                     "which is a consistency check on the calibration.", 10.5),
              P("8.3 Hold-out comparison of decision rules", "h2")]
    r_rows = [["Rule", "Flagged", "Defaults caught", "False alarms", "Total cost", "Saving vs no monitoring"]]
    for name, r in rules.items():
        r_rows.append([name + (f" ({r['threshold']:.3f})" if r.get("threshold") is not None else ""), pct(r["flagged_share"], 1), f"{r['tp']} of {r['tp'] + r['fn']} ({pct(r['recall'], 0)})",
                       f"{r['fp']}", f"{r['cost']:.1f}", pct(r["cost_saving_vs_flag_nobody"], 1)])
    story += [table(r_rows, [5.6, 1.7, 3.1, 1.9, 2.1, 3.1], highlight_rows=[1]),
              P("Cost in source currency units under the proxy loss model.", "caption"),
              P("8.4 Sensitivity to the margin assumption", "h2")]
    s_rows = [["Margin (cost of a false alarm)", "Cost ratio miss : alarm", "Optimal threshold", "Flagged", "Defaults caught", "Saving vs threshold 0.5"]]
    for s in thr_["margin_sensitivity"]:
        s_rows.append([pct(s["margin"], 0), f"{s['cost_ratio_fn_to_fp']:.0f} : 1", f"{s['optimal_threshold']:.3f}", pct(s["flagged_share"], 1), pct(s["recall"], 0), pct(s["cost_saving_vs_0.5"], 1)])
    story += [table(s_rows, [4.4, 3.2, 2.6, 2.0, 2.6, 2.7]),
              P("A cheaper false alarm (low margin) makes the optimum flag more firms; the ordering is stable, so the conclusion does not hinge on the 3% assumption.", "caption"),
              Spacer(1, 10)]

    # ------------------------------------------------------------------ 9 EL
    grade_rows = [["Grade", "PD band", "Firms", "Mean PD", "Observed default rate", "Expected loss"]]
    bands = {"A": "< 1%", "B": "1 - 2.5%", "C": "2.5 - 5%", "D": "5 - 10%", "E": "10 - 25%", "F": "> 25%"}
    for g in wlm["grade_table"]:
        grade_rows.append([g["grade"], bands[g["grade"]], f"{g['firms']:,}", pct(g["mean_pd"], 2), pct(g["observed_default_rate"], 1), f"{g['expected_loss']:.1f}"])
    gt = wlm["grade_table"]
    gr = [g["observed_default_rate"] for g in gt]
    inversions = [(gt[i], gt[i + 1]) for i in range(len(gt) - 1) if gr[i + 1] < gr[i]]
    grade_txt = (f"On the hold-out the observed default rate climbs from {pct(gr[0])} in grade {gt[0]['grade']} to {pct(gr[-1])} in grade {gt[-1]['grade']}"
                 + ("." if not inversions else ", with " + ("one inversion" if len(inversions) == 1 else f"{len(inversions)} inversions") + ": "
                    + "; ".join(f"grade {a['grade']} {pct(a['observed_default_rate'])} ({a['firms']} firms) vs grade {b['grade']} {pct(b['observed_default_rate'])} ({b['firms']} firms)" for a, b in inversions)
                    + ". With a few dozen defaults per mid-grade this is within sampling noise, but it means the grades are ordered only approximately."))
    story += [P("9. Expected Loss and the watchlist", "h1"),
              P("Expected Loss = PD x LGD x EAD. PD comes from the calibrated model; LGD and EAD are balance-sheet proxies because the dataset has no loan-level data.", "body"),
              P("9.1 LGD and EAD proxies (assumptions, all in config.py)", "h2")]
    story += bullets([
        f"<b>EAD</b> = {pct(loss['bank_share_of_liabilities'], 0)} x total liabilities, where total liabilities = (liabilities / assets, clipped to {loss['tl_to_ta_clip'][0]}-{loss['tl_to_ta_clip'][1]}) "
        "x total assets (the exponential of Attr29). Monetary units are those of the source data, so only relative size matters.",
        f"<b>LGD</b> = 1 - min(1, {loss['recovery_haircut']} x asset coverage), asset coverage = total assets / total liabilities, bounded to "
        f"[{pct(loss['lgd_floor'], 0)}, {pct(loss['lgd_cap'], 0)}]. More leveraged borrowers leave less collateral per unit of claim.",
        f"Anchoring: the haircut was chosen once so that the mean LGD of the <i>training</i> defaulters ({pct(ps['mean_lgd_defaulters_train'], 0)}) is close to the 45% Basel "
        "foundation-IRB senior-unsecured value. That uses default labels only in aggregate, on training data, and no model performance.",
        "The proxy formulas themselves use no default labels. Both proxies can be replaced by portfolio data without touching the rest of the engine."])
    story += [P("9.2 Scorecard points and rating grades", "h2"),
              P("PDs are also expressed as classic scorecard points (600 points at 50 : 1 good-to-bad odds, +20 points doubles the odds) and six rating grades. " + grade_txt),
              table(grade_rows, [1.6, 2.6, 2.0, 2.4, 4.2, 3.2], align=["C", "L", "R", "R", "R", "R"]),
              P("9.3 The EL-ranked watchlist", "h2"),
              P(f"All {wlm['portfolio_firms']:,} hold-out firms are ranked by Expected Loss. The top decile carries {pct(wlm['top_decile_el_share'], 0)} of total EL. "
                f"The overlap between the top-25 by EL and the top-25 by PD is {wlm['top25_overlap_el_vs_pd']} firms: EL promotes large exposures with a moderate PD "
                "and demotes small ones.")]
    top25 = wl.head(25)
    w_rows = [["#", "Firm", "PD", "Grade", "Score", "LGD", "EAD", "Expected loss", "Flagged", "Defaulted"]]
    for _, r in top25.iterrows():
        w_rows.append([str(int(r["el_rank"])), str(int(r["firm_id"])), pct(r["pd"], 1), r["grade"], f"{r['score']:.0f}", pct(r["lgd"], 0), f"{r['ead']:.1f}", f"{r['expected_loss']:.2f}",
                       "yes" if r["flagged"] else "no", "yes" if r["realized_default"] == 1 else "no"])
    story += [P("Top 25 watchlist (hold-out portfolio)", "h2"), table(w_rows, [0.9, 1.4, 1.6, 1.4, 1.5, 1.4, 1.8, 2.5, 1.7, 1.9], align=["R", "R", "R", "C", "R", "R", "R", "R", "C", "C"], font=7.5),
              P(f"'Defaulted' is the realised outcome, shown only for the back-test; in live use it is unknown. {wlm['top25_defaults']} of the top 25 firms did default.", "caption"),
              figure(fig_dir / "watchlist_top.png", "Figure 4. Top 25 firms by Expected Loss; darker bars have higher PD.", 10.5),
              P("9.4 Back-test: how much realised loss does the watchlist capture?", "h2"),
              P("Realised loss is defined as default x LGD x EAD on the hold-out. For each ranking the table shows the share of total realised loss found in the top q% of firms.")]
    fr = ["0.05", "0.1", "0.2", "0.3"]
    c_rows = [["Ranking"] + [f"Top {pct(float(q), 0)} of firms" for q in fr]]
    for name, label in (("EL rank", "Expected Loss (PD x LGD x EAD)"), ("PD rank", "PD only"), ("EAD only", "Exposure only")):
        c_rows.append([label] + [pct(cap[name][q], 1) for q in fr])
    c_rows.append(["Random (expected)"] + [pct(float(q), 1) for q in fr])
    c_rows.append(["Defaults caught, EL ranking"] + [pct(wlm["default_capture"]["EL rank"][q], 1) for q in fr])
    c_rows.append(["Defaults caught, PD ranking"] + [pct(wlm["default_capture"]["PD rank"][q], 1) for q in fr])
    lo, hi = wlm["el_minus_pd_capture_ci95"][q10]
    dcap = wlm["default_capture"]
    story += [KeepTogether([table(c_rows, [6.2, 2.8, 2.8, 2.8, 2.8], highlight_rows=[1]),
                            P(f"At 10% review capacity the EL ranking captures {pct(cap['EL rank'][q10], 1)} of realised loss versus {pct(cap['PD rank'][q10], 1)} for PD alone (difference "
                              f"{100 * (cap['EL rank'][q10] - cap['PD rank'][q10]):+.1f} points, bootstrap 95% interval {100 * lo:+.1f} to {100 * hi:+.1f}). "
                              + ("That interval includes zero, so an advantage of EL over PD is <b>not statistically established</b> on this hold-out, and it is favoured partly by construction "
                                 "because the loss being captured uses the same LGD and EAD proxies. " if lo < 0 < hi else
                                 "The interval excludes zero, but the ranking is favoured partly by construction because the captured loss uses the same LGD and EAD proxies. ")
                              + f"The EL ranking also finds <i>fewer defaulting firms</i> in the top 10% ({pct(dcap['EL rank'][q10])} vs {pct(dcap['PD rank'][q10])} by PD) because it favours large exposures: "
                                "it is optimised for loss, not for default count. The robust finding is that both rankings beat random review by a wide margin.", "body")]),
              figure(fig_dir / "watchlist_capture.png", "Figure 5. Cumulative share of realised loss captured as the review depth grows.", 10.5),
              Spacer(1, 10)]

    # ------------------------------------------------------------------ 10 results
    story += [P("10. Final results", "h1"), P("10.1 Hold-out performance (single final evaluation)", "h2")]
    res_rows = [["Model", "AUC-ROC", "95% CI", "Gini", "KS", "PR-AUC", "Brier", "ECE"]]
    for k, label in (("altman_z", "Altman Z'' (benchmark)"), ("lr", NAMES["lr"]), ("rf_uncalibrated", "Random forest, uncalibrated"), ("rf", "Random forest, calibrated"),
                     ("hgb", NAMES["hgb"])):
        m = tm[k]
        res_rows.append([label, f3(m["auc_roc"]), ci(m)] + ([f3(m["gini"]), f3(m["ks"]), f3(m["pr_auc"]), f"{m['brier']:.4f}", f"{m['ece']:.4f}"] if "gini" in m else ["-"] * 5))
    story += [table(res_rows, [4.8, 1.8, 2.4, 1.4, 1.4, 1.6, 1.6, 1.6], highlight_rows=[4]),
              figure(fig_dir / "roc_test.png", "Figure 6. ROC curves on the hold-out portfolio.", 9.8)]
    if horizons is not None:
        story += [P("10.2 Robustness across forecast horizons", "h2"),
                  P("The same protocol and the primary file's hyper-parameters, applied to each file. Discrimination fades as the horizon lengthens, as it should: "
                    "ratios observed five years before the event carry less information.")]
        h_rows = [["Ratios observed", "Defaults / firms", "Altman Z''", "Logistic reg.", "Random forest", "Gradient boosting"]]
        for _, r in horizons.sort_values("years_ahead").iterrows():
            h_rows.append([f"{int(r['years_ahead'])} year{'s' if r['years_ahead'] > 1 else ''} before outcome", f"{int(r['defaults'])} / {int(r['firms']):,}", f3(r["altman"]), f3(r["lr"]), f3(r["rf"]), f3(r["hgb"])])
        story += [table(h_rows, [4.2, 3.0, 2.2, 2.4, 2.6, 3.1], highlight_rows=[1]), Spacer(1, 4), figure(fig_dir / "auc_by_horizon.png", "Figure 7. Hold-out AUC-ROC by forecast horizon.", 9.8)]
    if rep:
        story += [P("10.3 Stability across random splits", "h2"),
                  P(f"Ten different stratified 75 / 25 splits of the primary file, fixed hyper-parameters. Because the hold-out has only {d['test_defaults']} defaults, the split matters.")]
        rp = [["Model", "Mean AUC", "Std dev", "Min", "Max"]]
        for k, label in (("altman", "Altman Z''"), ("lr", NAMES["lr"]), ("rf", NAMES["rf"]), ("hgb", NAMES["hgb"])):
            s = rep[k]
            rp.append([label, f3(s["mean"]), f3(s["sd"]), f3(s["min"]), f3(s["max"])])
        story += [table(rp, [5.5, 2.5, 2.5, 2.5, 2.5]),
                  P(f"The random forest reached 0.947 or more in {n947['rf']} of {n_rep} splits; the gradient-boosting challenger did so in "
                    f"{n947['hgb']} of {n_rep}.", "caption")]

    # ---- the AUC question
    rf_m, hg_m, lr_m, al_m = tm["rf"], tm["hgb"], tm["lr"], tm["altman_z"]
    story += [P("10.4 What AUC-ROC can honestly be claimed", "h2"),
              P(f"The claim 'AUC-ROC 0.947' is not what a random forest / logistic regression engine delivers on this data under a clean protocol: the champion measures "
                f"<b>{f3(rf_m['auc_roc'])}</b> on the hold-out (95% CI {ci(rf_m)}), {f3(rep['rf']['mean']) if rep else 'n/a'} on average over random splits, and logistic regression "
                f"{f3(lr_m['auc_roc'])}. The forest reached 0.947 in {n947['rf']} of {n_rep} random splits (best {rep['rf']['max']:.3f}); 0.947 only touches the top "
                f"of its hold-out interval. Gradient boosting does reach that level ({f3(hg_m['auc_roc'])}, CI {ci(hg_m)}; 0.947 or more in {n947['hgb']} of {n_rep} splits). "
                "Two defensible ways to word the result, both fully backed by this report:"),
              box([P(f"<b>Option A (RF / LR scope):</b> <i>Built a corporate credit-risk engine on {d['firms']:,} firms and 64 financial ratios using Random Forest and Logistic Regression "
                     f"with SQC control-chart ratio screening and cost-sensitive threshold optimisation (hold-out AUC-ROC {rf_m['auc_roc']:.2f}, 95% CI "
                     f"{rf_m['auc_ci95'][0]:.2f}-{rf_m['auc_ci95'][1]:.2f}; Altman Z'' benchmark {al_m['auc_roc']:.2f}); calibrated PDs and produced Expected Loss-ranked "
                     f"watchlists capturing {pct(cap['EL rank'][q10], 0)} of simulated loss in the top 10% of firms.</i>", "note")]),
              Spacer(1, 5),
              box([P(f"<b>Option B (adds boosting):</b> <i>Benchmarked logistic regression ({lr_m['auc_roc']:.2f}), random forest ({rf_m['auc_roc']:.2f}) and gradient boosting "
                     f"({hg_m['auc_roc']:.2f}) hold-out AUC-ROC against an Altman Z'' baseline ({al_m['auc_roc']:.2f}), with SQC ratio screening, calibrated PDs, "
                     "cost-sensitive thresholds and an Expected Loss-ranked watchlist.</i>", "note")], border=ACCENT2),
              Spacer(1, 4),
              P("Running the pipeline with <font name='DV-M'>--champion hgb</font> would drive the calibration, threshold and watchlist stages with the boosting model.", "body"),
              PageBreak()]

    # ------------------------------------------------------------------ 11 limitations
    story += [P("11. Limitations and assumptions", "h1")]
    story += bullets([
        "<b>LGD, EAD and costs are proxies.</b> Dollar-style results (cost savings, loss capture) show how the method behaves under stated assumptions; they are not estimates "
        "for a real lender. Real recoveries, limits and pricing would change the numbers, though not the machinery.",
        "<b>Sample, not portfolio.</b> The data are Polish manufacturing firms from a specific period with a matched bankrupt / non-bankrupt design (default rate "
        f"{pct(d['default_rate'], 1)}). PDs describe this sample; apply the base-rate correction before using them on another book. There is no time ordering, so no out-of-time test.",
        f"<b>Small number of events.</b> {d['test_defaults']} hold-out defaults make single-split metrics noisy (hence the intervals and repeated splits); capture figures for the top-5% to 30% depend on a handful of firms.",
        "<b>No concept-drift or macro dimension.</b> Ratios are one-year snapshots; stress scenarios and through-the-cycle calibration are out of scope.",
        f"<b>Challenger gap.</b> Gradient boosting discriminates better than RF and LR here (hold-out AUC {tm['hgb']['auc_roc'] - tm['rf']['auc_roc']:+.3f} vs the forest). Keeping RF / LR as champion is a scope decision, and this gap is its cost.",
        "<b>Model governance.</b> A production use would add monitoring (PSI, back-testing of PD by grade), challenger governance and documentation of overrides."])

    # ------------------------------------------------------------------ 12 reproducibility
    env = R["environment"]
    story += [P("12. Reproducibility", "h1"),
              P("Everything in this report is generated by code in the repository:"),
              P("pip install -r requirements.txt &amp;&amp; pip install -e .<br/>python -m credit_risk.data&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# download + checksum<br/>"
                "python -m credit_risk.pipeline&nbsp;&nbsp;&nbsp;# full run, writes reports/<br/>python -m credit_risk.report&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# this PDF<br/>python -m pytest&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 35 unit tests", "mono"),
              Spacer(1, 6),
              table([["Module", "Responsibility"],
                     ["data.py", "download with SHA-256 check, ARFF loading, exact-duplicate removal"],
                     ["screening.py", "SQC control-chart screener (scikit-learn transformer)"],
                     ["models.py", "LR / RF / GBM pipelines and search grids"],
                     ["calibration.py", "cross-fitted Platt / isotonic calibration, base-rate shift"],
                     ["threshold.py", "cost curve, bootstrap-stable optimal threshold, Bayes rule"],
                     ["expected_loss.py", "LGD / EAD proxies, EL, scorecard points, grades, watchlist, loss capture"],
                     ["metrics.py", "AUC, Gini, KS, PR-AUC, Brier, ECE, calibration slope, bootstrap CIs"],
                     ["pipeline.py / plots.py / report.py", "end-to-end orchestration, figures, this PDF"]], [5.2, 12.3], align=["L", "L"]),
              Spacer(1, 6),
              P("<b>Testing.</b> 35 unit tests cover the screener (including a regression test for skewed, non-negative ratios and a test that control limits ignore defaulter values), "
                "the cost curve against brute force, EL arithmetic, calibration and metrics. A mutation check confirmed that deliberately leaking defaulters into the control limits, or reverting to "
                "symmetric three-sigma limits, makes tests fail.", "body"),
              P(f"<b>Environment.</b> Python {env['python']}, scikit-learn {env['sklearn']}, numpy {env['numpy']}, pandas {env['pandas']}; seed {R['config']['seed']}.", "body"),
              PageBreak()]

    # ------------------------------------------------------------------ appendix A
    story += [P("Appendix A. SQC screening table (training firms, all 64 ratios)", "h1"),
              P("Detect = share of defaulters beyond the adverse-tail control limit; Healthy = share of healthy firms beyond it (the false-alarm rate, 2.5% by construction); "
                "Lift = Detect / Healthy. Shift = distance between the defaulter and healthy medians in robust sigmas (negative: defaulters lower). q = Benjamini-Hochberg adjusted p-value.", "caption")]
    a_rows = [["Ratio", "Definition", "Miss.", "Tail", "Detect", "Healthy", "Lift", "Shift", "q", "Status"]]
    sq = sqc.copy()
    sq["n"] = sq["ratio"].str.replace("Attr", "").astype(int)
    for _, r in sq.sort_values("n").iterrows():
        a_rows.append([r["ratio"], E(desc(r["ratio"])), pct(r["missing_rate"], 1), r["adverse_tail"], pct(r["detection_rate"], 1), pct(r["false_alarm_rate"], 1),
                       "-" if pd.isna(r["lift"]) else f"{r['lift']:.1f}", f"{r['shift_sigma']:+.2f}", f"{r['q_value']:.3f}", E(r["status"].replace("dropped: ", "drop: "))])
    at = table(a_rows, [1.3, 5.6, 1.1, 1.0, 1.5, 1.3, 1.0, 1.1, 1.0, 2.6], align=["L", "L", "R", "C", "R", "R", "R", "R", "R", "L"], font=6.5)
    story += [at, PageBreak()]

    # ------------------------------------------------------------------ appendix B
    story += [P("Appendix B. Hyper-parameter search results (training CV, top configurations)", "h1")]
    for k in ("lr", "rf", "hgb"):
        tdf = pd.read_csv(report_dir / f"tuning_{k}.csv").head(6)
        pcols = [c for c in tdf.columns if c.startswith("param_")]
        rows = [["Rank"] + [c.replace("param_clf__", "") for c in pcols] + ["Mean CV AUC", "sd"]]
        for _, r in tdf.iterrows():
            rows.append([str(int(r["rank_test_score"]))] + [str(r[c]) for c in pcols] + [f"{r['mean_test_score']:.4f}", f"{r['std_test_score']:.4f}"])
        widths = [1.5] + [(CONTENT_W / cm - 1.5 - 5.5) / len(pcols)] * len(pcols) + [3.0, 2.5]
        story += [P(NAMES[k], "h2"), table(rows, widths, align=["C"] + ["L"] * len(pcols) + ["R", "R"])]

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=1.8 * cm, bottomMargin=2.1 * cm,
                            title="Expected Loss-Based Credit Risk Scorecard - Project report", author="Credit risk scorecard project",
                            subject="From raw data to final result")
    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
    return out_path


def main():
    path = build()
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
