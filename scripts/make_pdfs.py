import os
import markdown
from xhtml2pdf import pisa
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

BASE = "/home/user/KK"

CSS = """
<style>
@page {
    size: A4;
    margin: 2cm 1.8cm;
    @frame footer_frame {
        -pdf-frame-content: footer_content;
        bottom: 1cm; margin-left: 1.8cm; margin-right: 1.8cm; height: 1cm;
    }
}
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 20pt; color: #0b3d91; border-bottom: 2px solid #0b3d91; padding-bottom: 6px; margin-top: 0; }
h2 { font-size: 15pt; color: #0b3d91; margin-top: 22px; border-bottom: 1px solid #ccc; padding-bottom: 3px; }
h3 { font-size: 12.5pt; color: #1a1a1a; margin-top: 16px; }
p { margin: 6px 0; text-align: justify; }
code { font-family: Courier, monospace; font-size: 9pt; }
pre { font-family: Courier, monospace; background-color: #f5f5f5; border: 0.5px solid #ccc;
      padding: 8px; font-size: 8.3pt; line-height: 1.3; }
table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 9.5pt; }
th, td { border: 0.75px solid #999; padding: 5px 7px; text-align: left; }
th { background-color: #0b3d91; color: white; }
tr:nth-child(even) { background-color: #f4f6fb; }
img { max-width: 100%; margin: 10px 0; }
blockquote { border-left: 3px solid #0b3d91; margin: 8px 0; padding-left: 10px; color: #444; }
a { color: #0b3d91; }
hr { border: none; border-top: 1px solid #ccc; margin: 14px 0; }
ul, ol { margin: 4px 0 8px 18px; }
li { margin: 3px 0; }
#footer_content { font-size: 8pt; color: #888; text-align: center; }
</style>
"""

def link_callback(uri, rel):
    if uri.startswith("http"):
        return uri
    path = os.path.join(BASE, uri)
    if os.path.isfile(path):
        return path
    return uri

def md_to_pdf(md_path, pdf_path, doc_title):
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    html_body = markdown.markdown(
        text,
        extensions=["extra", "tables", "fenced_code", "sane_lists", "toc"],
    )
    html = f"""<html><head><meta charset="utf-8">{CSS}</head>
<body>
<div id="footer_content">{doc_title} &nbsp;|&nbsp; <pdf:pagenumber /> / <pdf:pagecount /></div>
{html_body}
</body></html>"""
    with open(pdf_path, "wb") as out:
        result = pisa.CreatePDF(html, dest=out, link_callback=link_callback)
    print(pdf_path, "-> ERROR" if result.err else "OK")

def code_to_pdf(code_path, pdf_path, doc_title):
    with open(code_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    formatter = HtmlFormatter(style="default", noclasses=True, nowrap=True)
    numbered_lines = []
    for i, line in enumerate(lines, start=1):
        highlighted = highlight(line if line.strip() else " ", PythonLexer(), formatter)
        numbered_lines.append(
            f'<tr><td class="ln">{i}</td><td class="src">{highlighted}</td></tr>'
        )
    body = "<table class='code'>" + "".join(numbered_lines) + "</table>"
    extra_css = """
    <style>
    body { font-family: Helvetica, Arial, sans-serif; font-size: 9pt; }
    h1 { font-size: 16pt; color: #0b3d91; border-bottom: 2px solid #0b3d91; padding-bottom: 6px; }
    table.code { width: 100%; border-collapse: collapse; }
    table.code td { border: none; padding: 0; vertical-align: top; }
    table.code td.ln { color: #999; padding-right: 8px; text-align: right; width: 24px;
                        font-family: Courier, monospace; font-size: 7.3pt; }
    table.code td.src { font-family: Courier, monospace; font-size: 7.3pt; line-height: 1.3;
                         white-space: pre-wrap; }
    </style>
    """
    html = f"""<html><head><meta charset="utf-8">{extra_css}</head>
<body>
<h1>{doc_title}</h1>
<p style="font-size:9pt;color:#555;">Source file: {os.path.basename(code_path)} &mdash; {len(lines)} lines</p>
<div id="footer_content">{doc_title} &nbsp;|&nbsp; <pdf:pagenumber /> / <pdf:pagecount /></div>
{body}
</body></html>"""
    with open(pdf_path, "wb") as out:
        result = pisa.CreatePDF(html, dest=out, link_callback=link_callback)
    print(pdf_path, "-> ERROR" if result.err else "OK")


if __name__ == "__main__":
    pdf_dir = os.path.join(BASE, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    md_to_pdf(os.path.join(BASE, "report.md"), os.path.join(pdf_dir, "report.pdf"),
              "Explainable Histopathology Image Classification -- Project Report")
    md_to_pdf(os.path.join(BASE, "learning_notes.md"), os.path.join(pdf_dir, "learning_notes.pdf"),
              "Learning Notes -- Explainable Histopathology Image Classification")
    code_to_pdf(os.path.join(BASE, "code.py"), os.path.join(pdf_dir, "code.pdf"),
                "code.py -- Explainable Histopathology Image Classification")
