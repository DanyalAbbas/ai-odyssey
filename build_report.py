#!/usr/bin/env python3
"""Build REPORT.pdf from REPORT.md.

    .venv/bin/python build_report.py            # writes REPORT.pdf
    .venv/bin/python build_report.py --html     # keep the intermediate HTML

Needs `pandoc` and `google-chrome` on PATH.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "REPORT.md"
OUT = ROOT / "REPORT.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
html, body { margin: 0; padding: 0; }
body {
  font-family: "DejaVu Serif", Georgia, serif;
  font-size: 10.5pt; line-height: 1.45; color: #1b1b1b;
}
h1 {
  font-family: "DejaVu Sans", Arial, sans-serif;
  font-size: 21pt; line-height: 1.2; color: #123c5a;
  text-align: center; margin: 4mm 0 6mm 0;
}
h2 {
  font-family: "DejaVu Sans", Arial, sans-serif;
  font-size: 14pt; color: #123c5a;
  border-bottom: 0.4mm solid #123c5a;
  padding-bottom: 1.5mm; margin: 7mm 0 3mm 0;
  page-break-after: avoid;
}
h3 {
  font-family: "DejaVu Sans", Arial, sans-serif;
  font-size: 11.5pt; color: #123c5a;
  margin: 5mm 0 2mm 0; page-break-after: avoid;
}
p { margin: 0 0 2.4mm 0; text-align: justify; }
ul, ol { margin: 0 0 2.6mm 0; padding-left: 5.5mm; }
li { margin-bottom: 1mm; }
strong { color: #0f3350; }
a { color: #14507a; text-decoration: none; }
hr { border: none; border-top: 0.2mm solid #c9d3db; margin: 4mm 0; }
table {
  border-collapse: collapse; width: 100%;
  font-size: 8.4pt; line-height: 1.3;
  margin: 2mm 0 4mm 0; page-break-inside: avoid;
}
table.numeric { font-size: 8pt; }
table.numeric th:not(:first-child),
table.numeric td:not(:first-child) { white-space: nowrap; }
table.wide { font-size: 7.4pt; }
table.wide th, table.wide td { padding: 0.7mm 0.8mm; }
.nw { white-space: nowrap; }
th, td {
  border: 0.2mm solid #b9c4cc;
  padding: 1.2mm 1.6mm; text-align: left; vertical-align: top;
}
th { background: #eaf0f5; color: #123c5a; font-family: "DejaVu Sans", sans-serif; }
tr:nth-child(even) td { background: #f7f9fa; }
code {
  font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt;
  background: #f2f4f6; padding: 0 0.6mm;
}
pre {
  background: #f6f8f9; border-left: 0.8mm solid #123c5a;
  padding: 2.2mm 3mm; margin: 2mm 0 3mm 0;
  font-size: 7.9pt; line-height: 1.3;
  white-space: pre-wrap; page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 7.9pt; }
blockquote {
  margin: 2mm 0 3mm 0; padding-left: 4mm;
  border-left: 0.8mm solid #9fb3c2; color: #33414c;
}
img { max-width: 100%; display: block; margin: 3mm auto 2mm auto; }
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="store_true", help="keep the intermediate HTML")
    args = ap.parse_args()

    for tool in ("pandoc", "google-chrome"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found on PATH")

    proc = subprocess.run(
        ["pandoc", str(SRC), "-f", "markdown", "-t", "html5", "-s",
         "--metadata", "pagetitle=TechFest Face Recognition — Final Report"],
        cwd=ROOT, text=True, capture_output=True,
    )
    if proc.returncode != 0:
        sys.exit(f"pandoc failed:\n{proc.stderr}")

    html = proc.stdout.replace('src="results/', f'src="{ROOT}/results/')
    if "</head>" not in html:
        sys.exit("pandoc did not emit a <head>")
    html = html.replace("</head>", f"<style>{CSS}</style></head>")

    # The cover metadata is one markdown paragraph; put each field on its own line.
    for field in ("Institution:", "Date:", "Competition:"):
        html = html.replace(f"<strong>{field}</strong>", f"<br/><strong>{field}</strong>", 1)

    # Tables whose cells are all short keep values on one line so Chrome does
    # not break "98.63 %" across two lines; headers use non-breaking hyphens.
    def mark_numeric(match: re.Match[str]) -> str:
        table = match.group(0)
        for a, b in (("Rank-1", "Rank\u20111"), ("TAR@1e-3", "TAR@1e\u20113"),
                     ("TAR@1e-4", "TAR@1e\u20114")):
            table = table.replace(a, b)
        table = re.sub(r"(?<=[\d.])\s+%", "\u00a0%", table)
        table = re.sub(r"\[([\d.,\s]+)\]",
                       lambda m: "[" + m.group(1).replace(" ", "\u00a0") + "]", table)
        # Short headers should never break mid-token ("Rank-" / "1").
        def nw_th(m: re.Match[str]) -> str:
            inner = m.group(1)
            text = re.sub(r"<[^>]+>", "", inner).strip()
            return (f'<th><span class="nw">{inner}</span></th>'
                    if len(text) <= 14 else m.group(0))

        table = re.sub(r"<th(?:\s[^>]*)?>(.*?)</th>", nw_th, table, flags=re.S)
        # Pandoc's colgroup widths are computed from the markdown separator row
        # and routinely squeeze numeric columns; let the browser size them.
        table = re.sub(r"<colgroup>.*?</colgroup>\s*", "", table, flags=re.S)
        cells = re.findall(r"<t[dh](?:\s[^>]*)?>(.*?)</t[dh]>", table, re.S)
        if not cells:
            return table
        ncols = len(re.findall(r"<th(?:\s[^>]*)?>", table))
        longest = max(len(re.sub(r"<[^>]+>", "", c).strip()) for c in cells)
        if ncols >= 7:
            cls = "wide"
        elif ncols <= 6 and longest <= 22:
            cls = "numeric"
        else:
            cls = ""
        return re.sub(r"<table[^>]*>", f'<table class="{cls}">' if cls else "<table>", table, count=1)

    html = re.sub(r"<table[^>]*>.*?</table>", mark_numeric, html, flags=re.S)

    if args.html:
        (ROOT / "REPORT.html").write_text(html)

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="report_"))
    page = tmp / "report.html"
    page.write_text(html)

    cmd = [
        "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer", f"--user-data-dir={tmp / 'chrome'}",
        "--virtual-time-budget=30000", f"--print-to-pdf={OUT}", page.as_uri(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not OUT.exists():
        sys.exit(f"chrome failed:\n{res.stderr[-2000:]}")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
