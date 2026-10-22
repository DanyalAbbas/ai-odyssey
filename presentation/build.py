#!/usr/bin/env python3
"""Build PRESENTATION.pdf from presentation/slides.md.

Slides are separated by a line containing only `---`. Each chunk is converted
with pandoc, wrapped in a fixed 16:9 section, and printed with headless Chrome.

    python presentation/build.py            # writes PRESENTATION.pdf
    python presentation/build.py --html     # keep the intermediate HTML

Needs `pandoc` and `google-chrome` on PATH (both are present on this machine).
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SLIDES = HERE / "slides.md"
CSS = HERE / "style.css"
OUT = ROOT / "PRESENTATION.pdf"
BRAND = "Team Mamdani · TechFest 2026"


def pandoc(chunk: str) -> str:
    proc = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "html"],
        input=chunk, text=True, capture_output=True,
    )
    if proc.returncode != 0:
        sys.exit(f"pandoc failed:\n{proc.stderr}")
    return proc.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="store_true", help="keep the intermediate HTML")
    args = ap.parse_args()

    for tool in ("pandoc", "google-chrome"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found on PATH")

    chunks = [c for c in re.split(r"(?m)^---\s*$", SLIDES.read_text()) if c.strip()]
    sections = []
    for i, chunk in enumerate(chunks, 1):
        title = "<!-- title -->" in chunk
        chunk = chunk.replace("<!-- title -->", "")
        body = pandoc(chunk)
        # Chrome resolves relative image paths against the HTML file, which lives
        # in /tmp; make them absolute.
        body = body.replace('src="results/', f'src="{ROOT}/results/')
        if not title:
            m = re.match(r"\s*(<h1[^>]*>.*?</h1>)\s*(.*)", body, re.S)
            if m:
                body = f'{m.group(1)}\n<div class="content">{m.group(2)}</div>'
        cls = "slide title" if title else "slide"
        brand = "" if title else f'<div class="brand">{BRAND}</div>'
        sections.append(
            f'<section class="{cls}">\n{body}\n{brand}\n'
            f'<div class="num">{i} / {len(chunks)}</div>\n</section>'
        )

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>TechFest 2026 — Team Mamdani</title>"
        f"<style>{CSS.read_text()}</style></head><body>"
        + "\n".join(sections)
        + "</body></html>"
    )

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="deck_"))
    page = tmp / "deck.html"
    page.write_text(html)
    if args.html:
        (ROOT / "PRESENTATION.html").write_text(html)

    profile = tmp / "chrome"
    cmd = [
        "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer", f"--user-data-dir={profile}",
        "--virtual-time-budget=15000", f"--print-to-pdf={OUT}", page.as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not OUT.exists():
        sys.exit(f"chrome failed:\n{proc.stderr[-2000:]}")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {len(chunks)} slides)")


if __name__ == "__main__":
    main()
