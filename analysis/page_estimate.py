"""Rough MDPI page estimate for manuscript_draft_v1.md, calibrated on the control study [1]
(pdftotext of the published PDF: 17,328 words over 32 pages, tables and references included).

    .venv/bin/python paper/analysis/page_estimate.py
"""
import re
from pathlib import Path

WORDS_PER_PAGE = 17328 / 32
t = (Path(__file__).resolve().parents[1] / "manuscript_draft_v1.md").read_text()
t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
total = len(t.split())
refs = len(t[t.index("## References"):].split())
tables = sum(len(l.split()) for l in t.splitlines() if l.startswith("|"))
body = t[t.index("## 1. Introduction"):t.index("## Supplementary Materials")]
print(f"total words {total} (tables {tables}, references {refs})")
print(f"estimate: {total / WORDS_PER_PAGE:.1f} pages at {WORDS_PER_PAGE:.0f} words/page, plus Figure 1 (~0.5 page)")
print(f"body, Introduction to Conclusions incl. tables: {len(body.split())} words = {len(body.split()) / WORDS_PER_PAGE:.1f} pages")
