"""
Numeric audit of the manuscript.

Every number in the manuscript prose and tables (reference list excluded) is looked up in a universe built from
  * all generated tables and logs in paper/tables (CSV, Markdown, text logs)
  * the constants transcribed from the control study (published_ref1.py)
  * a small whitelist of design constants (N, reps, seeds, C, shots, section/table numbers ...)
A number is 'matched' when some universe value rounds to it at the number of decimals it is written with,
either as-is or multiplied by 100 (accuracy points). Unmatched numbers are listed for manual review;
they are either derived quantities (ratios, differences of two matched numbers) or errors.

    .venv/bin/python paper/analysis/audit_numbers.py
"""
from __future__ import annotations
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE.parent
sys.path.insert(0, str(HERE))
import published_ref1 as P1

NUM = re.compile(r"(?<![\w.])[−+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[−+-]?\d+(?:\.\d+)?")

def nums(text):
    out = []
    for m in NUM.finditer(text):
        s = m.group(0).replace("−", "-").replace(",", "")
        try:
            out.append((s, float(s)))
        except ValueError:
            pass
    return out

# ---------------- universe
uni = set()
def add_text(t):
    for s, v in nums(t):
        uni.add(abs(v))
for f in list((P / "tables").glob("*.csv")) + list((P / "tables").glob("*.txt")) + list((P / "tables" / "md").glob("*.md")):
    add_text(f.read_text())
def add_obj(o):
    if isinstance(o, (int, float)): uni.add(abs(float(o)))
    elif isinstance(o, dict):
        for v in o.values(): add_obj(v)
    elif isinstance(o, (list, tuple)):
        for v in o: add_obj(v)
for name in ("TABLE3", "A2", "TABLE5", "TABLEA1_N16", "TABLE6", "FACTS", "A2_REPS5_PRINTED"):
    add_obj(getattr(P1, name))
# design constants and structural numbers
for v in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,32,36,37,40,48,50,64,72,73,100,120,128,136,137,150,200,240,256,400,432,500,
          1000,2000,10000,100000,0.25,0.5,0.05,0.95,0.01,0.001,20153,5039,3759,3752,22544,20155,5037,125973,
          2026,2025,2024,2023,2021,2020,2019,2018]:
    uni.add(float(v))

def matches(tok, val):
    d = len(tok.split(".")[1]) if "." in tok else 0
    a = abs(val)
    tol = 0.5 * 10 ** (-d) + 1e-9
    for u in uni:
        for scale in (1.0, 100.0):
            if abs(u * scale - a) <= tol: return True
        # values stored as percentages in the universe (e.g. '+3.6') matched directly above
    return False

# ---------------- manuscript text
txt = (P / "manuscript_draft_v1.md").read_text()
body = txt[:txt.index("## References")]
body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
body = re.sub(r"\[\d+(?:[,–]\d+)*\]", " ", body)                # citation numbers
body = re.sub(r"\$\$.*?\$\$", " ", body, flags=re.S)             # equations
body = re.sub(r"`[^`]*`", " ", body)                             # code spans
body = re.sub(r"https?://\S+", " ", body)
body = re.sub(r"(?m)^#+ .*$", " ", body)                         # headings (section numbers)
body = re.sub(r"\bTable S?\d+[ab]?\b|\bFigure \d\b|\bSection \d(?:\.\d)?\b|\bE\d\b|\bRQ\d\b|\bS\d[ab]?\b|\bS8[ab]\b", " ", body)
body = re.sub(r"10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+", " ", body)                     # powers of ten
body = re.sub(r"2[ⁿ⁰¹²³]+|2¹⁶|2ⁿ|log₂N|ℓ₂|x̂|N = \d+|n = \d+", lambda m: " " + m.group(0) + " ", body)

bad = {}
for s, v in nums(body):
    if not matches(s, v):
        bad.setdefault(s, 0); bad[s] += 1
print(f"numbers checked: {len(nums(body))}   unmatched distinct tokens: {len(bad)}")
for s in sorted(bad, key=lambda x: float(x.replace(',',''))): 
    # show context of first occurrence
    i = body.replace("−","-").find(s)
    ctx = body.replace("−","-")[max(0,i-60):i+40].replace("\n"," ")
    print(f"  {s:>10s} x{bad[s]}  ...{ctx}...")
