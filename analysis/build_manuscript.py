"""
Build paper/manuscript_draft_v1.md from paper/manuscript_src.md

  * `[@key]` / `[@key1; @key2]` citation tokens -> numbered `[n]` in order of first appearance (MDPI style)
  * reference list generated from paper/references/refs_final.json (metadata pinned to Crossref/arXiv records)
  * prints word counts per section (body text only: tables, figure blocks and the reference list are excluded)
  * fails loudly on unknown keys; warns on unused pinned references

    .venv/bin/python paper/analysis/build_manuscript.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # the paper/ directory
PARTS = sorted((ROOT / "src").glob("part_*.md"))
TABLES = ROOT / "tables" / "md"
OUT = ROOT / "manuscript_draft_v1.md"
REFS = json.loads((ROOT / "references" / "refs_final.json").read_text())

CITE = re.compile(r"\[@([A-Za-z0-9_]+(?:\s*;\s*@[A-Za-z0-9_]+)*)\]")


def initials(given: str) -> str:
    parts = [p for p in re.split(r"[\s.]+", given.strip()) if p]
    out = []
    for p in parts:
        if "-" in p:
            out.append("-".join(q[0].upper() + "." for q in p.split("-") if q))
        else:
            out.append(p[0].upper() + ".")
    return "".join(out)


def fmt_author(a: str, from_arxiv: bool) -> str:
    if from_arxiv:                                   # "Given Middle Family"
        toks = a.split()
        return f"{toks[-1]}, {initials(' '.join(toks[:-1]))}" if len(toks) > 1 else a
    if "," in a:                                     # "Family, Given"
        fam, giv = [s.strip() for s in a.split(",", 1)]
        return f"{fam}, {initials(giv)}" if giv else fam
    return a


def fmt_ref(key: str, m: dict) -> str:
    arxiv = m.get("kind") == "arxiv"
    manual = key in ("jolliffe2002", "efron1993", "powell1994")
    authors = [a if manual else fmt_author(a, arxiv) for a in m.get("authors", [])]
    authors = [(a.split(",")[0].title() + "," + a.split(",", 1)[1]) if ("," in a and a.split(",")[0].isupper() and len(a.split(",")[0]) > 1) else a
               for a in authors]
    if key == "mottonen2005":
        authors = [a.replace("Mottonen", "Möttönen") for a in authors]
    if len(authors) > 10:
        au = "; ".join(authors[:10]) + "; et al."
    else:
        au = "; ".join(authors)
    au = au.rstrip(".") + "."            # exactly one full stop before the title
    title = m["title"].strip().rstrip(".")
    if key == "welch1947":
        title = "The generalization of ‘Student’s’ problem when several different population variances are involved"
    if key == "hedges1981":
        title = "Distribution theory for Glass’s estimator of effect size and related estimators"
    if key == "sim2019":
        title = "Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms"
    year = m.get("year")
    cont = {"J. Mach. Learn. Res.": "Journal of Machine Learning Research",
            "Trans. Mach. Learn. Res.": "Transactions on Machine Learning Research",
            "Adv. Neural Inf. Process. Syst. 34 (NeurIPS 2021)": "Advances in Neural Information Processing Systems 34 (NeurIPS 2021)"
            }.get(m.get("container") or "", m.get("container") or "")
    vol, iss, pages = m.get("volume"), m.get("issue"), m.get("pages")
    if pages:
        pages = str(pages).replace("-", "–")
    doi = m.get("doi")
    if key in ("jolliffe2002", "efron1993", "powell1994"):
        book = cont.split(";")[0]
        pub = cont.split(";", 1)[1].strip() if ";" in cont else ""
        if key == "powell1994":
            return f"{au} {title}. In *{book}*; {pub}, {year}; pp. 51–67. https://doi.org/{doi}"
        return f"{au} *{book}*; {pub}, {year}. https://doi.org/{doi}"
    if arxiv and cont == "arXiv preprint":
        return f"{au} {title}. *arXiv* **{year}**, arXiv:{m['id']}."
    if key == "pedregosa2011":
        return f"{au} {title}. *{cont}* **{year}**, *{vol}*, {pages}."
    if key in ("canatar2023",):
        return f"{au} {title}. *{cont}* **{year}**. arXiv:{m['id']}."
    if key == "kubler2021":
        return f"{au} {title}. In *{cont}*; {year}. arXiv:{m['id']}."
    s = f"{au} {title}. *{cont}* **{year}**"
    if vol:
        s += f", *{vol}*"
        if iss and cont not in ("Transactions on Machine Learning Research",):
            s += f"({iss})"
    if pages:
        s += f", {pages}"
    s += "."
    if doi:
        s += f" https://doi.org/{doi}"
    return s


def compress(nums):
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ",".join(out)


def main():
    text = "\n".join(p.read_text() for p in PARTS)
    (ROOT / "manuscript_src.md").write_text(text)          # assembled source (for inspection)

    def table(mo):
        f = TABLES / f"{mo.group(1)}.md"
        if not f.exists():
            raise SystemExit(f"MISSING TABLE FILE: {f}")
        return f.read_text().strip()

    text = re.sub(r"\{\{TABLE:([A-Za-z0-9_]+)\}\}", table, text)
    order: dict[str, int] = {"thomos2026": 1}     # the control study is [1], as in the authors' own convention

    def repl(mo):
        keys = [k.strip().lstrip("@") for k in mo.group(1).split(";")]
        for k in keys:
            if k not in REFS:
                raise SystemExit(f"UNKNOWN CITATION KEY: {k}")
            order.setdefault(k, len(order) + 1)
        return "[" + compress([order[k] for k in keys]) + "]"

    body = CITE.sub(repl, text)
    unused = [k for k in REFS if k not in order]

    # ---- word counts per section (body prose only)
    counts, cur = {}, None
    for line in body.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            counts[cur] = 0
            continue
        if line.startswith("### "):
            continue
        if cur is None or line.startswith("|") or line.startswith("![") or line.startswith("<") or line.startswith(">"):
            continue
        counts[cur] += len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-\.,/×±≈≤≥<>=%]*", line))

    refs_md = ["## References", ""]
    for k, n in sorted(order.items(), key=lambda t: t[1]):
        refs_md.append(f"{n}. {fmt_ref(k, REFS[k])}")
    OUT.write_text(body.rstrip() + "\n\n" + "\n".join(refs_md) + "\n")

    (ROOT / "references" / "citation_order.json").write_text(json.dumps(order, indent=1))
    print(f"wrote {OUT}  ({len(order)} references cited; {len(REFS)} pinned)")
    if unused:
        print("pinned but unused:", ", ".join(unused))
    tot = 0
    print("\nwords per section (prose only):")
    for k, v in counts.items():
        if k.lower().startswith("references"):
            continue
        print(f"  {v:6d}  {k}")
        tot += v
    print(f"  {tot:6d}  TOTAL prose (excl. tables/figures/references)")


if __name__ == "__main__":
    main()
