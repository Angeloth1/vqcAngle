"""Write paper/references/VERIFICATION.md: one row per cited reference with the registry record it was checked against."""
import json
from pathlib import Path
P = Path(__file__).resolve().parents[1]
refs = json.loads((P / "references" / "refs_final.json").read_text())
order = json.loads((P / "references" / "citation_order.json").read_text())
rows = []
for k, n in sorted(order.items(), key=lambda t: t[1]):
    m = refs[k]
    reg = "Crossref" if m["kind"] == "doi" else "arXiv API"
    ident = m["id"]
    au = (m["authors"][0] if m.get("authors") else "").split(",")[0]
    note = m.get("note", "")
    if k == "welch1947": note = (note + " OUP DOI used (JSTOR DOI 10.2307/2332510 also resolves)").strip()
    if k in ("bowles2024", "schuld2021kernel", "wang2025amp", "javadi2024"): note = (note + " arXiv preprint; no journal version located").strip()
    rows.append(f"| {n} | `{k}` | {reg} | {ident} | {m['title'][:90]} | {au} | {m.get('year')} | {m['title_check']} | {note} |")
out = ["# Reference verification report", "",
       "Every reference cited in `manuscript_draft_v1.md` was resolved against an authoritative registry (Crossref for DOIs, the arXiv API for preprints) by `paper/analysis/verify_refs.py` and pinned by `paper/analysis/final_refs.py`. "
       "A record passes only if its registry title and first author match the expectation. Metadata in the reference list is generated from these records, not typed. "
       "Venue overrides (JMLR pages, TMLR, NeurIPS, book publishers) were confirmed separately, as noted.", "",
       "**What this does and does not establish.** It establishes that each work exists and that title, first author, year and venue are those of the registry record. It does not establish that a cited work says what the manuscript says it does; that was checked against abstracts for the intrusion-detection and encoding papers (Crossref/arXiv abstracts), and by title only for well-known methodological references.", "",
       "| # | key | registry | identifier | registry title | first author | year | title check | note |", "|---|---|---|---|---|---|---|---|---|"] + rows
extra = ["", "## Candidates verified but not cited", "", "`sim2019` (Sim et al., expressibility), `cerezo2025sim` (classical simulability).", "",
         "## Corrections made to inherited citations", "",
         "* The paper on few-shot class-incremental learning for network intrusion detection (IEEE OJCOMS 2024) is listed in the control study with a different first author; the registry record gives Di Monda, Montieri, Persico, Voria, De Ieso, Pescapè, which is what this draft uses.",
         "* The control study cites the Kaggle mirror of NSL-KDD; this draft cites Tavallaee et al. (2009), the paper that defines the benchmark. **[TODO: authors to decide whether to add the dataset page.]**"]
(P / "references" / "VERIFICATION.md").write_text("\n".join(out + extra) + "\n")
print("written", len(rows), "rows")
