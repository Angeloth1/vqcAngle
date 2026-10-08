"""
How often the 2022-2025 literature on variational / quantum-neural-network classifiers names each component of the
classifier (encoding, ansatz, training, readout).  Supports the statement on the readout in the Introduction (Figure 1).

Corpus
  OpenAlex [Priem et al. 2022] works API, filter
      title_and_abstract.search: QUERY (below), publication_year: 2022-2025, is_paratext: false
  Cursor paging can return a work twice; a record that repeats an OpenAlex id is marked (column repeat) and not counted.
  A work is kept ("in scope") only if the title or abstract, as returned, contains one of the phrases in CORE and a word
  beginning with "classif" (OpenAlex search stems words, so its result set is wider than the phrases).
  Shares are computed over the in-scope works that have an abstract in OpenAlex, and over the 100 most cited of them:
  the first 100 when those works are sorted by citation count (descending), ties by OpenAlex id (ascending, as text).

Measure
  A work "names" a component when its title or abstract matches the component's keyword rule (case-insensitive).
  The title is the OpenAlex display_name; the abstract is rebuilt from the OpenAlex inverted index (words in position
  order, joined by single spaces); a rule is matched with re.search(rule, title + " " + abstract, re.IGNORECASE).
  A mention is not a study of the component; the measure counts attention, not results.  Three rule sets are
  reported: 'default' (used in Figure 1), 'strict' (narrower phrases) and 'no-entangle' (default without
  entangl* in the ansatz rule), as a sensitivity check.  OpenAlex lists some papers as several works (a preprint and
  a published version); the log also gives the shares with each title counted once.

    python3 paper/analysis/bibliometric_components.py             # queries OpenAlex (network), ~1 min
    python3 paper/analysis/bibliometric_components.py --recount   # no network: every value, from the stored corpus

Writes paper/tables/bibliometric_corpus.csv (Dataset S1: one row per record, with its citation count, rank and flags),
paper/tables/bibliometric_summary.csv and paper/tables/bibliometric_log.txt.  Citation counts change daily; the corpus
records the retrieval date.  --recount reads only the corpus, prints the log and checks it against
bibliometric_summary.csv; it writes no file.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = HERE.parent / "tables"
API = "https://api.openalex.org/works"
QUERY = ('("variational quantum classifier" OR "variational quantum classifiers" OR "variational quantum circuit" OR '
         '"variational quantum circuits" OR "quantum neural network" OR "quantum neural networks" OR '
         '"parameterized quantum circuit" OR "parameterized quantum circuits" OR "parametrized quantum circuit" OR '
         '"parametrized quantum circuits") AND (classification OR classifier OR classifiers)')
YEARS = "2022-2025"
CORE = r"variational quantum classifiers?|variational quantum circuits?|quantum neural networks?|parametri[sz]ed quantum circuits?"
TOP = 100
RULES = {
    "default": {
        "encoding": r"\bencod\w*|\bembedd?ing\w*|\bfeature[- ]maps?\b|\bdata[- ]loading\b|\bre-?upload\w*",
        "ansatz": r"\bans[aä]tz\w*|\bexpressib\w*|\bexpressiv\w*|\bcircuit (?:architecture|design|structure|depth|topology)\w*"
                  r"|\barchitecture search\b|\bentangl\w*",
        "training": r"\bbarren\b|\btrainab\w*|\bgradient\w*|\boptimi[sz]er\w*|\binitiali[sz]\w*|\blandscape\w*|\boverparametri[sz]\w*",
        "readout": r"\bread-?outs?\b|\bobservables?\b|\bmeasurements?\b|\bpost-?processing\b|\boutput (?:layer|decoding|mapping)\w*",
    },
    "strict": {
        "encoding": r"\bdata[- ]encoding|\bencoding (?:scheme|method|strateg|technique)\w*|\b(?:amplitude|angle|basis|dense|iqp) "
                    r"(?:encoding|embedding)\b|\bfeature[- ]maps?\b|\bquantum embedding\w*|\bre-?upload\w*",
        "ansatz": r"\bans[aä]tz\w*|\bexpressib\w*|\bcircuit architecture\w*|\barchitecture search\b",
        "training": r"\bbarren plateau\w*|\btrainab\w*|\boptimi[sz]ers?\b|\binitiali[sz]ation\b|\blandscape\w*",
        "readout": r"\bread-?outs?\b|\bobservables?\b|\bmeasurement (?:strateg|scheme|protocol|basis)\w*|\bpost-?processing\b",
    },
}
RULES["no-entangle"] = dict(RULES["default"], ansatz=r"\bans[aä]tz\w*|\bexpressib\w*|\bexpressiv\w*"
                            r"|\bcircuit (?:architecture|design|structure|depth|topology)\w*|\barchitecture search\b")
COMP = ("encoding", "ansatz", "training", "readout")
# Manual reading of the default-rule readout matches among the 100 most cited (Table S13): 1 = the work concerns the
# readout of a classifier or the processing of its measurement outcomes, 0 = it uses the words in another sense.
MANUAL_READOUT = {
    "W4293025139": 1,   # QuantumNAT: post-measurement normalisation and quantisation
    "W4220710689": 1,   # all-qubit multi-observable measurement strategy
    "W4210586732": 1,   # readout as a factor of classification performance
    "W4387846277": 1,   # Pauli-Z and basis measurement
    "W4410770879": 1,   # learnable observable
    "W4393200016": 0,   # measurement-based quantum computing
    "W4391243055": 0,   # "the observable universe"
    "W4365504435": 0,   # measurements as operations of the model
    "W4414349340": 0,   # observables in classical simulation
    "W4294335599": 0,   # measurement operators in topological field theory
    "W4385462354": 0,   # locality of observables as a cause of barren plateaus
    "W4390704502": 0,   # error mitigation by ensembles of classifiers
}
COLS = ["id", "doi", "year", "cited", "type", "in_scope", "has_abstract", "repeat", "rank", "top100", "manual_readout",
        "title"] + [f"{n}_{k}" for n in RULES for k in COMP] + [f"{n}_title_{k}" for n in RULES for k in COMP] + ["retrieved"]
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def get(url):
    for i in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "paper-analysis/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:                                                     # noqa: BLE001
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"OpenAlex request failed: {url}")


def abstract(w):
    inv = w.get("abstract_inverted_index") or {}
    pos = {p: word for word, ps in inv.items() for p in ps}
    return " ".join(pos[i] for i in sorted(pos))


def fetch():
    out, cursor = [], "*"
    flt = f"title_and_abstract.search:{QUERY},publication_year:{YEARS},is_paratext:false"
    while cursor:
        q = {"filter": flt, "per-page": 200, "cursor": cursor,
             "select": "id,doi,display_name,publication_year,cited_by_count,type,abstract_inverted_index"}
        d = get(API + "?" + urllib.parse.urlencode(q))
        for w in d["results"]:
            out.append(dict(id=w["id"], doi=w.get("doi") or "", title=w.get("display_name") or "", year=w["publication_year"],
                            cited=w["cited_by_count"], type=w.get("type") or "", abstract=abstract(w)))
        cursor = d["meta"].get("next_cursor") if d["results"] else None
        time.sleep(0.2)
    return out


def flags(text, rules):
    return {k: bool(re.search(v, text, re.I)) for k, v in rules.items()}


def flag(W):
    for w in W:
        text = w["title"] + " " + w["abstract"]
        w["in_scope"] = int(bool(re.search(CORE, text, re.I) and re.search(r"\bclassif\w*", text, re.I)))
        w["has_abstract"] = int(bool(w["abstract"]))
        w["manual_readout"] = MANUAL_READOUT.get(w["id"].rsplit("/", 1)[1], "")
        for name, rules in RULES.items():
            for k, v in flags(text, rules).items():
                w[f"{name}_{k}"] = int(v)
            for k, v in flags(w["title"], rules).items():
                w[f"{name}_title_{k}"] = int(v)


def rank(W):
    """Sort the records by citations (ties by OpenAlex id), mark repeated ids and rank the in-scope works with an abstract."""
    W.sort(key=lambda w: (-w["cited"], w["id"]))
    seen = set()
    for w in W:
        w["repeat"], w["rank"], w["top100"] = int(w["id"] in seen), "", 0
        seen.add(w["id"])
    base = [w for w in W if w["in_scope"] and w["has_abstract"] and not w["repeat"]]
    for i, w in enumerate(base, 1):
        w["rank"], w["top100"] = i, int(i <= TOP)
    return base


def write_corpus(W, retrieved):
    with open(T / "bibliometric_corpus.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        wr.writeheader()
        for w in W:
            wr.writerow(dict(w, retrieved=retrieved))


def read_corpus(path):
    W = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            w = dict(r, year=int(r["year"]), cited=int(r["cited"]), repeat=r.get("repeat", ""), rank=r.get("rank", ""),
                     top100=r.get("top100", ""), retrieved=r.get("retrieved", ""))
            for c in ["in_scope", "has_abstract"] + [c for c in r if c.split("_")[0] in RULES]:
                w[c] = int(r[c] in ("1", "True"))
            m = r.get("manual_readout", "")
            w["manual_readout"] = int(m) if m != "" else ""
            W.append(w)
    return W


def table(sets, indent):
    rows = []
    for name in RULES:
        for sname, sub in sets.items():
            n = len(sub)
            for k in COMP:
                kt = sum(w[f"{name}_{k}"] for w in sub)
                ktitle = sum(w[f"{name}_title_{k}"] for w in sub)
                rows.append(dict(rules=name, set=sname, n=n, component=k, named_title_or_abstract=kt,
                                 share_title_or_abstract=round(kt / n, 4), named_title=ktitle, share_title=round(ktitle / n, 4)))
            say(f"{indent}{name:12s} {sname:7s} n={n:4d}  title or abstract: " +
                "  ".join(f"{r['component']} {100 * r['named_title_or_abstract'] / n:5.1f}%" for r in rows[-4:]) +
                "   | title only: " + "  ".join(f"{r['component']} {100 * r['named_title'] / n:4.1f}%" for r in rows[-4:]))
    return rows


def count(W, base, retrieved):
    works = [w for w in W if not w["repeat"]]
    scope = [w for w in works if w["in_scope"]]
    top = base[:TOP]
    say(f"OpenAlex works API, retrieved {retrieved}")
    say(f"query: title_and_abstract.search:{QUERY}")
    say(f"years {YEARS}, is_paratext false: {len(W)} records, {len(W) - len(works)} of them repeat an id: {len(works)} works; "
        f"in scope (phrase + classif- in title/abstract): {len(scope)}; with abstract: {len(base)}; "
        f"{TOP} most cited: citations >= {top[-1]['cited']}")
    rows = table({"all": base, f"top{TOP}": top}, "  ")
    say("\nkeyword rules:")
    for name, rules in RULES.items():
        for k in COMP:
            say(f"  {name:12s} {k:9s} {rules[k]}")
    say(f"\nreadout matches (default rules) among the {TOP} most cited, for manual inspection:")
    for w in top:
        if w["default_readout"]:
            say(f"  [{w['cited']:4d}] {w['year']} {w['title'][:110]}  {w['doi']}")

    # The cut at rank TOP can fall inside a group of works with equal citations; the id order then decides which of
    # them are counted.  Report the range of every count over all choices of the tied works.
    cut = top[-1]["cited"]
    above = [w for w in base if w["cited"] > cut]
    tied = [w for w in base if w["cited"] == cut]
    k = TOP - len(above)
    say(f"\nties at the cut: {len(tied)} works have {cut} citations (ranks {len(above) + 1}-{len(above) + len(tied)}), "
        f"{k} of them among the {TOP} most cited; ties are broken by OpenAlex id (ascending, as text)")
    say(f"  counts among the {TOP} most cited, minimum-maximum over every choice of the tied works:")
    for name in RULES:
        rng = {}
        for col in [f"{name}_{c}" for c in COMP] + [f"{name}_title_{c}" for c in COMP]:
            fixed, ones = sum(w[col] for w in above), sum(w[col] for w in tied)
            rng[col] = f"{fixed + max(0, k - (len(tied) - ones))}-{fixed + min(k, ones)}"
        say(f"    {name:12s} title or abstract: " + "  ".join(f"{c} {rng[f'{name}_{c}']}" for c in COMP) +
            "   | title only: " + "  ".join(f"{c} {rng[f'{name}_title_{c}']}" for c in COMP))
    m = [w for w in top if w["default_readout"]]
    yes, no = sum(w["manual_readout"] == 1 for w in m), sum(w["manual_readout"] == 0 for w in m)
    say(f"\nmanual check of the {len(m)} readout matches among the {TOP} most cited (column manual_readout): {yes} concern "
        f"the readout of a classifier, {no} use the words in another sense" + (f", {len(m) - yes - no} unchecked" if len(m) - yes - no else ""))

    # Sensitivity: a paper listed as several works (preprint and published version) counted once.
    keep, titles = [], set()
    for w in base:
        t = re.sub(r"[^a-z0-9]+", " ", w["title"].lower()).strip()
        if t not in titles:
            titles.add(t)
            keep.append(w)
    say(f"\nsame title counted once (lower case, characters other than a-z and 0-9 read as spaces; the most cited record "
        f"kept): {len(keep)} works, {len(base) - len(keep)} records merged; {TOP} most cited: citations >= {keep[TOP - 1]['cited']}")
    table({"all": keep, f"top{TOP}": keep[:TOP]}, "  ")
    return rows


def write_outputs(rows):
    with open(T / "bibliometric_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (T / "bibliometric_log.txt").write_text("\n".join(LOG) + "\n")


def recount():
    W = read_corpus(T / "bibliometric_corpus.csv")
    stored = {id(w): (w["repeat"], w["rank"], w["top100"]) for w in W}
    base = rank(W)
    moved = sum(stored[id(w)] != (str(w["repeat"]), str(w["rank"]), str(w["top100"])) for w in W)
    rows = count(W, base, W[0]["retrieved"])
    with open(T / "bibliometric_summary.csv", newline="") as f:
        same = list(csv.DictReader(f)) == [{k: str(v) for k, v in r.items()} for r in rows]
    print(f"\nrecount: summary {'identical to' if same else 'DIFFERS from'} bibliometric_summary.csv; "
          f"{moved} stored repeat/rank/top100 values differ from the recomputed ones", file=sys.stderr)


def main():
    if "--recount" in sys.argv[1:]:
        recount()
        return
    retrieved = dt.date.today().isoformat()
    W = fetch()
    flag(W)
    base = rank(W)
    write_corpus(W, retrieved)
    write_outputs(count(W, base, retrieved))


if __name__ == "__main__":
    main()
