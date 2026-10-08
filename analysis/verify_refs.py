"""
Verify every candidate reference against an authoritative registry before it may be cited.

Rule: a reference enters the manuscript only if
  * its DOI resolves in Crossref and the registry title and first-author surname match the expectation, or
  * (arXiv-only works) the arXiv API returns the expected title and first author for the given identifier.
Anything that fails is reported and excluded. Nothing here is taken from memory: the expectations
below are only what we compare the registry against.

    .venv/bin/python paper/analysis/verify_refs.py
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "paper" / "references"
OUT.mkdir(parents=True, exist_ok=True)
UA = "Mozilla/5.0 (reference-verification script; anonymous)"


def get(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).split()


def sim(a, b):
    return difflib.SequenceMatcher(None, " ".join(norm(a)), " ".join(norm(b))).ratio()


def crossref_by_doi(doi):
    j = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    return j


def crossref_search(title, author):
    q = urllib.parse.urlencode({"query.bibliographic": f"{title} {author}", "rows": 5,
                                "select": "DOI,title,author,issued,container-title,type"})
    items = json.loads(get("https://api.crossref.org/works?" + q))["message"]["items"]
    return items


def meta_from_crossref(j):
    au = j.get("author", [])
    dp = (j.get("issued") or j.get("published-print") or j.get("published-online") or {}).get("date-parts", [[None]])[0]
    return dict(
        doi=j.get("DOI"),
        title=(j.get("title") or [""])[0],
        authors=[f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in au],
        first_family=(au[0].get("family") if au else (j.get("editor") or [{}])[0].get("family", "")),
        year=dp[0] if dp else None,
        container=(j.get("container-title") or [""])[0],
        volume=j.get("volume"), issue=j.get("issue"),
        pages=j.get("page") or j.get("article-number"),
        publisher=j.get("publisher"), type=j.get("type"),
    )


def arxiv_meta(aid):
    x = get("https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(aid))
    ent = re.search(r"<entry>(.*?)</entry>", x, flags=re.S)
    if not ent:
        return None
    e = ent.group(1)
    title = " ".join(re.search(r"<title>(.*?)</title>", e, flags=re.S).group(1).split())
    authors = [" ".join(a.split()) for a in re.findall(r"<author>\s*<name>(.*?)</name>", e, flags=re.S)]
    pub = re.search(r"<published>(\d{4})", e)
    doi = re.search(r"<arxiv:doi[^>]*>(.*?)</arxiv:doi>", e)
    jref = re.search(r"<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>", e, flags=re.S)
    return dict(title=title, authors=authors, first_family=authors[0].split()[-1] if authors else "",
                year=int(pub.group(1)) if pub else None, arxiv=aid,
                doi=doi.group(1) if doi else None, journal_ref=" ".join(jref.group(1).split()) if jref else None)


# key, kind ('doi'|'search'|'arxiv'), identifier or title, expected first-author family, expected title fragment
CANDIDATES = [
    # --- the control paper
    ("thomos2026", "doi", "10.3390/jsan15040067", "Thomos", "Evaluation of Variational Quantum Classifiers"),
    # --- QML background, trainability, expressibility
    ("cerezo2021vqa", "doi", "10.1038/s42254-021-00348-9", "Cerezo", "Variational quantum algorithms"),
    ("havlicek2019", "doi", "10.1038/s41586-019-0980-2", "Havlicek", "Supervised learning with quantum-enhanced feature spaces"),
    ("bowles2024", "arxiv", "2403.07059", "Bowles", "Better than classical"),
    ("preskill2018", "doi", "10.22331/q-2018-08-06-79", "Preskill", "Quantum computing in the NISQ era and beyond"),
    ("bharti2022", "doi", "10.1103/RevModPhys.94.015004", "Bharti", "Noisy intermediate-scale quantum"),
    ("mcclean2018", "doi", "10.1038/s41467-018-07090-4", "McClean", "Barren plateaus in quantum neural network training landscapes"),
    ("larocca2025", "doi", "10.1038/s42254-025-00813-9", "Larocca", "Barren plateaus in variational quantum computing"),
    ("cerezo2025sim", "doi", "10.1038/s41467-025-63099-6", "Cerezo", "Does provable absence of barren plateaus imply classical simulability"),
    ("sim2019", "doi", "10.1002/qute.201900070", "Sim", "Expressibility and entangling capability"),
    ("abbas2021", "doi", "10.1038/s43588-021-00084-1", "Abbas", "The power of quantum neural networks"),
    ("kandala2017", "doi", "10.1038/nature23879", "Kandala", "Hardware-efficient variational quantum eigensolver"),
    ("wang2021nibp", "search", "Noise-induced barren plateaus in variational quantum algorithms", "Wang", "Noise-induced barren plateaus"),
    ("sweke2020", "search", "Stochastic gradient descent for hybrid quantum-classical optimization", "Sweke", "Stochastic gradient descent for hybrid quantum-classical optimization"),
    # --- encodings and kernels
    ("schuld2021kernel", "arxiv", "2101.11020", "Schuld", "Supervised quantum machine learning models are kernel methods"),
    ("schuld2019feature", "search", "Quantum machine learning in feature Hilbert spaces", "Schuld", "Quantum machine learning in feature Hilbert spaces"),
    ("schuld2021encoding", "doi", "10.1103/PhysRevA.103.032430", "Schuld", "Effect of data encoding on the expressive power"),
    ("huang2021power", "doi", "10.1038/s41467-021-22539-9", "Huang", "Power of data in quantum machine learning"),
    ("thanasilp2024", "search", "Exponential concentration in quantum kernel methods", "Thanasilp", "Exponential concentration in quantum kernel methods"),
    ("shaydulin2022", "search", "Importance of kernel bandwidth in quantum machine learning", "Shaydulin", "Importance of kernel bandwidth"),
    ("canatar2023", "arxiv", "2206.06686", "Canatar", "Bandwidth enables generalization in quantum kernel models"),
    ("kubler2021", "arxiv", "2106.03747", "Kübler", "The inductive bias of quantum kernels"),
    ("jerbi2023", "search", "Quantum machine learning beyond kernel methods", "Jerbi", "Quantum machine learning beyond kernel methods"),
    ("gilfuster2024", "doi", "10.1038/s41467-024-45882-z", "Gil-Fuster", "Understanding quantum machine learning also requires rethinking generalization"),
    ("perezsalinas2020", "doi", "10.22331/q-2020-02-06-226", "Pérez-Salinas", "Data re-uploading for a universal quantum classifier"),
    ("larose2020", "search", "Robust data encodings for quantum classifiers", "LaRose", "Robust data encodings for quantum classifiers"),
    ("rath2024", "doi", "10.1140/epjqt/s40507-024-00285-3", "Rath", "Quantum data encoding"),
    ("kadi2025", "doi", "10.1109/OJCOMS.2025.3537957", "Kadi", "quantum-classical encoding"),
    ("helstrom1969", "search", "Quantum detection and estimation theory", "Helstrom", "Quantum detection and estimation theory"),
    ("cover1965", "search", "Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition", "Cover", "Geometrical and statistical properties of systems of linear inequalities"),
    # --- state preparation cost
    ("mottonen2005", "search", "Transformation of quantum states using uniformly controlled rotations", "Möttönen", "Transformation of quantum states using uniformly controlled rotations"),
    ("shende2006", "search", "Synthesis of quantum-logic circuits", "Shende", "Synthesis of quantum-logic circuits"),
    ("plesch2011", "search", "Quantum-state preparation with universal gate decompositions", "Plesch", "Quantum-state preparation with universal gate decompositions"),
    ("iten2016", "search", "Quantum circuits for isometries", "Iten", "Quantum circuits for isometries"),
    ("sun2023", "search", "Asymptotically optimal circuit depth for quantum state preparation and general unitary synthesis", "Sun", "Asymptotically optimal circuit depth"),
    ("huang2020shadows", "search", "Predicting many properties of a quantum system from very few measurements", "Huang", "Predicting many properties of a quantum system"),
    # --- intrusion detection
    ("tavallaee2009", "doi", "10.1109/CISDA.2009.5356528", "Tavallaee", "A detailed analysis of the KDD CUP 99 data set"),
    ("kukliansky2024", "doi", "10.1109/TQE.2024.3359574", "Kukliansky", "Network anomaly detection using quantum neural networks"),
    ("kumar2025quids", "doi", "10.1016/j.jnca.2024.104072", "Kumar", "QuIDS"),
    ("qi2024vqcnn", "doi", "10.1093/comjnl/bxae062", "Qi", "VQCNN"),
    ("abreu2024qmlids", "doi", "10.1109/ISCC61673.2024.10733655", "Abreu", "QML-IDS"),
    ("kim2024", "doi", "10.1038/s41598-024-78389-0", "Kim", "Quantum intrusion detection system using outlier analysis"),
    ("kaissar2026", "doi", "10.3390/fi18050234", "Kaissar", "Enhancing network intrusion detection with quantum machine learning"),
    ("eze2025", "doi", "10.3390/electronics14091827", "Eze", "Quantum-enhanced machine learning for cybersecurity"),
    ("montieri2024", "doi", "10.1109/OJCOMS.2024.3481895", "Montieri", "Few-shot class-incremental learning for network intrusion detection"),
    ("moustafa2015", "doi", "10.1109/MilCIS.2015.7348942", "Moustafa", "UNSW-NB15"),
    ("sharafaldin2018", "doi", "10.5220/0006639801080116", "Sharafaldin", "Toward generating a new intrusion detection dataset"),
    ("luque2019", "doi", "10.1016/j.patcog.2019.02.023", "Luque", "The impact of class imbalance in classification performance metrics"),
    ("buda2018", "doi", "10.1016/j.neunet.2018.07.011", "Buda", "A systematic study of the class imbalance problem"),
    # --- software, methods, statistics
    ("powell1994", "doi", "10.1007/978-94-015-8330-5_4", "Powell", "A direct search optimization method"),
    ("spall1998", "doi", "10.1109/7.705889", "Spall", "Implementation of the simultaneous perturbation algorithm"),
    ("pedregosa2011", "arxiv", "1201.0490", "Pedregosa", "Scikit-learn"),
    ("harris2020", "doi", "10.1038/s41586-020-2649-2", "Harris", "Array programming with NumPy"),
    ("virtanen2020", "doi", "10.1038/s41592-019-0686-2", "Virtanen", "SciPy 1.0"),
    ("javadi2024", "arxiv", "2405.08810", "Javadi-Abhari", "Quantum computing with Qiskit"),
    ("micci2001", "search", "A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems", "Micci-Barreca", "high-cardinality categorical attributes"),
    ("welch1947", "search", "The generalization of Student's problem when several different population variances are involved", "Welch", "generalization of"),
    ("hedges1981", "search", "Distribution theory for Glass's estimator of effect size and related estimators", "Hedges", "Distribution theory for Glass"),
    ("efron1993", "doi", "10.1007/978-1-4899-4541-9", "Efron", "An introduction to the bootstrap"),
    ("nielsen2010", "doi", "10.1017/CBO9780511976667", "Nielsen", "Quantum computation and quantum information"),
    ("jolliffe2002", "search", "Principal Component Analysis", "Jolliffe", "Principal component analysis"),
    ("hastie2009", "doi", "10.1007/978-0-387-84858-7", "Hastie", "The elements of statistical learning"),
]


def check(key, kind, ident, exp_author, exp_title):
    rec = dict(key=key, kind=kind, query=ident, status="FAIL", note="")
    try:
        if kind == "doi":
            m = meta_from_crossref(crossref_by_doi(ident))
        elif kind == "arxiv":
            m = arxiv_meta(ident)
            if m is None:
                rec["note"] = "arXiv returned no entry"
                return rec
        else:  # search: take the best title match among the top hits, then re-fetch by DOI
            items = crossref_search(ident, exp_author)
            scored = sorted(((sim(ident, (it.get("title") or [""])[0]), it) for it in items), key=lambda t: -t[0])
            best_s, best = scored[0]
            if best_s < 0.85:
                rec["note"] = f"no Crossref hit with title similarity >= 0.85 (best {best_s:.2f}: {(best.get('title') or [''])[0][:80]!r})"
                rec["candidates"] = [(round(s, 2), (i.get('title') or [''])[0][:90], i.get('DOI')) for s, i in scored[:3]]
                return rec
            m = meta_from_crossref(crossref_by_doi(best["DOI"]))
            rec["title_similarity_to_query"] = round(best_s, 3)
        t_ok = exp_title.lower() in m["title"].lower() or sim(exp_title, m["title"]) > 0.8
        a_ok = sim(exp_author, m["first_family"]) > 0.75 or norm(exp_author)[-1:] == norm(m["first_family"])[-1:]
        rec.update(meta=m, title_ok=bool(t_ok), author_ok=bool(a_ok))
        rec["status"] = "OK" if (t_ok and a_ok) else "MISMATCH"
        if not (t_ok and a_ok):
            rec["note"] = f"expected author~{exp_author!r}, title~{exp_title!r}; registry says {m['first_family']!r}, {m['title']!r}"
    except Exception as e:                            # noqa: BLE001
        rec["note"] = f"error: {e}"
    return rec


def main():
    out = []
    for k, kind, ident, au, ti in CANDIDATES:
        r = check(k, kind, ident, au, ti)
        out.append(r)
        m = r.get("meta") or {}
        print(f"{r['status']:8s} {k:20s} {str(m.get('first_family', ''))[:16]:16s} {str(m.get('year', '')):5s} {str(m.get('title', ''))[:70]:70s} {m.get('doi') or m.get('arxiv') or ''}  {r['note'][:110]}", flush=True)
        time.sleep(0.4)
    (OUT / "verification_raw.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    ok = sum(r["status"] == "OK" for r in out)
    print(f"\n{ok}/{len(out)} verified OK")


if __name__ == "__main__":
    main()
