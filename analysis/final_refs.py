"""
Pin every citable work to ONE authoritative registry record and cache the metadata.

Each entry is either a DOI (looked up in Crossref) or an arXiv id (looked up in the arXiv API).
Venue overrides are applied only where the registry record is the wrong *version* (preprint vs journal)
or where the registry has no venue field; each override was checked against the publisher/registry page
(see paper/references/VERIFICATION.md).  Output: paper/references/refs_final.json

    .venv/bin/python paper/analysis/final_refs.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_refs import arxiv_meta, crossref_by_doi, meta_from_crossref, sim   # noqa: E402

OUT = HERE.parents[1] / "paper" / "references"
OUT.mkdir(parents=True, exist_ok=True)

# key -> (kind, id, expected title fragment)
FINAL = {
    "thomos2026":        ("doi", "10.3390/jsan15040067", "Evaluation of Variational Quantum Classifiers"),
    "tavallaee2009":     ("doi", "10.1109/CISDA.2009.5356528", "A detailed analysis of the KDD CUP 99 data set"),
    "cerezo2021vqa":     ("doi", "10.1038/s42254-021-00348-9", "Variational quantum algorithms"),
    "havlicek2019":      ("doi", "10.1038/s41586-019-0980-2", "Supervised learning with quantum-enhanced feature spaces"),
    "kumar2025quids":    ("doi", "10.1016/j.jnca.2024.104072", "QuIDS"),
    "qi2024vqcnn":       ("doi", "10.1093/comjnl/bxae062", "VQCNN"),
    "abreu2024qmlids":   ("doi", "10.1109/ISCC61673.2024.10733655", "QML-IDS"),
    "kim2024":           ("doi", "10.1038/s41598-024-78389-0", "Quantum intrusion detection system using outlier analysis"),
    "kukliansky2024":    ("doi", "10.1109/TQE.2024.3359574", "Network anomaly detection using quantum neural networks"),
    "kaissar2026":       ("doi", "10.3390/fi18050234", "Enhancing network intrusion detection with quantum machine learning"),
    "bowles2024":        ("arxiv", "2403.07059", "Better than classical"),
    "preskill2018":      ("doi", "10.22331/q-2018-08-06-79", "Quantum computing in the NISQ era and beyond"),
    "schuld2021encoding": ("doi", "10.1103/PhysRevA.103.032430", "Effect of data encoding on the expressive power"),
    "larose2020":        ("doi", "10.1103/PhysRevA.102.032420", "Robust data encodings for quantum classifiers"),
    "rath2024":          ("doi", "10.1140/epjqt/s40507-024-00285-3", "Quantum data encoding"),
    "kadi2025":          ("doi", "10.1109/OJCOMS.2025.3537957", "Quantum-Classical Encoding Methods"),
    "wang2025amp":       ("arxiv", "2503.01545", "Limitations of amplitude encoding on quantum classification"),
    "schuld2021kernel":  ("arxiv", "2101.11020", "Supervised quantum machine learning models are kernel methods"),
    "schuld2019feature": ("doi", "10.1103/PhysRevLett.122.040504", "Quantum machine learning in feature Hilbert spaces"),
    "pedregosa2011":     ("arxiv", "1201.0490", "Scikit-learn"),
    "harris2020":        ("doi", "10.1038/s41586-020-2649-2", "Array programming with NumPy"),
    "virtanen2020":      ("doi", "10.1038/s41592-019-0686-2", "SciPy 1.0"),
    "javadi2024":        ("arxiv", "2405.08810", "Quantum computing with Qiskit"),
    "micci2001":         ("doi", "10.1145/507533.507538", "high-cardinality categorical attributes"),
    "jolliffe2002":      ("doi", "10.1007/b98835", "Principal Component Analysis"),
    "powell1994":        ("doi", "10.1007/978-94-015-8330-5_4", "A direct search optimization method"),
    "kandala2017":       ("doi", "10.1038/nature23879", "Hardware-efficient variational quantum eigensolver"),
    "welch1947":         ("doi", "10.1093/biomet/34.1-2.28", "generalization of"),
    "hedges1981":        ("doi", "10.3102/10769986006002107", "Distribution theory for Glass"),
    "efron1993":         ("doi", "10.1007/978-1-4899-4541-9", "An introduction to the bootstrap"),
    "shende2006":        ("doi", "10.1109/TCAD.2005.855930", "Synthesis of quantum-logic circuits"),
    "mottonen2005":      ("doi", "10.26421/qic5.6-5", "Transformation of quantum states using uniformly controlled rotations"),
    "plesch2011":        ("doi", "10.1103/PhysRevA.83.032302", "Quantum-state preparation with universal gate decompositions"),
    "iten2016":          ("doi", "10.1103/PhysRevA.93.032318", "Quantum circuits for isometries"),
    "sun2023":           ("doi", "10.1109/TCAD.2023.3244885", "Asymptotically optimal circuit depth"),
    "huang2021power":    ("doi", "10.1038/s41467-021-22539-9", "Power of data in quantum machine learning"),
    "thanasilp2024":     ("doi", "10.1038/s41467-024-49287-w", "Exponential concentration in quantum kernel methods"),
    "shaydulin2022":     ("doi", "10.1103/PhysRevA.106.042407", "Importance of kernel bandwidth"),
    "canatar2023":       ("arxiv", "2206.06686", "Bandwidth enables generalization in quantum kernel models"),
    "kubler2021":        ("arxiv", "2106.03747", "The inductive bias of quantum kernels"),
    "jerbi2023":         ("doi", "10.1038/s41467-023-36159-y", "Quantum machine learning beyond kernel methods"),
    "sim2019":           ("doi", "10.1002/qute.201900070", "Expressibility and entangling capability"),
    "mcclean2018":       ("doi", "10.1038/s41467-018-07090-4", "Barren plateaus in quantum neural network training landscapes"),
    "larocca2025":       ("doi", "10.1038/s42254-025-00813-9", "Barren plateaus in variational quantum computing"),
    "cerezo2025sim":     ("doi", "10.1038/s41467-025-63099-6", "Does provable absence of barren plateaus imply classical simulability"),
    "huang2020shadows":  ("doi", "10.1038/s41567-020-0932-7", "Predicting many properties of a quantum system"),
    "wang2021nibp":      ("doi", "10.1038/s41467-021-27045-6", "Noise-induced barren plateaus"),
    "sweke2020":         ("doi", "10.22331/q-2020-08-31-314", "Stochastic gradient descent for hybrid quantum-classical optimization"),
    "helstrom1969":      ("doi", "10.1007/BF01007479", "Quantum detection and estimation theory"),
    "cover1965":         ("doi", "10.1109/PGEC.1965.264137", "Geometrical and statistical properties of systems of linear inequalities"),
    "luque2019":         ("doi", "10.1016/j.patcog.2019.02.023", "The impact of class imbalance in classification performance metrics"),
    "moustafa2015":      ("doi", "10.1109/MilCIS.2015.7348942", "UNSW-NB15"),
    "sharafaldin2018":   ("doi", "10.5220/0006639801080116", "Toward generating a new intrusion detection dataset"),
    "montieri2024":      ("doi", "10.1109/OJCOMS.2024.3481895", "Few-shot class-incremental learning for network intrusion detection"),
    "gilfuster2024":     ("doi", "10.1038/s41467-024-45882-z", "Understanding quantum machine learning also requires rethinking generalization"),
    "perezsalinas2020":  ("doi", "10.22331/q-2020-02-06-226", "Data re-uploading for a universal quantum classifier"),
    "ali2025":           ("doi", "10.3390/app15041903", "Deep Learning vs. Machine Learning for Intrusion Detection in Computer Networks"),
}

# Venue overrides: applied on top of the registry record. Each was confirmed against the registry/publisher page.
OVERRIDES = {
    "pedregosa2011": dict(container="J. Mach. Learn. Res.", volume="12", pages="2825–2830", year=2011,
                          note="JMLR page confirmed at jmlr.org/papers/v12/pedregosa11a.html; arXiv:1201.0490"),
    "canatar2023":   dict(container="Trans. Mach. Learn. Res.", year=2023,
                          note="TMLR 2023 (web search: mlanthology.org/tmlr/2023/canatar2023tmlr-bandwidth); arXiv:2206.06686"),
    "kubler2021":    dict(container="Adv. Neural Inf. Process. Syst. 34 (NeurIPS 2021)", year=2021,
                          note="venue from the arXiv journal_ref field"),
    "bowles2024":    dict(container="arXiv preprint", year=2024),
    "schuld2021kernel": dict(container="arXiv preprint", year=2021),
    "wang2025amp":   dict(container="arXiv preprint", year=2025),
    "javadi2024":    dict(container="arXiv preprint", year=2024),
    "jolliffe2002":  dict(container="Principal Component Analysis, 2nd ed.; Springer: New York, NY, USA", year=2002,
                          note="book; Springer Series in Statistics; registry has no author list -> set manually"),
    "efron1993":     dict(container="An Introduction to the Bootstrap; Chapman & Hall/CRC: Boca Raton, FL, USA", year=1993),
    "powell1994":    dict(container="Advances in Optimization and Numerical Analysis; Kluwer: Dordrecht, The Netherlands", year=1994),
    "montieri2024":  dict(note="registry first author is Di Monda (not Montieri)"),
    "mottonen2005":  dict(first_family="Möttönen", note="registry omits diacritics"),
}
MANUAL_AUTHORS = {
    "jolliffe2002": ["Jolliffe, I.T."],
    "efron1993": ["Efron, B.", "Tibshirani, R.J."],
    "powell1994": ["Powell, M.J.D."],
}


def main():
    out = {}
    bad = []
    for key, (kind, ident, title_frag) in FINAL.items():
        try:
            if kind == "doi":
                m = meta_from_crossref(crossref_by_doi(ident))
            else:
                m = arxiv_meta(ident)
                m["container"] = "arXiv preprint"
            ok = title_frag.lower() in m["title"].lower() or sim(title_frag, m["title"]) > 0.8
            if not ok:
                bad.append((key, m["title"]))
            m.update(OVERRIDES.get(key, {}))
            if key in MANUAL_AUTHORS:
                m["authors"] = MANUAL_AUTHORS[key]
            m["kind"], m["id"] = kind, ident
            m["title_check"] = "OK" if ok else "MISMATCH"
            out[key] = m
            print(f"{'OK ' if ok else 'BAD'} {key:20s} {str(m.get('year')):5s} {m['title'][:80]}", flush=True)
        except Exception as e:                             # noqa: BLE001
            bad.append((key, f"ERROR {e}"))
            print(f"ERR {key}: {e}")
        time.sleep(0.3)
    (OUT / "refs_final.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n{len(out) - len(bad)}/{len(FINAL)} OK; problems: {bad}")


if __name__ == "__main__":
    main()
