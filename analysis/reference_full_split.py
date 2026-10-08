"""
Attainable reference of the anchor circuit (E1) on the records of E1.

E1 trains the published four-qubit circuit on the full training partition (20,153 records)
and scores it on the full internal partition (5,039 records), KDDTest+ (22,544) and the
3,752 novel-signature records. E2 trains its fidelity-kernel SVM on a 2,000-record
subsample and scores it on 2,000 internal records, so E2 is not the reference of E1.
This script trains the SVM of Section 3.2 (precomputed amplitude fidelity kernel,
F = (x_hat . y_hat)^2, N = 16, C in {1, 100}) on exactly the records of E1 and scores it
on exactly the records of E1. The training set is the whole partition, so there is no
seed: one value per C.

    VQCANGLE=/mnt/Main/src/ship/vqcAngle  $VQCANGLE/.venv/bin/python analysis/reference_full_split.py

Writes tables/reference_full_split.csv and tables/reference_full_split_log.txt.
Peak memory about 7 GB (the 20,153 x 20,153 kernel in float64 is 3.0 GiB).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC

VQC = Path(os.environ.get("VQCANGLE", "/mnt/Main/src/ship/vqcAngle"))
sys.path.insert(0, str(VQC))
from qids.data import load_arrays          # noqa: E402
from qids.fastsim import l2_normalise      # noqa: E402
from novel_ref1 import novel_mask_ref1     # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tables"
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def kernel(A, B):
    K = A @ B.T
    np.square(K, out=K)
    return K


def main():
    A = load_arrays(n_components=16)
    nov = novel_mask_ref1()
    Ztr, Zte, Zoo = l2_normalise(A.Xtr), l2_normalise(A.Xte), l2_normalise(A.Xood)
    say(f"records: train {len(Ztr)}  internal {len(Zte)}  KDDTest+ {len(Zoo)}  novel {int(nov.sum())}")
    t0 = time.perf_counter()
    K = kernel(Ztr, Ztr)
    say(f"training kernel {K.shape[0]} x {K.shape[1]} built in {time.perf_counter() - t0:.1f} s")

    rows = []
    for C in (1.0, 100.0):
        t0 = time.perf_counter()
        svm = SVC(kernel="precomputed", C=C, cache_size=2000).fit(K, A.ytr)

        def predict(Z, chunk=4000):
            return np.concatenate([svm.predict(kernel(Z[lo:lo + chunk], Ztr)) for lo in range(0, len(Z), chunk)])

        p_tr, p_te, p_oo = svm.predict(K), predict(Zte), predict(Zoo)
        r = dict(N=16, encoding="amplitude", C=C, n_train=len(Ztr), n_sv=int(svm.n_support_.sum()),
                 train_acc=float((p_tr == A.ytr).mean()),
                 internal=float((p_te == A.yte).mean()), internal_correct=int((p_te == A.yte).sum()),
                 kddtest=float((p_oo == A.yood).mean()),
                 novel_recall=float((p_oo[nov] == A.yood[nov]).mean()), novel_correct=int((p_oo[nov] == 1).sum()))
        rows.append(r)
        say(f"C = {C:5g}: fit + scoring {time.perf_counter() - t0:.1f} s  support vectors {r['n_sv']}  "
            f"train {r['train_acc']:.4f}  internal {r['internal']:.4f} ({r['internal_correct']}/{len(Zte)})  "
            f"KDDTest+ {r['kddtest']:.4f}  novel recall {r['novel_recall']:.4f} ({r['novel_correct']}/{int(nov.sum())})")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "reference_full_split.csv", index=False)

    # distance of every E1 readout from this reference (points; positive = readout below the reference)
    e1 = pd.read_csv(VQC / "results" / "exp1_readout.csv")
    say("\nE1 readouts against the reference on the same records (internal accuracy; points, + = below the reference)")
    for (mode, ro), g in e1.groupby(["mode", "readout"]):
        m = g.internal.mean()
        line = f"  {mode + '/' + ro:28s} {m:.4f}"
        for r in rows:
            line += f"   C={r['C']:<5g} {100 * (r['internal'] - m):+6.2f}"
        say(line)
    (OUT / "reference_full_split_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
