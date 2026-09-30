"""
EXPERIMENT 2 -- Separating the encoding from the readout.

Open question addressed
-----------------------
Abstract: "the separation is present in the encoded geometry and absent from the
classifier, which implicates the readout alongside the encoding, a distinction
this design does not separate."
Section 7.7.2: "The AUC in Equation (6) is not an upper limit on attainable
accuracy."
Section 9.2: a full-scale angle-encoded comparison was left out on compute cost.

Design
------
The fidelity K(x,y) = |<psi(x)|psi(y)>|^2 is a positive semi-definite kernel, so
an SVM with K precomputed is *the strongest classifier that uses nothing but the
pairwise geometry the paper measures*. It has no ansatz, no depth, no decoding
and no measurement basis. It therefore converts Table 5's AUC, which the paper
correctly refuses to call a bound, into an actual accuracy on the actual task.

  acc(kernel SVM)  -- what the encoded geometry supports
  acc(trained VQC) -- what the circuit plus parity readout recovers from it
  difference       -- the readout cost, in accuracy, per encoding

Both angle kernels are closed form (a product of per-qubit cosines), so N = 16
costs the same as N = 4. This is the 16-qubit angle-encoding comparison that
Section 9.2 sets aside as too expensive: it does not need a single state vector.
The caveat is honest and should be stated in any write-up -- a kernel machine is
not a VQC, and the result bounds the geometry, not a NISQ-deployable model.

Runtime: seconds per configuration at m = 2000. Quadratic in m, so raise --m
carefully.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from qids.data import angle_scale, load_arrays, subsample
from qids.fastsim import fidelity_kernel

ENCODINGS = ["amplitude", "angle_ry", "angle_phase"]


def separability_auc(X, y, rng, n_pairs=200_000):
    """Table 5's quantity, recomputed here so the two live side by side."""
    from sklearn.metrics import roc_auc_score
    i = rng.integers(0, len(X), n_pairs)
    j = rng.integers(0, len(X), n_pairs)
    keep = i != j
    return i[keep], j[keep], (y[i[keep]] == y[j[keep]]).astype(int)


def run(seeds, Ns, m, synthetic, data_dir, angle_mode='raw',
        angle_scale_=0.25, C=1.0):
    A = load_arrays(data_dir, synthetic=synthetic)
    from sklearn.metrics import roc_auc_score
    rows = []

    for seed in seeds:
        Xtr, ytr = subsample(A.Xtr, A.ytr, m, seed)
        Xte, yte = subsample(A.Xte, A.yte, m, seed + 100)
        Xa_tr, Xa_te = angle_scale(Xtr, Xte, mode=angle_mode,
                                   scale=angle_scale_, seed=seed)
        rng = np.random.default_rng(seed)

        for N in Ns:
            for enc in ENCODINGS:
                if enc == "amplitude" and 2 ** int(np.log2(N)) != N:
                    continue
                Atr = Xtr[:, :N] if enc == "amplitude" else Xa_tr[:, :N]
                Ate = Xte[:, :N] if enc == "amplitude" else Xa_te[:, :N]

                Ktr = fidelity_kernel(Atr, Atr, enc)
                Kte = fidelity_kernel(Ate, Atr, enc)
                svm = SVC(kernel="precomputed", C=C).fit(Ktr, ytr)

                i, j, same = separability_auc(Atr, ytr, rng)
                auc = roc_auc_score(same, Ktr[i, j])

                rows.append(dict(seed=seed, N=N, encoding=enc, C=C,
                                 qubits=int(np.log2(N)) if enc == "amplitude" else N,
                                 kernel_svm_acc=svm.score(Kte, yte),
                                 separability_auc=auc,
                                 n_sv=int(svm.n_support_.sum())))
        print(f"  seed {seed} done")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--Ns", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--m", type=int, default=2000)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--angle-mode", default="raw",
                    choices=["raw", "quantile"])
    ap.add_argument("--angle-scale", type=float, default=0.25)
    ap.add_argument("--C", type=float, default=1.0,
                    help="SVC soft-margin. Raise to 100 to check that the "
                         "reported ceiling is not a soft-margin artefact.")
    ap.add_argument("--out", default="results/exp2_geometry.csv")
    a = ap.parse_args()

    df = run(a.seeds, a.Ns, a.m, a.synthetic, a.data_dir,
             a.angle_mode, a.angle_scale, a.C)
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)

    piv = (df.groupby(["N", "encoding"])[["separability_auc", "kernel_svm_acc"]]
             .agg(["mean", "std"]).round(4))
    print("\n" + "=" * 72)
    print("EXP 2  geometry ceiling: what the encoded states alone support")
    print("=" * 72)
    print(piv.to_string())
    print("\nCompare kernel_svm_acc against the trained VQC accuracies in")
    print("Tables 3 and 6. The difference is the readout cost for that encoding.")
    print(f"\nwritten to {a.out}")
