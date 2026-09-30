"""
EXPERIMENT 3 -- Is the norm degeneracy the actual mechanism?

Open question addressed
-----------------------
Section 5.1, on the x versus 3x fidelity-1.0 example: "The example establishes
that the map is not injective; it does not by itself establish that
non-injectivity is the mechanism limiting the classifier on NSL-KDD, since the
construction is adversarial and need not be representative of the data."

The paper then measures pairwise fidelity (Section 7.7) but never measures the
one quantity the degeneracy argument is actually about: how much class
information lives in ||x||, the single scalar that l2 normalisation deletes.

Design
------
Three measurements of rising strength, all on the real records.

  1. UNIVARIATE. AUC of ||x|| alone as a score for the attack label. This is a
     one-line number and it is the cleanest statement available: if it sits well
     above 0.5, amplitude encoding is discarding a scalar that separates the
     classes by itself.

  2. CLASSICAL ABLATION. Logistic regression on x_hat versus on [x_hat, r] where
     r = log1p(||x||). The difference is the accuracy that l2 normalisation costs,
     measured with the quantum stage removed so nothing else can be blamed.

  3. QUANTUM ABLATION. A 5-qubit circuit: amplitude-encode x_hat on 4 qubits and
     write r into a fifth via RY(pi * r_scaled), then RealAmplitudes(5, reps=4) =
     25 parameters against the published 24. Matched budget, one extra qubit,
     nothing else changed. If this recovers the classical gap, the degeneracy is
     the operative mechanism and not merely a true statement about the map.

Note what this does NOT do: restoring the norm does not fix the Born rule or the
readout, so a partial recovery is the expected outcome and is still informative.
Report the three numbers together or the result is easy to over-read.

Runtime: a couple of minutes for five seeds.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from qids.data import load_arrays, subsample
from qids.decoders import ParityDecoder, accuracy
from qids.fastsim import l2_normalise
from qids.train import probs_for, train_fixed


def norm_feature(X, ref_max=None):
    r = np.log1p(np.linalg.norm(X, axis=1))
    hi = ref_max if ref_max is not None else r.max()
    return r, np.clip(r / max(hi, 1e-9), 0, 1)


def augmented_states(X, r_scaled):
    """|x_hat> (4 qubits, 16 amplitudes) tensor RY(pi*r)|0> (1 qubit).

    Kron ordering matches qiskit's little-endian convention: the appended qubit
    is the most significant, so outcome b = 16*q4 + b_low.
    """
    lo = l2_normalise(X)
    hi = np.stack([np.cos(np.pi * r_scaled / 2), np.sin(np.pi * r_scaled / 2)], 1)
    return (hi[:, :, None] * lo[:, None, :]).reshape(len(X), -1)


def run(seeds, maxiter, n_train, synthetic, data_dir):
    A = load_arrays(data_dir, synthetic=synthetic)
    rows = []
    nrm = lambda Z: l2_normalise(Z)

    for seed in seeds:
        Xtr, ytr = (subsample(A.Xtr, A.ytr, n_train, seed) if n_train
                    else (A.Xtr, A.ytr))
        r_tr, rs_tr = norm_feature(Xtr)
        hi = np.log1p(np.linalg.norm(Xtr, axis=1)).max()
        r_te, rs_te = norm_feature(A.Xte, hi)
        r_oo, rs_oo = norm_feature(A.Xood, hi)

        # 1. univariate signal in the deleted scalar
        rows.append(dict(seed=seed, model="auc_of_norm_alone",
                         internal=roc_auc_score(A.yte, r_te),
                         kddtest=roc_auc_score(A.yood, r_oo), novel_recall=np.nan))

        # 2. classical ablation
        for tag, Ztr, Zte, Zoo in [
            ("logreg_xhat", nrm(Xtr), nrm(A.Xte), nrm(A.Xood)),
            ("logreg_xhat+norm",
             np.c_[nrm(Xtr), rs_tr], np.c_[nrm(A.Xte), rs_te], np.c_[nrm(A.Xood), rs_oo]),
            ("logreg_raw_x", Xtr, A.Xte, A.Xood),
        ]:
            clf = LogisticRegression(max_iter=3000).fit(Ztr, ytr)
            nov = (clf.score(Zoo[A.novel_mask], A.yood[A.novel_mask])
                   if A.novel_mask.any() else np.nan)
            rows.append(dict(seed=seed, model=tag, internal=clf.score(Zte, A.yte),
                             kddtest=clf.score(Zoo, A.yood), novel_recall=nov))

        # 3. quantum ablation. Three arms, because two would confound the norm
        #    with the larger Hilbert space and the changed depth:
        #      4q  reps=5  -> 24 params, the published model
        #      5q  reps=4  -> 25 params, real norm on q4
        #      5q  reps=4  -> 25 params, norm SHUFFLED across records
        #    The third arm has the same dimension, the same parameter count and
        #    the same circuit as the second, and differs only in whether the
        #    extra qubit carries information. If arm 2 beats arm 3, the gain is
        #    the norm. If arms 2 and 3 are equal, the gain is the extra qubit and
        #    the norm story is not supported.
        perm = np.random.default_rng(9_000 + seed)
        for tag, nq, reps, S in [
            ("vqc_amplitude_4q", 4, 5, (nrm(Xtr), nrm(A.Xte), nrm(A.Xood))),
            # depth control: the two 5-qubit arms run at reps=4 to keep the
            # parameter count near 24, so this arm prices the depth change on
            # its own, with the qubit count held at four.
            ("vqc_amplitude_4q_reps4", 4, 4,
             (nrm(Xtr), nrm(A.Xte), nrm(A.Xood))),
            ("vqc_amplitude+norm_5q", 5, 4,
             (augmented_states(Xtr, rs_tr), augmented_states(A.Xte, rs_te),
              augmented_states(A.Xood, rs_oo))),
            ("vqc_shuffled_norm_5q", 5, 4,
             (augmented_states(Xtr, perm.permutation(rs_tr)),
              augmented_states(A.Xte, perm.permutation(rs_te)),
              augmented_states(A.Xood, perm.permutation(rs_oo)))),
        ]:
            dec = ParityDecoder(2 ** nq, 2)
            fit = train_fixed(S[0], ytr, nq, reps, dec, maxiter=maxiter, seed=seed)
            Pte = probs_for(S[1], fit.theta, nq, reps)
            Poo = probs_for(S[2], fit.theta, nq, reps)
            nov = (accuracy(dec.predict_proba(Poo[A.novel_mask]),
                            A.yood[A.novel_mask]) if A.novel_mask.any() else np.nan)
            rows.append(dict(seed=seed, model=f"{tag}_p{nq * (reps + 1)}",
                             internal=accuracy(dec.predict_proba(Pte), A.yte),
                             kddtest=accuracy(dec.predict_proba(Poo), A.yood),
                             novel_recall=nov))
        print(f"  seed {seed} done")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--n-train", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="results/exp3_norm.csv")
    a = ap.parse_args()

    df = run(a.seeds, a.maxiter, a.n_train, a.synthetic, a.data_dir)
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)

    print("\n" + "=" * 72)
    print("EXP 3  norm restoration: what l2 normalisation actually deletes")
    print("=" * 72)
    print(df.groupby("model")[["internal", "kddtest", "novel_recall"]]
            .agg(["mean", "std"]).round(4).to_string())
    print("\nRow 'auc_of_norm_alone' is an AUC, not an accuracy. Do not average")
    print("it with the rest.")
    print(f"\nwritten to {a.out}")
