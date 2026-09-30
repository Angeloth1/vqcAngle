"""
EXPERIMENT 1 -- How much does the decoding cost?

Open question addressed
-----------------------
Section 9.3: "the many-to-one decoding was held constant throughout, so no
experiment reported here bounds how much of the residual gap it contributes."
Section 7.8: "Their divergence pinpoints the residual bottleneck at reduced
scale to the readout rather than to the encoding."

Design
------
Encoding (amplitude, PCA-16), depth (reps=5), data and protocol are frozen at the
published values. Only the map from the 16 measured outcome probabilities to a
class decision varies.

  A. FROZEN THETA. Train once with the published parity decoding, then apply
     every other decoder to the same weights. Isolates what parity discards from
     the state the published circuit already produces. No confound whatever:
     identical circuit, identical weights, identical measurement.

  B. CO-TRAINED. Re-optimise theta against a free linear readout. Bounds what a
     readout-aware design could reach at the same depth and encoding.

Reference row: logistic regression straight on x_hat, i.e. no quantum stage at
all. Whatever gap survives after the best readout is attributable to the
encoding and the Born rule, not to the decoding -- which is the attribution the
paper says it cannot make.

Reading the output
------------------
  parity -> best_fixed     cost of *choosing* parity within the paper's own class
  best_fixed -> mlp_head   cost of *restricting to* many-to-one assignment
  mlp_head -> logreg(x_hat) what is left for encoding + Born rule to explain

Runtime: about a minute for all five seeds. The whole training set is used, not
the 2000-record subsample, because the matmul formulation makes it free.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from qids.data import load_arrays, subsample
from qids.decoders import (BestFixedDecoder, GlobalParityDecoder, LinearDecoder,
                           MLPDecoder, ParityDecoder, accuracy)
from qids.fastsim import amplitude_states
from qids.train import probs_for, train_cotrain_linear, train_fixed

N_QUBITS, REPS, N_OUT = 4, 5, 16


def run(seeds, maxiter, n_train, synthetic, data_dir):
    A = load_arrays(data_dir, synthetic=synthetic)
    rows = []

    for seed in seeds:
        Xtr, ytr = (subsample(A.Xtr, A.ytr, n_train, seed) if n_train
                    else (A.Xtr, A.ytr))
        Str, Ste = amplitude_states(Xtr), amplitude_states(A.Xte)
        Sood = amplitude_states(A.Xood)

        # ---- A. frozen theta -------------------------------------------------
        parity = ParityDecoder(N_OUT, 2)
        fit = train_fixed(Str, ytr, N_QUBITS, REPS, parity, maxiter=maxiter, seed=seed)
        Ptr = probs_for(Str, fit.theta, N_QUBITS, REPS)
        Pte = probs_for(Ste, fit.theta, N_QUBITS, REPS)
        Pood = probs_for(Sood, fit.theta, N_QUBITS, REPS)

        for dec in (ParityDecoder(N_OUT), GlobalParityDecoder(N_OUT),
                    BestFixedDecoder(N_OUT), LinearDecoder(N_OUT),
                    MLPDecoder(N_OUT, seed=seed)):
            dec.fit(Ptr, ytr)
            nov = accuracy(dec.predict_proba(Pood[A.novel_mask]),
                           A.yood[A.novel_mask]) if A.novel_mask.any() else np.nan
            rows.append(dict(seed=seed, mode="frozen_theta", readout=dec.name,
                             internal=accuracy(dec.predict_proba(Pte), A.yte),
                             kddtest=accuracy(dec.predict_proba(Pood), A.yood),
                             novel_recall=nov, seconds=fit.seconds))

        # ---- B. co-trained ---------------------------------------------------
        cofit, head = train_cotrain_linear(Str, ytr, N_QUBITS, REPS,
                                           maxiter=maxiter, seed=seed)
        Pte2 = probs_for(Ste, cofit.theta, N_QUBITS, REPS)
        Pood2 = probs_for(Sood, cofit.theta, N_QUBITS, REPS)
        nov = accuracy(head.predict_proba(Pood2[A.novel_mask]),
                       A.yood[A.novel_mask]) if A.novel_mask.any() else np.nan
        rows.append(dict(seed=seed, mode="cotrained", readout="linear_head",
                         internal=accuracy(head.predict_proba(Pte2), A.yte),
                         kddtest=accuracy(head.predict_proba(Pood2), A.yood),
                         novel_recall=nov, seconds=cofit.seconds))

        # ---- reference: no quantum stage ------------------------------------
        ref = LogisticRegression(max_iter=2000).fit(
            Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True), ytr)
        nrm = lambda Z: Z / np.linalg.norm(Z, axis=1, keepdims=True)
        nov = ref.score(nrm(A.Xood[A.novel_mask]),
                        A.yood[A.novel_mask]) if A.novel_mask.any() else np.nan
        rows.append(dict(seed=seed, mode="reference", readout="logreg_on_xhat",
                         internal=ref.score(nrm(A.Xte), A.yte),
                         kddtest=ref.score(nrm(A.Xood), A.yood),
                         novel_recall=nov, seconds=0.0))

        print(f"  seed {seed} done ({fit.seconds:.1f}s + {cofit.seconds:.1f}s)")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--n-train", type=int, default=0,
                    help="0 = full training split; 2000 reproduces Section 7.4")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="results/exp1_readout.csv")
    a = ap.parse_args()

    df = run(a.seeds, a.maxiter, a.n_train, a.synthetic, a.data_dir)
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)

    agg = (df.groupby(["mode", "readout"])[["internal", "kddtest", "novel_recall"]]
             .agg(["mean", "std"]).round(4))
    print("\n" + "=" * 72)
    print("EXP 1  readout ablation  (encoding, depth, data, protocol all fixed)")
    print("=" * 72)
    print(agg.to_string())
    print(f"\nwritten to {a.out}")
