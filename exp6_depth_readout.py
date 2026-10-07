"""
EXPERIMENT 6 -- Is the depth plateau of [1] a readout plateau?

Open question addressed
-----------------------
Section 7.6 of [1]: internal accuracy saturates at 0.880 by reps = 9 while the
training objective keeps falling, read there as "it has run out of information
to fit them to". exp1 shows that on the full training split a change of readout
alone moves the same 24-parameter circuit from 0.856 (parity) to 0.908 (frozen
linear head) and 0.948 (co-trained head). But exp1 also changes the training-set
size, so it cannot say whether the plateau of Section 7.6 belongs to the encoding
or to the parity readout.

Design
------
The depth sweep of [1] at its own protocol -- 2,000-record subsample, reps in
{1, 3, 5, 9, 17}, maxiter = max(150, 24 (reps + 1)), theta0 ~ U(-0.1, 0.1), five
seeds, the published RealAmplitudes (reverse_linear) on 4 qubits -- with the
readout as the second factor:

  parity            trained against qubit-0 parity (replicates Table A2 of [1];
                    the reps 1/5/17 rows must equal exp4 at rho = 0)
Novel recall uses the 3,752-record definition of [1] (see novel_mask_ref1).
  best_fixed        exhaustive many-to-one assignment, frozen parity weights
  linear_frozen     logistic regression on the 16 outcome probabilities, frozen weights
  linear_cotrained  weights re-optimised against the linear head (refit per evaluation)

If the co-trained head keeps rising with depth where parity stops, the plateau
belongs to the readout. If both stop at the same place, it belongs to the encoding.

Runtime: ~5-8 min for five seeds.
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from qids.data import load_arrays, subsample
from qids.decoders import BestFixedDecoder, LinearDecoder, ParityDecoder, accuracy
from qids.fastsim import amplitude_states, n_params
from qids.train import probs_for, train_cotrain_linear, train_fixed
from novel_ref1 import novel_mask_ref1

warnings.filterwarnings("ignore", category=ConvergenceWarning)
N_QUBITS, N_OUT = 4, 16


def scores(proba_fn, Ptr, ytr, Pte, yte, Poo, yoo, nov):
    return dict(train_acc=accuracy(proba_fn(Ptr), ytr), internal=accuracy(proba_fn(Pte), yte),
                kddtest=accuracy(proba_fn(Poo), yoo), novel_recall=accuracy(proba_fn(Poo[nov]), yoo[nov]))


def run(seeds, reps_grid, m, floor, synthetic, data_dir, out):
    A = load_arrays(data_dir, synthetic=synthetic)
    Ste, Soo = amplitude_states(A.Xte), amplitude_states(A.Xood)
    nov = novel_mask_ref1(data_dir)
    rows = []
    for seed in seeds:
        t_seed = time.perf_counter()
        Xtr, ytr = subsample(A.Xtr, A.ytr, m, seed)
        Str = amplitude_states(Xtr)
        for reps in reps_grid:
            it = max(floor, 6 * N_QUBITS * (reps + 1))        # the schedule of exp4 / Table A2
            base = dict(seed=seed, reps=reps, params=n_params(N_QUBITS, reps), maxiter=it)

            fit = train_fixed(Str, ytr, N_QUBITS, reps, ParityDecoder(N_OUT, 2), maxiter=it, seed=seed)
            Ptr, Pte, Poo = (probs_for(S, fit.theta, N_QUBITS, reps) for S in (Str, Ste, Soo))
            for name, dec in (("parity", ParityDecoder(N_OUT, 2)), ("best_fixed", BestFixedDecoder(N_OUT)),
                              ("linear_frozen", LinearDecoder(N_OUT))):
                dec.fit(Ptr, ytr)
                rows.append({**base, "readout": name,
                             **scores(dec.predict_proba, Ptr, ytr, Pte, A.yte, Poo, A.yood, nov),
                             "loss": fit.loss, "evals": fit.n_eval, "seconds": fit.seconds})

            cofit, head = train_cotrain_linear(Str, ytr, N_QUBITS, reps, maxiter=it, seed=seed)
            P2 = [probs_for(S, cofit.theta, N_QUBITS, reps) for S in (Str, Ste, Soo)]
            rows.append({**base, "readout": "linear_cotrained",
                         **scores(head.predict_proba, P2[0], ytr, P2[1], A.yte, P2[2], A.yood, nov),
                         "loss": cofit.loss, "evals": cofit.n_eval, "seconds": cofit.seconds})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  seed {seed} done ({time.perf_counter() - t_seed:.0f}s)", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 3, 5, 9, 17])
    ap.add_argument("--m", type=int, default=2000)
    ap.add_argument("--floor", type=int, default=150, help="minimum COBYLA budget, as in exp4 / [1]")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="results/exp6_depth_readout.csv")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df = run(a.seeds, a.reps, a.m, a.floor, a.synthetic, a.data_dir, a.out)

    print("\n" + "=" * 72)
    print("EXP 6  depth x readout at the 2,000-record protocol of [1]  (internal accuracy)")
    print("=" * 72)
    print(df.pivot_table(index="reps", columns="readout", values="internal", aggfunc="mean").round(4).to_string())
    print("\nnovel-signature recall:")
    print(df.pivot_table(index="reps", columns="readout", values="novel_recall", aggfunc="mean").round(4).to_string())
    print(f"\nwritten to {a.out}")
