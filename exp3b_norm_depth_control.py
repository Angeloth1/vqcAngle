"""
EXPERIMENT 3b -- exp3 again, with the depth-control arm that results/exp3_norm.csv lacks.

Why
---
exp3_norm_restoration.py defines four circuit arms, but results/exp3_norm.csv holds only
three: it was written by an earlier version of the script, before the depth control
'vqc_amplitude_4q_reps4' was added. Without that arm, the two 5-qubit arms (reps = 4)
differ from the 4-qubit baseline (reps = 5) in depth as well as in the extra qubit, and the
norm result cannot be read cleanly.

What this does
--------------
Runs exp3_norm_restoration.run() unchanged, with one substitution: the novel-signature mask
is the 3,752-record definition of [1] (novel_ref1.py), which is the definition the existing
exp3/exp1/exp4 result files were computed with. The arms already in exp3_norm.csv must
therefore come out identical; the new arm is the only new information.

Runtime: ~2 min.
"""
from __future__ import annotations

import argparse
import os

import exp3_norm_restoration as E3
from novel_ref1 import novel_mask_ref1
from qids.data import load_arrays as _load_arrays


def load_arrays_ref1(data_dir=None, *args, **kwargs):
    A = _load_arrays(data_dir, *args, **kwargs)
    if not kwargs.get("synthetic", False):
        A.novel_mask = novel_mask_ref1(data_dir)
    return A


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--n-train", type=int, default=0)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="results/exp3_norm_v2.csv")
    a = ap.parse_args()

    E3.load_arrays = load_arrays_ref1            # the only change to exp3's run()
    df = E3.run(a.seeds, a.maxiter, a.n_train, False, a.data_dir)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)

    print("\n" + "=" * 72)
    print("EXP 3b  norm restoration with the depth-control arm")
    print("=" * 72)
    print(df.groupby("model")[["internal", "kddtest", "novel_recall"]].agg(["mean", "std"]).round(4).to_string())
    print(f"\nwritten to {a.out}")
