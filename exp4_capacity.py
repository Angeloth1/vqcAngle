"""
EXPERIMENT 4 -- A mechanism for the out-of-distribution result.

Open question addressed
-----------------------
Contribution list: on novel-signature recall, "We present it as an observation on
a single dataset, without identifying a mechanism."
Section 7.5 proposes one -- "Reduced effective capacity would then act as an
implicit normalizer" -- and Section 7.6 offers indirect support through the depth
sweep. What is missing is a direct measurement of effective capacity.

Design
------
The standard instrument: fit random labels. Train every model on the same
records with a fraction rho of labels randomly reassigned, and record TRAINING
accuracy. A model that can drive training accuracy high on noise has spare
capacity to memorise; a model that cannot, does not. Parameter count does not
predict this and that is exactly the point, since the paper already notes that
MLP-1 at 19 parameters does not behave like the 24-parameter circuit.

Then correlate the capacity measure against novel-signature recall across the
model set. The capacity account predicts a negative relationship. Six models is
a small n and the correlation should be reported as descriptive, in the same
register the paper already uses for its five-seed tests.

Models: VQC at reps 1/5/17, logistic regression, MLP-1, MLP-4 -- the published
set plus the two depth extremes, so the within-family trend is visible too.

Runtime: dominated by the reps=17 rows. Ten minutes or so at the defaults.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from qids.data import load_arrays, subsample
from qids.decoders import ParityDecoder, accuracy
from qids.fastsim import amplitude_states
from qids.train import probs_for, train_fixed

RHOS = [0.0, 0.25, 0.5, 1.0]


def corrupt(y, rho, rng):
    y = y.copy()
    k = rng.random(len(y)) < rho
    y[k] = rng.integers(0, 2, k.sum())
    return y


def classical_models(seed):
    return {
        "logreg_17p": LogisticRegression(max_iter=3000),
        "mlp1_19p": MLPClassifier((1,), max_iter=2000, random_state=seed),
        "mlp4_73p": MLPClassifier((4,), max_iter=2000, random_state=seed),
    }


def run(seeds, maxiter, m, reps_grid, synthetic, data_dir):
    A = load_arrays(data_dir, synthetic=synthetic)
    rows = []

    for seed in seeds:
        Xtr, ytr = subsample(A.Xtr, A.ytr, m, seed)
        Str = amplitude_states(Xtr)
        Xn = Str  # classical baselines see the same l2-normalised vectors
        Ste, Soo = amplitude_states(A.Xte), amplitude_states(A.Xood)

        for rho in RHOS:
            rng = np.random.default_rng(1000 * seed + int(100 * rho))
            yc = corrupt(ytr, rho, rng)

            for reps in reps_grid:
                dec = ParityDecoder(16, 2)
                it = max(maxiter, 6 * 4 * (reps + 1))     # scale as Section 7.6 does
                fit = train_fixed(Str, yc, 4, reps, dec, maxiter=it, seed=seed)
                Ptr = probs_for(Str, fit.theta, 4, reps)
                row = dict(seed=seed, rho=rho, model=f"vqc_reps{reps}",
                           params=4 * (reps + 1),
                           fit_acc=accuracy(dec.predict_proba(Ptr), yc))
                if rho == 0.0:
                    Poo = probs_for(Soo, fit.theta, 4, reps)
                    row["internal"] = accuracy(
                        dec.predict_proba(probs_for(Ste, fit.theta, 4, reps)), A.yte)
                    row["novel_recall"] = (
                        accuracy(dec.predict_proba(Poo[A.novel_mask]),
                                 A.yood[A.novel_mask]) if A.novel_mask.any() else np.nan)
                rows.append(row)

            for name, clf in classical_models(seed).items():
                clf.fit(Xn, yc)
                row = dict(seed=seed, rho=rho, model=name,
                           params={"logreg_17p": 17, "mlp1_19p": 19, "mlp4_73p": 73}[name],
                           fit_acc=clf.score(Xn, yc))
                if rho == 0.0:
                    row["internal"] = clf.score(Ste, A.yte)
                    row["novel_recall"] = (clf.score(Soo[A.novel_mask],
                                                     A.yood[A.novel_mask])
                                           if A.novel_mask.any() else np.nan)
                rows.append(row)
        print(f"  seed {seed} done")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--m", type=int, default=2000)
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 5, 17])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="results/exp4_capacity.csv")
    a = ap.parse_args()

    df = run(a.seeds, a.maxiter, a.m, a.reps, a.synthetic, a.data_dir)
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)

    print("\n" + "=" * 72)
    print("EXP 4  memorisation capacity vs novel-signature recall")
    print("=" * 72)
    fit = df.pivot_table(index="model", columns="rho", values="fit_acc")
    print(fit.round(4).to_string())

    # Raw fit accuracy at rho=1 is the wrong statistic: it is dominated by each
    # model's clean accuracy, so models that fit the signal poorly look like they
    # memorise little. `corrupt` reassigns labels at random, so a fraction rho/2
    # actually flips, and a model that learns ONLY the true signal is expected to
    # reach a + (rho/2)(1 - 2a) on the corrupted training set, where a is its
    # clean fit. Excess over that is memorisation, and it is comparable across
    # models because the clean accuracy has been divided out.
    clean = fit[0.0]
    excess = pd.DataFrame({r: fit[r] - (clean + (r / 2) * (1 - 2 * clean))
                           for r in fit.columns if r > 0})
    cap = excess.mean(axis=1).rename("memorisation")

    nov = (df[df.rho == 0.0].groupby("model").novel_recall.mean()
             .rename("novel_recall"))
    j = pd.concat([cap, nov], axis=1).dropna()
    print("\nexcess fit over a signal-only learner (higher = memorises more):")
    print(pd.concat([excess.round(4), j.round(4)], axis=1).to_string())

    if len(j) > 2:
        rho_s, p = spearmanr(j.memorisation, j.novel_recall)
        print(f"\nSpearman(memorisation, novel recall) = {rho_s:.3f}   "
              f"p = {p:.3f}   n = {len(j)}")
        spread = float(j.memorisation.max() - j.memorisation.min())
        if spread < 0.05:
            print("INCONCLUSIVE: the models barely differ in memorisation "
                  f"(range {spread:.3f}), so this instrument cannot resolve them.")
        elif rho_s < 0 and p < 0.10:
            print("Negative and reasonably separated: consistent with the "
                  "implicit-regularisation account of Section 7.5.")
        elif rho_s < 0:
            print("Negative but not separated from chance at n=6. Descriptive only.")
        else:
            print("POSITIVE sign, i.e. the OPPOSITE of what Section 7.5 predicts. "
                  "The capacity account is not supported by this measurement.")
        print("\nCheck the within-family trend too: if VQC reps 1/5/17 do not "
              "order the same way as the across-family comparison, the "
              "separation is family, not capacity.")
    print(f"\nwritten to {a.out}")
