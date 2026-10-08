"""
Why do angle-encoded classifiers miss unseen attack types? A diagnostic on existing fits (no training).

Hypothesis. At scale s = 0.25 the angle fidelity kernel is close to a Gaussian kernel, so a record far from every
training record has near-zero similarity to all of them and a kernel model falls back to its bias. Amplitude
encoding depends only on the direction of x (the kernel is (x_hat . y_hat)^2), so a record that is far away in
norm but points in an attack-like direction is still scored by that direction.

Measurements, N = 8 and 16, seed 0 (the exp5 training subsample):
  * nearest-neighbour similarity: max over training records of the angle-RY fidelity, for internal test records,
    known-type KDDTest+ attacks and novel-type KDDTest+ attacks
  * norm ||x|| of the same groups
  * recall on novel records, binned by that similarity, for the exp5 co-trained-head circuits of both arms
    (weights from results/exp5_thetas.json at N = 8; the notebook checkpoint at N = 16 for the angle arm)

    .venv/bin/python paper/analysis/novel_mechanism.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebook"))
import angle_fast_v2 as AF                                          # noqa: E402
import exp5_matched_sweep as E5                                    # noqa: E402
from novel_ref1 import novel_mask_ref1                              # noqa: E402
from qids.data import _read, load_arrays, subsample                 # noqa: E402
from qids.fastsim import fidelity_kernel                            # noqa: E402

OUT = ROOT / "paper" / "tables"
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def max_sim(Z, Ztr, enc, chunk=4000):
    out = np.empty(len(Z))
    for lo in range(0, len(Z), chunk):
        out[lo:lo + chunk] = fidelity_kernel(Z[lo:lo + chunk], Ztr, enc).max(axis=1)
    return out


def main():
    nov = novel_mask_ref1()
    te_raw = _read(str(ROOT / "data" / "KDDTest+.txt"))
    known_attack = (te_raw["y"].to_numpy() == 1) & ~nov
    thetas = json.loads((ROOT / "results" / "exp5_thetas.json").read_text())
    rows = []
    for N in (8, 16):
        A = load_arrays(n_components=N)
        Xtr, ytr = subsample(A.Xtr, A.ytr, 2000, 0)
        say(f"\n=== N = {N}, seed 0 ===")
        groups = {"internal test": A.Xte, "KDDTest+ known attacks": A.Xood[known_attack], "KDDTest+ novel attacks": A.Xood[nov],
                  "KDDTest+ normal": A.Xood[te_raw["y"].to_numpy() == 0]}
        sims = {}
        for g, X in groups.items():
            s_ang = max_sim(X * E5.SCALE, Xtr * E5.SCALE, "angle_ry")
            s_amp = max_sim(X, Xtr, "amplitude")
            nrm = np.linalg.norm(X, axis=1)
            sims[g] = s_ang
            say(f"  {g:24s} n={len(X):5d}  max angle similarity: median {np.median(s_ang):.3f}, share < 0.5: {np.mean(s_ang < 0.5):.3f}"
                f"   | max amplitude similarity: median {np.median(s_amp):.3f}   | ||x||: median {np.median(nrm):.2f}")
            rows.append(dict(N=N, group=g, n=len(X), angle_maxsim_median=np.median(s_ang), angle_maxsim_lt05=np.mean(s_ang < 0.5),
                             amp_maxsim_median=np.median(s_amp), norm_median=np.median(nrm)))
        say(f"  training records: ||x|| median {np.median(np.linalg.norm(Xtr, axis=1)):.2f}")

        # co-trained-head circuits of both arms on the novel records, binned by angle nearest-neighbour similarity
        Xn = A.Xood[nov]
        preds = {}
        for arm_cls, arm_name in ((E5.AmplitudeArm, "amplitude"), (E5.AngleArm, "angle")):
            arm = arm_cls(N)
            if arm_name == "angle" and N == 16:
                ck = E5.load_ckpt("fit_n16_s0_head")
                th, coef, b0 = np.array(ck["theta"]), np.array(ck["head"]["coef"]), np.array(ck["head"]["intercept"])
            else:
                rec = thetas[f"N{N}_s0_{arm_name}_head"]
                th, coef, b0 = np.array(rec["theta"]), np.array(rec["head"]["coef"]), np.array(rec["head"]["intercept"])
            R = arm.readout(arm.inputs(Xn), th)
            preds[arm_name] = ((R @ coef.T + b0).ravel() > 0).astype(int)
        s = sims["KDDTest+ novel attacks"]
        edges = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0001]
        say("  novel-attack recall of the co-trained-head circuits, by nearest-neighbour angle similarity:")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (s >= lo) & (s < hi)
            if m.sum() == 0:
                continue
            say(f"    similarity [{lo:.2f}, {min(hi, 1):.2f}): n={int(m.sum()):5d}   amplitude {preds['amplitude'][m].mean():.3f}   angle {preds['angle'][m].mean():.3f}")
            rows.append(dict(N=N, group=f"novel, sim [{lo:.2f},{min(hi,1):.2f})", n=int(m.sum()),
                             recall_amplitude_head=preds["amplitude"][m].mean(), recall_angle_head=preds["angle"][m].mean()))
        say(f"    all novel: amplitude {preds['amplitude'].mean():.3f}   angle {preds['angle'].mean():.3f}")
    pd.DataFrame(rows).to_csv(OUT / "novel_mechanism.csv", index=False)
    (OUT / "novel_mechanism_log.txt").write_text("\n".join(LOG))


if __name__ == "__main__" and "--nn" not in sys.argv:
    main()


def nearest_label():
    """Class of the nearest training record (by fidelity) for each novel attack, under each encoding."""
    nov = novel_mask_ref1()
    rows = []
    for N in (4, 8, 16):
        A = load_arrays(n_components=N)
        for seed in range(5):
            Xtr, ytr = subsample(A.Xtr, A.ytr, 2000, seed)
            Xn = A.Xood[nov]
            for enc, f in (("angle_ry", lambda X: X * E5.SCALE), ("amplitude", lambda X: X)):
                lab = np.empty(len(Xn), dtype=int)
                for lo in range(0, len(Xn), 2000):
                    K = fidelity_kernel(f(Xn[lo:lo + 2000]), f(Xtr), enc)
                    lab[lo:lo + 2000] = ytr[K.argmax(axis=1)]
                rows.append(dict(N=N, seed=seed, encoding=enc, share_nn_attack=float(lab.mean())))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "novel_nearest_label.csv", index=False)
    say("\nshare of novel attacks whose nearest training record (by fidelity) is an attack, mean ± SD over 5 seeds:")
    say(D.groupby(["N", "encoding"]).share_nn_attack.agg(["mean", "std"]).round(3).to_string())
    (OUT / "novel_mechanism_log.txt").write_text("\n".join(LOG))


if __name__ == "__main__" and "--nn" in sys.argv:
    nearest_label()
