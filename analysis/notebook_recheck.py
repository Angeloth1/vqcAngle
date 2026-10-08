"""
Re-analysis of the companion notebook (notebook/vqc_ceiling_and_shots.ipynb) from its checkpoints.

Purpose
-------
1. Confirm that the checkpointed thetas / heads reproduce the numbers printed in the notebook
   (exact parity accuracy, exact linear-head accuracy, kernel 'ceiling').
2. Test whether the finite-shot LINEAR-HEAD accuracy of ~0.50 printed in the notebook (cell 15)
   is physics or a bug. Hypothesis: `head_shot_curve` assembles the shot-estimated feature matrix
   in the column order [singles..., pairs..., parity], while the head was trained on
   `forward_readout` output, whose columns follow `readout_index` order: ascending bitmask S, i.e.
   singles and pairs INTERLEAVED, parity last. The trained coefficients are then applied to a
   permuted feature matrix.
3. Report record-level bootstrap intervals for the single-seed accuracies.

Nothing is retrained. Run:  .venv/bin/python paper/analysis/notebook_recheck.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebook"))
sys.path.insert(0, str(HERE))

import angle_fast_v2 as A                                     # noqa: E402
from qids.data import load_arrays, subsample                  # noqa: E402
from stats_utils import boot_acc_ci, wilson                   # noqa: E402

CK = ROOT / "notebook" / "ckpt"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
CFG = dict(reps=2, angle_scale=0.25, m_train=2000, m_eval=2000, max_weight=2, seed=0)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def ck(name):
    """Load a notebook checkpoint. The notebook's own `ckpt()` reloads with `.item()`, which raises for
    list-valued stages (ceiling / shots / headshots: np.array(list_of_dicts, dtype=object) has size > 1),
    so its documented resume-after-interruption only works for the dict-valued `fit_*` stages."""
    obj = np.load(CK / f"{name}.npz", allow_pickle=True)["obj"]
    return obj.item() if obj.ndim == 0 else list(obj)


def prep(n):
    """Same data path as the notebook: build(n) -> stratified subsample of train (seed) and eval (seed+100)."""
    D = load_arrays(n_components=n)
    Xa, ya = subsample(D.Xtr * CFG["angle_scale"], D.ytr, CFG["m_train"], CFG["seed"])
    Xb, yb = subsample(D.Xte * CFG["angle_scale"], D.yte, CFG["m_eval"], CFG["seed"] + 100)
    return D, Xa, ya, Xb, yb


def decide_head(R, coef, b0):
    return (R @ np.asarray(coef).T + np.asarray(b0) > 0).astype(int).ravel()


def est_columns_notebook_order(sg, n, keep, w):
    """Reproduces the notebook's assembly: [singles..., pairs..., parity]."""
    singles = [int(np.log2(S)) for S in keep[w == 1]]
    pairs = [(int(np.log2(S & -S)), int(np.log2(S ^ (S & -S)))) for S in keep[w == 2]]
    out = np.empty(len(keep))
    out[:len(singles)] = sg[:, singles].mean(0)
    out[len(singles):len(singles) + len(pairs)] = [(sg[:, p] * sg[:, q]).mean() for p, q in pairs]
    out[-1] = np.prod(sg, axis=1).mean()
    return out


def est_columns_keep_order(sg, n, keep):
    """Estimate every column S in `keep` order: <Z_S> = mean over shots of prod_{q in S} sigma_q."""
    out = np.empty(len(keep))
    for j, S in enumerate(keep):
        qs = [q for q in range(n) if (int(S) >> q) & 1]
        out[j] = np.prod(sg[:, qs], axis=1).mean()
    return out


def head_shots(n, th, coef, b0, Xs, ys, keep, w, shots_grid, repeats, order, seed=0, one_rep_check=False):
    P = A.forward_probs(Xs, th, CFG["reps"], max_bytes=8_000_000_000)
    cdf = np.cumsum(P, axis=1)
    cdf /= cdf[:, -1:]
    rows = []
    for s in shots_grid:
        for r in range(repeats):
            rng = np.random.default_rng(99_000 + 31 * r + s + seed)      # same stream as the notebook
            est = np.empty((len(Xs), len(keep)))
            for i in range(len(Xs)):
                b = np.searchsorted(cdf[i], rng.random(s))
                sg = 1.0 - 2.0 * ((b[:, None] >> np.arange(n)[None, :]) & 1)
                est[i] = (est_columns_notebook_order(sg, n, keep, w) if order == "notebook"
                          else est_columns_keep_order(sg, n, keep))
            acc = float((decide_head(est, coef, b0) == ys).mean())
            rows.append(dict(n=n, shots=s, rep=r, order=order, acc=acc))
    return pd.DataFrame(rows)


def main():
    all_rows = []
    for n in (8, 16):
        say("\n" + "=" * 90 + f"\nn = {n}\n" + "=" * 90)
        t0 = time.perf_counter()
        D, Xa, ya, Xb, yb = prep(n)
        say(f"data: train {len(D.ytr)} test {len(D.yte)} KDDTest+ {len(D.yood)} novel {int(D.novel_mask.sum())} "
            f"explained variance {D.explained_variance:.4f}   (notebook printed {0.6511 if n == 8 else 0.8550})")
        keep, w = A.readout_index(n, CFG["max_weight"])
        say(f"readout columns: {len(keep)}   first 12 bitmasks in keep order: {keep[:12].tolist()}  weights {w[:12].tolist()}")
        say("  -> keep order interleaves singles and pairs:", bool(np.any(np.diff(w[:-1]) < 0)))

        # ---- 1. kernel reference reproduces ceiling.npz
        cz = pd.DataFrame(ck("ceiling"))
        Kf = lambda P_, Q_: np.prod([np.cos(0.5 * (P_[:, i:i + 1] - Q_[None, :, i])) ** 2 for i in range(P_.shape[1])], axis=0)
        Ktr, Kte = Kf(Xa, Xa), Kf(Xb, Xa)
        for C in (1.0, 100.0):
            acc = float(SVC(kernel="precomputed", C=C).fit(Ktr, ya).score(Kte, yb))
            nb = float(cz[(cz.n == n) & (cz.C == C)].ceiling.iloc[0])
            say(f"kernel SVM C={C:5.0f}: recomputed {acc:.4f}   notebook checkpoint {nb:.4f}   {'OK' if abs(acc - nb) < 1e-9 else 'MISMATCH'}")

        # ---- 2. exact accuracies from checkpointed weights
        fp, fh = ck(f"fit_n{n}_s0_parity"), ck(f"fit_n{n}_s0_head")
        thp, thh = np.array(fp["theta"]), np.array(fh["theta"])
        Rp = A.forward_readout(Xb, thp, CFG["reps"], keep=keep)
        Rh = A.forward_readout(Xb, thh, CFG["reps"], keep=keep)
        pred_par = (A.probs_from_z(Rp[:, -1]).argmax(1) == yb)
        coef, b0 = np.array(fh["head"]["coef"]), np.array(fh["head"]["intercept"])
        pred_head = (decide_head(Rh, coef, b0) == yb)
        for name, c in (("parity (global, as in notebook)", pred_par), ("linear head (37/137 cols)", pred_head)):
            m, lo, hi = boot_acc_ci(c, B=2000, seed=0)
            say(f"exact {name:34s} acc {m:.4f}  95% record-bootstrap [{lo:.4f}, {hi:.4f}]   (2000 eval records, ONE seed)")
        # q0 parity on the parity-trained circuit and on the head-trained circuit (informative, not a trained model)
        z0 = A.forward_readout(Xb, thp, CFG["reps"], max_weight=1, include_parity=False)[:, 0]
        say(f"  (diagnostic) q0-parity decision on the parity-trained theta: acc {float((A.probs_from_z(z0).argmax(1) == yb).mean()):.4f}")
        say(f"  loss at checkpoint: parity {fp['loss']:.4f}  head {fh['loss']:.4f};  evals {fp['evals']} / {fh['evals']};  "
            f"train time {fp['seconds']/60:.1f} / {fh['seconds']/60:.1f} min")
        say(f"  head coefficient L2 norm {np.linalg.norm(coef):.2f}   max |coef| {np.abs(coef).max():.2f}   intercept {b0.ravel()}")

        # ---- 3. parity shots curve, recomputed
        sh = pd.DataFrame(ck(f"shots_n{n}_s0"))
        g = sh.groupby("shots").acc.agg(["mean", "std"])
        say("parity finite-shot curve from checkpoint (20 repeats/point):")
        say(g.round(4).to_string())

        # ---- 4. head at finite shots: notebook assembly vs keep-order assembly
        hs_ck = pd.DataFrame(ck(f"headshots_n{n}_s0"))
        say("head finite-shot values printed by the notebook (checkpoint):")
        say(hs_ck.groupby("shots").acc.agg(["mean", "std"]).round(4).to_string())

        ns = 500                              # notebook: head_m_eval
        Xs, ys = Xb[:ns], yb[:ns]
        grid, reps = [1000, 10000], 5         # notebook: head_shots_grid, head_repeats
        # logit-noise diagnostic: could real shot noise alone destroy the head?
        z = Rh[:ns]
        for s in grid:
            var = np.clip(1.0 - z ** 2, 0, None) / s                    # Var of a +-1 mean estimator
            sd_logit = np.sqrt((var * (coef.ravel() ** 2)).sum(1))       # independent-column approximation
            margin = np.abs(z @ coef.T + b0).ravel()
            say(f"  shots={s:6d}: median logit-noise sd {np.median(sd_logit):.3f}   median |logit margin| {np.median(margin):.3f}   "
                f"records with margin < 1 sd: {float((margin < sd_logit).mean()):.3f}")

        t1 = time.perf_counter()
        buggy = head_shots(n, thh, coef, b0, Xs, ys, keep, w, grid, reps, "notebook", seed=0)
        fixed = head_shots(n, thh, coef, b0, Xs, ys, keep, w, grid, reps, "keep", seed=0)
        say(f"head shots recomputed in {time.perf_counter() - t1:.0f}s")
        exact_prefix = float((decide_head(Rh[:ns], coef, b0) == ys).mean())
        say(f"exact head accuracy on the same {ns}-record prefix: {exact_prefix:.4f}")
        cmp = pd.concat([buggy, fixed]).groupby(["order", "shots"]).acc.agg(["mean", "std"]).round(4)
        say(cmp.to_string())
        all_rows += [buggy.assign(kind="head_shots"), fixed.assign(kind="head_shots")]
        say(f"(n={n} done in {time.perf_counter() - t0:.0f}s)")

    pd.concat(all_rows).to_csv(OUT / "notebook_head_shots_recheck.csv", index=False)
    (OUT / "notebook_recheck_stage1_log.txt").write_text("\n".join(LOG))


if __name__ == "__main__" and "--ladder-only" not in sys.argv:
    main()


# ======================================================================================
# Stage 2: parallel decomposition for the angle arm, with paired record-level intervals
# ======================================================================================
def paired_boot(c1, c2, B=2000, seed=0, level=0.95):
    """Paired bootstrap over evaluation records for acc(c1) - acc(c2); c* are 0/1 correctness vectors."""
    c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c1), size=(B, len(c1)))
    d = (c1[idx] - c2[idx]).mean(axis=1)
    lo, hi = np.quantile(d, [(1 - level) / 2, 0.5 + level / 2])
    return float((c1 - c2).mean()), float(lo), float(hi)


def angle_ladder():
    from sklearn.linear_model import LogisticRegression

    say("\n" + "#" * 90 + "\nSTAGE 2  angle-arm decomposition (seed 0, 2000 train / 2000 eval records)\n" + "#" * 90)
    rows = []
    for n in (8, 16):
        D, Xa, ya, Xb, yb = prep(n)
        keep, w = A.readout_index(n, CFG["max_weight"])
        fp, fh = ck(f"fit_n{n}_s0_parity"), ck(f"fit_n{n}_s0_head")
        thp, thh = np.array(fp["theta"]), np.array(fh["theta"])
        Kf = lambda P_, Q_: np.prod([np.cos(0.5 * (P_[:, i:i + 1] - Q_[None, :, i])) ** 2 for i in range(P_.shape[1])], axis=0)
        Ktr, Kte = Kf(Xa, Xa), Kf(Xb, Xa)
        corr = {}
        for C in (1.0, 100.0):
            corr[f"kernel C={C:g}"] = (SVC(kernel="precomputed", C=C).fit(Ktr, ya).predict(Kte) == yb)
        Rp_b = A.forward_readout(Xb, thp, CFG["reps"], keep=keep)
        corr["VQC parity (global), trained"] = (A.probs_from_z(Rp_b[:, -1]).argmax(1) == yb)
        # frozen-theta head: theta from the PARITY fit, linear head refit on the same 2000 training records
        Rp_a = A.forward_readout(Xa, thp, CFG["reps"], keep=keep)
        head = LogisticRegression(C=1.0, max_iter=2000).fit(Rp_a, ya)
        corr["VQC parity theta + refit linear head (frozen theta)"] = (head.predict(Rp_b) == yb)
        Rh_b = A.forward_readout(Xb, thh, CFG["reps"], keep=keep)
        fhh = fh["head"]
        corr["VQC linear head, co-trained"] = (decide_head(Rh_b, fhh["coef"], fhh["intercept"]) == yb)
        base = corr["VQC parity (global), trained"]
        say(f"\nn = {n}   ({len(keep)} readout columns)")
        for name, c in corr.items():
            m, lo, hi = boot_acc_ci(c, B=2000, seed=0)
            dm, dlo, dhi = paired_boot(c, base)
            rows.append(dict(n=n, model=name, acc=m, ci_lo=lo, ci_hi=hi, diff_vs_parity=dm, diff_lo=dlo, diff_hi=dhi))
            say(f"  {name:58s} acc {m:.4f} [{lo:.4f},{hi:.4f}]   vs parity {100*dm:+.2f} pts [{100*dlo:+.2f},{100*dhi:+.2f}]")
        for ref in ("kernel C=1", "kernel C=100"):
            for name in ("VQC parity (global), trained", "VQC linear head, co-trained"):
                dm, dlo, dhi = paired_boot(corr[name], corr[ref])
                rows.append(dict(n=n, model=f"{name} minus {ref}", acc=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                                 diff_vs_parity=dm, diff_lo=dlo, diff_hi=dhi))
                say(f"  gap  {name:32s} - {ref:12s}: {100*dm:+.2f} pts [{100*dlo:+.2f},{100*dhi:+.2f}]")
    pd.DataFrame(rows).to_csv(OUT / "angle_arm_ladder.csv", index=False)
    (OUT / "notebook_recheck_stage2_log.txt").write_text("\n".join(LOG))


if __name__ == "__main__" and ("--ladder" in sys.argv or "--ladder-only" in sys.argv):
    angle_ladder()
