"""
Statistics for the gap-filling runs (exp5 matched sweep, exp6 depth x readout, exp3b depth control),
plus the reproduction checks that tie them to the earlier result files.

    .venv/bin/python paper/analysis/analyze_gapfill.py

Writes CSV tables to paper/tables/ and a log to paper/tables/gapfill_log.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import published_ref1 as P1                                                  # noqa: E402
from stats_utils import msd, paired, welch_summary, hedges_g_summary         # noqa: E402

ROOT = HERE.parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def pv(p):
    return "n/a" if p is None or np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


# ====================================================================================== checks
def checks():
    say("=" * 100 + "\nREPRODUCTION CHECKS\n" + "=" * 100)
    # exp3b vs exp3
    old, new = pd.read_csv(RES / "exp3_norm.csv"), pd.read_csv(RES / "exp3_norm_v2.csv")
    m = old.merge(new, on=["seed", "model"], suffixes=("_old", "_new"))
    for k in ("internal", "kddtest", "novel_recall"):
        d = (m[f"{k}_old"] - m[f"{k}_new"]).abs().max()
        say(f"exp3b vs exp3_norm.csv, {k:13s}: {len(m)} common rows, max |diff| = {d:.2e}")
    say("exp3b models:", sorted(new.model.unique()))

    # exp6 parity vs exp4 (rho = 0)
    e6, e4 = pd.read_csv(RES / "exp6_depth_readout.csv"), pd.read_csv(RES / "exp4_capacity.csv")
    e4 = e4[(e4.rho == 0.0) & e4.model.str.startswith("vqc_reps")].copy()
    e4["reps"] = e4.model.str.replace("vqc_reps", "").astype(int)
    m = e6[e6.readout == "parity"].merge(e4, on=["seed", "reps"], suffixes=("_6", "_4"))
    for k6, k4 in (("internal_6", "internal_4"), ("novel_recall_6", "novel_recall_4"), ("train_acc", "fit_acc")):
        say(f"exp6 parity vs exp4 (reps 1/5/17), {k6:15s}: {len(m)} rows, max |diff| = {(m[k6] - m[k4]).abs().max():.2e}")

    # exp5 N=8 seed 0 angle fits vs notebook checkpoints
    th = json.loads((RES / "exp5_thetas.json").read_text())
    for mode, key in (("parity", "N8_s0_angle_global"), ("head", "N8_s0_angle_head")):
        ck = np.load(ROOT / "notebook" / "ckpt" / f"fit_n8_s0_{mode}.npz", allow_pickle=True)["obj"].item()
        d = float(np.abs(np.array(th[key]["theta"]) - np.array(ck["theta"])).max())
        say(f"exp5 {key} vs notebook fit_n8_s0_{mode}: max |d theta| = {d:.2e}; loss {th[key]['loss']:.6f} vs {ck['loss']:.6f}")


# ====================================================================================== exp5
def exp5():
    say("\n" + "=" * 100 + "\nEXP5  matched encoding comparison\n" + "=" * 100)
    d = pd.read_csv(RES / "exp5_matched.csv")
    a16 = RES / "exp5_angle16.csv"
    if a16.exists():
        d = pd.concat([d, pd.read_csv(a16)], ignore_index=True)
    d.to_csv(OUT / "exp5_all_rows.csv", index=False)

    rows = []
    for (N, arm, ro), g in d.groupby(["N", "arm", "readout"]):
        r = dict(N=N, arm=arm, readout=ro, n_seeds=g.seed.nunique(), qubits=g.qubits.iloc[0],
                 params=g.params.iloc[0] if "params" in g else np.nan)
        for k in ("train_acc", "internal", "kddtest", "novel_recall"):
            r[k + "_mean"], r[k + "_sd"] = msd(g[k])
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "exp5_summary.csv", index=False)
    say(S[["N", "arm", "readout", "n_seeds", "qubits", "internal_mean", "internal_sd", "kddtest_mean",
           "novel_recall_mean", "train_acc_mean"]].round(4).to_string(index=False))

    # paired angle - amplitude, per N and readout, by seed
    say("\n-- paired angle - amplitude (same seed = same subsample and theta0 draw)")
    rows = []
    for N in sorted(d.N.unique()):
        for ro in ("q0", "global", "frozen_head_after_q0", "frozen_head_after_global", "head"):
            a = d[(d.N == N) & (d.arm == "angle") & (d.readout == ro)].set_index("seed")
            b = d[(d.N == N) & (d.arm == "amplitude") & (d.readout == ro)].set_index("seed")
            common = sorted(set(a.index) & set(b.index))
            if not common:
                continue
            for k in ("internal", "kddtest", "novel_recall"):
                x, y = a.loc[common, k].to_numpy(), b.loc[common, k].to_numpy()
                if len(common) >= 2:
                    pr = paired(x, y)
                    rows.append(dict(N=N, readout=ro, metric=k, n=len(common), diff=pr["diff"], ci_lo=pr["ci"][0],
                                     ci_hi=pr["ci"][1], p=pr["p"], dz=pr["dz"]))
                else:
                    rows.append(dict(N=N, readout=ro, metric=k, n=1, diff=float(x[0] - y[0]), ci_lo=np.nan,
                                     ci_hi=np.nan, p=np.nan, dz=np.nan))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "exp5_angle_minus_amplitude.csv", index=False)
    say(D[D.metric == "internal"].round(4).to_string(index=False))
    say("\n  KDDTest+ and novel:")
    say(D[D.metric != "internal"].round(4).to_string(index=False))

    # N=16: angle seed 0 against the amplitude seed distribution
    say("\n-- N=16: angle (seed 0, notebook fits) against amplitude (5 seeds)")
    for ro in ("global", "frozen_head_after_global", "head"):
        a = d[(d.N == 16) & (d.arm == "angle") & (d.readout == ro)]
        b = d[(d.N == 16) & (d.arm == "amplitude") & (d.readout == ro)]
        if len(a) and len(b):
            for k in ("internal", "kddtest", "novel_recall"):
                bm, bs = msd(b[k])
                b0 = b[b.seed == 0][k].iloc[0]
                say(f"  {ro:26s} {k:13s} angle s0 {a[k].iloc[0]:.4f}  amplitude s0 {b0:.4f}  amplitude 5 seeds {bm:.4f} ± {bs:.4f}"
                    f"  (angle s0 - amp mean = {100*(a[k].iloc[0]-bm):+.2f} pts)")

    # distance to the kernel reference, per arm (internal), same training records and same test set
    say("\n-- distance to the fidelity-kernel reference of the same encoding (points, internal; positive = below)")
    rows = []
    for N in sorted(d.N.unique()):
        for arm, ref in (("amplitude", "kernel_amplitude"), ("angle", "kernel_angle_ry")):
            for C in ("C=1", "C=100"):
                kr = d[(d.N == N) & (d.arm == ref) & (d.readout == C)].set_index("seed")["internal"]
                for ro in ("q0", "global", "frozen_head_after_global", "head"):
                    g = d[(d.N == N) & (d.arm == arm) & (d.readout == ro)].set_index("seed")["internal"]
                    common = sorted(set(kr.index) & set(g.index))
                    if not common:
                        continue
                    x = kr.loc[common].to_numpy() - g.loc[common].to_numpy()
                    if len(common) >= 2:
                        pr = paired(kr.loc[common].to_numpy(), g.loc[common].to_numpy())
                        rows.append(dict(N=N, arm=arm, ref=C, readout=ro, n=len(common), gap=pr["diff"],
                                         ci_lo=pr["ci"][0], ci_hi=pr["ci"][1], p=pr["p"]))
                    else:
                        rows.append(dict(N=N, arm=arm, ref=C, readout=ro, n=1, gap=float(x[0]), ci_lo=np.nan,
                                         ci_hi=np.nan, p=np.nan))
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "exp5_gap_to_kernel.csv", index=False)
    say(G.round(4).to_string(index=False))

    # share of the parity gap closed by the head, per arm and N (mean over seeds, C=1)
    say("\n-- share of the parity-to-reference distance closed by the head (C = 1, global parity)")
    for N in sorted(d.N.unique()):
        for arm, ref in (("amplitude", "kernel_amplitude"), ("angle", "kernel_angle_ry")):
            k = d[(d.N == N) & (d.arm == ref) & (d.readout == "C=1")]
            p = d[(d.N == N) & (d.arm == arm) & (d.readout == "global")]
            fh = d[(d.N == N) & (d.arm == arm) & (d.readout == "frozen_head_after_global")]
            h = d[(d.N == N) & (d.arm == arm) & (d.readout == "head")]
            if len(p) == 0:
                continue
            seeds = sorted(set(p.seed))
            km = k[k.seed.isin(seeds)].internal.mean()
            gap = km - p.internal.mean()
            say(f"  N={N:2d} {arm:9s} ref {km:.4f}  parity {p.internal.mean():.4f}  frozen head {fh.internal.mean():.4f}  "
                f"co-trained {h.internal.mean():.4f}  | gap {100*gap:+.2f} pts; frozen closes {100*(fh.internal.mean()-p.internal.mean()):+.2f}"
                f" ({(fh.internal.mean()-p.internal.mean())/gap*100 if gap>0 else float('nan'):.0f}%), co-trained closes "
                f"{100*(h.internal.mean()-p.internal.mean()):+.2f} ({(h.internal.mean()-p.internal.mean())/gap*100 if gap>0 else float('nan'):.0f}%)")

    # references: kernel on the full internal test set vs exp2 (2,000 test records)
    say("\n-- kernel reference: full internal test set (exp5) vs 2,000-record test subsample (exp2), C=1 and 100")
    e2 = pd.concat([pd.read_csv(RES / "exp2_geometry.csv"), pd.read_csv(RES / "exp2_C100.csv")])
    for N in sorted(d.N.unique()):
        for enc5, enc2 in (("kernel_amplitude", "amplitude"), ("kernel_angle_ry", "angle_ry")):
            for C in (1.0, 100.0):
                a = d[(d.N == N) & (d.arm == enc5) & (d.readout == f"C={C:g}")].internal
                b = e2[(e2.N == N) & (e2.encoding == enc2) & (e2.C == C)].kernel_svm_acc
                say(f"  N={N:2d} {enc2:10s} C={C:5g}: exp5 {a.mean():.4f} ± {a.std(ddof=1):.4f}   exp2 {b.mean():.4f} ± {b.std(ddof=1):.4f}"
                    f"   diff {100*(a.mean()-b.mean()):+.2f} pts")
    # logistic regression per N
    say("\n-- logistic regression per N (internal / KDDTest+ / novel), mean over seeds")
    for N in sorted(d.N.unique()):
        for arm in ("logreg_xhat", "logreg_x"):
            g = d[(d.N == N) & (d.arm == arm)]
            say(f"  N={N:2d} {arm:12s} {g.internal.mean():.4f} ± {g.internal.std(ddof=1):.4f}   {g.kddtest.mean():.4f}   {g.novel_recall.mean():.4f}")
    return d


# ====================================================================================== exp6
def exp6():
    say("\n" + "=" * 100 + "\nEXP6  depth x readout, 2,000-record protocol of [1]\n" + "=" * 100)
    d = pd.read_csv(RES / "exp6_depth_readout.csv")
    rows = []
    for (reps, ro), g in d.groupby(["reps", "readout"]):
        r = dict(reps=reps, params=int(g.params.iloc[0]), maxiter=int(g.maxiter.iloc[0]), readout=ro, n=g.seed.nunique())
        for k in ("train_acc", "internal", "kddtest", "novel_recall", "loss"):
            r[k + "_mean"], r[k + "_sd"] = msd(g[k])
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "exp6_summary.csv", index=False)
    say(S.pivot_table(index="reps", columns="readout", values="internal_mean").round(4).to_string())
    say("SD:")
    say(S.pivot_table(index="reps", columns="readout", values="internal_sd").round(4).to_string())
    say("KDDTest+:")
    say(S.pivot_table(index="reps", columns="readout", values="kddtest_mean").round(4).to_string())
    say("novel recall:")
    say(S.pivot_table(index="reps", columns="readout", values="novel_recall_mean").round(4).to_string())
    say("loss (parity fit for the frozen readouts; cotrain loss for linear_cotrained):")
    say(S.pivot_table(index="reps", columns="readout", values="loss_mean").round(4).to_string())

    say("\n-- replication of Table A2 at all five depths (parity)")
    a2 = pd.DataFrame(P1.A2, columns=P1.A2_COLUMNS)
    rows = []
    for reps in sorted(d.reps.unique()):
        g = d[(d.reps == reps) & (d.readout == "parity")].sort_values("seed")
        a = a2[a2.reps == reps].sort_values("seed")
        for kn, kp in (("internal", "internal"), ("kddtest", "kddtest"), ("novel_recall", "novel")):
            m, s = msd(g[kn]); pm, ps = msd(a[kp])
            t, df_, p = welch_summary(m, s, 5, pm, ps, 5)
            rows.append(dict(reps=reps, metric=kp, new_mean=m, new_sd=s, A2_mean=pm, A2_sd=ps, diff=m - pm, welch_p=p))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "exp6_vs_A2.csv", index=False)
    say(R.round(4).to_string(index=False))

    say("\n-- paired contrasts (internal accuracy, points)")
    piv = d.pivot_table(index=["seed"], columns=["reps", "readout"], values="internal")
    rows = []

    def add(label, x, y):
        pr = paired(x, y)
        rows.append(dict(contrast=label, diff_pts=100 * pr["diff"], ci_lo=100 * pr["ci"][0], ci_hi=100 * pr["ci"][1],
                         p=pr["p"], dz=pr["dz"]))

    for reps in sorted(d.reps.unique()):
        for ro in ("best_fixed", "linear_frozen", "linear_cotrained"):
            add(f"reps={reps}: {ro} - parity", piv[(reps, ro)].to_numpy(), piv[(reps, "parity")].to_numpy())
    for ro in ("parity", "linear_cotrained"):
        add(f"{ro}: reps 9 -> 17", piv[(17, ro)].to_numpy(), piv[(9, ro)].to_numpy())
        add(f"{ro}: reps 5 -> 9", piv[(9, ro)].to_numpy(), piv[(5, ro)].to_numpy())
        add(f"{ro}: reps 1 -> 5", piv[(5, ro)].to_numpy(), piv[(1, ro)].to_numpy())
    add("cotrained reps=1 - parity reps=17", piv[(1, "linear_cotrained")].to_numpy(), piv[(17, "parity")].to_numpy())
    add("cotrained reps=5 - parity reps=17", piv[(5, "linear_cotrained")].to_numpy(), piv[(17, "parity")].to_numpy())
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "exp6_contrasts.csv", index=False)
    say(C.round(3).to_string(index=False))

    say("\n-- novel recall contrasts (points)")
    pivn = d.pivot_table(index=["seed"], columns=["reps", "readout"], values="novel_recall")
    rows = []
    for reps in sorted(d.reps.unique()):
        for ro in ("best_fixed", "linear_frozen", "linear_cotrained"):
            pr = paired(pivn[(reps, ro)].to_numpy(), pivn[(reps, "parity")].to_numpy())
            rows.append(dict(contrast=f"reps={reps}: {ro} - parity", diff_pts=100 * pr["diff"], ci_lo=100 * pr["ci"][0],
                             ci_hi=100 * pr["ci"][1], p=pr["p"]))
    for ro in ("parity", "linear_cotrained"):
        rho, p = stats.spearmanr(d[d.readout == ro].reps, d[d.readout == ro].novel_recall)
        rows.append(dict(contrast=f"{ro}: Spearman(reps, novel) over 25 runs", diff_pts=rho, ci_lo=np.nan, ci_hi=np.nan, p=p))
    Cn = pd.DataFrame(rows)
    Cn.to_csv(OUT / "exp6_contrasts_novel.csv", index=False)
    say(Cn.round(3).to_string(index=False))
    return d


# ====================================================================================== exp3b
def exp3b():
    say("\n" + "=" * 100 + "\nEXP3b  norm restoration with the depth control\n" + "=" * 100)
    d = pd.read_csv(RES / "exp3_norm_v2.csv")
    rows = []
    for m, g in d.groupby("model"):
        r = dict(model=m, n=len(g))
        for k in ("internal", "kddtest", "novel_recall"):
            if g[k].notna().any():
                r[k + "_mean"], r[k + "_sd"] = msd(g[k])
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "exp3b_summary.csv", index=False)
    say(S.round(4).to_string(index=False))

    def arm(name):
        return d[d.model == name].sort_values("seed")

    b5, b4 = arm("vqc_amplitude_4q_p24"), arm("vqc_amplitude_4q_reps4_p20")
    real, shuf = arm("vqc_amplitude+norm_5q_p25"), arm("vqc_shuffled_norm_5q_p25")
    rows = []
    for label, x, y in (("norm 5q (reps 4) - 4q reps 4  [extra qubit + norm, depth matched]", real, b4),
                        ("shuffled 5q (reps 4) - 4q reps 4  [extra qubit only, depth matched]", shuf, b4),
                        ("norm 5q - shuffled 5q  [norm only]", real, shuf),
                        ("4q reps 4 - 4q reps 5  [depth only]", b4, b5)):
        for k in ("internal", "kddtest", "novel_recall"):
            pr = paired(x[k].to_numpy(), y[k].to_numpy())
            rows.append(dict(contrast=label, metric=k, diff=pr["diff"], ci_lo=pr["ci"][0], ci_hi=pr["ci"][1],
                             p=pr["p"], dz=pr["dz"]))
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "exp3b_contrasts.csv", index=False)
    say(C.round(4).to_string(index=False))


if __name__ == "__main__":
    which = sys.argv[1:] or ["checks", "exp5", "exp6", "exp3b"]
    for w in which:
        {"checks": checks, "exp5": exp5, "exp6": exp6, "exp3b": exp3b}[w]()
    (OUT / "gapfill_log.txt").write_text("\n".join(LOG))
