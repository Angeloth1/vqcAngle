"""
Statistics for every experiment in results/*.csv, plus the anchor comparison against the published
control paper [1]. Run from anywhere:

    .venv/bin/python paper/analysis/analyze_results.py

Writes CSV tables to paper/tables/ and prints a readable log. Every number that appears in the
manuscript's Results section is produced here (or in notebook_recheck.py / resources.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import published_ref1 as P1                                  # noqa: E402
from stats_utils import (diff_ci_welch, fmt, hedges_g_summary, msd, paired,    # noqa: E402
                         one_sample_vs_constant, welch_summary)

ROOT = HERE.parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 400)

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)


# ======================================================================================
# EXP 1 -- readout ladder (amplitude, 4 qubits, reps=5, full training split, COBYLA 150)
# ======================================================================================
def exp1():
    d = pd.read_csv(RES / "exp1_readout.csv")
    d["arm"] = d["mode"] + "/" + d["readout"]
    order = ["frozen_theta/parity", "frozen_theta/global_parity", "frozen_theta/best_fixed",
             "frozen_theta/linear_head", "frozen_theta/mlp_head", "cotrained/linear_head",
             "reference/logreg_on_xhat"]
    say("\n" + "=" * 100 + "\nEXP1  readout ladder  (5 seeds; seed = COBYLA initialisation only, data split fixed)\n" + "=" * 100)
    say("seeds per arm:", d.groupby("arm").seed.nunique().to_dict())
    rows = []
    for arm in order:
        g = d[d.arm == arm].sort_values("seed")
        r = dict(arm=arm)
        for m in ["internal", "kddtest", "novel_recall"]:
            mu, sd = msd(g[m])
            r[m + "_mean"], r[m + "_sd"] = mu, sd
        rows.append(r)
    T = pd.DataFrame(rows)
    save(T, "exp1_ladder_summary")
    say(T.round(4).to_string(index=False))

    base = d[d.arm == "frozen_theta/parity"].sort_values("seed")
    say("\n-- paired vs the published decoder (parity = b mod 2 = qubit 0) on the SAME frozen theta / same seed")
    prow = []
    for arm in order[1:]:
        g = d[d.arm == arm].sort_values("seed")
        for m in ["internal", "kddtest", "novel_recall"]:
            if arm == "reference/logreg_on_xhat":
                # deterministic baseline: identical in every seed
                res = one_sample_vs_constant(base[m].to_numpy(), g[m].iloc[0])
                res = {k: (-v if k in ("diff", "t", "dz") else v) for k, v in res.items()}
                res["ci"] = (-res["ci"][1], -res["ci"][0]) if isinstance(res["ci"], tuple) else res["ci"]
                # sign: report arm - parity
                res["diff"] = float(g[m].iloc[0] - base[m].mean())
            else:
                res = paired(g[m].to_numpy(), base[m].to_numpy())
            t_w, df_w, p_w = welch_summary(g[m].mean(), g[m].std(ddof=1), 5,
                                           base[m].mean(), base[m].std(ddof=1), 5)
            prow.append(dict(arm=arm, metric=m, diff_vs_parity=res["diff"], ci_lo=res["ci"][0], ci_hi=res["ci"][1],
                             paired_p=res["p"], dz=res["dz"], welch_p=p_w))
    PR = pd.DataFrame(prow)
    save(PR, "exp1_ladder_vs_parity")
    say(PR.round(4).to_string(index=False))

    # within-ladder increments (frozen theta): parity -> best_fixed -> linear -> mlp
    say("\n-- ladder increments on internal accuracy (points), mean over seeds")
    pt = {a: d[d.arm == a].sort_values("seed").internal.to_numpy() for a in order}
    say(f"parity->best_fixed  {100*(pt['frozen_theta/best_fixed']-pt['frozen_theta/parity']).mean():+.2f}")
    say(f"best_fixed->linear  {100*(pt['frozen_theta/linear_head']-pt['frozen_theta/best_fixed']).mean():+.2f}")
    say(f"linear->mlp         {100*(pt['frozen_theta/mlp_head']-pt['frozen_theta/linear_head']).mean():+.2f}")
    say(f"parity->cotrained   {100*(pt['cotrained/linear_head']-pt['frozen_theta/parity']).mean():+.2f}")
    say(f"cotrained vs logreg {100*(pt['cotrained/linear_head'].mean()-pt['reference/logreg_on_xhat'].mean()):+.2f}")

    # seed-level correlation between the three metrics (does internal accuracy predict KDDTest+ / novel?)
    say("\n-- across-seed correlation of the parity anchor's metrics (n=5, descriptive)")
    say(base[["internal", "kddtest", "novel_recall"]].corr(method="spearman").round(3).to_string())
    return d, T


# ======================================================================================
# ANCHOR vs published (Table 3, Table A2) + depth-sweep replication (exp4 rho=0)
# ======================================================================================
def anchor(d1, d4):
    say("\n" + "=" * 100 + "\nANCHOR  new protocol vs published [1]\n" + "=" * 100)
    a2 = pd.DataFrame(P1.A2, columns=P1.A2_COLUMNS)
    r5 = a2[a2.reps == 5]
    chk = {m: msd(r5[m]) for m in ["internal", "kddtest", "novel"]}
    say("transcription check, A2 reps=5 recomputed from the 5 printed rows:",
        {k: (round(v[0], 3), round(v[1], 3)) for k, v in chk.items()}, " printed:", P1.A2_REPS5_PRINTED)

    anc = d1[d1.arm == "frozen_theta/parity"].sort_values("seed")
    ref = d1[d1.arm == "reference/logreg_on_xhat"].iloc[0]
    rows = []
    for label, m_new, m_pub in [("internal", "internal", "internal"), ("kddtest", "kddtest", "kddtest"),
                                ("novel", "novel_recall", "novel")]:
        mu, sd = msd(anc[m_new])
        for src, (pm, ps) in [("Table 3 (original build, 2000-rec., maxiter 150)", P1.TABLE3["VQC (COBYLA)"][label]),
                              ("Table A2 reps=5 (independent build, 2000-rec.)", chk[m_pub])]:
            t, df, p = welch_summary(mu, sd, 5, pm, ps, 5)
            lo, hi = diff_ci_welch(mu, sd, 5, pm, ps, 5)
            g = hedges_g_summary(mu, sd, 5, pm, ps, 5)
            rows.append(dict(metric=label, source=src, anchor_mean=mu, anchor_sd=sd, published_mean=pm, published_sd=ps,
                             diff=mu - pm, ci_lo=lo, ci_hi=hi, welch_t=t, welch_df=df, welch_p=p, hedges_g=g))
    A = pd.DataFrame(rows)
    save(A, "anchor_vs_published")
    say(A.round(4).to_string(index=False))

    say("\nlogistic regression on x_hat (full training split; deterministic, no seed variance) vs Table 3 LogReg (2000-record subsample)")
    for label, key in [("internal", "internal"), ("kddtest", "kddtest"), ("novel", "novel_recall")]:
        pm, ps = P1.TABLE3["Logistic Regression"][label]
        v = float(ref[key])
        say(f"  {label:9s} ours {v:.4f}   published {pm:.3f} ± {ps:.3f}   diff {v-pm:+.4f}  ({(v-pm)/ps:+.1f} published SDs)")

    # ---- depth-sweep replication: exp4 rho=0 rows vs A2 (same 2000-record protocol)
    say("\n-- depth-sweep replication at the published protocol: exp4 (rho=0) vs Table A2, reps in {1,5,17}")
    v = d4[(d4.rho == 0.0) & d4.model.str.startswith("vqc_reps")].copy()
    v["reps"] = v.model.str.replace("vqc_reps", "").astype(int)
    rows = []
    for reps in [1, 5, 17]:
        g = v[v.reps == reps].sort_values("seed")
        a = a2[a2.reps == reps].sort_values("seed")
        for mn, mp in [("internal", "internal"), ("novel_recall", "novel")]:
            mu, sd = msd(g[mn])
            pm, ps = msd(a[mp])
            t, df, p = welch_summary(mu, sd, 5, pm, ps, 5)
            pr = paired(g[mn].to_numpy(), a[mp].to_numpy())
            rows.append(dict(reps=reps, metric=mp, new_mean=mu, new_sd=sd, A2_mean=pm, A2_sd=ps, diff=mu - pm,
                             welch_p=p, paired_p=pr["p"], per_seed_absdiff_max=float(np.abs(g[mn].to_numpy() - a[mp].to_numpy()).max())))
    R = pd.DataFrame(rows)
    save(R, "depth_replication_vs_A2")
    say(R.round(4).to_string(index=False))
    say("per-seed internal accuracy, exp4 vs A2:")
    for reps in [1, 5, 17]:
        g = v[v.reps == reps].sort_values("seed").internal.round(3).tolist()
        a = a2[a2.reps == reps].sort_values("seed").internal.round(3).tolist()
        say(f"  reps={reps:2d}  new {g}   A2 {a}")


# ======================================================================================
# EXP 2 -- geometry: separability AUC and fidelity-kernel SVM accuracy
# ======================================================================================
def exp2():
    d1 = pd.read_csv(RES / "exp2_geometry.csv")     # C = 1
    d100 = pd.read_csv(RES / "exp2_C100.csv")       # C = 100
    d = pd.concat([d1, d100], ignore_index=True)
    say("\n" + "=" * 100 + "\nEXP2  encoded geometry: separability AUC and fidelity-kernel accuracy (5 seeds; seed = subsample draw)\n" + "=" * 100)
    say("rows:", len(d), " seeds:", sorted(d.seed.unique()), " C:", sorted(d.C.unique()))
    # AUC does not depend on C -> verify
    chk = d.groupby(["seed", "N", "encoding"]).separability_auc.nunique().max()
    say("AUC identical across C for every (seed,N,encoding):", chk == 1)

    rows = []
    for (C, N, enc), g in d.groupby(["C", "N", "encoding"]):
        g = g.sort_values("seed")
        am, asd = msd(g.separability_auc)
        km, ksd = msd(g.kernel_svm_acc)
        rows.append(dict(C=C, N=N, encoding=enc, qubits=int(g.qubits.iloc[0]), auc_mean=am, auc_sd=asd,
                         acc_mean=km, acc_sd=ksd, nsv_mean=g.n_sv.mean()))
    T = pd.DataFrame(rows)
    save(T, "exp2_summary")
    say(T.round(4).to_string(index=False))

    say("\n-- paired differences (same seed = same subsample): angle - amplitude")
    rows = []
    for C in [1.0, 100.0]:
        for N in [4, 8, 16]:
            for enc in ["angle_ry", "angle_phase"]:
                a = d[(d.C == C) & (d.N == N) & (d.encoding == enc)].sort_values("seed")
                b = d[(d.C == C) & (d.N == N) & (d.encoding == "amplitude")].sort_values("seed")
                for m in ["separability_auc", "kernel_svm_acc"]:
                    pr = paired(a[m].to_numpy(), b[m].to_numpy())
                    rows.append(dict(C=C, N=N, contrast=f"{enc} - amplitude", metric=m, diff=pr["diff"],
                                     ci_lo=pr["ci"][0], ci_hi=pr["ci"][1], paired_p=pr["p"], dz=pr["dz"]))
    D = pd.DataFrame(rows)
    save(D, "exp2_angle_minus_amplitude")
    say(D.round(4).to_string(index=False))

    say("\n-- within-encoding change N=4 -> N=16 (paired across seeds)")
    rows = []
    for C in [1.0, 100.0]:
        for enc in ["amplitude", "angle_ry", "angle_phase"]:
            a = d[(d.C == C) & (d.N == 16) & (d.encoding == enc)].sort_values("seed")
            b = d[(d.C == C) & (d.N == 4) & (d.encoding == enc)].sort_values("seed")
            for m in ["separability_auc", "kernel_svm_acc"]:
                pr = paired(a[m].to_numpy(), b[m].to_numpy())
                rows.append(dict(C=C, encoding=enc, metric=m, diff_16_minus_4=pr["diff"], ci_lo=pr["ci"][0],
                                 ci_hi=pr["ci"][1], paired_p=pr["p"], dz=pr["dz"]))
    W = pd.DataFrame(rows)
    save(W, "exp2_n4_to_n16")
    say(W.round(4).to_string(index=False))

    say("\n-- C=1 -> C=100 (does the 'ceiling' move?)  mean change in kernel accuracy, points")
    for enc in ["amplitude", "angle_ry", "angle_phase"]:
        for N in [4, 8, 16]:
            a = d[(d.C == 100.0) & (d.N == N) & (d.encoding == enc)].sort_values("seed").kernel_svm_acc.to_numpy()
            b = d[(d.C == 1.0) & (d.N == N) & (d.encoding == enc)].sort_values("seed").kernel_svm_acc.to_numpy()
            pr = paired(a, b)
            say(f"  {enc:12s} N={N:2d}  {100*pr['diff']:+.2f} pts  (p={pr['p']:.4f})")

    say("\n-- AUC vs published Table 5 (mean ± SD, 5 seeds), this build minus published")
    rows = []
    for enc in ["amplitude", "angle_ry", "angle_phase"]:
        for N in [4, 8, 16]:
            g = d[(d.C == 1.0) & (d.N == N) & (d.encoding == enc)].separability_auc
            mu, sd = msd(g)
            pm, ps = P1.TABLE5[enc][N]
            t, df, p = welch_summary(mu, sd, 5, pm, ps, 5)
            rows.append(dict(encoding=enc, N=N, this_mean=mu, this_sd=sd, pub_mean=pm, pub_sd=ps, diff=mu - pm, welch_p=p))
    V = pd.DataFrame(rows)
    save(V, "exp2_auc_vs_table5")
    say(V.round(4).to_string(index=False))
    return d


# ======================================================================================
# EXP 3 -- norm restoration
# ======================================================================================
def exp3(d1):
    d = pd.read_csv(RES / "exp3_norm.csv")
    say("\n" + "=" * 100 + "\nEXP3  norm restoration\n" + "=" * 100)
    say("models:", sorted(d.model.unique()))
    say("NOTE: results CSV has no 'vqc_amplitude_4q_reps4' arm although exp3_norm_restoration.py defines it -> "
        "CSV was produced by an earlier version of the script.")
    rows = []
    for m, g in d.groupby("model"):
        g = g.sort_values("seed")
        r = dict(model=m, n=len(g))
        for k in ["internal", "kddtest", "novel_recall"]:
            if g[k].notna().any():
                r[k + "_mean"], r[k + "_sd"] = msd(g[k])
        rows.append(r)
    T = pd.DataFrame(rows)
    save(T, "exp3_summary")
    say(T.round(4).to_string(index=False))

    say("\n-- the deleted scalar: AUC of log1p(||x||) alone (deterministic, identical across seeds)")
    a = d[d.model == "auc_of_norm_alone"].iloc[0]
    say(f"   internal AUC {a.internal:.4f}   KDDTest+ AUC {a.kddtest:.4f}")

    say("\n-- classical ablation: adding the norm back to logistic regression on x_hat (deterministic)")
    lx = d[d.model == "logreg_xhat"].iloc[0]
    ln = d[d.model == "logreg_xhat+norm"].iloc[0]
    lr = d[d.model == "logreg_raw_x"].iloc[0]
    for k in ["internal", "kddtest", "novel_recall"]:
        say(f"   {k:13s} x_hat {lx[k]:.4f}  x_hat+norm {ln[k]:.4f} ({100*(ln[k]-lx[k]):+.2f} pts)  raw x {lr[k]:.4f} ({100*(lr[k]-lx[k]):+.2f} pts)")

    say("\n-- quantum ablation, paired by seed (all arms share seed -> same theta0 draw within an arm family)")
    base = d[d.model == "vqc_amplitude_4q_p24"].sort_values("seed")
    real = d[d.model == "vqc_amplitude+norm_5q_p25"].sort_values("seed")
    shuf = d[d.model == "vqc_shuffled_norm_5q_p25"].sort_values("seed")
    rows = []
    for name, x, y in [("norm5q - shuffled5q (does the norm carry the gain?)", real, shuf),
                       ("norm5q - base4q (does restoring the norm beat the published circuit?)", real, base),
                       ("shuffled5q - base4q (price of the extra qubit alone)", shuf, base)]:
        for k in ["internal", "kddtest", "novel_recall"]:
            pr = paired(x[k].to_numpy(), y[k].to_numpy())
            t, df, p = welch_summary(x[k].mean(), x[k].std(ddof=1), 5, y[k].mean(), y[k].std(ddof=1), 5)
            rows.append(dict(contrast=name, metric=k, diff=pr["diff"], ci_lo=pr["ci"][0], ci_hi=pr["ci"][1],
                             paired_p=pr["p"], welch_p=p, dz=pr["dz"]))
    Q = pd.DataFrame(rows)
    save(Q, "exp3_quantum_ablation")
    say(Q.round(4).to_string(index=False))

    say("\n-- base 4q arm of exp3 is the same computation as exp1 parity? (per-seed identical)")
    p1 = d1[d1.arm == "frozen_theta/parity"].sort_values("seed")
    say("   max |exp3 base - exp1 parity| internal:", float(np.abs(base.internal.to_numpy() - p1.internal.to_numpy()).max()))
    say("   max |exp3 base - exp1 parity| kddtest :", float(np.abs(base.kddtest.to_numpy() - p1.kddtest.to_numpy()).max()))
    return d


# ======================================================================================
# EXP 4 -- capacity via random-label fitting
# ======================================================================================
def exp4():
    d = pd.read_csv(RES / "exp4_capacity.csv")
    say("\n" + "=" * 100 + "\nEXP4  memorisation capacity vs novel-signature recall\n" + "=" * 100)
    say("models:", sorted(d.model.unique()), " rho:", sorted(d.rho.unique()), " seeds:", sorted(d.seed.unique()))
    fit = d.pivot_table(index="model", columns="rho", values="fit_acc", aggfunc="mean")
    fit_sd = d.pivot_table(index="model", columns="rho", values="fit_acc", aggfunc=lambda x: x.std(ddof=1))
    say("training accuracy on (corrupted) labels, mean over seeds:")
    say(fit.round(4).to_string())
    clean = fit[0.0]
    excess = pd.DataFrame({r: fit[r] - (clean + (r / 2) * (1 - 2 * clean)) for r in fit.columns if r > 0})
    excess["mean_excess"] = excess.mean(axis=1)
    nov = d[d.rho == 0.0].groupby("model").novel_recall.mean().rename("novel_recall")
    inte = d[d.rho == 0.0].groupby("model").internal.mean().rename("internal")
    J = pd.concat([excess, nov, inte], axis=1)
    say("\nexcess fit over a signal-only learner (memorisation) and clean-data metrics:")
    say(J.round(4).to_string())
    save(J.reset_index(), "exp4_memorisation")

    # per-seed memorisation to get seed-level dispersion
    rows = []
    for seed, g in d.groupby("seed"):
        f = g.pivot_table(index="model", columns="rho", values="fit_acc")
        c = f[0.0]
        ex = pd.DataFrame({r: f[r] - (c + (r / 2) * (1 - 2 * c)) for r in f.columns if r > 0}).mean(axis=1)
        for m, v in ex.items():
            rows.append(dict(seed=seed, model=m, memorisation=v))
    S = pd.DataFrame(rows)
    say("\nper-model memorisation, mean ± SD over seeds:")
    say(S.groupby("model").memorisation.agg(["mean", "std"]).round(4).to_string())

    j = J.dropna(subset=["novel_recall"])
    rho_s, p = stats.spearmanr(j["mean_excess"], j["novel_recall"])
    say(f"\nSpearman(memorisation, novel recall) across the 6 models: rho={rho_s:.3f} p={p:.3f}  spread={j.mean_excess.max()-j.mean_excess.min():.3f}")

    say("\nwithin the VQC family (reps 1/5/17):")
    V = J.loc[["vqc_reps1", "vqc_reps5", "vqc_reps17"], ["mean_excess", "novel_recall", "internal"]]
    say(V.round(4).to_string())
    per = d[(d.rho == 0.0) & d.model.str.startswith("vqc_reps")].pivot_table(index="seed", columns="model", values="novel_recall")
    say("per-seed novel recall, VQC by reps:\n" + per.round(3).to_string())
    for a_, b_ in [("vqc_reps5", "vqc_reps1"), ("vqc_reps17", "vqc_reps5"), ("vqc_reps17", "vqc_reps1")]:
        pr = paired(per[a_].to_numpy(), per[b_].to_numpy())
        say(f"  novel recall {a_} - {b_}: {100*pr['diff']:+.2f} pts  95%CI[{100*pr['ci'][0]:+.2f},{100*pr['ci'][1]:+.2f}] p={pr['p']:.3f} dz={pr['dz']:.2f}")
    return d


if __name__ == "__main__":
    d1, _ = exp1()
    d4 = pd.read_csv(RES / "exp4_capacity.csv")
    anchor(d1, d4)
    exp2()
    exp3(d1)
    exp4()
    (OUT / "analysis_log.txt").write_text("\n".join(LOG))
    print("\n[written] tables ->", OUT)
