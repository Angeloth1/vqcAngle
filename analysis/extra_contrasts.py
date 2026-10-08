"""
Extra contrasts needed for Sec. 3.1 and Sec. 4.1 (run after analyze_results.py).
  A. effect of training-set size on the published circuit: exp1 (full split) vs exp4 reps=5, rho=0 (2000-record subsample).
     Everything else is identical: exact simulator, COBYLA 150, theta0 ~ U(-0.1,0.1) seeded per run, q0-parity, same test sets.
  B. classical baselines under the matched 2000-record protocol (exp4, rho=0) vs Table 3 (same protocol, original build).
  C. distance from the fidelity-kernel reference to each classifier rung, amplitude arm (N=16) and angle arm (n=8,16).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import published_ref1 as P1
from stats_utils import msd, welch, welch_summary, diff_ci_welch, hedges_g, hedges_g_summary

R = ROOT / "results"; T = ROOT / "paper" / "tables"
e1 = pd.read_csv(R / "exp1_readout.csv"); e4 = pd.read_csv(R / "exp4_capacity.csv")
e2 = pd.concat([pd.read_csv(R / "exp2_geometry.csv"), pd.read_csv(R / "exp2_C100.csv")])

print("=== A. training-set size effect on the published circuit (reps=5, parity q0), internal accuracy")
full = e1[(e1["mode"] == "frozen_theta") & (e1.readout == "parity")].sort_values("seed")
sub = e4[(e4.rho == 0.0) & (e4.model == "vqc_reps5")].sort_values("seed")
for col_full, col_sub, name in [("internal", "internal", "internal"), ("novel_recall", "novel_recall", "novel")]:
    a, b = full[col_full].to_numpy(), sub[col_sub].to_numpy()
    t, df, p = welch(a, b); g = hedges_g(a, b)
    lo, hi = diff_ci_welch(a.mean(), a.std(ddof=1), 5, b.mean(), b.std(ddof=1), 5)
    print(f"  {name:9s} full {a.mean():.4f}±{a.std(ddof=1):.4f}   2000-rec {b.mean():.4f}±{b.std(ddof=1):.4f}   diff {100*(a.mean()-b.mean()):+.2f} pts  95%CI[{100*lo:+.2f},{100*hi:+.2f}]  Welch p={p:.3f}  g={g:.2f}")
print("  (KDDTest+ not recorded in exp4)")

print("\n=== B. classical baselines at the matched 2000-record protocol (exp4, rho=0) vs Table 3")
for m_new, m_pub in [("logreg_17p", "Logistic Regression"), ("mlp4_73p", "MLP-4"), ("mlp1_19p", "MLP-1")]:
    g = e4[(e4.rho == 0.0) & (e4.model == m_new)].sort_values("seed")
    for col, lab in [("internal", "internal"), ("novel_recall", "novel")]:
        mu, sd = msd(g[col]); pm, ps = P1.TABLE3[m_pub][lab]
        t, df, p = welch_summary(mu, sd, 5, pm, ps, 5)
        print(f"  {m_new:11s} {lab:9s} this {mu:.3f}±{sd:.3f}  Table3 {pm:.3f}±{ps:.3f}  diff {100*(mu-pm):+.1f} pts  Welch p={p:.3f}")
mlp1 = e4[(e4.rho == 0.0) & (e4.model == "mlp1_19p")].sort_values("seed")[["seed", "internal", "novel_recall"]]
print("  MLP-1 per seed (degenerate runs predict the negative class only):\n", mlp1.round(3).to_string(index=False))

print("\n=== C. distance from the kernel reference to each classifier rung (accuracy points; positive = rung BELOW reference)")
def kern(enc, N, C):
    g = e2[(e2.encoding == enc) & (e2.N == N) & (e2.C == C)]
    return msd(g.kernel_svm_acc)
rows = []
# amplitude arm at N=16: exp1 rungs (full training split, full 5039 internal test) vs kernel (2000/2000)
for C in (1.0, 100.0):
    km, ks = kern("amplitude", 16, C)
    for arm in ["frozen_theta/parity", "frozen_theta/best_fixed", "frozen_theta/linear_head", "frozen_theta/mlp_head",
                "cotrained/linear_head", "reference/logreg_on_xhat"]:
        mo, rd = arm.split("/")
        g = e1[(e1["mode"] == mo) & (e1.readout == rd)]
        mu, sd = msd(g.internal)
        t, df, p = welch_summary(km, ks, 5, mu, sd, 5)
        lo, hi = diff_ci_welch(km, ks, 5, mu, sd, 5)
        rows.append(dict(arm="amplitude N=16", C=C, rung=arm, ref=km, rung_acc=mu, gap_pts=100 * (km - mu), ci_lo=100 * lo, ci_hi=100 * hi))
        print(f"  amplitude N=16  C={C:5.0f}  ref {km:.4f}  {arm:28s} {mu:.4f}  gap {100*(km-mu):+6.2f} pts  95%CI[{100*lo:+.2f},{100*hi:+.2f}]")
# matched-protocol amplitude number: 2000-record VQC (exp4 reps5) vs kernel
for C in (1.0, 100.0):
    km, ks = kern("amplitude", 16, C)
    b = sub.internal.to_numpy()
    print(f"  amplitude N=16  C={C:5.0f}  ref {km:.4f}  VQC parity trained on the SAME 2000 records (exp4 reps=5)   {b.mean():.4f}  gap {100*(km-b.mean()):+.2f} pts")
pd.DataFrame(rows).to_csv(T / "gap_amplitude_arm.csv", index=False)

print("\n=== D. other numbers used in the text")
a = e2[(e2.C == 1.0) & (e2.N == 16)].groupby("encoding").separability_auc.mean()
print("  AUC-0.5 at N=16:", {k: round(v - 0.5, 3) for k, v in a.items()}, " ratio angle_ry/amplitude:", round((a['angle_ry'] - .5) / (a['amplitude'] - .5), 1), " phase/amp:", round((a['angle_phase'] - .5) / (a['amplitude'] - .5), 1))
print("  exp1 training seconds per fit (full split, 150 evals): median %.2f  range [%.2f, %.2f]" % (
    e1[e1.readout == 'parity'].seconds.median(), e1[e1.readout == 'parity'].seconds.min(), e1[e1.readout == 'parity'].seconds.max()))
print("  co-trained fit seconds: median %.1f range [%.1f, %.1f]" % (
    e1[e1['mode'] == 'cotrained'].seconds.median(), e1[e1['mode'] == 'cotrained'].seconds.min(), e1[e1['mode'] == 'cotrained'].seconds.max()))
print("  published: 15.6 ± 2.1 min per COBYLA run on 2000 records -> speed-up per run %.0fx (median), hardware nominally the same CPU model" % (15.6 * 60 / e1[e1.readout == 'parity'].seconds.median()))
