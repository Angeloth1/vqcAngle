"""
Trained circuits of the matched comparison (E5) against the classical models on the same inputs.

For every N, metric and circuit arm, the circuit is compared, paired by seed (same 2,000 training records), with
  * logistic regression on the same input   (x for the angle arm, x_hat for the amplitude arm)
  * the fidelity-kernel SVM of its own encoding, C = 1 and C = 100 (the attainable reference)
The 16-qubit angle circuit exists for seed 0 only; it is compared with the seed-0 value of each baseline.

    .venv/bin/python paper/analysis/circuit_vs_classical.py

Reads  paper/tables/exp5_all_rows.csv  (written by analyze_gapfill.py).
Writes paper/tables/exp5_circuit_vs_classical.csv and paper/tables/circuit_vs_classical_log.txt.
Differences are in accuracy points (x 100); positive = circuit above the baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stats_utils import paired                                               # noqa: E402

T = HERE.parent / "tables"
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def pv(p):
    return "n/a" if p is None or np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


d = pd.read_csv(T / "exp5_all_rows.csv")
CIRCUITS = [("angle", "head"), ("angle", "global"), ("amplitude", "head"), ("amplitude", "global"), ("amplitude", "q0")]
BASE = {
    "angle": [("logreg_x", "-"), ("kernel_angle_ry", "C=1"), ("kernel_angle_ry", "C=100")],
    "amplitude": [("logreg_xhat", "-"), ("kernel_amplitude", "C=1"), ("kernel_amplitude", "C=100")],
}


def series(N, arm, readout, metric):
    return d[(d.N == N) & (d.arm == arm) & (d.readout == readout)].set_index("seed")[metric]


rows = []
for metric in ("internal", "kddtest", "novel_recall"):
    for N in (4, 8, 16):
        for arm, ro in CIRCUITS:
            a = series(N, arm, ro, metric)
            if a.empty:
                continue
            for base, bro in BASE[arm]:
                b = series(N, base, bro, metric)
                s = a.index.intersection(b.index)
                if len(s) > 1:
                    r = paired(a.loc[s].values, b.loc[s].values)
                    rows.append(dict(metric=metric, N=N, arm=arm, readout=ro, baseline=f"{base} {bro}".strip(" -"), n=len(s),
                                     circuit=a.loc[s].mean(), base=b.loc[s].mean(), diff_pts=100 * r["diff"],
                                     ci_lo_pts=100 * r["ci"][0], ci_hi_pts=100 * r["ci"][1], p=r["p"]))
                else:
                    rows.append(dict(metric=metric, N=N, arm=arm, readout=ro, baseline=f"{base} {bro}".strip(" -"), n=len(s),
                                     circuit=a.loc[s].mean(), base=b.loc[s].mean(), diff_pts=100 * (a.loc[s].mean() - b.loc[s].mean()),
                                     ci_lo_pts=np.nan, ci_hi_pts=np.nan, p=np.nan))
R = pd.DataFrame(rows)
R.to_csv(T / "exp5_circuit_vs_classical.csv", index=False)

say("=" * 100 + "\nE5 circuits against classical models on the same inputs (paired by seed; points; + = circuit above)\n" + "=" * 100)
for metric in ("internal", "kddtest", "novel_recall"):
    say(f"\n-- {metric}")
    say(f"{'N':>3s} {'circuit':22s} {'baseline':24s} {'n':>2s} {'circuit':>8s} {'base':>8s} {'diff':>7s} {'95% CI':>17s} {'p':>7s}")
    for _, r in R[R.metric == metric].iterrows():
        ci = "" if np.isnan(r.ci_lo_pts) else f"[{r.ci_lo_pts:+.2f},{r.ci_hi_pts:+.2f}]"
        say(f"{r.N:3d} {r.arm + ' ' + r.readout:22s} {r.baseline:24s} {r.n:2d} {r.circuit:8.4f} {r.base:8.4f} {r.diff_pts:+7.2f} {ci:>17s} {pv(r.p):>7s}")

# novel recall of the co-trained circuits against the share of novel records whose nearest training record
# (by fidelity) is an attack (novel_mechanism.py --nn), paired by seed
nn = pd.read_csv(T / "novel_nearest_label.csv")
say("\n-- novel recall of the co-trained head minus the nearest-neighbour attack share (points, mean over seeds)")
for arm, enc in (("angle", "angle_ry"), ("amplitude", "amplitude")):
    parts = []
    for N in (4, 8, 16):
        r = series(N, arm, "head", "novel_recall")
        s = nn[(nn.N == N) & (nn.encoding == enc)].set_index("seed").share_nn_attack
        c = r.index.intersection(s.index)
        parts.append(f"N={N}: recall {r.loc[c].mean():.3f}  share {s.loc[c].mean():.3f}  diff {100 * (r.loc[c] - s.loc[c]).mean():+.1f}"
                     f"{'  (seed 0)' if len(c) == 1 else ''}")
    say(f"  {arm:9s} " + " | ".join(parts))
(T / "circuit_vs_classical_log.txt").write_text("\n".join(LOG) + "\n")
