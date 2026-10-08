"""
Generate every table of the manuscript (Markdown) directly from the result files and the analysis
outputs, so that no number in a table is typed by hand.

  main text:   t3_anchor, t4_geometry, t5a_ladder, t5b_matched, t6_resources
  supplement:  s1 ... s8  (per-seed values, controls, recomputations)
  figure data: fig1_data.csv

Run after analyze_results.py, notebook_recheck.py (both stages) and resources.py:
    .venv/bin/python paper/analysis/make_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import published_ref1 as P1                                                         # noqa: E402
from stats_utils import (diff_ci_welch, hedges_g_summary, msd, paired, welch,        # noqa: E402
                         welch_summary)

R = ROOT / "results"
T = ROOT / "paper" / "tables"
MD = T / "md"
MD.mkdir(parents=True, exist_ok=True)


def w(name, text):
    (MD / f"{name}.md").write_text(text.strip() + "\n")


def f3(m, s=None):
    return f"{m:.3f}" if s is None else f"{m:.3f} ± {s:.3f}"


def sg(x, nd=1):
    return f"{x:+.{nd}f}".replace("-", "−")


def pv(p):
    if np.isnan(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.2f}" if p >= 0.1 else f"{p:.3f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------------------------- data
e1 = pd.read_csv(R / "exp1_readout.csv")
e2 = pd.concat([pd.read_csv(R / "exp2_geometry.csv"), pd.read_csv(R / "exp2_C100.csv")], ignore_index=True)
e3 = pd.read_csv(R / "exp3_norm_v2.csv")          # superset of exp3_norm.csv (identical common rows) + depth control
e4 = pd.read_csv(R / "exp4_capacity.csv")
e5 = pd.concat([pd.read_csv(R / "exp5_matched.csv"), pd.read_csv(R / "exp5_angle16.csv")], ignore_index=True)
e6 = pd.read_csv(R / "exp6_depth_readout.csv")
a2 = pd.DataFrame(P1.A2, columns=P1.A2_COLUMNS)


def arm(mode, readout):
    return e1[(e1["mode"] == mode) & (e1.readout == readout)].sort_values("seed")


# =========================================================================== Table 3: anchor
def table3():
    anc = arm("frozen_theta", "parity")
    r5 = a2[a2.reps == 5]
    ref = arm("reference", "logreg_on_xhat").iloc[0]
    sub = e4[(e4.rho == 0.0) & (e4.model == "vqc_reps5")].sort_values("seed")
    lr2 = e4[(e4.rho == 0.0) & (e4.model == "logreg_17p")].sort_values("seed")
    cols = [("internal", "internal", "internal"), ("kddtest", "kddtest", "kddtest"), ("novel_recall", "novel", "novel")]

    def cell(df, col):
        m, s = msd(df[col])
        return f3(m, s)

    rows = []
    rows.append(["**Anchor**: amplitude, 4 qubits, reps 5, qubit-0 parity, 20,153 records (E1)"] + [cell(anc, c[0]) for c in cols])
    t3 = P1.TABLE3["VQC (COBYLA)"]
    rows.append(["[@thomos2026] Table 3: VQC (COBYLA), 2,000 records, original build"] + [f3(*t3[c[2]]) for c in cols])
    a2s = {c[2]: msd(r5[c[2]]) for c in cols}
    rows.append(["[@thomos2026] Table A2: reps = 5, 2,000 records, build of the anchor"] + [f3(*a2s[c[2]]) for c in cols])
    for label, pub in (("Anchor − Table 3", {c[2]: t3[c[2]] for c in cols}), ("Anchor − Table A2", a2s)):
        cells = []
        for (k_new, _, k_pub) in cols:
            m, s = msd(anc[k_new])
            pm, ps = pub[k_pub]
            t, df, p = welch_summary(m, s, 5, pm, ps, 5)
            lo, hi = diff_ci_welch(m, s, 5, pm, ps, 5)
            g = hedges_g_summary(m, s, 5, pm, ps, 5)
            cells.append(f"{sg(100*(m-pm))} [{sg(100*lo)}, {sg(100*hi)}]; p = {pv(p)}; g = {sg(g, 2)}")
        rows.append([f"*{label}*, points [95% CI]"] + cells)
    rows.append(["Same circuit on 2,000 records (E4, ρ = 0)", cell(sub, "internal"), "not recorded", cell(sub, "novel_recall")])
    cells = []
    for k in ("internal", "novel_recall"):
        a, b = anc[k].to_numpy(), sub[k].to_numpy()
        t, df, p = welch(a, b)
        lo, hi = diff_ci_welch(a.mean(), a.std(ddof=1), 5, b.mean(), b.std(ddof=1), 5)
        cells.append(f"{sg(100*(a.mean()-b.mean()))} [{sg(100*lo)}, {sg(100*hi)}]; p = {pv(p)}")
    rows.append(["*Full split − 2,000 records*, points [95% CI]", cells[0], "n/a", cells[1]])
    rows.append(["Logistic regression on x̂, 20,153 records (no seed variance)", f3(ref.internal), f3(ref.kddtest), f3(ref.novel_recall)])
    rows.append(["Logistic regression on x̂, 2,000 records (E4)", cell(lr2, "internal"), "not recorded", cell(lr2, "novel_recall")])
    lr = P1.TABLE3["Logistic Regression"]
    rows.append(["[@thomos2026] Table 3: logistic regression, 2,000 records"] + [f3(*lr[c[2]]) for c in cols])
    w("t3_anchor", md_table(["Model / source", "Internal", "KDDTest+", "Novel recall"], rows))

    # replication of the depth sweep at all five depths (exp6, parity readout)
    rows = []
    for reps in (1, 3, 5, 9, 17):
        g = e6[(e6.reps == reps) & (e6.readout == "parity")].sort_values("seed")
        a = a2[a2.reps == reps].sort_values("seed")
        cells = []
        for kn, kp in (("internal", "internal"), ("kddtest", "kddtest"), ("novel_recall", "novel")):
            m, s_ = msd(g[kn]); pm, ps = msd(a[kp])
            t, df, p = welch_summary(m, s_, 5, pm, ps, 5)
            cells += [f3(m, s_), f3(pm, ps), f"{sg(100*(m-pm))}; p = {pv(p)}"]
        rows.append([f"{reps} ({4*(reps+1)})"] + cells)
    w("s7_depth_replication", md_table(["reps (params)", "Internal, new", "Internal, Table A2", "Δ points; Welch p",
                                        "KDDTest+, new", "KDDTest+, Table A2", "Δ points; Welch p",
                                        "Novel, new", "Novel, Table A2", "Δ points; Welch p"], rows))


# =========================================================================== Table 4: geometry
def table4():
    rows = []
    enc_name = {"amplitude": "Amplitude", "angle_ry": "Angle-RY", "angle_phase": "Angle-phase"}
    for N in (4, 8, 16):
        d = {}
        for enc in ("amplitude", "angle_ry", "angle_phase"):
            g1 = e2[(e2.N == N) & (e2.encoding == enc) & (e2.C == 1.0)].sort_values("seed")
            g100 = e2[(e2.N == N) & (e2.encoding == enc) & (e2.C == 100.0)].sort_values("seed")
            d[enc] = (g1, g100)
            rows.append([N, enc_name[enc], int(g1.qubits.iloc[0]), f3(*msd(g1.separability_auc)),
                         f3(*msd(g1.kernel_svm_acc)), f3(*msd(g100.kernel_svm_acc))])
        for enc in ("angle_ry", "angle_phase"):
            a1, a100 = d[enc]; b1, b100 = d["amplitude"]
            pa = paired(a1.separability_auc.to_numpy(), b1.separability_auc.to_numpy())
            p1 = paired(a1.kernel_svm_acc.to_numpy(), b1.kernel_svm_acc.to_numpy())
            p100 = paired(a100.kernel_svm_acc.to_numpy(), b100.kernel_svm_acc.to_numpy())
            fmt = lambda pr: f"{sg(100*pr['diff'])} [{sg(100*pr['ci'][0])}, {sg(100*pr['ci'][1])}]"
            rows.append(["", f"*Δ {enc_name[enc]} − amplitude*", "", f"{sg(pa['diff'], 3)}", fmt(p1), fmt(p100)])
    hdr = ["N", "Encoding", "Qubits", "Separability AUC", "Kernel accuracy, C = 1", "Kernel accuracy, C = 100"]
    w("t4_geometry", md_table(hdr, rows))


# =========================================================================== Table 5: classifier level
def table5():
    names = [("frozen_theta", "parity", "Parity, qubit 0 (published readout)"),
             ("frozen_theta", "best_fixed", "Best fixed many-to-one assignment (all 2¹⁶ searched)"),
             ("frozen_theta", "linear_head", "Linear head on the 16 outcome probabilities"),
             ("frozen_theta", "mlp_head", "MLP head (32 hidden units) on the same probabilities"),
             ("cotrained", "linear_head", "Co-trained linear head"),
             ("reference", "logreg_on_xhat", "Logistic regression on x̂ (no quantum stage)")]
    base = arm("frozen_theta", "parity")
    rows = []
    for mode, ro, label in names:
        g = arm(mode, ro)
        if ro == "logreg_on_xhat":
            r0 = g.iloc[0]
            rows.append([label, f3(r0.internal), f3(r0.kddtest), f3(r0.novel_recall),
                         sg(100 * (r0.internal - base.internal.mean())) + " (vs constant)"])
        elif ro == "parity":
            rows.append([label, f3(*msd(g.internal)), f3(*msd(g.kddtest)), f3(*msd(g.novel_recall)), "reference row"])
        else:
            pr = paired(g.internal.to_numpy(), base.internal.to_numpy())
            rows.append([label, f3(*msd(g.internal)), f3(*msd(g.kddtest)), f3(*msd(g.novel_recall)),
                         f"{sg(100*pr['diff'])} [{sg(100*pr['ci'][0])}, {sg(100*pr['ci'][1])}]"])
    k = {C: msd(e2[(e2.N == 16) & (e2.encoding == "amplitude") & (e2.C == C)].kernel_svm_acc) for C in (1.0, 100.0)}
    rows.append(["Fidelity-kernel reference, N = 16, 2,000 records (C = 1; C = 100)", f"{f3(*k[1.0])}; {f3(*k[100.0])}", "n/a", "n/a", "n/a"])
    w("t5a_ladder", md_table(["Readout on the same circuit (amplitude, N = 16, 4 qubits)", "Internal", "KDDTest+",
                              "Novel recall", "Δ internal vs parity, points [95% CI]"], rows))

    w("t5b_matched", table5b_matched())


def table5b_matched():
    """exp5: both encodings under one protocol. Cells: internal accuracy, mean ± SD over the seeds the circuit
    arm has (5; seed 0 only for the 16-qubit angle arm, whose references are then also seed 0)."""
    cols = [("-", "lr"), ("C=1", "kern"), ("q0", None), ("global", None), ("frozen_head_after_global", None), ("head", None)]
    rows = []
    for N in (4, 8, 16):
        q = int(np.log2(N))
        arms = (("amplitude", f"Amplitude, {q} qubits", "logreg_xhat", "kernel_amplitude"),
                ("angle", f"Angle-RY, {N} qubits", "logreg_x", "kernel_angle_ry"))
        seeds = {}
        for arm_, label, lr, kern in arms:
            seeds[arm_] = sorted(e5[(e5.N == N) & (e5.arm == arm_)].seed.unique())
        common = sorted(set(seeds["amplitude"]) & set(seeds["angle"]))
        vals = {}
        for arm_, label, lr, kern in arms:
            cells = []
            for ro, kind in cols:
                src = {"lr": lr, "kern": kern, None: arm_}[kind]
                x = e5[(e5.N == N) & (e5.arm == src) & (e5.readout == ro) & (e5.seed.isin(seeds[arm_]))].sort_values("seed")
                vals[(arm_, ro)] = x.set_index("seed").internal
                cells.append("not run" if len(x) == 0 else (f"{x.internal.iloc[0]:.3f}" if len(x) == 1 else f3(*msd(x.internal))))
            tag = "" if len(seeds[arm_]) > 1 else " (seed 0)"
            rows.append([N if arm_ == "amplitude" else "", f"{label}, {3*N} params{tag}"] + cells)
        cells = []
        for ro, kind in cols:
            a_, b_ = vals[("angle", ro)], vals[("amplitude", ro)]
            cs = sorted(set(a_.index) & set(b_.index) & set(common))
            if not cs:
                cells.append("")
            elif len(cs) == 1:
                cells.append(f"{sg(100*(a_.loc[cs[0]] - b_.loc[cs[0]]))}")
            else:
                pr = paired(a_.loc[cs].to_numpy(), b_.loc[cs].to_numpy())
                cells.append(f"{sg(100*pr['diff'])} [{sg(100*pr['ci'][0])}, {sg(100*pr['ci'][1])}]")
        rows.append(["", "*Δ angle − amplitude, points*" + (" (seed 0)" if len(common) == 1 else " [95% CI]")] + cells)
    hdr = ["N", "Arm (qubits, parameters)", "Logistic regression on the arm's input", "Fidelity-kernel reference (C = 1)",
           "Qubit-0 parity", "Global parity", "Frozen head (global-parity weights)", "Co-trained head"]
    return md_table(hdr, rows)


# =========================================================================== Table 6: resources
def table6():
    """Resources of the exp5 circuits: amplitude (matched n_local ansatz) and angle-RY, both 3N parameters."""
    r = pd.read_csv(T / "resources.csv")
    full, prep = "full circuit (prep + ansatz)", "state preparation"

    def get(topo, N, arm_, part):
        return r[(r.topology == topo) & (r.N == N) & (r.arm == arm_) & (r.part == part)].iloc[0]

    dep = lambda q: f"{int(q.depth_ucx_med)}/{int(q.depth_native_med)}"
    rows = []
    for N in (4, 8, 16):
        for enc_arm, full_arm, label in (("amplitude", "amplitude (matched)", "Amplitude"), ("angle-RY", "angle-RY", "Angle-RY")):
            pa, pc = get("all-to-all", N, enc_arm, prep), get("linear chain", N, enc_arm, prep)
            fa, fc = get("all-to-all", N, full_arm, full), get("linear chain", N, full_arm, full)
            rows.append([N, label, int(pa.qubits), 3 * N,
                         f"{int(pa.cx_med)} ({int(pc.cx_med)})", f"{dep(pa)} ({dep(pc)})",
                         f"{int(fa.cx_med)} ({int(fc.cx_med)})", f"{dep(fa)} ({dep(fc)})"])
    hdr = ["N", "Encoding", "Qubits", "Parameters", "Encoding stage: two-qubit gates", "Encoding stage: depth {u,cx}/native",
           "Full circuit: two-qubit gates", "Full circuit: depth {u,cx}/native"]
    w("t6_resources", md_table(hdr, rows))

    s = pd.read_csv(T / "resources_amplitude_scaling.csv")
    rows = [[int(x.qubits), int(x.N), int(x.cx), int(x.cx_2n_minus_n_minus_1), int(x.depth_ucx), int(x.depth_native)]
            for x in s.itertuples()]
    w("s8b_amp_scaling", md_table(["Qubits n", "N = 2ⁿ", "CX (transpiled)", "2ⁿ − n − 1", "Depth {u,cx}", "Depth native"], rows))
    full = r.copy()
    rows = [[x.topology, x.N, x.arm, x.qubits, x.part, f"{x.cx_med:g}", f"{x.depth_ucx_med:g}",
             f"{x.depth_native_med:g} [{x.depth_native_min}–{x.depth_native_max}]", f"{x.oneq_native_med:g}"] for x in full.itertuples()]
    w("s8a_resources_full", md_table(["Topology", "N", "Arm", "Qubits", "Part", "CX", "Depth {u,cx}", "Depth native (range)", "1-qubit gates (native)"], rows))


# =========================================================================== figure data
def fig1():
    """Both panels from exp5: same training records, same internal test set, same metric."""
    rows = []
    for (N, arm_, ro), g in e5.groupby(["N", "arm", "readout"]):
        m, sd_ = msd(g.internal)
        kind = f"SD over {g.seed.nunique()} seeds" if g.seed.nunique() > 1 else "single seed (0)"
        if arm_.startswith("kernel"):
            rows.append(dict(panel="a_encoding_level", encoding=arm_.replace("kernel_", ""), N=N, qubits=int(g.qubits.iloc[0]),
                             variant=f"fidelity-kernel SVM, {ro}", acc=m, spread=sd_, spread_kind=kind))
        elif arm_.startswith("logreg"):
            rows.append(dict(panel="both_reference_line", encoding=arm_, N=N, qubits=0, variant="logistic regression",
                             acc=m, spread=sd_, spread_kind=kind))
        elif ro in ("q0", "global", "frozen_head_after_global", "head"):
            rows.append(dict(panel="b_classifier_level", encoding=arm_, N=N, qubits=int(g.qubits.iloc[0]),
                             variant={"q0": "qubit-0 parity", "global": "global parity",
                                      "frozen_head_after_global": "frozen head", "head": "co-trained head"}[ro],
                             acc=m, spread=sd_, spread_kind=kind))
    pd.DataFrame(rows).to_csv(T / "fig1_data.csv", index=False)


# =========================================================================== supplement
def supplement():
    # S1: E1 per seed
    d = e1.copy(); d["arm"] = d["mode"] + "/" + d["readout"]
    rows = [[x.seed, x.arm, f"{x.internal:.4f}", f"{x.kddtest:.4f}", f"{x.novel_recall:.4f}"] for x in d.sort_values(["arm", "seed"]).itertuples()]
    w("s1_exp1_per_seed", md_table(["Seed", "Arm", "Internal", "KDDTest+", "Novel recall"], rows))
    # S2: E2 per seed
    rows = [[x.seed, x.N, x.encoding, x.C, x.qubits, f"{x.separability_auc:.4f}", f"{x.kernel_svm_acc:.4f}", x.n_sv]
            for x in e2.sort_values(["C", "N", "encoding", "seed"]).itertuples()]
    w("s2_exp2_per_seed", md_table(["Seed", "N", "Encoding", "C", "Qubits", "AUC", "Kernel accuracy", "Support vectors"], rows))
    # S3: E3
    rows = [[x.seed, x.model, "" if pd.isna(x.internal) else f"{x.internal:.4f}", "" if pd.isna(x.kddtest) else f"{x.kddtest:.4f}",
             "" if pd.isna(x.novel_recall) else f"{x.novel_recall:.4f}"] for x in e3.sort_values(["model", "seed"]).itertuples()]
    w("s3_exp3_per_seed", md_table(["Seed", "Model", "Internal (AUC for the norm row)", "KDDTest+ (AUC for the norm row)", "Novel recall"], rows))
    # S4: E4 summary
    fit = e4.pivot_table(index="model", columns="rho", values="fit_acc", aggfunc="mean")
    clean = fit[0.0]
    exc = pd.DataFrame({r_: fit[r_] - (clean + (r_ / 2) * (1 - 2 * clean)) for r_ in fit.columns if r_ > 0})
    exc["mean"] = exc.mean(axis=1)
    nov = e4[e4.rho == 0].groupby("model").novel_recall.mean()
    inte = e4[e4.rho == 0].groupby("model").internal.mean()
    rows = [[m, f"{fit.loc[m, 0.0]:.3f}", f"{fit.loc[m, 0.25]:.3f}", f"{fit.loc[m, 0.5]:.3f}", f"{fit.loc[m, 1.0]:.3f}",
             f"{exc.loc[m, 0.25]:+.3f}", f"{exc.loc[m, 0.5]:+.3f}", f"{exc.loc[m, 1.0]:+.3f}", f"{exc.loc[m, 'mean']:+.3f}",
             f"{inte[m]:.3f}", f"{nov[m]:.3f}"] for m in fit.index]
    w("s4_exp4_capacity", md_table(["Model", "Fit acc. ρ=0", "ρ=0.25", "ρ=0.5", "ρ=1", "Excess ρ=0.25", "ρ=0.5", "ρ=1", "Mean excess",
                                    "Internal (ρ=0)", "Novel (ρ=0)"], rows))
    # S5: angle kernel vs Gaussian
    g = pd.read_csv(T / "angle_kernel_vs_rbf.csv").groupby(["N", "C"]).agg(
        acc_ry=("acc_ry", "mean"), acc_rbf=("acc_rbf", "mean"), corr=("corr", "min"), meandiff=("meandiff", "mean"), worst=("acc_rbf", "count")).reset_index()
    raw = pd.read_csv(T / "angle_kernel_vs_rbf.csv")
    raw["d"] = 100 * (raw.acc_rbf - raw.acc_ry).abs()
    worst = raw.groupby(["N", "C"]).d.max()
    rows = [[int(x.N), f"{x.C:g}", f"{x.acc_ry:.4f}", f"{x.acc_rbf:.4f}", f"{100*(x.acc_rbf-x.acc_ry):+.2f}",
             f"{worst.loc[(x.N, x.C)]:.2f}", f"{x.corr:.4f}"] for x in g.itertuples()]
    w("s5_angle_kernel_vs_gaussian", md_table(["N", "C", "Angle-RY kernel accuracy", "Gaussian (γ = s²/4) accuracy", "Mean Δ, points",
                                               "Worst per-seed absolute Δ, points", "Min. kernel-matrix correlation"], rows))
    # S6: head shots
    h = pd.read_csv(T / "notebook_head_shots_recheck.csv")
    gg = h.groupby(["n", "order", "shots"]).acc.agg(["mean", "std"]).reset_index()
    rows = [[int(x.n), "assembled as in the notebook (singles, pairs, parity)" if x.order == "notebook" else "assembled in the head's column order",
             int(x.shots), f"{x.mean:.4f} ± {x.std:.4f}"] for x in gg.itertuples()]
    w("s6_head_shots", md_table(["n", "Column order of the shot-estimated features", "Shots", "Accuracy (5 draws, 500 records)"], rows))


def supplement_gapfill():
    # S10: depth x readout (exp6)
    lab = {"parity": "Qubit-0 parity", "best_fixed": "Best fixed assignment", "linear_frozen": "Linear head, frozen",
           "linear_cotrained": "Linear head, co-trained"}
    rows = []
    for (reps, ro), g in e6.groupby(["reps", "readout"]):
        rows.append([f"{reps} ({4*(reps+1)})", int(g.maxiter.iloc[0]), lab[ro], f3(*msd(g.internal)), f3(*msd(g.kddtest)),
                     f3(*msd(g.novel_recall)), f3(*msd(g.train_acc))])
    order = {v: i for i, v in enumerate(lab.values())}
    rows.sort(key=lambda r: (int(r[0].split()[0]), order[r[2]]))
    w("s10_exp6_depth_readout", md_table(["reps (params)", "COBYLA budget", "Readout", "Internal", "KDDTest+", "Novel recall",
                                           "Training accuracy"], rows))
    # S11: exp5, every arm and readout
    names = {"q0": "qubit-0 parity", "global": "global parity", "frozen_head_after_q0": "frozen head (q0 weights)",
             "frozen_head_after_global": "frozen head (global weights)", "head": "co-trained head", "C=1": "C = 1",
             "C=100": "C = 100", "-": "-"}
    arms = {"amplitude": "Amplitude circuit", "angle": "Angle-RY circuit", "kernel_amplitude": "Kernel, amplitude",
            "kernel_angle_ry": "Kernel, angle-RY", "logreg_xhat": "Logistic regression on x̂", "logreg_x": "Logistic regression on x"}
    rows = []
    for (N, arm_, ro), g in e5.groupby(["N", "arm", "readout"]):
        rows.append([N, arms[arm_], names[ro], g.seed.nunique(), f3(*msd(g.internal)) if len(g) > 1 else f"{g.internal.iloc[0]:.3f}",
                     f3(*msd(g.kddtest)) if len(g) > 1 else f"{g.kddtest.iloc[0]:.3f}",
                     f3(*msd(g.novel_recall)) if len(g) > 1 else f"{g.novel_recall.iloc[0]:.3f}",
                     f3(*msd(g.train_acc)) if len(g) > 1 else f"{g.train_acc.iloc[0]:.3f}"])
    ordr = list(arms.keys())
    rows.sort(key=lambda r: (r[0], ordr.index([k for k, v in arms.items() if v == r[1]][0]), r[2]))
    w("s11_exp5_all", md_table(["N", "Model", "Readout / C", "Seeds", "Internal", "KDDTest+", "Novel recall", "Training accuracy"], rows))


if __name__ == "__main__":
    table3(); table4(); table5(); table6(); fig1(); supplement(); supplement_gapfill()
    print("tables written to", MD)
    for p in sorted(MD.glob("*.md")):
        print("  ", p.name, f"({len(p.read_text().splitlines())} lines)")
