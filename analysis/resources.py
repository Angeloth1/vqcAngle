"""
Resource counts for the circuits defined in this repository (paper Sec. 3.4).

NOT an experiment: a deterministic transpilation of circuits already specified in the folder
  * amplitude arm : StatePreparation(x_hat) on ceil(log2 N) qubits  +  RealAmplitudes(reps=5, reverse_linear)   [1] / exp1-4
  * angle arm     : RY(x_i) on N qubits                             +  n_local(['ry'], 'cx', linear, reps=2)   notebook
Counts are for the noiseless, ideal gate set; no error model is involved.

Transpiler settings (fixed, reported in the paper): Qiskit 2.3.0, optimization_level=3, seed_transpiler=7.
Two gate sets are reported because depth depends on the single-qubit basis while the CX count does not:
  native  : {cx, rz, sx, x}  (IBM-style; RY is not native and expands to several rz/sx gates)
  u-cx    : {cx, u}          (gate-set-agnostic: any single-qubit gate is one layer; the convention that reproduces [1]'s 'depth 17, 15 CNOT'
                              for the 4-qubit RealAmplitudes reps=5 ansatz)
Two connectivity assumptions:
  all-to-all            : abstract cost of the circuit
  linear chain (NN)     : every 2-qubit gate must act on neighbouring physical qubits (adds SWAPs)
Amplitude state preparation is data dependent in principle, so it is transpiled for real records
(first 25 test vectors) and the median and range are reported.

Run:  .venv/bin/python paper/analysis/resources.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, n_local, real_amplitudes
from qiskit.transpiler import CouplingMap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from qids.data import load_arrays                                    # noqa: E402

OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
BASIS = ["cx", "rz", "sx", "x"]  # kept for reference; see BASES
SEED_T = 7
N_RECORDS = 25
REPS_AMP = 5          # published circuit ([1] Sec. 5.2)
REPS_ANG = 2          # notebook
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


BASES = {"native": ["cx", "rz", "sx", "x"], "ucx": ["cx", "u"]}


def tstat(qc: QuantumCircuit, topo: str):
    """depth in both gate sets, CX count (identical in both), single-qubit gate count in the native set."""
    cm = None if topo == "all-to-all" else CouplingMap.from_line(qc.num_qubits)
    out = {}
    for tag, basis in BASES.items():
        t = transpile(qc, basis_gates=basis, coupling_map=cm, optimization_level=3, seed_transpiler=SEED_T)
        ops = t.count_ops()
        out[f"depth_{tag}"] = t.depth()
        out[f"cx_{tag}"] = int(ops.get("cx", 0))
        if tag == "native":
            out["oneq"] = int(sum(v for k, v in ops.items() if k not in ("cx", "measure", "barrier")))
    assert out["cx_native"] == out["cx_ucx"], "CX count should not depend on the single-qubit basis here"
    out["cx"] = out["cx_native"]
    return out


def amp_prep(x: np.ndarray) -> QuantumCircuit:
    n = int(np.log2(len(x)))
    qc = QuantumCircuit(n)
    qc.append(StatePreparation(x / np.linalg.norm(x)), range(n))
    return qc


def amp_ansatz(n: int, reps: int) -> QuantumCircuit:
    qc = real_amplitudes(n, reps=reps, entanglement="reverse_linear")
    return qc.assign_parameters(np.full(qc.num_parameters, 0.3))


def angle_encode(n: int, x: np.ndarray | None = None) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(0.4 if x is None else float(x[q]), q)
    return qc


def angle_ansatz(n: int, reps: int) -> QuantumCircuit:
    qc = n_local(n, ["ry"], "cx", entanglement="linear", reps=reps)
    return qc.assign_parameters(np.full(qc.num_parameters, 0.3))


def summarise(rows):
    """median and (min,max) of each count over the records"""
    df = pd.DataFrame(rows)
    return {k: (float(df[k].median()), int(df[k].min()), int(df[k].max())) for k in df.columns}


def main():
    say(f"qiskit {qiskit.__version__}; bases {BASES}; optimization_level=3; seed_transpiler={SEED_T}")
    D = load_arrays(n_components=16)
    X = D.Xte
    records = []
    for topo in ("all-to-all", "linear chain"):
        for N in (4, 8, 16):
            nq_a = int(np.log2(N))
            # ------------------------------------------------ amplitude arm
            prep_rows, full_rows = [], []
            for i in range(N_RECORDS):
                x = X[i, :N]
                p = amp_prep(x)
                prep_rows.append(tstat(p, topo))
                full = p.compose(amp_ansatz(nq_a, REPS_AMP))
                full_rows.append(tstat(full, topo))
            ans = tstat(amp_ansatz(nq_a, REPS_AMP), topo)
            sp, sf = summarise(prep_rows), summarise(full_rows)
            for part, s in (("state preparation", sp), ("full circuit (prep + ansatz)", sf)):
                records.append(dict(topology=topo, N=N, arm="amplitude", qubits=nq_a, part=part,
                                    depth_native_med=s["depth_native"][0], depth_native_min=s["depth_native"][1], depth_native_max=s["depth_native"][2],
                                    depth_ucx_med=s["depth_ucx"][0], depth_ucx_min=s["depth_ucx"][1], depth_ucx_max=s["depth_ucx"][2],
                                    cx_med=s["cx"][0], cx_min=s["cx"][1], cx_max=s["cx"][2],
                                    oneq_native_med=s["oneq"][0]))
            records.append(dict(topology=topo, N=N, arm="amplitude", qubits=nq_a, part=f"ansatz alone (RealAmplitudes reps={REPS_AMP})",
                                depth_native_med=ans["depth_native"], depth_native_min=ans["depth_native"], depth_native_max=ans["depth_native"],
                                depth_ucx_med=ans["depth_ucx"], depth_ucx_min=ans["depth_ucx"], depth_ucx_max=ans["depth_ucx"],
                                cx_med=ans["cx"], cx_min=ans["cx"], cx_max=ans["cx"], oneq_native_med=ans["oneq"]))
            # ------------------------------------------------ angle arm
            enc = tstat(angle_encode(N), topo)
            ans = tstat(angle_ansatz(N, REPS_ANG), topo)
            full = tstat(angle_encode(N).compose(angle_ansatz(N, REPS_ANG)), topo)
            for part, s in (("state preparation", enc), (f"ansatz alone (n_local RY+CX linear reps={REPS_ANG})", ans),
                            ("full circuit (prep + ansatz)", full)):
                records.append(dict(topology=topo, N=N, arm="angle-RY", qubits=N, part=part,
                                    depth_native_med=s["depth_native"], depth_native_min=s["depth_native"], depth_native_max=s["depth_native"],
                                    depth_ucx_med=s["depth_ucx"], depth_ucx_min=s["depth_ucx"], depth_ucx_max=s["depth_ucx"],
                                    cx_med=s["cx"], cx_min=s["cx"], cx_max=s["cx"], oneq_native_med=s["oneq"]))
            # ------------------------------------------------ amplitude arm with the parameter-matched ansatz of exp5
            # same n_local(RY, CX, linear) family as the angle arm, reps = 3N/log2(N) - 1, so both carry 3N parameters
            reps_m = 3 * N // nq_a - 1
            sf = summarise([tstat(amp_prep(X[i, :N]).compose(angle_ansatz(nq_a, reps_m)), topo) for i in range(N_RECORDS)])
            ans = tstat(angle_ansatz(nq_a, reps_m), topo)
            records.append(dict(topology=topo, N=N, arm="amplitude (matched)", qubits=nq_a, part="full circuit (prep + ansatz)",
                                depth_native_med=sf["depth_native"][0], depth_native_min=sf["depth_native"][1], depth_native_max=sf["depth_native"][2],
                                depth_ucx_med=sf["depth_ucx"][0], depth_ucx_min=sf["depth_ucx"][1], depth_ucx_max=sf["depth_ucx"][2],
                                cx_med=sf["cx"][0], cx_min=sf["cx"][1], cx_max=sf["cx"][2], oneq_native_med=sf["oneq"][0]))
            records.append(dict(topology=topo, N=N, arm="amplitude (matched)", qubits=nq_a,
                                part=f"ansatz alone (n_local RY+CX linear reps={reps_m})",
                                depth_native_med=ans["depth_native"], depth_native_min=ans["depth_native"], depth_native_max=ans["depth_native"],
                                depth_ucx_med=ans["depth_ucx"], depth_ucx_min=ans["depth_ucx"], depth_ucx_max=ans["depth_ucx"],
                                cx_med=ans["cx"], cx_min=ans["cx"], cx_max=ans["cx"], oneq_native_med=ans["oneq"]))
    R = pd.DataFrame(records)
    R.to_csv(OUT / "resources.csv", index=False)
    say(R.to_string(index=False))

    # ---------- scaling of amplitude state preparation beyond the N used here (all-to-all, real vectors)
    say("\nscaling of amplitude state preparation with the number of qubits (all-to-all; random real vectors, median of 5)")
    rng = np.random.default_rng(0)
    rows = []
    for n in range(2, 9):
        res = []
        for _ in range(5):
            x = rng.normal(size=2 ** n)
            res.append(tstat(amp_prep(x), "all-to-all"))
        s = summarise(res)
        rows.append(dict(qubits=n, N=2 ** n, cx=s["cx"][0], depth_native=s["depth_native"][0],
                         depth_ucx=s["depth_ucx"][0], cx_2n_minus_n_minus_1=2 ** n - n - 1))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "resources_amplitude_scaling.csv", index=False)
    say(S.to_string(index=False))
    say("angle encoding: N qubits, N single-qubit RY gates, 0 two-qubit gates, depth 1, for every N.")

    # ---------- circuit executions needed to TRAIN with a shot-based device (arithmetic from the protocol)
    say("\ncircuit executions to train on a shot-based device = evaluations x records x shots (per COBYLA objective call, whole training subsample)")
    for name, ev, m in (("amplitude anchor, [1] Table 3 protocol (150 evals, 2000 records)", 150, 2000),
                        ("angle, notebook protocol (400 evals, 2000 records)", 400, 2000)):
        for shots in (100, 1000, 10000):
            say(f"  {name:66s} shots={shots:6d}: {ev*m*shots:.2e} executions")
    (OUT / "resources_log.txt").write_text("\n".join(LOG))


if __name__ == "__main__":
    main()
