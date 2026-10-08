"""
Numerical checks of the four steps behind the attainable reference (Section 3.2).

  1. A circuit that encodes once and measures an observable is linear in rho(x):
     f(x) = <phi(x)| U^dag O U |phi(x)> = Tr[M rho(x)],  M = U^dag O U.
  2. The Hilbert-Schmidt inner product of two encoded states is the fidelity of Eq. (1):
     Tr[rho(x) rho(y)] = |<phi(x)|phi(y)>|^2, and the closed forms of Eq. (1) hold.
  3. The decision function of an SVM trained on the precomputed kernel F,
     g(x) = sum_j alpha_j y_j F(x_j, x) + b, equals Tr[M_svm rho(x)] + b with
     M_svm = sum_j alpha_j y_j rho(x_j): the SVM picks one member of the same linear family.
  4. Every linear readout of the paper (parity, best fixed assignment, linear head on <Z_S>)
     classifies by the sign of Tr[M rho(x)] + b for some M, so it lies in that family.

Small sizes (N = 4, 300 records), so it runs in seconds:

    VQCANGLE=/mnt/Main/src/ship/vqcAngle  $VQCANGLE/.venv/bin/python analysis/check_attainable_reference.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.svm import SVC

sys.path.insert(0, os.environ.get("VQCANGLE", "/mnt/Main/src/ship/vqcAngle"))
from qids.data import load_arrays, subsample                          # noqa: E402
from qids.fastsim import angle_states, ansatz_unitary, fidelity_kernel, l2_normalise  # noqa: E402

rng = np.random.default_rng(0)
A = load_arrays(n_components=16)
X, y = subsample(A.Xtr, A.ytr, 300, 0)
worst = {}


def note(name, value):
    worst[name] = max(worst.get(name, 0.0), float(value))


# ---- states: amplitude (N = 16, 4 qubits), angle-RY and angle-phase (N = 4, 4 qubits, s = 0.25)
states = {
    "amplitude": (l2_normalise(X), X),
    "angle_ry": (angle_states(X[:, :4] * 0.25, "ry"), X[:, :4] * 0.25),
    "angle_phase": (angle_states(X[:, :4] * 0.25, "phase"), X[:, :4] * 0.25),
}
Z = np.diag([1.0 - 2.0 * (b & 1) for b in range(16)])            # Z on qubit 0 (least significant bit)

for enc, (psi, feats) in states.items():
    rho = np.einsum("ma,mb->mab", psi, psi.conj())                 # rho(x) = |phi><phi|

    # step 1: f(x) = <phi|U^dag O U|phi> = Tr[M rho(x)]
    U = ansatz_unitary(rng.normal(size=24), 4, 5)                  # RealAmplitudes, 4 qubits, 5 reps
    M = U.conj().T @ Z @ U
    f_direct = np.real(np.einsum("ma,ab,mb->m", psi.conj(), M, psi))
    f_trace = np.real(np.einsum("ab,mba->m", M, rho))
    note("step 1  |f - Tr[M rho]|", np.abs(f_direct - f_trace).max())

    # step 2: Tr[rho(x) rho(y)] = |<phi(x)|phi(y)>|^2 = closed form of Eq. (1)
    K_trace = np.real(np.einsum("iab,jba->ij", rho[:50], rho[:50]))
    K_overlap = np.abs(psi[:50].conj() @ psi[:50].T) ** 2
    K_closed = fidelity_kernel(feats[:50], feats[:50], enc)
    note("step 2  |Tr[rho rho'] - |<phi|phi'>|^2|", np.abs(K_trace - K_overlap).max())
    note("step 2  |closed form (Eq. 1) - |<phi|phi'>|^2|", np.abs(K_closed - K_overlap).max())

    # step 3: SVM decision function = Tr[M_svm rho(x)] + b
    K = fidelity_kernel(feats, feats, enc)
    svm = SVC(kernel="precomputed", C=1.0).fit(K, y)
    coef = np.zeros(len(feats))
    coef[svm.support_] = svm.dual_coef_[0]                          # alpha_j y_j
    M_svm = np.einsum("j,jab->ab", coef, rho)
    g_svm = svm.decision_function(K)
    g_trace = np.real(np.einsum("ab,mba->m", M_svm, rho)) + svm.intercept_[0]
    note("step 3  |SVM decision - (Tr[M_svm rho] + b)|", np.abs(g_svm - g_trace).max())

# step 4: the linear readouts are of the form Tr[M rho] + b (amplitude circuit, 4 qubits)
psi = l2_normalise(X)
U = ansatz_unitary(rng.normal(size=24), 4, 5)
P = np.abs(psi @ U.T) ** 2                                          # outcome probabilities p(b)
pop = np.array([bin(b).count("1") for b in range(16)])
for name, w, c in (("qubit-0 parity", 1.0 - 2.0 * (np.arange(16) & 1), 0.0),
                   ("global parity", 1.0 - 2.0 * (pop % 2), 0.0),
                   ("fixed assignment", rng.integers(0, 2, 16).astype(float), -0.5),
                   ("linear head", rng.normal(size=16), rng.normal())):
    D = U.T @ np.diag(w) @ U                                        # M = U^dag diag(w) U
    lhs = P @ w + c
    rhs = np.einsum("ma,ab,mb->m", psi, D, psi) + c
    note(f"step 4  {name}: |sum_b w_b p(b) - Tr[M rho]|", np.abs(lhs - rhs).max())

print("largest deviation of each identity (should be ~1e-12 or smaller):")
for k, v in worst.items():
    print(f"  {k:52s} {v:.1e}")
