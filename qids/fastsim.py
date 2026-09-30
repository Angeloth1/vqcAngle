"""
Exact, noiseless forward pass for the paper's VQC, written as dense linear algebra.

Why this exists
---------------
In the published pipeline the ansatz is RealAmplitudes and the simulator is
noiseless. Two consequences that the paper does not exploit:

  1. U(theta) is a *real orthogonal* 2^n x 2^n matrix and depends only on theta,
     never on the data.
  2. Amplitude encoding maps record x to the unit vector x_hat directly.

So the whole forward pass over the entire dataset is

        P = (X_hat @ U(theta).T) ** 2                       # (m, 2^n)

one matrix multiply, not m circuit simulations. verify_fastsim.py checks this
against qiskit_machine_learning.SamplerQNN to ~1e-12.

Practical effect: the 15.6 min/run COBYLA figure in Section 7.4 becomes ~1 s.
Every experiment in this repo is therefore full-scale, not subsampled, unless a
script says otherwise.

Angle encoding produces a product state, built by iterated Kronecker product.
Tractable to about n = 12 for batch training; the *kernel* (exp2) is closed form
and has no qubit-count limit at all.
"""
from __future__ import annotations

import numpy as np

try:  # qiskit >= 2.1 moved to the function form
    from qiskit.circuit.library import real_amplitudes as _real_amplitudes

    def _ansatz(n: int, reps: int):
        return _real_amplitudes(n, reps=reps, entanglement="reverse_linear")
except ImportError:  # pragma: no cover
    from qiskit.circuit.library import RealAmplitudes

    def _ansatz(n: int, reps: int):
        return RealAmplitudes(n, reps=reps, entanglement="reverse_linear")

from qiskit.quantum_info import Operator


# --------------------------------------------------------------------------- #
#  ansatz
# --------------------------------------------------------------------------- #
def n_params(n_qubits: int, reps: int) -> int:
    """RealAmplitudes parameter count: one RY per qubit per rotation layer."""
    return n_qubits * (reps + 1)


def ansatz_unitary(theta: np.ndarray, n_qubits: int, reps: int) -> np.ndarray:
    """Dense real orthogonal 2^n x 2^n matrix for RealAmplitudes(theta).

    Built once per theta evaluation. Cost is independent of the record count,
    which is the whole point.
    """
    qc = _ansatz(n_qubits, reps).assign_parameters(np.asarray(theta, dtype=float))
    U = Operator(qc).data
    assert np.abs(U.imag).max() < 1e-12, "RealAmplitudes should be real-valued"
    return np.ascontiguousarray(U.real)


# --------------------------------------------------------------------------- #
#  encodings -> state vectors
# --------------------------------------------------------------------------- #
def l2_normalise(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, eps)


def amplitude_states(X: np.ndarray) -> np.ndarray:
    """Amplitude encoding. X is (m, 2^n); returns (m, 2^n) unit rows.

    Qiskit's raw_feature_vector places x[i] on basis state |i> in the same index
    convention that measurement outcomes use, so no re-ordering is needed.
    """
    m, d = X.shape
    n = int(np.round(np.log2(d)))
    if 2 ** n != d:
        raise ValueError(f"amplitude encoding needs a power-of-two width, got {d}")
    return l2_normalise(X)


def angle_states(X: np.ndarray, convention: str = "ry") -> np.ndarray:
    """Angle encoding. One feature per qubit; returns (m, 2^N) product states.

    convention='ry'    : RY(x_i)|0>  -> [cos(x_i/2), sin(x_i/2)]
    convention='phase' : matches ZFeatureMap's 2*phi(x) scaling, i.e. the
                         per-qubit overlap factor becomes cos^2(x_i - y_i).
                         Amplitudes are complex here, so we return the complex
                         state.
    """
    X = np.asarray(X, dtype=float)
    m, N = X.shape
    if convention == "ry":
        blocks = [np.stack([np.cos(X[:, i] / 2), np.sin(X[:, i] / 2)], 1) for i in range(N)]
        psi = blocks[0]
        for b in blocks[1:]:
            psi = (psi[:, :, None] * b[:, None, :]).reshape(m, -1)
        return psi
    if convention == "phase":
        blocks = [
            np.stack([np.ones(m), np.exp(2j * X[:, i])], 1) / np.sqrt(2.0)
            for i in range(N)
        ]
        psi = blocks[0]
        for b in blocks[1:]:
            psi = (psi[:, :, None] * b[:, None, :]).reshape(m, -1)
        return psi
    raise ValueError(convention)


# --------------------------------------------------------------------------- #
#  forward pass
# --------------------------------------------------------------------------- #
def outcome_probs(states: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Born-rule probabilities over the 2^n computational basis outcomes.

    states : (m, 2^n) encoded states (real or complex)
    U      : (2^n, 2^n) ansatz unitary
    returns: (m, 2^n), rows sum to 1
    """
    amps = states @ U.T
    P = np.abs(amps) ** 2
    return P / P.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
#  fidelity kernels  (closed form -- no simulation, no qubit-count limit)
# --------------------------------------------------------------------------- #
def fidelity_kernel(A: np.ndarray, B: np.ndarray, encoding: str) -> np.ndarray:
    """K_ij = |<psi(a_i)|psi(b_j)>|^2, the quantity Section 7.7 measures.

    This is PSD (it equals Tr[rho_i rho_j] for pure states), so it can be
    handed straight to SVC(kernel='precomputed').
    """
    if encoding == "amplitude":
        Ah, Bh = l2_normalise(A), l2_normalise(B)
        return (Ah @ Bh.T) ** 2

    if encoding in ("angle_ry", "angle_phase"):
        half = 0.5 if encoding == "angle_ry" else 1.0
        K = np.ones((A.shape[0], B.shape[0]))
        # accumulate feature by feature: m*m per step instead of m*m*N at once
        for i in range(A.shape[1]):
            d = A[:, i][:, None] - B[:, i][None, :]
            K *= np.cos(half * d) ** 2
        return K

    raise ValueError(encoding)


def separability_auc(K_same_scores: np.ndarray, is_same: np.ndarray) -> float:
    """Equation (6) of the paper: AUC of fidelity as a score for 'same class'."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(is_same.astype(int), K_same_scores))
