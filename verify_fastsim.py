"""
Correctness gate. Everything in this repo rests on the claim that the noiseless
forward pass equals a single matrix multiply, so that claim is checked against
qiskit_machine_learning. SamplerQNN before any experiment is trusted.

Run this first. If it does not pass on your Qiskit version, nothing else is valid.
"""
import numpy as np
from qiskit.quantum_info import Operator
from qiskit_machine_learning.circuit.library import raw_feature_vector
from qiskit_machine_learning.neural_networks import SamplerQNN

from qids.fastsim import amplitude_states, ansatz_unitary, outcome_probs, _ansatz

TOL = 1e-9


def check(n_qubits=4, reps=5, m=8, seed=0):
    rng = np.random.default_rng(seed)
    fm = raw_feature_vector(2 ** n_qubits)
    ans = _ansatz(n_qubits, reps)
    qc = fm.compose(ans)

    theta = rng.normal(size=ans.num_parameters)
    X = rng.normal(size=(m, 2 ** n_qubits))

    qnn = SamplerQNN(circuit=qc, input_params=fm.parameters,
                     weight_params=ans.parameters, interpret=None,
                     output_shape=2 ** n_qubits, sparse=False)
    ref = qnn.forward(amplitude_states(X), theta)
    fast = outcome_probs(amplitude_states(X), ansatz_unitary(theta, n_qubits, reps))

    U = ansatz_unitary(theta, n_qubits, reps)
    d_prob = float(np.abs(ref - fast).max())
    d_orth = float(np.abs(U @ U.T - np.eye(2 ** n_qubits)).max())
    ok = d_prob < TOL and d_orth < TOL
    print(f"  n={n_qubits} reps={reps}:  max|p_qiskit - p_fast| = {d_prob:.2e}"
          f"   ||UU^T - I|| = {d_orth:.2e}   {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("verifying the matmul forward pass against Qiskit\n")
    allok = all(check(n, r) for n, r in [(2, 3), (3, 1), (4, 5), (4, 17), (5, 4)])
    print("\n" + ("ALL PASS -- fast path is exact" if allok else "FAILED"))
    raise SystemExit(0 if allok else 1)
