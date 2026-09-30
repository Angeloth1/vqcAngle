"""
Training. COBYLA on cross-entropy, same objective and optimiser family as the
paper -- qiskit_algorithms.optimizers.COBYLA is a thin wrapper over
scipy.optimize.minimize(method='COBYLA'), so results are directly comparable.

Two training modes matter for the readout question:

  mode='fixed'    theta is optimised against a fixed decoding (parity).
                  Other decoders are then applied to the FROZEN theta. This
                  measures how much class information the parity map throws away
                  from the state the published circuit actually produces.

  mode='cotrain'  theta is optimised against the decoder itself, with the linear
                  head refit at every objective evaluation. This measures the
                  ceiling when the circuit is allowed to adapt to its readout.

Both are needed: 'fixed' answers "what does parity discard", 'cotrain' answers
"what could a better-designed readout reach". They are different questions and
the paper asks neither.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .decoders import accuracy, cross_entropy
from .fastsim import ansatz_unitary, n_params, outcome_probs


@dataclass
class FitResult:
    theta: np.ndarray
    loss: float
    n_eval: int
    seconds: float
    history: list = field(default_factory=list)


def init_theta(n_qubits: int, reps: int, seed: int, scale: float = 0.1) -> np.ndarray:
    """U(-0.1, 0.1), matching the initialisation noted in Table A2."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=n_params(n_qubits, reps))


def train_fixed(states, y, n_qubits, reps, decoder, maxiter=150, seed=0,
                record_history=False) -> FitResult:
    theta0 = init_theta(n_qubits, reps, seed)
    hist, n_eval = [], 0
    t0 = time.perf_counter()

    def obj(theta):
        nonlocal n_eval
        n_eval += 1
        P = outcome_probs(states, ansatz_unitary(theta, n_qubits, reps))
        loss = cross_entropy(decoder.predict_proba(P), y)
        if record_history:
            hist.append(loss)
        return loss

    res = minimize(obj, theta0, method="COBYLA", options={"maxiter": maxiter})
    return FitResult(np.asarray(res.x), float(res.fun), n_eval,
                     time.perf_counter() - t0, hist)


def train_cotrain_linear(states, y, n_qubits, reps, maxiter=150, seed=0,
                         C: float = 1.0) -> tuple[FitResult, object]:
    """theta and a linear readout trained jointly (inner refit per evaluation)."""
    from sklearn.linear_model import LogisticRegression

    theta0 = init_theta(n_qubits, reps, seed)
    n_eval = 0
    t0 = time.perf_counter()

    def obj(theta):
        nonlocal n_eval
        n_eval += 1
        P = outcome_probs(states, ansatz_unitary(theta, n_qubits, reps))
        head = LogisticRegression(C=C, max_iter=300).fit(P, y)
        return cross_entropy(head.predict_proba(P), y)

    res = minimize(obj, theta0, method="COBYLA", options={"maxiter": maxiter})
    theta = np.asarray(res.x)
    P = outcome_probs(states, ansatz_unitary(theta, n_qubits, reps))
    head = LogisticRegression(C=C, max_iter=2000).fit(P, y)
    return (FitResult(theta, float(res.fun), n_eval, time.perf_counter() - t0),
            head)


def evaluate(states, y, theta, n_qubits, reps, decoder) -> dict:
    P = outcome_probs(states, ansatz_unitary(theta, n_qubits, reps))
    proba = decoder.predict_proba(P)
    return {"acc": accuracy(proba, y), "loss": cross_entropy(proba, y)}


def probs_for(states, theta, n_qubits, reps) -> np.ndarray:
    return outcome_probs(states, ansatz_unitary(theta, n_qubits, reps))
