"""
Readout maps: 2^n measured basis probabilities -> class scores.

Section 9.3 of the paper states plainly that the many-to-one decoding was held
constant throughout and that no experiment bounds its contribution. This module
provides the ladder that bounds it:

  1. ParityDecoder        b mod k                 -- what the paper uses
  2. BestFixedDecoder     argmax over ALL 2^16 subsets of outcomes
                          -- the best member of the paper's own decoding class,
                             found exhaustively, not heuristically
  3. LinearDecoder        free weights on p(b)    -- best linear readout
  4. MLPDecoder           small net on p(b)       -- near-Bayes readout

The gap 1 -> 2 is the cost of *choosing* parity.
The gap 2 -> 4 is the cost of *restricting to* many-to-one assignment at all.
Whatever remains against a classical model on x_hat is attributable to the
encoding plus the Born rule, not to the decoding.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

EPS = 1e-12


def _agg(P: np.ndarray, assign: np.ndarray, n_classes: int) -> np.ndarray:
    """Sum outcome probabilities into classes according to `assign`."""
    out = np.zeros((P.shape[0], n_classes))
    for c in range(n_classes):
        out[:, c] = P[:, assign == c].sum(axis=1)
    return out


class ParityDecoder:
    """The published decoding: class(b) = b mod n_classes."""

    name = "parity"

    def __init__(self, n_outcomes: int, n_classes: int = 2):
        self.n_classes = n_classes
        self.assign = np.arange(n_outcomes) % n_classes

    def fit(self, P, y):  # nothing to learn
        return self

    def predict_proba(self, P):
        return _agg(P, self.assign, self.n_classes)


class GlobalParityDecoder(ParityDecoder):
    """class(b) = popcount(b) mod 2. Uses all qubits, not just q0."""

    name = "global_parity"

    def __init__(self, n_outcomes: int, n_classes: int = 2):
        if n_classes != 2:
            raise ValueError("global parity is binary only")
        self.n_classes = 2
        self.assign = np.array([bin(b).count("1") % 2 for b in range(n_outcomes)])


class BestFixedDecoder:
    """Exhaustive search over every many-to-one assignment of outcomes to 2 classes.

    For 4 qubits there are only 2^16 = 65536 candidate subsets, so the optimum is
    found exactly rather than approximated. Selected on the TRAINING split only.
    """

    name = "best_fixed"

    def __init__(self, n_outcomes: int, n_classes: int = 2, chunk: int = 4096):
        if n_classes != 2:
            raise NotImplementedError("binary only; extend by k-way partition search")
        self.n_outcomes = n_outcomes
        self.n_classes = 2
        self.chunk = chunk
        self.assign = np.arange(n_outcomes) % 2

    def fit(self, P: np.ndarray, y: np.ndarray):
        n = self.n_outcomes
        bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
        best_acc, best_w = -1.0, bits[1]
        for s in range(0, bits.shape[0], self.chunk):
            W = bits[s: s + self.chunk]              # (c, n)
            scores = P @ W.T                          # (m, c) -- P(class 1)
            acc = ((scores > 0.5).astype(int) == y[:, None]).mean(axis=0)
            j = int(np.argmax(acc))
            if acc[j] > best_acc:
                best_acc, best_w = float(acc[j]), W[j]
        self.assign = best_w.astype(int)
        self.train_acc_ = best_acc
        return self

    def predict_proba(self, P):
        p1 = P[:, self.assign == 1].sum(axis=1)
        return np.stack([1.0 - p1, p1], axis=1)


class LinearDecoder:
    """Free linear weights on the outcome distribution, then a logistic link."""

    name = "linear_head"

    def __init__(self, n_outcomes: int, n_classes: int = 2, C: float = 1.0):
        self.clf = LogisticRegression(C=C, max_iter=2000)

    def fit(self, P, y):
        self.clf.fit(P, y)
        return self

    def predict_proba(self, P):
        return self.clf.predict_proba(P)


class MLPDecoder:
    """Non-linear readout of the outcome distribution. Upper bound of the ladder."""

    name = "mlp_head"

    def __init__(self, n_outcomes: int, n_classes: int = 2, hidden=(32,), seed: int = 0):
        self.clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=2000,
                                 random_state=seed)

    def fit(self, P, y):
        self.clf.fit(P, y)
        return self

    def predict_proba(self, P):
        return self.clf.predict_proba(P)


def cross_entropy(proba: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(proba[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.mean(np.log(p)))


def accuracy(proba: np.ndarray, y: np.ndarray) -> float:
    return float((np.argmax(proba, axis=1) == y).mean())
