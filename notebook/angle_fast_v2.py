"""
Exact noiseless forward pass for the angle-RY + n_local(ry, cx, linear) VQC,
written as batched numpy. Companion to fastsim.py, NOT a reuse of it.

Why a separate module
---------------------
fastsim.py materialises the dense 2^n x 2^n ansatz unitary and does one matmul.
That is optimal at n=4 (U is 16x16). At n=16, U is 34.4 GB. The dense path has a
hard ceiling around n=13.

This module never builds U. Each gate is applied directly to the batch of state
vectors as a reshape plus a contraction over one axis, cost O(m * 2^n) per gate,
memory O(chunk * 2^n). The win over Qiskit is not asymptotic, it is constant
factor: all m records go through one vectorised numpy call instead of m
Python-level circuit simulations, and shot sampling is skipped entirely.

Two architecture-specific reductions, both exact:
  1. The feature map RY(x_i) and the first ansatz rotation layer RY(theta_i)
     act on the same qubit with nothing between them, and RY is a one-parameter
     group: RY(theta)RY(x) = RY(theta + x). The feature map is absorbed for
     free and the encoded state is never built separately.
  2. CX is a permutation. Zero arithmetic, one slice swap.

Qubit convention: Qiskit little-endian. Reshaped to (m,) + (2,)*n, qubit q is
axis 1 + (n - 1 - q). Getting this backwards produces a silently different model
that still trains, so verify_against_qiskit() checks against Statevector.

WHAT CHANGED IN v2
------------------
1. forward_readout() replaces forward_probs() as the working entry point.
   Returning the full 2^n distribution is what breaks at scale: at n=16 with the
   20153-record training split that array is 9.8 GB, allocated once per COBYLA
   evaluation. Nothing downstream ever needed it -- parity_probs collapsed it to
   one number on the next line.

   forward_readout returns the Pauli-Z expectations <Z_S> for |S| <= max_weight
   instead. At n=16, max_weight=2 that is 136 columns (16 singles + 120 pairs),
   plus parity, so 137 in total and 22 MB. The count is not a coincidence worth
   ignoring: 136 is exactly the dimension of Sym(16), the hypothesis class of the
   4-qubit amplitude VQC, so a free readout over these features is parameter-
   matched against that result by construction.

2. All <Z_S> at once via an in-place fast Walsh-Hadamard transform. <Z_S> is
   sum_b p(b) (-1)^popcount(S & b), which is the Walsh-Hadamard transform of p
   evaluated at S. Cost n*2^n per record instead of K*2^n for K observables
   computed separately: at n=16, max_weight=2 that is 8x cheaper, and it yields
   every Pauli-Z string, not only the requested ones.

3. Memory. v1's _ry_layer_ allocated three full-width temporaries per qubit, so
   measured peak RSS was 2.83 GB against a nominal 1 GB chunk budget. v2
   preallocates one scratch buffer, reuses it in the rotation layer, the CX and
   the transform, and sizes the default chunk against measured peak rather than
   against the psi array alone.

4. Both parity conventions are exposed and named. Qiskit's VQC default interpret
   is GLOBAL parity, popcount(b) mod 2. A decoder written as b mod 2 reads qubit
   0 only. These are different models and neither is more standard than the
   other -- v1's parity_probs was the global one. Check which the published
   notebooks used before comparing any number against the paper.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
#  sizes and conventions
# --------------------------------------------------------------------------- #
# Chunks are sized against CACHE, not RAM. The kernel is memory-bandwidth bound:
# every gate is one streaming pass over the chunk, and there are ~5n passes per
# rep. Keeping a chunk inside L2/L3 measured 2x faster at n=16 than a 500 MB
# chunk (48.9 s -> 24.4 s, flat from 64 records down to 4). Chunking is exact, so
# this only trades Python call overhead against cache residency.
DEFAULT_CHUNK_BYTES = 16_000_000
_BYTES_PER_AMP = 8
_PEAK_FACTOR = 2.0          # psi + scratch(half) + one transient half


def n_params(n_qubits: int, reps: int) -> int:
    """n_local(['ry'], reps=r) parameter count: r+1 rotation layers."""
    return n_qubits * (reps + 1)


def default_chunk(n: int, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> int:
    """Records per batch, sized so the working set stays cache-resident."""
    per_record = 2 ** n * _BYTES_PER_AMP * _PEAK_FACTOR
    return max(1, int(chunk_bytes / per_record))


def _axis(n: int, q: int) -> int:
    """Axis index of qubit q in a (m,) + (2,)*n view (little-endian)."""
    return 1 + (n - 1 - q)


# --------------------------------------------------------------------------- #
#  gates
# --------------------------------------------------------------------------- #
def _product_state(angles: np.ndarray) -> np.ndarray:
    """Product state (x) RY(a_i)|0>, shape (m, 2^n), little-endian.

    Built most-significant qubit first so that axis order matches Qiskit's
    index convention.
    """
    m, n = angles.shape
    half = angles * 0.5
    c, s = np.cos(half), np.sin(half)
    psi = np.ones((m, 1), dtype=np.float64)
    for q in range(n - 1, -1, -1):
        blk = np.stack([c[:, q], s[:, q]], axis=1)
        psi = (psi[:, :, None] * blk[:, None, :]).reshape(m, -1)
    return psi


def _ry_layer_(psi: np.ndarray, thetas: np.ndarray, n: int,
               scratch: np.ndarray) -> None:
    """In-place RY(theta_q) on every qubit. psi is (m, 2^n).

    v1 wrote `v0 = c*a - s*b; v1 = s*a + c*b`, which numpy evaluates with three
    full-width temporaries per qubit. Here `a` is the only copy and `scratch` is
    reused across qubits, layers and calls, so peak is psi + one half-width copy
    + one half-width scratch.
    """
    m = psi.shape[0]
    half = thetas * 0.5
    cos_t, sin_t = np.cos(half), np.sin(half)
    flat = scratch[: m * 2 ** (n - 1)]
    for q in range(n):
        ax = _axis(n, q)
        left, right = 2 ** (ax - 1), 2 ** (n - ax)
        v = psi.reshape(m, left, 2, right)
        v0, v1 = v[:, :, 0, :], v[:, :, 1, :]
        a = flat.reshape(m, left, right)
        np.copyto(a, v0)                       # a = old v0
        # v0 <- c*a - s*v1      (v1 still original)
        v0 *= cos_t[q]
        v0 -= sin_t[q] * v1
        # v1 <- s*a + c*v1
        v1 *= cos_t[q]
        v1 += sin_t[q] * a


def _cx_(psi: np.ndarray, ctrl: int, targ: int, n: int,
         scratch: np.ndarray) -> None:
    """In-place CX. Pure permutation: swap the two target slices where ctrl=1."""
    m = psi.shape[0]
    v = psi.reshape((m,) + (2,) * n)
    ac, at = _axis(n, ctrl), _axis(n, targ)
    i0: list = [slice(None)] * (n + 1)
    i0[ac] = 1
    i0[at] = 0
    i1: list = [slice(None)] * (n + 1)
    i1[ac] = 1
    i1[at] = 1
    blk0, blk1 = v[tuple(i0)], v[tuple(i1)]
    tmp = scratch[: blk0.size].reshape(blk0.shape)
    np.copyto(tmp, blk0)
    np.copyto(blk0, blk1)
    np.copyto(blk1, tmp)


def _fwht_(a: np.ndarray, n: int, scratch: np.ndarray) -> None:
    """In-place fast Walsh-Hadamard transform along the 2^n axis.

    Afterwards a[:, S] == sum_b p(b) (-1)^popcount(S & b) == <Z_S>, for every
    S in 0 .. 2^n - 1 simultaneously. a[:, 0] == 1 up to rounding, which is a
    free normalisation check.
    """
    m = a.shape[0]
    flat = scratch[: m * 2 ** (n - 1)]
    for ax in range(n):
        left, right = 2 ** ax, 2 ** (n - ax - 1)
        v = a.reshape(m, left, 2, right)
        v0, v1 = v[:, :, 0, :], v[:, :, 1, :]
        u = flat.reshape(m, left, right)
        np.copyto(u, v0)
        v0 += v1
        np.subtract(u, v1, out=v1)


# --------------------------------------------------------------------------- #
#  readout feature selection
# --------------------------------------------------------------------------- #
def readout_index(n: int, max_weight: int = 2,
                  include_parity: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Which Pauli-Z strings to keep.

    Returns (indices, weights). Index S is a bitmask over qubits; weight is
    popcount(S). Weight 0 is excluded -- <Z_empty> is identically 1 and carries
    no information. Parity (S = all qubits) is appended when requested even
    though its weight exceeds max_weight, because it is the published readout
    and must stay comparable.

    At n=16, max_weight=2: 16 + 120 + 1 = 137 columns.
    """
    pc = np.array([bin(i).count("1") for i in range(2 ** n)], dtype=np.int16)
    keep = np.flatnonzero((pc >= 1) & (pc <= max_weight))
    if include_parity:
        full = (1 << n) - 1
        if pc[full] > max_weight:
            keep = np.append(keep, full)
    return keep, pc[keep]


def readout_labels(idx: np.ndarray, n: int) -> list[str]:
    """Human-readable names, e.g. 'Z3', 'Z3Z7', 'parity'. For tables and plots."""
    out = []
    for S in idx:
        qs = [q for q in range(n) if S >> q & 1]
        out.append("parity" if len(qs) == n else "".join(f"Z{q}" for q in qs))
    return out


# --------------------------------------------------------------------------- #
#  forward passes
# --------------------------------------------------------------------------- #
def _evolve(psi: np.ndarray, th: np.ndarray, n: int, reps: int,
            scratch: np.ndarray) -> None:
    for r in range(1, reps + 1):
        for q in range(n - 1):           # entanglement='linear'
            _cx_(psi, q, q + 1, n, scratch)
        _ry_layer_(psi, th[r], n, scratch)


def _prepare(X: np.ndarray, theta: np.ndarray, reps: int):
    X = np.asarray(X, dtype=np.float64)
    m, n = X.shape
    theta = np.asarray(theta, dtype=np.float64)
    if theta.size != n_params(n, reps):
        raise ValueError(f"expected {n_params(n, reps)} parameters, got {theta.size}")
    return X, m, n, theta.reshape(reps + 1, n)


def forward_readout(X: np.ndarray, theta: np.ndarray, reps: int = 2,
                    max_weight: int = 2, include_parity: bool = True,
                    chunk: int | None = None,
                    keep: np.ndarray | None = None) -> np.ndarray:
    """Pauli-Z expectations <Z_S> for the selected strings. Returns (m, K).

    This is the entry point to use at n > 8. The 2^n distribution is formed one
    chunk at a time, transformed in place and discarded; only the K selected
    columns survive, so the output is K/2^n of the naive size -- 0.2% at n=16.

    Pass `keep` from readout_index() to reuse the index across calls; at n=16 the
    popcount table costs about 0.1 s to build and there is no reason to pay it
    on every COBYLA evaluation.
    """
    X, m, n, th = _prepare(X, theta, reps)
    if keep is None:
        keep, _ = readout_index(n, max_weight, include_parity)
    if chunk is None:
        chunk = default_chunk(n)

    scratch = np.empty(min(chunk, m) * 2 ** (n - 1), dtype=np.float64)
    out = np.empty((m, len(keep)), dtype=np.float64)
    for lo in range(0, m, chunk):
        hi = min(lo + chunk, m)
        psi = _product_state(X[lo:hi] + th[0])    # reduction 1
        _evolve(psi, th, n, reps, scratch)
        psi *= psi                                 # now probabilities, real
        _fwht_(psi, n, scratch)
        out[lo:hi] = psi[:, keep]
    return out


def forward_probs(X: np.ndarray, theta: np.ndarray, reps: int = 2,
                  chunk: int | None = None,
                  max_bytes: int = 2_000_000_000) -> np.ndarray:
    """Full Born-rule distribution over all 2^n outcomes. Returns (m, 2^n).

    Kept for verification and for small n. Guarded, because the array this
    allocates is the thing that made v1 unusable at scale: at n=16 with m=20153
    it is 9.8 GB, and it is allocated once per objective evaluation. Raise
    max_bytes only if you have actually checked you can afford it.
    """
    X, m, n, th = _prepare(X, theta, reps)
    need = m * 2 ** n * _BYTES_PER_AMP
    if need > max_bytes:
        raise MemoryError(
            f"the full distribution is {need / 2**30:.1f} GB for m={m}, n={n}. "
            f"Use forward_readout() instead, or raise max_bytes deliberately.")
    if chunk is None:
        chunk = default_chunk(n)

    scratch = np.empty(min(chunk, m) * 2 ** (n - 1), dtype=np.float64)
    out = np.empty((m, 2 ** n), dtype=np.float64)
    for lo in range(0, m, chunk):
        hi = min(lo + chunk, m)
        psi = _product_state(X[lo:hi] + th[0])
        _evolve(psi, th, n, reps, scratch)
        out[lo:hi] = psi * psi
    return out


# --------------------------------------------------------------------------- #
#  decoding
# --------------------------------------------------------------------------- #
def probs_from_z(z: np.ndarray) -> np.ndarray:
    """A single <Z> expectation in [-1, 1] to two class probabilities.

    z = P(even) - P(odd), so P(class 0) = (1 + z)/2. Works for any Pauli-Z
    column, not only parity -- which is the point of the ladder.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    return np.stack([(1.0 + z) * 0.5, (1.0 - z) * 0.5], axis=1)


def global_parity_probs(P: np.ndarray) -> np.ndarray:
    """Qiskit VQC's default interpret: class = popcount(b) mod 2.

    Equivalent to probs_from_z(<Z^(x)n>), i.e. the last column returned by
    forward_readout with include_parity=True. Takes the full distribution, so
    only usable at small n; prefer the readout column at scale.
    """
    d = P.shape[1]
    sign = 1.0 - 2.0 * (np.array(
        [bin(i).count("1") for i in range(d)], dtype=np.float64) % 2.0)
    return probs_from_z(P @ sign)


def q0_parity_probs(P: np.ndarray) -> np.ndarray:
    """class = b mod 2, i.e. measure qubit 0 alone. NOT the same model as
    global parity -- this is the convention decoders.py uses in exp1."""
    d = P.shape[1]
    sign = 1.0 - 2.0 * (np.arange(d) % 2)
    return probs_from_z(P @ sign.astype(np.float64))


# Back-compatible alias. v1's parity_probs was the global convention.
parity_probs = global_parity_probs


# --------------------------------------------------------------------------- #
#  verification
# --------------------------------------------------------------------------- #
def verify_against_qiskit(cases=((3, 2), (4, 1), (5, 3), (6, 2)), m: int = 4,
                          seed: int = 0, tol: float = 1e-9, verbose: bool = True):
    """Check conventions against Qiskit Statevector. Run before trusting output.

    Endianness, layer ordering and the RY(theta)RY(x) = RY(theta + x) absorption
    all fail silently: a wrong convention still trains and still converges, it
    just fits a different model. This is the only guard against that.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import n_local
    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(seed)
    ok = True
    for n, reps in cases:
        X = rng.normal(size=(m, n))
        th = rng.normal(size=n_params(n, reps))
        ans = n_local(n, ["ry"], "cx", entanglement="linear", reps=reps)
        ref = np.empty((m, 2 ** n))
        for k in range(m):
            qc = QuantumCircuit(n)
            for q in range(n):
                qc.ry(X[k, q], q)
            qc.compose(ans.assign_parameters(th), inplace=True)
            ref[k] = np.abs(np.asarray(Statevector(qc))) ** 2

        d_prob = float(np.abs(ref - forward_probs(X, th, reps)).max())

        keep, _ = readout_index(n, max_weight=2, include_parity=True)
        R = forward_readout(X, th, reps, keep=keep)
        d_par = float(np.abs(R[:, -1] - (global_parity_probs(ref)[:, 0] * 2 - 1)).max())

        sub = keep[:min(8, len(keep))]
        b = np.arange(2 ** n)
        d_z = 0.0
        for j, S in enumerate(sub):
            sgn = 1.0 - 2.0 * (np.array([bin(int(i) & int(S)).count("1")
                                         for i in b]) % 2)
            d_z = max(d_z, float(np.abs(ref @ sgn - R[:, j]).max()))

        good = max(d_prob, d_par, d_z) < tol
        ok &= good
        if verbose:
            print(f"  n={n} reps={reps}:  probs {d_prob:.2e}   <Z_S> {d_z:.2e}   "
                  f"parity {d_par:.2e}   {'PASS' if good else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("verifying angle simulator against Qiskit\n")
    good = verify_against_qiskit()
    print("\n" + ("ALL PASS" if good else "FAILED"))
    n = 16
    keep, w = readout_index(n, 2)
    print(f"\nn={n}: {len(keep)} readout columns "
          f"({int((w == 1).sum())} single + {int((w == 2).sum())} pair + parity)")
    print(f"default chunk: {default_chunk(n)} records")
    raise SystemExit(0 if good else 1)
