"""
EXPERIMENT 5 -- The matched comparison: one protocol, two encodings, N = 4, 8, 16.

Open question addressed
-----------------------
Every trained comparison so far changes more than the encoding. exp1-4 train the
published 4-qubit amplitude circuit (RealAmplitudes reverse_linear, qubit-0
parity, 150 evaluations); the notebook trains angle-encoded circuits (n_local
RY+CX linear, global parity, 400 evaluations). Ansatz, readout, training-set size
and budget all differ between them, and exp2 compares the encodings without any
training. So far the paper can say what the encoded geometry supports and what the
readout costs inside each arm, but not which encoding trains to the better
classifier. This experiment makes that comparison.

Design
------
One protocol for both arms at N in {4, 8, 16} principal components:

  data      the 2,000-record stratified subsample of Section 7.4 of [1], drawn per
            seed; evaluation on the full internal test split, KDDTest+, and the
            3,752 KDDTest+ attack records of a type absent from KDDTrain+_20Percent
            ("novel", the definition of [1] and of the exp1/3/4 result files)
  ansatz    n_local(['ry'], 'cx', entanglement='linear') in BOTH arms (the
            notebook's family), reps chosen so both carry 3N parameters:
              angle-RY   N qubits         reps = 2              (12, 24, 48 params)
              amplitude  log2(N) qubits   reps = 3N/log2(N) - 1 (5, 7, 11)
  budget    COBYLA, 400 evaluations, theta0 ~ U(-0.1, 0.1) from the seed
  readouts  each trained separately
              q0      class = b mod 2            (qubit 0: published / Qiskit default)
              global  class = popcount(b) mod 2  (the notebook's)
              head    logistic regression on <Z_S>, |S| <= 2, plus global parity,
                      refit at every objective evaluation (co-trained)
            plus a frozen head refit once on the weights of each parity fit
  references, on the same training records
              fidelity-kernel SVM, amplitude and angle-RY, C in {1, 100}
              logistic regression on x_hat and on x

At a given N the two circuit arms differ only in the encoding and in what it
implies: the qubit count, and with it the number of |S| <= 2 strings the head
reads (3, 7, 11 for amplitude; 11, 37, 137 for angle). A seed pairs the arms:
same subsample, same theta0 draw.

Caveat, stated before the numbers: with a *linear* CX chain and reps = 2 the
qubit-0 observable of the angle circuit depends on x_0, x_1, x_2 only (its
backward light cone; checked numerically at start-up). The q0 rows of the angle
arm measure that restriction as much as the encoding. The global-parity and head
rows read every qubit in both arms.

N = 16, angle arm: one 16-qubit fit takes ~2.3 h, so the notebook's seed-0 fits
(global parity and co-trained head, same protocol) are loaded from notebook/ckpt
and evaluated (--part angle16). The q0 fit and seeds 1-4 at N = 16 are not run.

Runtime: ~15 min for --part main, ~11 min for --part angle16.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "notebook"))
import angle_fast_v2 as AF                                   # noqa: E402
from qids.data import load_arrays, subsample                  # noqa: E402
from qids.fastsim import fidelity_kernel, l2_normalise        # noqa: E402
from novel_ref1 import novel_mask_ref1                        # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

EPS = 1e-12
SCALE = 0.25          # angle scale of the notebook and exp2
REPS_ANGLE = 2        # notebook depth: 3N parameters
M_TRAIN = 2000
CKPT = ROOT / "notebook" / "ckpt"


def xent(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0))))


# --------------------------------------------------------------------------- #
#  amplitude arm: dense n_local(ry, cx, linear) unitary, checked against Qiskit
# --------------------------------------------------------------------------- #
def _ry(t):
    c, s = np.cos(t / 2.0), np.sin(t / 2.0)
    return np.array([[c, -s], [s, c]])


def chain_perm(n):
    """Image of each basis index under CX(0,1), CX(1,2), ..., CX(n-2,n-1), in that order."""
    b = np.arange(2 ** n)
    for q in range(n - 1):
        b = b ^ (((b >> q) & 1) << (q + 1))
    return b


def nlocal_linear_unitary(theta, n, reps, perm):
    th = np.asarray(theta, dtype=float).reshape(reps + 1, n)

    def layer(r):                      # kron(RY_{n-1}, ..., RY_0): qubit 0 is the least significant bit
        L = np.ones((1, 1))
        for q in range(n - 1, -1, -1):
            L = np.kron(L, _ry(th[r, q]))
        return L

    U = layer(0)
    for r in range(1, reps + 1):
        V = np.empty_like(U)
        V[perm] = U                    # CX chain: row b moves to row perm[b]
        U = layer(r) @ V
    return U


def verify_unitary(cases=((2, 5), (3, 7), (4, 11), (4, 2), (3, 1)), seed=0, tol=1e-12):
    from qiskit.circuit.library import n_local
    from qiskit.quantum_info import Operator
    rng = np.random.default_rng(seed)
    worst = 0.0
    for n, reps in cases:
        th = rng.normal(size=n * (reps + 1))
        ref = Operator(n_local(n, ["ry"], "cx", entanglement="linear", reps=reps).assign_parameters(th)).data
        mine = nlocal_linear_unitary(th, n, reps, chain_perm(n))
        worst = max(worst, float(np.abs(ref.imag).max()), float(np.abs(ref.real - mine).max()))
    if worst > tol:
        raise SystemExit(f"amplitude-arm unitary does not match Qiskit (max deviation {worst:.2e})")
    return worst


def check_lightcone(n=8, seed=0):
    """<Z_0> of the angle circuit must not depend on x_3..x_{n-1} (linear chain, reps = 2)."""
    rng = np.random.default_rng(seed)
    th = rng.normal(size=AF.n_params(n, REPS_ANGLE))
    x = rng.normal(size=(1, n))
    base = AF.forward_readout(x, th, REPS_ANGLE, max_weight=1, include_parity=False)[0, 0]
    dev_out, dev_in = 0.0, 0.0
    for i in range(n):
        x2 = x.copy()
        x2[0, i] += 0.7
        d = abs(AF.forward_readout(x2, th, REPS_ANGLE, max_weight=1, include_parity=False)[0, 0] - base)
        if i >= REPS_ANGLE + 1:
            dev_out = max(dev_out, d)
        else:
            dev_in = max(dev_in, d)
    return dev_in, dev_out


def z_table(n, keep):
    """(-1)^popcount(b & S) for every outcome b and every kept string S."""
    b = np.arange(2 ** n)
    out = np.empty((2 ** n, len(keep)))
    for j, S in enumerate(keep):
        out[:, j] = 1.0 - 2.0 * (np.array([bin(int(v) & int(S)).count("1") for v in b]) % 2)
    return out


class AmplitudeArm:
    name = "amplitude"

    def __init__(self, N):
        self.q = int(round(np.log2(N)))
        assert 2 ** self.q == N, "amplitude arm needs a power-of-two N"
        self.reps = 3 * N // self.q - 1
        self.n_params = self.q * (self.reps + 1)
        assert self.n_params == 3 * N
        self.keep, _ = AF.readout_index(self.q, 2)
        self.H = z_table(self.q, self.keep)
        self.perm = chain_perm(self.q)
        self.qubits = self.q

    def inputs(self, X):
        return l2_normalise(X)

    def readout(self, Z, theta):
        U = nlocal_linear_unitary(theta, self.q, self.reps, self.perm)
        P = (Z @ U.T) ** 2
        P /= P.sum(axis=1, keepdims=True)
        return P @ self.H


class AngleArm:
    name = "angle"

    def __init__(self, N):
        self.q = N
        self.reps = REPS_ANGLE
        self.n_params = AF.n_params(N, REPS_ANGLE)
        assert self.n_params == 3 * N
        self.keep, _ = AF.readout_index(N, 2)
        self.qubits = N

    def inputs(self, X):
        return X * SCALE

    def readout(self, Z, theta):
        return AF.forward_readout(Z, theta, self.reps, keep=self.keep)


def check_columns(arm):
    assert arm.keep[0] == 1, "first readout column must be Z on qubit 0"
    assert arm.keep[-1] == 2 ** arm.q - 1, "last readout column must be the global parity"


# --------------------------------------------------------------------------- #
#  training, prediction, metrics
# --------------------------------------------------------------------------- #
def train(arm, Ztr, ytr, readout, seed, maxiter):
    """Same objective and optimiser as the notebook's train_one()."""
    th0 = np.random.default_rng(seed).uniform(-0.1, 0.1, arm.n_params)
    calls = {"n": 0}

    def obj(th):
        calls["n"] += 1
        R = arm.readout(Ztr, th)
        if readout == "q0":
            return xent(AF.probs_from_z(R[:, 0]), ytr)
        if readout == "global":
            return xent(AF.probs_from_z(R[:, -1]), ytr)
        head = LogisticRegression(C=1.0, max_iter=300).fit(R, ytr)
        return xent(head.predict_proba(R), ytr)

    t0 = time.perf_counter()
    res = minimize(obj, th0, method="COBYLA", options={"maxiter": maxiter})
    th = np.asarray(res.x)
    out = dict(theta=th, loss=float(res.fun), evals=calls["n"], seconds=time.perf_counter() - t0, head=None)
    if readout == "head":
        out["head"] = LogisticRegression(C=1.0, max_iter=2000).fit(arm.readout(Ztr, th), ytr)
    return out


def predict(R, readout, head=None):
    if readout == "q0":
        return (R[:, 0] < 0).astype(int)          # class 1 iff P(odd) > P(even)
    if readout == "global":
        return (R[:, -1] < 0).astype(int)
    return np.asarray(head.predict(R)).astype(int)


def metrics(pred, y, nov):
    """pred / y: dicts with keys tr, te, oo."""
    return dict(train_acc=float((pred["tr"] == y["tr"]).mean()),
                internal=float((pred["te"] == y["te"]).mean()),
                kddtest=float((pred["oo"] == y["oo"]).mean()),
                novel_recall=float((pred["oo"][nov] == y["oo"][nov]).mean()))


def head_record(head):
    return dict(coef=np.asarray(head.coef_).tolist(), intercept=np.asarray(head.intercept_).tolist())


def references(A, Xtr, ytr, base, chunk=4000):
    """Fidelity-kernel SVMs and logistic regression on exactly the circuits' training records."""
    rows = []
    y = dict(tr=ytr, te=A.yte, oo=A.yood)
    for arm, enc, f, qb in (("kernel_amplitude", "amplitude", lambda X: X, int(np.log2(Xtr.shape[1]))),
                            ("kernel_angle_ry", "angle_ry", lambda X: X * SCALE, Xtr.shape[1])):
        Ztr = f(Xtr)
        Ktr = fidelity_kernel(Ztr, Ztr, enc)
        svms = {C: SVC(kernel="precomputed", C=C).fit(Ktr, ytr) for C in (1.0, 100.0)}
        preds = {C: dict(tr=svms[C].predict(Ktr)) for C in svms}
        for key, X in (("te", A.Xte), ("oo", A.Xood)):
            Z = f(X)
            parts = {C: [] for C in svms}
            for lo in range(0, len(Z), chunk):
                K = fidelity_kernel(Z[lo:lo + chunk], Ztr, enc)
                for C, s in svms.items():
                    parts[C].append(s.predict(K))
            for C in svms:
                preds[C][key] = np.concatenate(parts[C])
        for C, s in svms.items():
            rows.append({**base, "arm": arm, "readout": f"C={C:g}", "source": "reference", "qubits": qb,
                         "n_sv": int(s.n_support_.sum()), **metrics(preds[C], y, NOVEL)})
    for arm, f in (("logreg_xhat", l2_normalise), ("logreg_x", lambda X: X)):
        clf = LogisticRegression(max_iter=2000).fit(f(Xtr), ytr)
        pred = dict(tr=clf.predict(f(Xtr)), te=clf.predict(f(A.Xte)), oo=clf.predict(f(A.Xood)))
        rows.append({**base, "arm": arm, "readout": "-", "source": "reference", "qubits": 0,
                     "params": Xtr.shape[1] + 1, **metrics(pred, y, NOVEL)})
    return rows


# --------------------------------------------------------------------------- #
#  parts
# --------------------------------------------------------------------------- #
def run_main(Ns, seeds, maxiter, out, thetas_out):
    rows, thetas = [], {}
    for N in Ns:
        A = load_arrays(n_components=N)
        arms = [AmplitudeArm(N)] + ([AngleArm(N)] if N < 16 else [])
        for arm in arms:
            check_columns(arm)
        y_eval = dict(te=A.yte, oo=A.yood)
        for seed in seeds:
            t_seed = time.perf_counter()
            Xtr, ytr = subsample(A.Xtr, A.ytr, M_TRAIN, seed)
            base = dict(N=N, seed=seed)
            rows += references(A, Xtr, ytr, base)
            y = dict(tr=ytr, **y_eval)
            for arm in arms:
                Z = dict(tr=arm.inputs(Xtr), te=arm.inputs(A.Xte), oo=arm.inputs(A.Xood))
                geo = dict(arm=arm.name, source="trained", qubits=arm.qubits, params=arm.n_params,
                           reps=arm.reps, n_features=len(arm.keep))
                for readout in ("q0", "global", "head"):
                    f = train(arm, Z["tr"], ytr, readout, seed, maxiter)
                    R = {k: arm.readout(Z[k], f["theta"]) for k in Z}
                    pred = {k: predict(R[k], readout, f["head"]) for k in R}
                    rows.append({**base, **geo, "readout": readout, **metrics(pred, y, NOVEL),
                                 "loss": f["loss"], "evals": f["evals"], "seconds": f["seconds"]})
                    rec = dict(theta=f["theta"].tolist(), loss=f["loss"], evals=f["evals"], seconds=f["seconds"])
                    if f["head"] is not None:
                        rec["head"] = head_record(f["head"])
                    thetas[f"N{N}_s{seed}_{arm.name}_{readout}"] = rec
                    if readout in ("q0", "global"):             # frozen weights, head refit once
                        head = LogisticRegression(C=1.0, max_iter=2000).fit(R["tr"], ytr)
                        pred = {k: np.asarray(head.predict(R[k])).astype(int) for k in R}
                        rows.append({**base, **geo, "readout": f"frozen_head_after_{readout}",
                                     **metrics(pred, y, NOVEL)})
            pd.DataFrame(rows).to_csv(out, index=False)
            Path(thetas_out).write_text(json.dumps(thetas))
            print(f"  N={N:2d} seed {seed} done ({time.perf_counter() - t_seed:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def load_ckpt(name):
    obj = np.load(CKPT / f"{name}.npz", allow_pickle=True)["obj"]
    return obj.item() if obj.ndim == 0 else list(obj)


def run_angle16(out):
    """Evaluate the notebook's 16-qubit seed-0 fits under this experiment's protocol and metrics."""
    N, seed = 16, 0
    A = load_arrays(n_components=N)
    arm = AngleArm(N)
    check_columns(arm)
    Xtr, ytr = subsample(A.Xtr, A.ytr, M_TRAIN, seed)
    y = dict(tr=ytr, te=A.yte, oo=A.yood)
    Z = dict(tr=arm.inputs(Xtr), te=arm.inputs(A.Xte), oo=arm.inputs(A.Xood))
    geo = dict(N=N, seed=seed, arm=arm.name, source="notebook_ckpt", qubits=arm.qubits,
               params=arm.n_params, reps=arm.reps, n_features=len(arm.keep))
    rows = []
    fp, fh = load_ckpt("fit_n16_s0_parity"), load_ckpt("fit_n16_s0_head")

    t0 = time.perf_counter()
    R = {k: arm.readout(Z[k], np.array(fp["theta"])) for k in Z}
    pred = {k: predict(R[k], "global") for k in R}
    rows.append({**geo, "readout": "global", **metrics(pred, y, NOVEL),
                 "loss": fp["loss"], "evals": fp["evals"], "seconds": fp["seconds"]})
    head = LogisticRegression(C=1.0, max_iter=2000).fit(R["tr"], ytr)
    pred = {k: np.asarray(head.predict(R[k])).astype(int) for k in R}
    rows.append({**geo, "readout": "frozen_head_after_global", **metrics(pred, y, NOVEL)})
    print(f"  parity weights evaluated ({time.perf_counter() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    t0 = time.perf_counter()
    R = {k: arm.readout(Z[k], np.array(fh["theta"])) for k in Z}
    coef, b0 = np.array(fh["head"]["coef"]), np.array(fh["head"]["intercept"])
    pred = {k: ((R[k] @ coef.T + b0).ravel() > 0).astype(int) for k in R}
    rows.append({**geo, "readout": "head", **metrics(pred, y, NOVEL),
                 "loss": fh["loss"], "evals": fh["evals"], "seconds": fh["seconds"]})
    refit = LogisticRegression(C=1.0, max_iter=2000).fit(R["tr"], ytr)
    print(f"  head weights evaluated ({time.perf_counter() - t0:.0f}s); stored head vs refit: "
          f"max |coef diff| = {float(np.abs(refit.coef_ - coef).max()):.2e}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["main", "angle16"], default="main")
    ap.add_argument("--Ns", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--thetas", default="results/exp5_thetas.json")
    a = ap.parse_args()

    NOVEL = novel_mask_ref1()
    dev = verify_unitary()
    d_in, d_out = check_lightcone()
    print(f"amplitude-arm unitary vs Qiskit: max deviation {dev:.2e}")
    print(f"angle circuit, <Z_0> light cone: change from x_0..x_2 {d_in:.3f}, from x_3..x_7 {d_out:.1e}")

    if a.part == "main":
        out = a.out or "results/exp5_matched.csv"
        df = run_main(a.Ns, a.seeds, a.maxiter, out, a.thetas)
    else:
        out = a.out or "results/exp5_angle16.csv"
        df = run_angle16(out)

    print("\n" + "=" * 72)
    print("EXP 5  matched encoding comparison  (internal accuracy, mean over seeds)")
    print("=" * 72)
    print(df.pivot_table(index=["N", "readout"], columns="arm", values="internal", aggfunc="mean").round(4).to_string())
    print(f"\nwritten to {out}")
