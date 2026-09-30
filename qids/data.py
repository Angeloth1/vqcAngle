"""
NSL-KDD pipeline, rebuilt from the specification in Sections 3.1 and 4.

IMPORTANT. This is an independent build, exactly like the one behind Table A2.
If you want numbers that are directly comparable to Table 3 rather than merely
replicating it, replace `load_arrays` with a thin wrapper around your own
notebooks/ pipeline and keep everything else. The paper already documents that
the two builds agree on internal accuracy and novel recall but not on aggregate
KDDTest+ accuracy, so which build you use decides which table you can compare to.

Set QIDS_DATA to the directory holding KDDTrain+_20Percent.txt and KDDTest+.txt.
With no data present, `load_arrays(synthetic=True)` returns a structurally
similar fake dataset so the scripts can be smoke-tested.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, TargetEncoder

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag", "src_bytes_band"]

FAMILY = {
    **{a: "DoS" for a in ["back", "land", "neptune", "pod", "smurf", "teardrop",
                          "apache2", "udpstorm", "processtable", "mailbomb"]},
    **{a: "Probe" for a in ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]},
    **{a: "R2L" for a in ["guess_passwd", "ftp_write", "imap", "phf", "multihop",
                          "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
                          "snmpguess", "snmpgetattack", "httptunnel", "sendmail",
                          "named"]},
    **{a: "U2R" for a in ["buffer_overflow", "loadmodule", "rootkit", "perl",
                          "sqlattack", "xterm", "ps"]},
}
FAMILY_IDX = {"DoS": 0, "Probe": 1, "R2L": 2, "U2R": 3}


@dataclass
class Arrays:
    """PCA-space arrays. `Xtr`/`Xte` are the internal 80/20 split of the 20%
    subset; `Xood` is the full KDDTest+ file."""
    Xtr: np.ndarray
    ytr: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    Xood: np.ndarray
    yood: np.ndarray
    novel_mask: np.ndarray          # KDDTest+ records with an unseen attack type
    fam_tr: np.ndarray              # attack-family index, -1 for normal traffic
    fam_te: np.ndarray
    explained_variance: float
    pca: PCA


def _read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, names=COLUMNS)
    df["attack"] = df["label"].str.strip()
    df["y"] = (df["attack"] != "normal").astype(int)
    df["family"] = df["attack"].map(FAMILY).fillna("unknown")
    return df


def load_arrays(data_dir: str | None = None, n_components: int = 16,
                test_size: float = 0.2, seed: int = 42,
                synthetic: bool = False) -> Arrays:
    if synthetic:
        return _synthetic(n_components, seed)

    if data_dir is None:
        here = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        data_dir = os.environ.get("QIDS_DATA") or (here if os.path.isdir(here) else ".")
    tr_raw = _read(os.path.join(data_dir, "KDDTrain+_20Percent.txt"))
    te_raw = _read(os.path.join(data_dir, "KDDTest+.txt"))

    # src_bytes -> equal-frequency bands, fitted on the training file only
    # src_bytes is ~39% zeros in KDDTrain+_20Percent, so three equal-frequency
    # bins do not exist: qcut drops a duplicate edge and returns two. Passing a
    # fixed three-label list therefore fails. labels=False returns integer codes
    # whose count always matches the edges that actually survived.
    codes, edges = pd.qcut(tr_raw["src_bytes"], 3, labels=False,
                           duplicates="drop", retbins=True)
    names = ["low", "med", "high"][: len(edges) - 1]
    print(f"[data] src_bytes bands: {names}  "
          f"(zeros: {(tr_raw['src_bytes'] == 0).mean():.1%})")
    tr_raw["src_bytes_band"] = pd.Series(codes, index=tr_raw.index).map(
        dict(enumerate(names))).astype(str)
    edges[0], edges[-1] = -np.inf, np.inf
    te_raw["src_bytes_band"] = pd.cut(te_raw["src_bytes"], edges,
                                      labels=names).astype(str)

    strat = tr_raw["y"].astype(str) + "_" + tr_raw["src_bytes_band"]
    counts = strat.value_counts()
    rare = strat.isin(counts[counts < 2].index)          # the 'outlier' injection
    idx = np.arange(len(tr_raw))
    tr_idx, te_idx = train_test_split(idx[~rare], test_size=test_size,
                                      random_state=seed, stratify=strat[~rare])
    tr_idx = np.concatenate([tr_idx, idx[rare]])

    feats = [c for c in COLUMNS if c not in ("label", "difficulty")] + ["src_bytes_band"]
    num = [c for c in feats if c not in CATEGORICAL]

    def frame(df, rows=None):
        d = df if rows is None else df.iloc[rows]
        return d[num].astype(float).to_numpy(), d[CATEGORICAL].astype(str).to_numpy()

    Ntr, Ctr = frame(tr_raw, tr_idx)
    Nte, Cte = frame(tr_raw, te_idx)
    Nood, Cood = frame(te_raw)
    ytr = tr_raw["y"].to_numpy()[tr_idx]
    yte = tr_raw["y"].to_numpy()[te_idx]
    yood = te_raw["y"].to_numpy()

    enc = TargetEncoder(smooth="auto", random_state=seed).fit(Ctr, ytr)
    scal = StandardScaler().fit(np.hstack([Ntr, enc.transform(Ctr)]))
    pca = PCA(n_components=n_components, random_state=seed)

    def project(N, C):
        return pca.transform(scal.transform(np.hstack([N, enc.transform(C)])))

    pca.fit(scal.transform(np.hstack([Ntr, enc.transform(Ctr)])))

    seen = set(tr_raw["attack"].to_numpy()[tr_idx])
    novel = (~te_raw["attack"].isin(seen)).to_numpy() & (yood == 1)

    fam = tr_raw["family"].map(FAMILY_IDX).fillna(-1).to_numpy().astype(int)

    return Arrays(
        Xtr=project(Ntr, Ctr), ytr=ytr,
        Xte=project(Nte, Cte), yte=yte,
        Xood=project(Nood, Cood), yood=yood,
        novel_mask=novel, fam_tr=fam[tr_idx], fam_te=fam[te_idx],
        explained_variance=float(pca.explained_variance_ratio_.sum()), pca=pca,
    )


def _synthetic(n_components: int, seed: int) -> Arrays:
    """Structurally similar stand-in: correlated blobs plus a magnitude signal,
    so the norm-restoration experiment has something to find."""
    rng = np.random.default_rng(seed)

    def make(m, shift):
        y = rng.integers(0, 2, m)
        X = rng.normal(size=(m, n_components))
        X += y[:, None] * rng.normal(shift, 0.4, size=(1, n_components))
        X *= np.exp(0.6 * y + rng.normal(0, 0.3, m))[:, None]   # class-linked scale
        return X, y

    Xtr, ytr = make(4000, 0.5)
    Xte, yte = make(1200, 0.5)
    Xo, yo = make(2000, 0.35)
    novel = (yo == 1) & (rng.random(len(yo)) < 0.4)
    fam_tr = np.where(ytr == 1, rng.integers(0, 4, len(ytr)), -1)
    fam_te = np.where(yte == 1, rng.integers(0, 4, len(yte)), -1)
    return Arrays(Xtr, ytr, Xte, yte, Xo, yo, novel, fam_tr, fam_te, 0.0,
                  PCA(n_components=n_components))


def subsample(X, y, n=2000, seed=0):
    """Stratified draw matching the Section 7.4 protocol."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        pool = np.flatnonzero(y == c)
        k = int(round(n * len(pool) / len(y)))
        idx.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def angle_scale(Xtr, *others, mode="raw", scale=0.25, seed=0):
    """Map features to rotation angles for angle encoding.

    mode='raw' keeps the PCA variance hierarchy: later components carry little
    variance, contribute a per-qubit overlap factor of roughly cos^2(0) = 1, and
    therefore do not collapse the product as N grows. That is what reproduces
    the mild N=4 -> N=16 degradation of Table 5.

    mode='quantile' forces every component to be uniform on [0, scale], which
    destroys that hierarchy. The product of N cosines then decays towards zero
    for almost every pair and the separability AUC falls apart with N. Kept only
    so the failure is reproducible.

    scale=0.25 with mode='raw' matched Table 5 on the build in this repo. Verify
    against PoC/ before trusting the angle rows.
    """
    if mode == "raw":
        out = [Xtr * scale] + [o * scale for o in others]
    elif mode == "quantile":
        qt = QuantileTransformer(output_distribution="uniform", random_state=seed,
                                 n_quantiles=min(1000, len(Xtr))).fit(Xtr)
        out = [qt.transform(Xtr) * scale] + [qt.transform(o) * scale for o in others]
    else:
        raise ValueError(f"unknown angle scaling mode: {mode}")
    return out if len(out) > 1 else out[0]