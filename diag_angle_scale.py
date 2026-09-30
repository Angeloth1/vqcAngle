"""Which angle scaling reproduces Table 5? Sweep, do not guess."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from qids.data import load_arrays, subsample
from qids.fastsim import fidelity_kernel

A = load_arrays()
X, y = subsample(A.Xtr, A.ytr, 1000, 0)
rng = np.random.default_rng(0)
i, j = rng.integers(0, len(X), 200_000), rng.integers(0, len(X), 200_000)
k = i != j; i, j = i[k], j[k]
same = (y[i] == y[j]).astype(int)

rows = []
for mode in ["raw", "quantile"]:
    Z = (X if mode == "raw" else
         QuantileTransformer(output_distribution="uniform",
                             random_state=0, n_quantiles=1000).fit_transform(X))
    for s in [0.25, 0.5, 1.0, 2.0, np.pi]:
        for enc in ["angle_ry", "angle_phase"]:
            r = dict(mode=mode, scale=round(s, 3), enc=enc)
            for N in [4, 8, 16]:
                K = fidelity_kernel(Z[:, :N] * s, Z[:, :N] * s, enc)
                r[N] = round(roc_auc_score(same, K[i, j]), 3)
            rows.append(r)

df = pd.DataFrame(rows).set_index(["mode", "scale", "enc"])
print(df.to_string())
print("\nTable 5 targets at N=16:  angle_ry 0.749   angle_phase 0.762")
print("amplitude reference:")
for N in [4, 8, 16]:
    K = fidelity_kernel(X[:, :N], X[:, :N], "amplitude")
    print(f"   N={N:2d}  {roc_auc_score(same, K[i, j]):.3f}   (paper: 0.647/0.563/0.560)")
