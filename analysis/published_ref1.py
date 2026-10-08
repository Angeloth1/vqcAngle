"""
Numbers transcribed from the published control paper [1]:
  Thomos & Andronikos, J. Sens. Actuator Netw. 2026, 15(4), 67, doi:10.3390/jsan15040067
Source PDF: https://mdpi-res.com/d_attachment/jsan/jsan-15-00067/article_deploy/jsan-15-00067.pdf
Page numbers are the journal's own ("n of 32") pagination.

Nothing in this file is computed. Every value is copied from the table named in the comment;
derived quantities (means/SDs of Table A2 rows) are recomputed by the caller from the per-run rows
and compared with the paper's printed summary as a transcription check.
"""
import numpy as np

# ---- Table 3 (p.15): five seeds, 2000-record training subsample, maxiter=150, reps=5.
# mean, sample SD (n=5)
TABLE3 = {
    "VQC (COBYLA)":        dict(params=24, internal=(0.854, 0.018), kddtest=(0.765, 0.049), known=(0.639, 0.159), novel=(0.653, 0.062)),
    "VQC (SPSA)":          dict(params=24, internal=(0.873, 0.020), kddtest=(0.758, 0.036), known=(0.665, 0.089), novel=(0.579, 0.042)),
    "Linear SVC":          dict(params=17, internal=(0.926, 0.005), kddtest=(0.769, 0.002), known=(0.721, 0.013), novel=(0.495, 0.015)),
    "Logistic Regression": dict(params=17, internal=(0.927, 0.005), kddtest=(0.764, 0.007), known=(0.714, 0.013), novel=(0.482, 0.017)),
    "MLP-1":               dict(params=19, internal=(0.850, 0.176), kddtest=(0.707, 0.156), known=(0.590, 0.331), novel=(0.393, 0.237)),
    "MLP-4":               dict(params=73, internal=(0.958, 0.003), kddtest=(0.752, 0.017), known=(0.719, 0.010), novel=(0.395, 0.120)),
}

# ---- Table A2 (p.28): individual runs of the ansatz-depth sweep, independent build
# (partition 20,153/5,039, theta0 ~ U(-0.1, 0.1), 2000-record subsample), COBYLA.
# columns: reps, params, maxiter, seed, loss, internal, kddtest, novel
A2 = [
    (1, 8, 150, 0, 0.6832, 0.778, 0.613, 0.573),
    (1, 8, 150, 1, 0.6319, 0.806, 0.636, 0.614),
    (1, 8, 150, 2, 0.6593, 0.776, 0.614, 0.580),
    (1, 8, 150, 3, 0.6676, 0.775, 0.604, 0.538),
    (1, 8, 150, 4, 0.6830, 0.777, 0.617, 0.601),
    (3, 16, 150, 0, 0.5558, 0.905, 0.821, 0.677),
    (3, 16, 150, 1, 0.6161, 0.836, 0.659, 0.632),
    (3, 16, 150, 2, 0.5846, 0.823, 0.653, 0.434),
    (3, 16, 150, 3, 0.6219, 0.862, 0.723, 0.660),
    (3, 16, 150, 4, 0.6227, 0.831, 0.672, 0.661),
    (5, 24, 150, 0, 0.6252, 0.842, 0.666, 0.616),
    (5, 24, 150, 1, 0.5789, 0.847, 0.657, 0.659),
    (5, 24, 150, 2, 0.5537, 0.814, 0.642, 0.583),
    (5, 24, 150, 3, 0.6015, 0.844, 0.763, 0.742),
    (5, 24, 150, 4, 0.6207, 0.840, 0.686, 0.485),
    (9, 40, 240, 0, 0.5516, 0.896, 0.807, 0.588),
    (9, 40, 240, 1, 0.5107, 0.897, 0.820, 0.577),
    (9, 40, 240, 2, 0.5947, 0.830, 0.631, 0.562),
    (9, 40, 240, 3, 0.5443, 0.900, 0.826, 0.659),
    (9, 40, 240, 4, 0.5547, 0.867, 0.690, 0.660),
    (17, 72, 432, 0, 0.5322, 0.884, 0.804, 0.546),
    (17, 72, 432, 1, 0.5153, 0.874, 0.724, 0.505),
    (17, 72, 432, 2, 0.5217, 0.884, 0.807, 0.507),
    (17, 72, 432, 3, 0.5295, 0.880, 0.724, 0.570),
    (17, 72, 432, 4, 0.5397, 0.875, 0.725, 0.645),
]
A2_COLUMNS = ["reps", "params", "maxiter", "seed", "loss", "internal", "kddtest", "novel"]
# summary printed in the text of Sec. 7.6 for the reps=5 row (p.18), used only as a transcription check
A2_REPS5_PRINTED = dict(internal=(0.837, 0.013), kddtest=(0.683, 0.048), novel=(0.617, 0.095))

# ---- Table 5 (p.20): separability AUC of the encoded states, mean ± sample SD over five seeds
TABLE5 = {
    "amplitude":   {4: (0.647, 0.018), 8: (0.563, 0.018), 16: (0.560, 0.018)},
    "angle_ry":    {4: (0.763, 0.014), 8: (0.752, 0.018), 16: (0.749, 0.016)},
    "angle_phase": {4: (0.787, 0.017), 8: (0.763, 0.021), 16: (0.762, 0.020)},
}
# ---- Table A1 (p.27): per-seed AUC at N=16
TABLEA1_N16 = {
    "amplitude":   [0.564, 0.560, 0.581, 0.562, 0.531],
    "angle_ry":    [0.746, 0.768, 0.764, 0.737, 0.732],
    "angle_phase": [0.771, 0.730, 0.780, 0.775, 0.754],
}

# ---- Table 6 (p.22): first four PCs, both circuits carry 8 trainable parameters, COBYLA 100 iterations,
# 2000-record subsample, five seeds; angle = ZFeatureMap (1 rep) + RealAmplitudes reps=1 on 4 qubits.
TABLE6 = {
    "amplitude (2 qubits, reps=3)": dict(internal=(0.813, 0.015), kddtest=(0.635, 0.030), known=(0.335, 0.093), novel=(0.594, 0.024)),
    "angle (4 qubits, reps=1)":     dict(internal=(0.796, 0.012), kddtest=(0.630, 0.010), known=(0.300, 0.036), novel=(0.586, 0.013)),
}

# ---- scalar facts used in the text
FACTS = dict(
    partition_table3=(20155, 5037),          # Sec. 3.1, Table 1
    partition_rebuild=(20153, 5039),         # Sec. 7.6
    retained_variance_pca16_table3=0.8678,   # Sec. 4.2
    kdd_test_records=22544, kdd_novel_records=3752,
    kdd_composition=dict(normal=0.431, known=0.403, novel=0.166),   # Sec. 7.5
    ansatz_reps5_cnot=15, ansatz_reps5_transpiled_depth=17,          # Sec. 7.3 / 8.2.2 (ansatz only)
    build_gap_kddtest=(0.765, 0.683), build_gap_points=8.2,          # Sec. 7.5 / 7.6
    optimizer_budget_single_run=500, optimizer_budget_multiseed=150,
    published_decoder="b mod 2 (least-significant qubit q0), Sec. 5.3",
)
