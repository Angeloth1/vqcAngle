"""
Small statistics toolkit. Tools mirror the control paper [1] (Welch, paired t across matched seeds,
percentile bootstrap over evaluation records) and add effect sizes with intervals.

Design notes
------------
* n = 5 seeds everywhere. A percentile bootstrap over five seed values is badly calibrated, so
  seed-level intervals are Student-t. The bootstrap is used the way [1] uses it (Table 4): over
  EVALUATION RECORDS of one fixed model, which measures test-sampling uncertainty, not seed variance.
* Effect sizes: Hedges' g for independent groups, d_z (with the small-sample factor) for paired data.
* Nothing here is corrected for multiplicity. Tests are descriptive, as in [1].
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def msd(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def fmt(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


# ---------------------------------------------------------------- independent groups
def welch_summary(m1, s1, n1, m2, s2, n2):
    """Welch's t from summary statistics. Returns (t, df, p)."""
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    se = np.sqrt(v1 + v2)
    if se == 0:
        return float("nan"), float("nan"), float("nan")
    t = (m1 - m2) / se
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    return float(t), float(df), float(2 * stats.t.sf(abs(t), df))


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return welch_summary(a.mean(), a.std(ddof=1), len(a), b.mean(), b.std(ddof=1), len(b))


def hedges_g_summary(m1, s1, n1, m2, s2, n2):
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if sp == 0:
        return float("nan")
    d = (m1 - m2) / sp
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(d * j)


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return hedges_g_summary(a.mean(), a.std(ddof=1), len(a), b.mean(), b.std(ddof=1), len(b))


def diff_ci_welch(m1, s1, n1, m2, s2, n2, level=0.95):
    """Welch-Satterthwaite interval for the difference of means (summary statistics)."""
    t, df, _ = welch_summary(m1, s1, n1, m2, s2, n2)
    se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    q = stats.t.ppf(0.5 + level / 2, df)
    d = m1 - m2
    return float(d - q * se), float(d + q * se)


# ---------------------------------------------------------------- paired
def paired(a, b, level=0.95):
    """Paired comparison a - b across matched seeds.

    Returns dict(diff, sd_diff, ci=(lo,hi), t, p, dz)  -- dz is the small-sample corrected d_z.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m, s = d.mean(), d.std(ddof=1)
    if s == 0:
        return dict(diff=float(m), sd_diff=0.0, ci=(float(m), float(m)), t=float("nan"),
                    p=float("nan"), dz=float("nan"), n=n)
    se = s / np.sqrt(n)
    t = m / se
    p = 2 * stats.t.sf(abs(t), n - 1)
    q = stats.t.ppf(0.5 + level / 2, n - 1)
    j = 1 - 3 / (4 * (n - 1) - 1)
    return dict(diff=float(m), sd_diff=float(s), ci=(float(m - q * se), float(m + q * se)),
                t=float(t), p=float(p), dz=float(m / s * j), n=n)


def one_sample_vs_constant(a, c, level=0.95):
    """a (n seeds) against a deterministic constant c (e.g. a classical baseline with no seed variance)."""
    return paired(np.asarray(a, float), np.full(len(a), float(c)), level)


# ---------------------------------------------------------------- record-level bootstrap
def boot_acc_ci(correct, B=2000, seed=0, level=0.95):
    """Percentile bootstrap over evaluation records (as in [1], Table 4: 2000 resamples).

    `correct` is a 0/1 vector: per-record correctness of ONE fixed model.
    """
    correct = np.asarray(correct, float)
    rng = np.random.default_rng(seed)
    n = len(correct)
    idx = rng.integers(0, n, size=(B, n))
    accs = correct[idx].mean(axis=1)
    lo, hi = np.quantile(accs, [(1 - level) / 2, 0.5 + level / 2])
    return float(correct.mean()), float(lo), float(hi)


def wilson(k, n, level=0.95):
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return float(p), float(c - h), float(c + h)
