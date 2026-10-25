"""v3 data generation and duplicate-record pollution/cleaning (see README.md)."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


def make_regression_data(n_samples, n_features, n_informative, seed, noise_ratio=0.2):
    """
    Linear regression data with SNR-based noise.

    The noiseless signal comes from sklearn ``make_regression(noise=0)``; Gaussian noise with
    std = noise_ratio * std(signal) is added from ``np.random.default_rng(seed)``.
    Returns (X DataFrame with integer column labels, y ndarray, y_scale) where y_scale = std(y) of this
    clean noisy target and is the denominator of every nRMSE.
    """
    X, signal = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                                noise=0.0, random_state=seed)
    rng = np.random.default_rng(seed)
    y = signal + rng.normal(0.0, noise_ratio * np.std(signal), size=signal.shape)
    return pd.DataFrame(X), y, float(np.std(y))


def realised_meta(X):
    """Realised pollution of a table: missing rate and distinct fraction (over non-missing cells) per column."""
    n = len(X)
    return {
        "n_rows": int(n),
        "missing_rate": [float(X[c].isna().mean()) for c in X.columns],
        "distinct_fraction": [float(X[c].nunique(dropna=True) / n) if n else None for c in X.columns],
    }


def inject_duplicate_rows(X, y, rate, seed):
    """
    Append exact copies of randomly chosen source rows (with replacement) so that the appended copies make up
    `rate` of the final table. Returns (X, y, groups) where groups[i] is the source row id of row i
    (original rows keep their position 0..n-1 and are their own source).
    """
    if not 0 <= rate < 1:
        raise ValueError("rate must be in [0, 1)")
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    n = len(X)
    n_dup = int(round(rate * n / (1 - rate)))
    rng = np.random.default_rng(seed)
    sources = rng.choice(n, size=n_dup, replace=True)
    groups = np.concatenate([np.arange(n), sources])
    X_out = pd.concat([X, X.iloc[sources]], ignore_index=True)
    y_out = np.concatenate([y, y[sources]])
    assert abs(n_dup - rate * len(X_out)) <= 1, "realised duplicate rate off by more than one row"
    return X_out, y_out, groups


def deduplicate(X, y, groups=None):
    """Drop exact duplicate rows (same features and target), keeping the first occurrence."""
    table = X.reset_index(drop=True).copy()
    table.columns = range(table.shape[1])
    table["__y__"] = np.asarray(y)
    keep = ~table.duplicated(keep="first").to_numpy()
    X_out = X.iloc[np.flatnonzero(keep)]
    y_out = np.asarray(y)[keep]
    if groups is None:
        return X_out, y_out
    return X_out, y_out, np.asarray(groups)[keep]


def pollute_duplicates(X, y, x, seed):
    """Ready-made `pollute` for run_experiment: duplicate rate x. Returns (X, y, meta, groups)."""
    X_out, y_out, groups = inject_duplicate_rows(X, y, x, seed)
    meta = {"duplicate_rate": float((len(X_out) - len(X)) / len(X_out)),
            "n_duplicates": int(len(X_out) - len(X)),
            "n_sources_duplicated": int(len(np.unique(groups[len(X):])))}
    return X_out, y_out, meta, groups
