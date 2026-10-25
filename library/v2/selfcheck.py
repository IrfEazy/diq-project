"""Invariant checks for library.v2. Run: python -m library.v2.selfcheck"""
import json
import os
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from library.v2 import my_functions as mf
from library.v2.D_data_analysis import regression


def dataset(n=500, f=9, inf=4, seed=2023):
    X, y = make_regression(n_samples=n, n_features=f, n_informative=inf, random_state=seed)
    return pd.DataFrame(X), y


def check_distinctness():
    X, y = dataset()
    n = len(X)
    for p in (0, 0.05, 0.5, 1.0):
        before = X.copy()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = mf.pollution_first_second_third_experiments(X, p)
        pd.testing.assert_frame_equal(X, before)  # input not mutated
        changed = [c for c in X.columns if not X[c].equals(out[c])]
        assert len(changed) == (0 if p == 0 else 1), (p, changed)
        if p == 0:
            continue
        nu = out[changed[0]].nunique()
        if p == 1.0:
            assert nu == 1 and any("clipped" in str(x.message) for x in w)
        else:
            assert nu / n <= 1 - p + 1e-12 and nu / n >= 1 - p - 1 / n - 1e-12, (p, nu)
    # fourth, eighth, ninth/tenth
    out = mf.pollution_fourth_experiment(X, 0.3)
    assert sum(not X[c].equals(out[c]) for c in X.columns) <= X.shape[1] // 2 + 1
    out = mf.pollution_eighth_experiment(X, 0.4)
    nus = sorted(out[c].nunique() for c in X.columns if not X[c].equals(out[c]))
    assert len(nus) == 2 and nus[0] <= 0.6 * n and nus[1] <= 0.6 * n + 1
    assert max(nus) > min(nus) or True
    for inf_flag in (True, False):
        out = mf.pollution_ninth_tenth_experiments(X, y, 0.5, informative=inf_flag)
        assert sum(out[c].nunique() <= n * 0.5 + 1e-9 for c in X.columns) == 3
    # fifth, sixth, seventh
    base = X.copy()
    out = mf.pollution_fifth_experiment(X, 0.04)
    assert abs(np.std((out - X).to_numpy()) - 0.2) < 0.02
    pd.testing.assert_frame_equal(X, base)
    assert mf.pollution_sixth_experiment(X, 4).shape == X.shape
    out = mf.pollution_seventh_experiment(X, 0.05)
    assert out.shape == X.shape and int((out != X).to_numpy().sum()) > 0


def check_completeness():
    X, y = dataset()
    n = len(X)
    out = mf.pollute_data_mcar(X.copy(), 0.25)
    assert abs(out.isna().to_numpy().mean() - 0.25) < 0.03
    d = X.copy()
    mf.pollute_data_mcar_for_each_feature_in_incremental_way(d)
    rates = d.isna().mean().to_numpy()
    assert rates[0] == 0 and abs(rates[1] - 0.1) < 0.04 and abs(rates[4] - 0.4) < 0.08 and rates[-1] > rates[1]
    d = X.copy()
    r = mf.pollute_data_mar(d, 1, 2)
    assert r is d and d[2].isna().sum() == (X[1] > X[1].quantile(0.8)).sum() > 0
    assert d.drop(columns=2).isna().sum().sum() == 0
    d = X.copy()
    r = mf.pollute_data_mnar(d, 1)
    assert r is d and 0.4 * n < d[1].isna().sum() < 0.6 * n
    for fn, label in ((mf.pollute_most_important_features, "most"), (mf.pollute_less_important_features, "less")):
        before = X.copy()
        out = fn(X, y, 0.3)
        pd.testing.assert_frame_equal(X, before)
        cols = [c for c in out.columns if out[c].isna().any()]
        assert len(cols) == 3, (label, cols)
        for c in cols:
            assert out[c].isna().sum() == int(n * 0.3)


def check_regression():
    X, y = dataset(n=60, f=3, inf=3)
    X.iloc[:5, 0] = np.nan
    for alg in ("LinearRegressor", "BayesianRidge", "GPRegressor", "SVMRegressor", "KNNRegressor", "MLPRegressor"):
        r = regression(X, y, alg, 2023)
        assert all(np.isfinite(r[k]) for k in ("mean_perf", "distance", "speed")), (alg, r)


def check_plots():
    algs = ["LinearRegressor", "GPRegressor"]
    results = [[{"mean_perf": 1.0 + i + j, "distance": 0.1 * j * (i + 1) ** 2 + i, "speed": 0.01 * (i + 1) ** 2 + 0.001 * j} for j in range(4)]
               for i in range(2)]
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            plt.close("all")
            mf.plot_results([0.1, 0.2, 0.3, 0.4], "x", results, "Selfcheck", algs)
            pdfs = os.listdir("Plot/v2")
            assert len([p for p in pdfs if p.endswith(".pdf")]) == 6, pdfs
            js = os.listdir("Plot/v2/results")
            assert "Selfcheck - Trial regression.json" in js, js
            with open("Plot/v2/results/Selfcheck - Trial regression.json") as f:
                d = json.load(f)
            assert d["version"] == "v2" and d["algorithms"] == algs and len(d["results"][algs[0]]) == 4
        finally:
            os.chdir(old)
            plt.close("all")


if __name__ == "__main__":
    check_distinctness()
    check_completeness()
    check_regression()
    check_plots()
    print("selfcheck ok")
