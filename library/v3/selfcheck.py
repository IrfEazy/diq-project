"""Protocol invariant checks for library.v3. Run: uv run --frozen python -m library.v3.selfcheck"""
import json
import os
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from library.v2 import my_functions as mf
from library.v3 import evaluation as ev
from library.v3 import plots, runner
from library.v3.data import deduplicate, make_regression_data, pollute_duplicates

SEED = 2023


class SpyImputer(TransformerMixin, BaseEstimator):
    """Pass-through that records the row labels it is fitted on."""
    fitted_on = []

    def fit(self, X, y=None):
        SpyImputer.fitted_on.append(np.asarray(X.index))
        self.fitted_ = True
        return self

    def transform(self, X):
        return X


class SpyModel(RegressorMixin, BaseEstimator):
    """Predicts 0 in the standardised target space; records the targets it is fitted on and predict calls."""
    fits = []
    predicts = 0

    def fit(self, X, y):
        SpyModel.fits.append(np.array(y, dtype=float))
        self.fitted_ = True
        return self

    def predict(self, X):
        SpyModel.predicts += 1
        return np.zeros(len(X))


def _without_time(records):
    return [{k: v for k, v in r.items() if k != "fit_time"} for r in records]


def check_fold_fitting():
    """(1) preprocessing and target transformer fitted on the training fold only; (4) target units; (5) one fit."""
    X, y, y_scale = make_regression_data(400, 3, 3, SEED)
    y = y + 1000.0  # a missing inverse transform would show up as a huge nRMSE
    ev.IMPUTERS["spy"] = lambda seed: [("imputer", SpyImputer())]
    ev.MODELS["Spy"] = lambda seed: SpyModel()
    try:
        SpyImputer.fitted_on.clear()
        SpyModel.fits.clear()
        SpyModel.predicts = 0
        recs = ev.evaluate(X, y, "Spy", "spy", SEED, y_scale)
    finally:
        del ev.IMPUTERS["spy"], ev.MODELS["Spy"]
    splits = list(ShuffleSplit(ev.N_SPLITS, test_size=ev.TEST_SIZE, random_state=SEED).split(X))
    assert len(recs) == len(SpyImputer.fitted_on) == len(SpyModel.fits) == ev.N_SPLITS
    assert SpyModel.predicts == 2 * ev.N_SPLITS  # train and test scored with the same fitted model, no refit
    for (tr, _), seen, y_fit in zip(splits, SpyImputer.fitted_on, SpyModel.fits):
        assert np.array_equal(np.sort(seen), np.sort(X.index[tr]))
        manual = StandardScaler().fit_transform(y[tr].reshape(-1, 1)).ravel()
        assert np.allclose(y_fit, manual)
    # (4) predicting the training mean in target units gives nRMSE ~ 1; a perfect predictor gives 0
    assert all(0.85 < r["test_nrmse"] < 1.15 and 0.95 < r["train_nrmse"] < 1.05 for r in recs), recs
    assert ev.nrmse(y, y, y_scale) == 0.0
    Xc, yc, sc = make_regression_data(400, 3, 3, SEED, noise_ratio=0.0)
    recs = ev.evaluate(Xc, yc + 1000.0, "LinearRegressor", None, SEED, sc)
    assert all(r["test_nrmse"] < 0.01 for r in recs), recs
    assert all(r["train_nrmse"] <= r["test_nrmse"] + 0.01 for r in recs)


def check_missing():
    """(2) fold-fitted mean imputation differs from the v2 scale-then-zero path; None raises on NaN."""
    X, y, y_scale = make_regression_data(500, 3, 3, SEED)
    Xm = mf.pollute_data_mcar(X.copy(), 0.2, SEED)
    assert Xm.isna().to_numpy().any()
    new = ev.evaluate(Xm, y, "LinearRegressor", "mean", SEED, y_scale)
    Z = StandardScaler().fit_transform(Xm)  # v2: scaler on all rows, then NaN -> 0
    old_X = Xm.copy()
    old_X[:] = np.where(np.isnan(Z), 0.0, Z)
    old = ev.evaluate(old_X, y, "LinearRegressor", None, SEED, y_scale)
    diff = max(abs(a["test_nrmse"] - b["test_nrmse"]) for a, b in zip(new, old))
    assert diff > 1e-6, diff
    # on complete data mean imputation is a no-op
    a = ev.evaluate(X, y, "LinearRegressor", "mean", SEED, y_scale)
    b = ev.evaluate(X, y, "LinearRegressor", None, SEED, y_scale)
    assert _without_time(a) == _without_time(b)
    for bad in ((Xm, "LinearRegressor", None), (Xm, "LinearRegressor", "native")):
        try:
            ev.evaluate(bad[0], y, bad[1], bad[2], SEED, y_scale)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad[1:]}")
    native = ev.evaluate(Xm, y, "HGBRegressor", "native", SEED, y_scale, n_splits=1)
    assert native[0]["test_nrmse"] < 1
    for imp in ("median", "most_frequent", "std_constant", "knn", "iterative", "mean_indicator"):
        r = ev.evaluate(Xm, y, "LinearRegressor", imp, SEED, y_scale, n_splits=1)
        assert np.isfinite(r[0]["test_nrmse"]), imp


def _make_data(seed):
    return make_regression_data(200, 3, 3, seed)


def _pollute_mcar(X, y, x, seed):
    return mf.pollute_data_mcar(X, x, seed), y, {"requested_rate": x}


def check_runner(tmp):
    """(3) paired design and n_jobs independence; (7) strict JSON round trip and plots from JSON only."""
    kw = dict(exp_id="selfcheck", title="selfcheck", x_label="missing rate", x_axis=[0.1, 0.3],
              make_data=_make_data, pollute=_pollute_mcar, models=["LinearRegressor", "KNNRegressor"],
              conditions=["clean", "mean", "median", "std_constant"], seeds=[0, 1], out_dir=tmp)
    p1 = runner.run_experiment(**kw, n_jobs=1)
    p2 = runner.run_experiment(**kw, n_jobs=2)
    assert _without_time(p1["records"]) == _without_time(p2["records"])
    recs = p1["records"]
    assert len(recs) == 2 * 2 * 4 * 2 * ev.N_SPLITS
    by_cell = {}
    for r in recs:
        by_cell.setdefault((r["seed"], r["x"], r["condition"] == "clean"), set()).add(r["data_hash"])
        by_cell.setdefault(("folds", r["seed"], r["fold"]), set()).add(r["test_hash"])
    assert all(len(v) == 1 for v in by_cell.values()), "data or folds differ across conditions/models"
    assert by_cell[(0, 0.1, False)] != by_cell[(1, 0.1, False)], "seeds must vary the data"
    assert by_cell[("folds", 0, 0)] != by_cell[("folds", 1, 0)], "seeds must vary the splits"
    for r in recs:
        rate = np.mean(r["meta"]["realised"]["missing_rate"])
        assert (rate == 0) if r["condition"] == "clean" else abs(rate - r["x"]) < 0.08, (r["condition"], rate)
    # strict JSON round trip
    path = runner.results_path("selfcheck", tmp)
    with open(path) as f:
        text = f.read()
    assert "NaN" not in text and "Infinity" not in text
    assert runner.load_results(path) == p2 == json.loads(json.dumps(p2, allow_nan=False))
    assert runner.to_strict({"a": [np.float64("nan"), np.inf, np.int64(3)]}) == {"a": [None, None, 3]}
    # plots from the JSON alone: building any estimator now is an error
    original = ev.make_estimator
    ev.make_estimator = lambda *a, **k: (_ for _ in ()).throw(AssertionError("plotting reran a model"))
    try:
        paths = plots.plot_experiment(path, out_dir=tmp)
    finally:
        ev.make_estimator = original
    assert len(paths) == 3 * 4 + 2 and all(os.path.getsize(p) > 0 for p in paths), paths


def check_duplicates():
    """(6) duplicate groups never straddle train/test; one scored row per held-out source; identical test rows."""
    X, y, y_scale = make_regression_data(300, 3, 3, SEED)
    Xd, yd, meta, groups = pollute_duplicates(X, y, 0.3, SEED)
    assert abs(meta["n_duplicates"] - 0.3 * len(Xd)) <= 1 and len(Xd) == len(groups)
    assert np.array_equal(groups[:len(X)], np.arange(len(X)))
    assert len(deduplicate(Xd, yd)[0]) == len(X)
    tests = {}
    for cond in ("duplicated", "deduplicated"):
        tests[cond] = []
        for tr, te in ev.duplicate_splits(groups, cond, SEED):
            assert not set(groups[tr]) & set(groups[te])
            assert len(te) == len(np.unique(groups[te]))
            assert np.array_equal(groups[te], te)  # the scored row is the original of each source
            tests[cond].append(te)
    assert all(np.array_equal(a, b) for a, b in zip(tests["duplicated"], tests["deduplicated"]))
    leaks = [bool(set(groups[tr]) & set(groups[te])) for tr, te in ev.duplicate_splits(groups, "naive_split", SEED)]
    assert any(leaks), "the naive split should leak duplicates into the test fold"
    recs = {c: ev.evaluate_duplicates(Xd, yd, groups, "KNNRegressor", c, SEED, y_scale)
            for c in ev.DUPLICATE_CONDITIONS}
    for a, b in zip(recs["duplicated"], recs["deduplicated"]):
        assert a["test_hash"] == b["test_hash"] and a["n_test"] == b["n_test"] and a["n_train"] > b["n_train"]
    assert recs["naive_split"][0]["n_test"] > recs["duplicated"][0]["n_test"]


def check_gp():
    """(8) GP on a heavily tied dataset does not collapse to the mean predictor."""
    X, y, y_scale = make_regression_data(1000, 3, 3, SEED)
    Xt = mf.pollution_eighth_experiment(X, 0.5, SEED)
    gp = ev.evaluate(Xt, y, "GPRegressor", None, SEED, y_scale, n_splits=1)[0]
    ridge = ev.evaluate(Xt, y, "LinearRegressor", None, SEED, y_scale, n_splits=1)[0]
    assert gp["test_nrmse"] < 0.95 and abs(gp["test_nrmse"] - ridge["test_nrmse"]) < 0.03, (gp, ridge)
    assert "lower_bound" not in gp["status"], gp["status"]


def main():
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        for check in (check_fold_fitting, check_missing, check_duplicates, check_runner, check_gp):
            t = time.perf_counter()
            if check is check_runner:
                with tempfile.TemporaryDirectory() as tmp:
                    check(tmp)
            else:
                check()
            print(f"{check.__name__}: {time.perf_counter() - t:.1f} s")
    print(f"selfcheck ok ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
