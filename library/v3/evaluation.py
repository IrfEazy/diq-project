"""
v3 evaluation protocol (see README.md).

Per fold: TransformedTargetRegressor(Pipeline([imputer], StandardScaler, [finite check], model),
transformer=StandardScaler()), fitted on the training fold only. Metrics are in target units:
nRMSE = RMSE / y_scale, R2; train metrics come from the same fitted fold model.
"""
import hashlib
import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (enables IterativeImputer)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from library.v3.data import deduplicate

N_SPLITS = 5
TEST_SIZE = 0.3
GP_LENGTH_SCALE_BOUNDS = (0.1, 1e3)
GP_RESTARTS = 1


def _gp(seed):
    kernel = ConstantKernel(1.0) * RBF(1.0, length_scale_bounds=GP_LENGTH_SCALE_BOUNDS) + WhiteKernel(1e-2)
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=GP_RESTARTS,
                                    random_state=seed)


# name -> factory(seed). The six study models first; HGB is the native-NaN reference model.
MODELS = {
    "LinearRegressor": lambda seed: Ridge(),
    "BayesianRidge": lambda seed: BayesianRidge(),
    "GPRegressor": _gp,
    "SVMRegressor": lambda seed: LinearSVR(random_state=seed, max_iter=100_000, dual="auto"),
    "KNNRegressor": lambda seed: KNeighborsRegressor(),
    "MLPRegressor": lambda seed: MLPRegressor(max_iter=2000, early_stopping=True, random_state=seed),
    "HGBRegressor": lambda seed: HistGradientBoostingRegressor(random_state=seed),
}
STUDY_MODELS = list(MODELS)[:6]
NATIVE_NAN_MODELS = {"HGBRegressor"}


class StdConstantImputer(TransformerMixin, BaseEstimator):
    """Labelled naive constant: fill each column with its training-fold standard deviation (ddof=0)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column -> nan, replaced below
            std = np.nanstd(X, axis=0)
        self.statistics_ = np.where(np.isfinite(std), std, 0.0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        return np.where(np.isnan(X), self.statistics_, X)


class FiniteCheck(TransformerMixin, BaseEstimator):
    """Pass-through that raises if the model would receive NaN or inf."""

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        if not np.isfinite(np.asarray(X, dtype=float)).all():
            raise ValueError("non-finite input reached the model (use an imputer, or 'native' with HGBRegressor)")
        return X


# name -> factory(seed) returning a list of pipeline steps placed before the model scaler.
IMPUTERS = {
    "mean": lambda seed: [("imputer", SimpleImputer(strategy="mean", keep_empty_features=True))],
    "median": lambda seed: [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))],
    "most_frequent": lambda seed: [("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True))],
    "std_constant": lambda seed: [("imputer", StdConstantImputer())],
    "knn": lambda seed: [("pre_scaler", StandardScaler()), ("imputer", KNNImputer(keep_empty_features=True))],
    "iterative": lambda seed: [("imputer", IterativeImputer(estimator=BayesianRidge(), random_state=seed,
                                                            keep_empty_features=True))],
    "mean_indicator": lambda seed: [("imputer", SimpleImputer(strategy="mean", add_indicator=True,
                                                              keep_empty_features=True))],
    None: lambda seed: [],
    "native": lambda seed: [],
}


def make_estimator(model, imputer, seed):
    """Unfitted per-fold estimator for a model name and imputer name (see IMPUTERS)."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {list(MODELS)}")
    if imputer not in IMPUTERS:
        raise ValueError(f"unknown imputer {imputer!r}; choose from {list(IMPUTERS)}")
    if imputer == "native" and model not in NATIVE_NAN_MODELS:
        raise ValueError(f"imputer 'native' is only allowed for {sorted(NATIVE_NAN_MODELS)}")
    steps = IMPUTERS[imputer](seed) + [("scaler", StandardScaler())]
    if imputer != "native":
        steps.append(("finite", FiniteCheck()))
    steps.append(("model", MODELS[model](seed)))
    return TransformedTargetRegressor(regressor=Pipeline(steps), transformer=StandardScaler())


def nrmse(y_true, y_pred, y_scale):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)) / y_scale)


def _gp_flags(fitted, tol=1e-2):
    """Flags for GP hyperparameters that end at (within tol in log space of) a bound."""
    model = fitted.regressor_.named_steps["model"]
    if not isinstance(model, GaussianProcessRegressor):
        return []
    kernel = model.kernel_
    params = kernel.get_params()
    flags = []
    for hp in kernel.hyperparameters:
        if hp.fixed:
            continue
        values = np.log(np.atleast_1d(params[hp.name]))
        bounds = np.log(np.atleast_2d(hp.bounds))
        for side, column in (("lower", 0), ("upper", 1)):
            if np.any(np.abs(values - bounds[:, column]) < tol):
                flags.append(f"gp_at_{side}_bound:{hp.name}")
    return flags


def _status(flags):
    return "ok" if not flags else ";".join(sorted(set(flags)))


def fit_score(estimator, X_train, y_train, X_test, y_test, y_scale):
    """Fit once on the training fold and score train and test with that same fitted model."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        pred_train = estimator.predict(X_train)
        pred_test = estimator.predict(X_test)
    flags = ["convergence_warning" if issubclass(w.category, ConvergenceWarning) else f"warning:{w.category.__name__}"
             for w in caught]
    flags += _gp_flags(estimator)
    return {
        "test_nrmse": nrmse(y_test, pred_test, y_scale),
        "test_r2": float(r2_score(y_test, pred_test)),
        "train_nrmse": nrmse(y_train, pred_train, y_scale),
        "train_r2": float(r2_score(y_train, pred_train)),
        "fit_time": float(fit_time),
        "status": _status(flags),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def _index_hash(idx):
    return hashlib.sha1(np.sort(np.asarray(idx, dtype=np.int64)).tobytes()).hexdigest()[:12]


def _check_input(X, imputer):
    if imputer is None and X.isna().to_numpy().any():
        raise ValueError("imputer None (complete data) received NaN; choose an imputer")


def evaluate(X, y, model, imputer, seed, y_scale, n_splits=N_SPLITS, test_size=TEST_SIZE):
    """Per-fold records for ShuffleSplit(n_splits, test_size, random_state=seed)."""
    _check_input(X, imputer)
    y = np.asarray(y)
    records = []
    for fold, (tr, te) in enumerate(ShuffleSplit(n_splits, test_size=test_size, random_state=seed).split(X)):
        est = make_estimator(model, imputer, seed)
        rec = fit_score(est, X.iloc[tr], y[tr], X.iloc[te], y[te], y_scale)
        records.append({"fold": fold, "test_hash": _index_hash(te), **rec})
    return records


DUPLICATE_CONDITIONS = ("duplicated", "deduplicated", "naive_split")


def duplicate_splits(groups, condition, seed, n_splits=N_SPLITS, test_size=TEST_SIZE):
    """
    Yield (train_idx, test_idx) row positions for a duplicate-records condition.

    duplicated/deduplicated: GroupShuffleSplit on source ids (test_size is a fraction of source ids), test is
    exactly one row (the first, i.e. the original) per held-out source id. naive_split: ShuffleSplit on all rows,
    all test rows scored (labelled leakage demonstration). Training-fold deduplication happens in
    evaluate_duplicates, not here.
    """
    groups = np.asarray(groups)
    if condition not in DUPLICATE_CONDITIONS:
        raise ValueError(f"condition must be one of {DUPLICATE_CONDITIONS}")
    if condition == "naive_split":
        yield from ShuffleSplit(n_splits, test_size=test_size, random_state=seed).split(groups)
        return
    splitter = GroupShuffleSplit(n_splits, test_size=test_size, random_state=seed)
    for tr, te in splitter.split(groups, groups=groups):
        te = np.sort(te)
        _, first = np.unique(groups[te], return_index=True)
        yield tr, te[np.sort(first)]


def evaluate_duplicates(X, y, groups, model, condition, seed, y_scale, n_splits=N_SPLITS, test_size=TEST_SIZE,
                        imputer=None):
    """Per-fold records for a duplicate-records condition (see duplicate_splits)."""
    _check_input(X, imputer)
    y = np.asarray(y)
    groups = np.asarray(groups)
    records = []
    for fold, (tr, te) in enumerate(duplicate_splits(groups, condition, seed, n_splits, test_size)):
        X_tr, y_tr = X.iloc[tr], y[tr]
        if condition == "deduplicated":
            X_tr, y_tr = deduplicate(X_tr, y_tr)
        est = make_estimator(model, imputer, seed)
        rec = fit_score(est, X_tr, y_tr, X.iloc[te], y[te], y_scale)
        records.append({"fold": fold, "test_hash": _index_hash(te), **rec})
    return records
