# library/v3: evaluation protocol

v3 replaces the v2 evaluation (`library/v2/D_data_analysis.py`) with a per-fold protocol and reuses the frozen v2
polluters by import (`library.v2.my_functions`). Like `library/` and `library/v2/` it is a namespace package (no
`__init__.py`); import it from the repository root. Differences from v2 are listed in `CHANGES.md`.

## Modules

- `data.py`: `make_regression_data(n_samples, n_features, n_informative, seed, noise_ratio=0.2)` returns
  `(X, y, y_scale)`. The signal comes from sklearn `make_regression(noise=0)`, then Gaussian noise with
  std = `noise_ratio * std(signal)` is added. `y_scale = std(y)` of this clean noisy target and is the nRMSE
  denominator. The module also has `inject_duplicate_rows(X, y, rate, seed)` returning `(X, y, groups)`,
  `deduplicate(X, y[, groups])`, the ready-made polluter `pollute_duplicates(X, y, x, seed)` and
  `realised_meta(X)`.
- `evaluation.py`: the model and imputer registries `MODELS`, `STUDY_MODELS` (the six study models) and
  `IMPUTERS`, plus `make_estimator`, `evaluate`, `evaluate_duplicates`, `duplicate_splits` and `nrmse`.
- `runner.py`: `run_experiment`, `load_results`, `results_path`, `records_frame` and `summarise`.
- `plots.py`: `plot_experiment(path_or_exp_id)` builds every figure from the results JSON alone.
- `selfcheck.py`: protocol invariants. Run it with `uv run --frozen python -m library.v3.selfcheck`. It prints
  `selfcheck ok` and takes about 15 s.

## Protocol

- Each fold fits `TransformedTargetRegressor(Pipeline([imputer steps], StandardScaler, FiniteCheck, model),
  transformer=StandardScaler())` on the training fold only. The imputer, the scaler and the target scaler never see
  test rows. Predictions are inverse-transformed, so metrics are in target units.
- Metrics per fold are `test_nrmse` and `train_nrmse` (RMSE / `y_scale`), `test_r2`, `train_r2`, `fit_time` (seconds
  for one fit), `n_train`, `n_test` and `status`. Train metrics come from the same fitted fold model; there is no
  extra refit.
- Splits: `ShuffleSplit(n_splits=5, test_size=0.3, random_state=split_seed)`.
- `FiniteCheck` raises if NaN or inf would reach a model. Imputer `None` raises on NaN input. Only `HGBRegressor`
  accepts `'native'` (raw NaN).
- `status` is `ok`, or a `;`-joined set of flags. The flags are `convergence_warning`, `warning:<Category>` and
  `gp_at_lower_bound:<param>` / `gp_at_upper_bound:<param>`, where a GP hyperparameter counts as at a bound when it is
  within 0.01 of it in log space.

### Models (`MODELS`)

| name | estimator |
|---|---|
| `LinearRegressor` | `Ridge()` |
| `BayesianRidge` | `BayesianRidge()` |
| `GPRegressor` | `GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.0, length_scale_bounds=(0.1, 1e3)) + WhiteKernel(1e-2), normalize_y=True, n_restarts_optimizer=1, random_state=seed)` |
| `SVMRegressor` | `LinearSVR(random_state=seed, max_iter=100000, dual='auto')` |
| `KNNRegressor` | `KNeighborsRegressor()` |
| `MLPRegressor` | `MLPRegressor(max_iter=2000, early_stopping=True, random_state=seed)` |
| `HGBRegressor` | `HistGradientBoostingRegressor(random_state=seed)`: the native-NaN reference model, not one of the six study models |

The GP length-scale lower bound is 0.1, in standardised feature units. The lead's probe on v2
`pollution_eighth_experiment` at 0.5 used the sklearn default bounds `(1e-5, 1e5)`. There the optimiser converged to
`length_scale = 1e-5` with a worse log marginal likelihood (-993 against -940.7 with the 0.1 bound), and the GP
collapsed to the mean predictor (RMSE 73.8 against Ridge 66.1). Using `n_restarts_optimizer` 0 or 3 gave identical
results, so restarts did not fix it. With the bound at 0.1 the RMSE was 66.0. One restart is kept, and every fold
records bound hits in `status`. Selfcheck (8) guards against this collapse.

### Imputers (`IMPUTERS`; the runner condition has the same name)

| name | steps (fitted on the training fold) |
|---|---|
| `mean`, `median`, `most_frequent` | `SimpleImputer(strategy=...)` (`mean` is the default baseline) |
| `std_constant` | fills each column with its training-fold standard deviation. This is a labelled naive constant that reproduces the original study's method. |
| `knn` | `StandardScaler -> KNNImputer`, then the model scaler. KNN distances need scaled features. |
| `iterative` | `IterativeImputer(estimator=BayesianRidge(), random_state=seed)` |
| `mean_indicator` | `SimpleImputer(strategy='mean', add_indicator=True)` |
| `None` | complete data. Raises if X contains NaN. |
| `native` | no imputation, `HGBRegressor` only. The runner skips the other models for this condition. |

All SimpleImputer, KNN and iterative imputers use `keep_empty_features=True`, so the column count stays stable.

### Duplicate records

`inject_duplicate_rows(X, y, rate, seed)` appends exact copies of randomly chosen source rows (with replacement), so
the copies make up `rate` of the final table (asserted to within one row). `groups[i]` is the source id of row `i`.
The original rows keep positions `0..n-1`. `evaluate_duplicates(X, y, groups, model, condition, seed, y_scale)` takes
one of these conditions:

- `duplicated`: `GroupShuffleSplit` on source ids, so a source never appears in both train and test. The test fold
  scores exactly one row (the original) per held-out source.
- `deduplicated`: the same splits and the same test rows as `duplicated`, but `deduplicate` drops exact duplicate rows
  from the training fold only.
- `naive_split`: a labelled leakage demonstration. It uses `ShuffleSplit` on all rows and scores all test rows, so
  copies of training rows leak into the test fold.

## Seeds, pairing and bands

`run_experiment` takes one repetition seed `s`. `runner.derive_seeds(s)` expands it with `np.random.SeedSequence(s)`
into independent `data`, `pollute` and `split` seeds. `make_data` receives the data seed and `pollute` the pollute
seed. The split seed is used for the CV splits and as the model and imputer `random_state`. The sub-seeds are stored
in each record's `meta.seeds`. Consequences:

- Paired design: for a given seed, every condition and model sees the same data and the same pollution mask (it does
  not depend on the condition), and all conditions, models and x values share the same folds. Each record has
  `data_hash` (the polluted X and y) and `test_hash` (the test rows), so the pairing can be audited.
- Different seeds vary the data generation, the pollution and the splits.
- Tasks are `(seed, x, condition, model)`. `clean` does not depend on x, so it is fitted once per (seed, model) and
  its fold records are repeated for every x. They run through joblib (loky) with BLAS/OpenMP limited to one thread per
  task (`threadpoolctl`), so results do not depend on `n_jobs` apart from `fit_time`. Selfcheck (3) checks this.
- Band semantics: within each seed the folds are averaged first. Then the mean and SD (ddof=1) are taken over the
  seeds. The band is labelled "mean +- SD over seeds (descriptive)". It describes spread across repetitions and is
  not a confidence interval.

## Results JSON and figures

`run_experiment(...)` writes `Plot/v3/results/<exp_id>.json` (set `out_dir` to change `Plot/v3`) and returns the same
payload. The payload contains `version: "v3"`, `exp_id`, `title`, `x_label`, `x_axis`, `models`, `conditions`,
`seeds`, `config` (n_splits, test_size, noise_ratio, GP bounds and restarts, library versions, plus the optional
`config=` dict) and `records`. Each record is one fold:
`{seed, x, condition, model, imputer, data_hash, y_scale, fold, test_hash, test_nrmse, test_r2, train_nrmse, train_r2,
fit_time, status, n_train, n_test, meta}`. `meta` is the polluter's own meta plus `realised` (`n_rows`, the realised
`missing_rate` and `distinct_fraction` per column) and `seeds`. The JSON is strict: non-finite values become `null`,
and `load_results` rejects NaN and Infinity.

Conditions: `clean` (unpolluted, imputer None, the reference), `polluted` (polluted, imputer None, for example
distinctness experiments), any imputer name above, `native`, and `duplicated` / `deduplicated` / `naive_split`. The
duplicate conditions need `pollute` to return `groups`.

`plot_experiment(exp_id_or_path)` writes to `Plot/v3/<exp_id>/`. For each condition it writes
`<condition>_test_nrmse.pdf`, `<condition>_gap.pdf` (test minus train nRMSE) and `<condition>_fit_time.pdf`, each
with one line per model. For each model it writes `model_<model>_test_nrmse.pdf`, comparing the conditions. Colours
and markers are fixed per model and per condition.

## Minimal notebook usage

```python
from library.v2 import my_functions as mf
from library.v3 import evaluation as ev
from library.v3.data import make_regression_data, pollute_duplicates
from library.v3.runner import run_experiment, load_results, summarise
from library.v3.plots import plot_experiment

def make_data(seed):                      # -> (X, y, y_scale)
    return make_regression_data(1000, 3, 3, seed)

def pollute_mcar(X, y, x, seed):          # called on copies; -> (X, y, meta[, groups])
    return mf.pollute_data_mcar(X, x, seed), y, {"requested_rate": x}

run_experiment("c01_mcar", "MCAR on all features", "missing rate", [0.0, 0.1, 0.2, 0.3],
               make_data, pollute_mcar, models=ev.STUDY_MODELS + ["HGBRegressor"],
               conditions=["clean", "mean", "knn", "std_constant", "native"], seeds=[0, 1, 2], n_jobs=8)
plot_experiment("c01_mcar")               # figures from Plot/v3/results/c01_mcar.json only
summarise(load_results("c01_mcar"))       # DataFrame: condition, model, x, mean, sd, n_seeds

run_experiment("d11_duplicates", "Duplicate records", "duplicate rate", [0.0, 0.2, 0.4],
               make_data, pollute_duplicates, models=ev.STUDY_MODELS,
               conditions=["duplicated", "deduplicated", "naive_split"], seeds=[0, 1, 2], n_jobs=8)
```

Define `make_data` and `pollute` at notebook top level. joblib's loky backend pickles them with cloudpickle.

## Runtime probe (for budgeting seeds)

This was measured on 2026-10-25 on this machine (8 logical CPUs, WSL2). The call was one `run_experiment` with 1000x3
data (`make_regression_data(1000, 3, 3, seed)`), 10 x-points (MCAR 0.0 to 0.45), 1 seed, the 6 `STUDY_MODELS`, 1
condition (`mean`), 5 folds and `n_jobs=8`. Wall time was **about 61 s** (60.7 s and 61.4 s in two runs).
`GPRegressor` takes about 6.4 s per fold fit and accounts for about 320 s of the 334 s total fit CPU time. The other
models together take about 12 s (MLP 9 s, the rest under 1 s each). The cost therefore scales with the number of
GP tasks: roughly `ceil(GP tasks / 8) * 32 s`, plus the cheap models.
