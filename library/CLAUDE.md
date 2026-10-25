# library/

This file describes v1, the top-level modules. Modules are prefixed `A_`, `D_`, `E_`; no B or C modules exist.

- `A_data_collection.py`: dataset generators wrapping sklearn `make_regression`, `make_classification`, `make_blobs`.
- `my_functions.py` pollution (functions default to `seed=2023`):
  - Completeness: `pollute_data_mcar*`, `pollute_data_mar`, `pollute_data_mnar` (MCAR/MAR/MNAR missing values), plus `pollute_most/less_important_features` (feature-importance-targeted missingness, used only in `completeness.ipynb`).
  - Distinctness: `pollution_*_experiment(s)` functions (first through tenth experiment).
- `D_data_analysis.py`: `regression(X, y, regression, seed)` picks the model by name string (`LinearRegressor` is Ridge and also the fallback, `BayesianRidge`, `GPRegressor`, `SVMRegressor`, `KNNRegressor`, `MLPRegressor`). It uses ShuffleSplit (8 splits, test size 0.3) and returns a dict with `mean_perf`, `distance`, `speed`. It imports `ignore_warnings` from the private `sklearn.utils._testing`.
- `my_functions.plot_results` appends ' - Trial regression ' or ' - Cleaned regression ' to the title, while `zoomed_plot` receives an already-titled string; both then call `E_plot_results.py` `plot()`, which dispatches on plot type (performance, distance train-test, speed) and saves a PDF named from the title.

## v2 and v3

- `library/v2/`: faithful fixes of the v1 modules (same names, plus `selfcheck.py`). Differences and reasons are in `CHANGES.md`. Imports are `library.v2.*`.
- `library/v3/`: per-fold evaluation protocol that reuses the frozen v2 polluters. Modules, protocol, and results JSON are described in `README.md`; differences from v2 are in `CHANGES.md`.
