# library/v3 vs v2: intentional differences

These follow the Phase 4 design of run 20261024T180844Z-0d73a8e3 (methodology consult: fold-fitted
preprocessing, group-aware duplicate splits, repetitions over seeds). The v2 polluters are reused unchanged
by import. v2 itself is frozen.

- Preprocessing: v2 `regression()` fitted `StandardScaler` on all rows, zero-filled the NaN in the scaled
  matrix (an implicit global-mean imputation that saw the test rows), and then cross-validated. In v3 every
  imputer, the feature scaler and a target scaler are fitted per training fold inside
  `TransformedTargetRegressor(Pipeline(...))`. Missing values are handled by a named imputer or rejected
  (imputer `None` raises on NaN), and nothing is silently zero-filled.
- Imputation is now an explicit experimental condition: `mean` (the baseline), `median`, `most_frequent`,
  `std_constant` (the labelled naive constant of the original study), `knn` (scaled before KNN), `iterative`,
  `mean_indicator`, and `native` NaN handling with the new `HGBRegressor` reference model. `clean` is an
  unpolluted reference.
- Metrics: v2 reported cross-validated RMSE in raw target units, plus a train/test "distance" from a separate
  `train_test_split` refit. v3 reports nRMSE (RMSE / std of the clean noisy target) and R2, for test and train.
  Both come from the same fitted fold model, together with the fit time per fold.
- Splits: v2 used `ShuffleSplit(8, 0.3)` with one seed. v3 uses `ShuffleSplit(5, 0.3)` and repetition seeds.
  Each seed is expanded through `SeedSequence` into independent data, pollution and split seeds. The design is
  paired across conditions and models, and the results do not depend on `n_jobs`.
- Data: the v2 notebooks generated noiseless targets (`make_regression(noise=0.0)`). v3 `make_regression_data`
  adds SNR-based noise (std = 0.2 * std(signal)) from a seeded `Generator`.
- GP: v2 used `ConstantKernel * RBF + WhiteKernel` with the default length-scale bounds `(1e-5, 1e5)` and no
  restarts. On heavily tied data this collapsed to the mean predictor. v3 uses `length_scale_bounds=(0.1, 1e3)`
  and `n_restarts_optimizer=1`, and records bound hits in each fold's `status`.
- `SVMRegressor` now has an explicit `random_state`, `max_iter=100000` and `dual='auto'`. `MLPRegressor` has
  `max_iter=2000` and `early_stopping=True`. Convergence warnings are recorded per fold in `status` rather than
  suppressed through the private `sklearn.utils._testing.ignore_warnings`.
- New duplicate-records experiment: `inject_duplicate_rows`, `deduplicate`, and `evaluate_duplicates` with the
  `duplicated` / `deduplicated` conditions (group-aware splits, one scored row per held-out source) and
  `naive_split` (a labelled leakage demonstration).
- Results: v2 stored one aggregate per x and model in `Plot/v2/results/<title>.json`. v3 stores every fold
  record, with seeds, hashes, pollution meta and config, in `Plot/v3/results/<exp_id>.json`. Figures are drawn
  from that JSON only, as mean +- SD bands over seeds (descriptive) under `Plot/v3/<exp_id>/`.
- Classification and clustering helpers from v2 `D_data_analysis` are not carried over.
