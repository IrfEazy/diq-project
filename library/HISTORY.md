# Differences from the original implementation

The original code (the first study: `library/my_functions.py`, `library/D_data_analysis.py`, `library/E_plot_results.py`,
`library/A_data_collection.py` and the first notebooks) is available at the local git tag `pre-flatten`
(`git show pre-flatten:library/my_functions.py`). That tag also holds the intermediate corrected-polluter variant and the
evaluation-protocol variant that this directory flattens. Every result-changing difference is listed here, in two parts:
what was wrong in the original code, then what the evaluation protocol changed.

Findings referenced (F*, C*, D*) come from an investigation of the original code whose findings are summarised in
this document.

## Polluters (`pollution.py`) versus the original `my_functions.py`

- `pollution_first_second_third_experiments`: the original never recomputed `n_unique_values`, so its loop never
  ended. The polluter now follows the report ("randomly selected a value ... pasted it into another cell ... until the
  distinctness index ... reached the target") with incremental unique counting (`collections.Counter`), asserts the stop
  is at the first crossing, clips to 1 distinct value (with a warning) when the target needs fewer than 1 (p = 1), and
  returns a copy. It selects the column by position (the original used a label). Reason: theory.md F8/F9 (target
  never hit, old algorithm).
- `pollution_fourth_experiment`: columns are chosen without replacement and addressed by name; the percentage is drawn
  from the original value each time (the original reassigned `percentage`, a random walk); no `pd.DataFrame(values)` trick.
  Report: "percentages ... similar, although not identical".
- `pollution_fifth_experiment`: `np.random.normal(0, sqrt(noise_variance))` (the original passed the variance as the
  standard deviation); the report says "noise's variance (0.00625 to 0.0625)". Returns a copy instead of `data +=`.
- `pollution_sixth_experiment`, `pollution_seventh_experiment`: same semantics, but they copy and do not mutate the input;
  the seventh addresses rows positionally (`iloc`) rather than by index label.
- `pollution_eighth_experiment`: the original passed Series into a function expecting a DataFrame (TypeError). It now
  works on arrays of two distinct columns: the first with `percentage`, the second with `1 - percentage` (the original
  intent; report "kept it high, albeit slightly lower").
- `pollution_ninth_tenth_experiments`: distinctness is applied per selected feature column (the original passed
  one-column frames and assigned wrongly); returns a copy.
- `pollute_data_mcar_for_each_feature_in_incremental_way`: `percent_incomplete = col_index * 0.1` (the original used
  `col * 10`, i.e. 100% NaN for every column but the first; theory.md exp. 4). Report: "first feature 10%, second 20%".
- `pollute_data_mar`: NaN where `feature_referenced` is above its q-quantile (parameter `q`, default 0.8), and returns
  `data`. The original used a `> 100` threshold that never fires on N(0,1) data and returned None. Reason: theory.md exp. 5.
- `pollute_data_mnar`: NaN where the feature's own value is inside the `[q_low, q_high]` quantile band (defaults
  0.25/0.75); the original used `50..150`, which never fires on N(0,1) data. Reason: theory.md exp. 6.
- `pollute_most_important_features`: writes `np.nan` (the original wrote `data.iloc[0, col]`, a constant) to rows sampled
  WITHOUT replacement; returns a copy. Report exp. 7: "incomplete values".
- `pollute_less_important_features`: uses the last three features of the ranking (`[-3:]`; the original used `[3:]`)
  and writes `np.nan`. Report exp. 8: "the last three features of the ranking".
- All completeness and distinctness polluters assert that the achieved NaN counts or distinct counts match the rule. All
  keep `np.random.seed(seed)` on entry. The MCAR/MAR/MNAR/incremental polluters still mutate in place (the original
  notebooks call them without using the return value) and also return `data`; assignment uses explicit column
  replacement, not chained assignment.
- The plotting helpers of the original `my_functions.py` (`plot_results`, `zoomed_plot`) and the modules
  `E_plot_results.py`, `D_data_analysis.py` and `A_data_collection.py` are not carried over; figures are built by
  `plots.py` from the results JSON.

## Evaluation protocol (`evaluation.py`, `runner.py`, `data.py`, `plots.py`) versus the original `D_data_analysis.py`

These follow the design of run 20261024T180844Z-0d73a8e3 (methodology consult: fold-fitted preprocessing, group-aware
duplicate splits, repetitions over seeds).

- Gaussian process: the original used the scikit-learn default kernel, which changed in scikit-learn 1.9 and made
  results environment dependent (differential.md). The corrected variant used `ConstantKernel(1.0) * RBF(1.0) +
  WhiteKernel(1e-2)` with the default length-scale bounds `(1e-5, 1e5)` and no restarts; on heavily tied data this
  collapsed to the mean predictor. Now `length_scale_bounds=(0.1, 1e3)` and `n_restarts_optimizer=1`, and bound hits are
  recorded in each fold's `status`.
- Preprocessing: the original `regression()` fitted `StandardScaler` on all rows, zero-filled the NaN in the scaled
  matrix (an implicit global-mean imputation that saw the test rows), and then cross-validated. Now every imputer, the
  feature scaler and a target scaler are fitted per training fold inside
  `TransformedTargetRegressor(Pipeline(...))`. Missing values are handled by a named imputer or rejected (imputer `None`
  raises on NaN), and nothing is silently zero-filled.
- Imputation is an explicit experimental condition: `mean` (the baseline), `median`, `most_frequent`, `std_constant` (the
  labelled naive constant of the original study), `knn` (scaled before KNN), `iterative`, `mean_indicator`, and `native`
  NaN handling with the new `HGBRegressor` reference model. `clean` is an unpolluted reference.
- Metrics: the original reported cross-validated RMSE in raw target units, plus a train/test "distance" from a separate
  `train_test_split` refit. Now nRMSE (RMSE / std of the clean noisy target) and R2, for test and train, both from the
  same fitted fold model, together with the fit time per fold.
- Splits: the original used `ShuffleSplit(8, 0.3)` with one seed (2023). Now `ShuffleSplit(5, 0.3)` and repetition seeds;
  each seed is expanded through `SeedSequence` into independent data, pollution and split seeds. The design is paired
  across conditions and models, and the results do not depend on `n_jobs`.
- Data: the original notebooks generated noiseless targets (`make_regression(noise=0.0)`). `make_regression_data` adds
  SNR-based noise (std = 0.2 * std(signal)) from a seeded `Generator`.
- `SVMRegressor` has an explicit `random_state`, `max_iter=100000` and `dual='auto'`. `MLPRegressor` has `max_iter=2000`
  and `early_stopping=True`. Convergence warnings are recorded per fold in `status` rather than suppressed through the
  private `sklearn.utils._testing.ignore_warnings`.
- New duplicate-records experiment: `inject_duplicate_rows`, `deduplicate`, and `evaluate_duplicates` with the
  `duplicated` / `deduplicated` conditions (group-aware splits, one scored row per held-out source) and `naive_split` (a
  labelled leakage demonstration).
- Results: the original stored figures only (and later one aggregate per x and model). Now every fold record, with
  seeds, hashes, pollution meta and config, is stored in `Plot/results/<exp_id>.json` (`schema_version` 1). Figures are
  drawn from that JSON only, as mean +- SD bands over seeds (descriptive) under `Plot/<exp_id>/`.
- Classification and clustering helpers of the original `D_data_analysis` are not carried over.
- `make_data` may take `(seed, x)` for sample-size axes, in which case `clean` is fitted per x. With `make_data(seed)`,
  `clean` is reused across x, records carry `reused_from_x`, and a `pollute` that changes the row count raises instead of
  silently comparing different datasets. `summarise` reports `n_null`, warns on null metrics and plots annotate them.
  `status` only reflects convergence warnings and GP bound hits; other warning categories go to a separate `warnings`
  field. The runner restores the global numpy RNG state around `pollute`.

## Flattening

The repository was later flattened to a single unversioned variant. That change moved the modules to `library/`, put the
polluters in `pollution.py`, renamed the notebooks, changed the default output directory to `Plot/`, and replaced the
payload field `version` by `schema_version: 1`. Results are unchanged: a quick run of both notebooks was compared
record by record with the stored full results: data hashes, test hashes, metrics, status and sizes are identical. The only
difference was the extra `requested_not_distinct_low` key in the `meta` of the polluted D8 records. The stored D8
results were then regenerated with the current code: hashes and metrics are identical and the records now include that
meta key.
