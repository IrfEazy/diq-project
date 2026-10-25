# library/v2 vs v1: intentional differences

Findings referenced (F*, C*, D*) come from the run investigation `investigation/theory.md` and
`investigation/differential.md` (GP root cause) of run 20261024T180844Z-0d73a8e3.

- `my_functions.pollution_first_second_third_experiments` (v1 `library/my_functions.py:72-137`, loop at ~l.128): v1 never
  recomputes `n_unique_values` so it loops forever. v2 follows the report ("randomly selected a value ... pasted it into
  another cell ... until the distinctness index ... reached the target", Project.tex ~l.252) with incremental unique
  counting (`collections.Counter`), asserts the stop is at the first crossing, clips to 1 distinct value (+warning) when
  the target needs fewer than 1 (p=1), and returns a copy. Also selects the column by position (v1 `data[column]` used a
  label). Reason: theory.md F8/F9 (target never hit, old algorithm).
- `pollution_fourth_experiment` (v1 :139-200): columns chosen without replacement and addressed by name; percentage is
  drawn from the original value each time (v1 reassigned `percentage`, a random walk); no `pd.DataFrame(values)` trick.
  Report: "percentages ... similar, although not identical".
- `pollution_fifth_experiment` (v1 :~200-250): `np.random.normal(0, sqrt(noise_variance))` (v1 passed the variance as std);
  report says "noise's variance (0.00625 to 0.0625)". Returns a copy instead of `data +=`.
- `pollution_sixth_experiment`, `pollution_seventh_experiment`: same semantics, but copy, no mutation of the input; seventh
  addresses rows positionally (`iloc`) rather than index labels.
- `pollution_eighth_experiment` (v1 :351-435): v1 passed Series into a function expecting a DataFrame (TypeError). v2
  works on arrays of two distinct columns: first with `percentage`, second with `1 - percentage` (v1 intent; report
  "kept it high, albeit slightly lower").
- `pollution_ninth_tenth_experiments`: distinctness applied per selected feature column (v1 passed one-column frames and
  assigned wrongly); returns a copy.
- `pollute_data_mcar_for_each_feature_in_incremental_way` (v1 :~490): `percent_incomplete = col_index * 0.1` (v1 used
  `col * 10`, i.e. 100% NaN for every column but the first; theory.md exp. 4). Report: "first feature 10%, second 20%".
- `pollute_data_mar` (v1 :~540): NaN where `feature_referenced` > its q-quantile (param `q`, default 0.8) and returns
  `data`. v1 used a `> 100` threshold that never fires on N(0,1) data and returned None. Reason: theory.md exp. 5.
- `pollute_data_mnar` (v1 :~585): NaN where the feature's own value is inside the `[q_low, q_high]` quantile band
  (defaults 0.25/0.75); v1 used `50..150`, never firing on N(0,1) data. Reason: theory.md exp. 6.
- `pollute_most_important_features` (v1 :12-45): writes `np.nan` (v1 wrote `data.iloc[0, col]`, a constant) to rows sampled
  WITHOUT replacement; returns a copy. Report exp. 7: "incomplete values".
- `pollute_less_important_features` (v1 :48-70): uses the last three features of the ranking (`[-3:]`; v1 `[3:]`) and
  writes `np.nan`. Report exp. 8: "the last three features of the ranking".
- All completeness/distinctness polluters assert that the achieved NaN counts / distinct counts match the rule. All keep
  `np.random.seed(seed)` on entry. MCAR/MAR/MNAR/incremental polluters still mutate in place (the v1 notebooks call them
  without using the return value) and also return `data`; assignment uses explicit column replacement, not chained
  assignment.
- `D_data_analysis.regression` / `distance_measurement` (v1 :~96, :~190): `GaussianProcessRegressor(kernel=
  ConstantKernel(1.0)*RBF(1.0)+WhiteKernel(1e-2), normalize_y=True, random_state=seed)` instead of the library default,
  which changed in scikit-learn 1.9 (differential.md, GP root cause) and made results environment dependent.
  Scaler + `nan_to_num` + ShuffleSplit unchanged.
- `E_plot_results.your_path = "Plot/v2/"`, created with `os.makedirs(exist_ok=True)` before each save (v1 required a
  pre-existing `Plot/`). Titles and file names are unchanged.
- `my_functions.plot_results` / `zoomed_plot` also write `Plot/v2/results/<title>.json`
  (`version, title, x_label, x_axis, algorithms, results`), so figures can be rebuilt from data.
- Module imports are `library.v2.*` only; `selfcheck.py` and this file are new.
