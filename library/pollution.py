import collections
import heapq
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

_EPS = 1e-12


def _check_dataframe(data, name="Input data must be a pandas.DataFrame"):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(name)


def _feature_ranking(data, target, seed):
    """Return [(importance, column_position), ...] sorted from most to least important (Random Forest)."""
    data_train, data_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=seed)
    model = RandomForestRegressor(random_state=seed)
    model.fit(data_train, y_train)
    return sorted(zip(model.feature_importances_, range(data.shape[1])), reverse=True)


def _set_nan_at_positions(data, features, percentage):
    """Write NaN, without replacement, into int(n * percentage) random rows of each feature position."""
    out = data.copy()
    n_rows = out.shape[0]
    n_nan = int(n_rows * percentage)
    for feature in features:
        rows = np.random.choice(n_rows, n_nan, replace=False)
        column = out.iloc[:, feature].astype(float).to_numpy(copy=True)
        previous_nan = np.isnan(column)
        column[rows] = np.nan
        out.isetitem(feature, column)
        expected = int((previous_nan | np.isin(np.arange(n_rows), rows)).sum())
        assert int(out.iloc[:, feature].isna().sum()) == expected, "NaN count does not match the requested rule"
        assert out.iloc[:, feature].isna().iloc[rows].all()
    return out


def pollute_most_important_features(data, target, percentage, seed=2023):
    _check_dataframe(data)
    feature_importance = _feature_ranking(data, target, seed)

    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Missing completely at random in the 3 most important features
    return _set_nan_at_positions(data, [f for _, f in feature_importance[:3]], percentage)


def pollute_less_important_features(data, target, percentage, seed=2023):
    _check_dataframe(data)
    feature_importance = _feature_ranking(data, target, seed)

    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Missing completely at random in the LAST three features of the ranking
    return _set_nan_at_positions(data, [f for _, f in feature_importance[-3:]], percentage)


def _reduce_distinctness(values, percentage):
    """
    Report algorithm: copy a random existing value of the column into a random other cell until
    n_unique / n <= 1 - percentage. The number of unique values is tracked incrementally.
    Does not seed the random generator. Returns a new numpy array.
    """
    arr = np.asarray(values)
    n = len(arr)
    if percentage <= 0 or n == 0:
        return arr.copy()
    target = 1 - percentage
    vals = arr.tolist()
    counter = collections.Counter(vals)
    start_unique = len(counter)

    if 1 / n > target + _EPS:
        # The target requires fewer than one distinct value: clip to exactly one.
        warnings.warn("Target distinctness below 1/n: the column is clipped to exactly 1 distinct value")
        out = np.full(n, vals[np.random.randint(n)], dtype=arr.dtype)
        assert len(set(out.tolist())) == 1
        return out

    while len(counter) / n > target + _EPS:
        value = vals[np.random.randint(n)]
        index = np.random.randint(n)
        old = vals[index]
        if old == value:
            continue
        counter[old] -= 1
        if counter[old] == 0:
            del counter[old]
        counter[value] += 1
        vals[index] = value

    achieved = len(counter) / n
    assert achieved <= target + _EPS, "distinctness above the target"
    if len(counter) < start_unique:
        assert achieved >= target - 1 / n - _EPS, "overshot the target distinctness"
    out = np.asarray(vals, dtype=arr.dtype)
    assert len(np.unique(out)) == len(counter)
    return out


def pollution_first_second_third_experiments(data, percentage, seed=2023):
    """
    Introduces non-distinctness in a randomly selected column of a pandas DataFrame (report algorithm): a random
    value of the column is pasted into a random other cell until n_unique / n <= 1 - percentage.

    Returns a new DataFrame; the input is not modified. If percentage requires fewer than one distinct value
    (e.g. 1.0) the column is clipped to exactly one distinct value and a warning is emitted.
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    _check_dataframe(data)

    out = data.copy()
    # If the percentage is 0, nothing changes
    if percentage == 0:
        return out

    position = np.random.randint(data.shape[1])
    out.isetitem(position, _reduce_distinctness(data.iloc[:, position].to_numpy(), percentage))
    return out


def pollution_fourth_experiment(data, percentage, seed=2023):
    """
    Introduces non-distinctness to a subset of columns (without replacement), each one with a percentage drawn from
    N(percentage, 0.1) clipped to [0, 1]. Returns a new DataFrame.
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    _check_dataframe(data, "Input data must be a pandas DataFrame")

    out = data.copy()
    columns = np.random.choice(data.columns, data.shape[1] // 2 + 1, replace=False)

    for column in columns:
        # Always drawn around the ORIGINAL percentage (no random walk)
        column_percentage = float(np.clip(np.random.normal(percentage, 0.1), 0, 1))
        position = data.columns.get_loc(column)
        out.isetitem(position, _reduce_distinctness(data.iloc[:, position].to_numpy(), column_percentage))

    return out


def pollution_fifth_experiment(data, noise_variance=0, seed=2023):
    """
    Adds Gaussian noise with variance `noise_variance` (standard deviation sqrt(noise_variance)) to every cell.
    Returns a new DataFrame.
    """
    # Set the seed for the random number generator to ensure reproducible results
    np.random.seed(seed)

    _check_dataframe(data)

    out = data.copy()
    if noise_variance > 0:
        noise = np.random.normal(0, np.sqrt(noise_variance), data.shape)
        out = out + noise

    return out


def pollution_sixth_experiment(data, number_of_variables, seed=2023):
    """
    Randomly transforms a column into categorical data by binning its values. Returns a new DataFrame.
    """
    # Set the seed for the random number generator to ensure consistent results
    np.random.seed(seed)

    _check_dataframe(data)

    out = data.copy()
    column = out.sample(axis=1).columns[0]
    out[column] = pd.cut(out[column], number_of_variables, labels=False)
    return out


def pollution_seventh_experiment(data, percentage, seed=2023):
    """
    Introduces random outliers (random integers in [0, 100)) in randomly chosen cells. Rows and columns are
    addressed by position. Returns a new DataFrame.
    """
    # Set the seed for the random number generator to ensure consistent results
    np.random.seed(seed)

    _check_dataframe(data)

    out = data.copy()
    n_outliers = int(len(out) * percentage)
    outliers_rows = np.random.choice(len(out), size=n_outliers, replace=False)

    for i in outliers_rows:
        feature = np.random.randint(0, len(out.columns))
        out.iloc[i, feature] = np.random.randint(0, 100)

    return out


def pollution_eighth_experiment(data, percentage, seed=2023):
    """
    Reduces the distinctness of two distinct random columns: the first with `percentage`, the second with
    `1 - percentage`. Returns a new DataFrame.
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    _check_dataframe(data)

    out = data.copy()
    if percentage == 0:
        return out

    column_high, column_low = np.random.choice(data.columns, 2, replace=False)
    for column, column_percentage in ((column_high, percentage), (column_low, 1 - percentage)):
        position = data.columns.get_loc(column)
        out.isetitem(position, _reduce_distinctness(data.iloc[:, position].to_numpy(), column_percentage))
    return out


def pollution_ninth_tenth_experiments(data, target, percentage, informative=True, seed=2023):
    # Initialize the random number generator with a specific seed to ensure reproducibility of results
    np.random.seed(seed)

    _check_dataframe(data)

    data_train, data_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=seed)
    model = RandomForestRegressor(random_state=seed)
    model.fit(data_train, y_train)
    importance = model.feature_importances_

    # Depending on the 'informative' flag, select the top 3 most or least important features
    if informative:
        features_of_interest = heapq.nlargest(3, zip(importance, range(data.shape[1])))
    else:
        features_of_interest = heapq.nsmallest(3, zip(importance, range(data.shape[1])))

    out = data.copy()
    for _, feature in features_of_interest:
        out.isetitem(feature, _reduce_distinctness(data.iloc[:, feature].to_numpy(), percentage))

    return out


def pollute_data_mcar(data, percent_incomplete=0.2, seed=2023):
    """
    Generate a random mask (MCAR) and introduce NaN values in the DataFrame (in place; also returned).
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    _check_dataframe(data)

    mask = np.random.rand(*data.shape) < percent_incomplete
    previous_nan = data.isna().to_numpy()

    for j in range(data.shape[1]):
        column = data.iloc[:, j].astype(float).to_numpy(copy=True)
        column[mask[:, j]] = np.nan
        data.isetitem(j, column)

    assert (data.isna().to_numpy() == (previous_nan | mask)).all(), "NaN pattern does not match the MCAR mask"
    return data


def pollute_data_mcar_for_each_feature_in_incremental_way(data, seed=2023):
    """
    MCAR in an incremental way: the column in position k gets a fraction k * 0.1 of missing values
    (first column 0%, second 10%, ...). In place; also returned.
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    _check_dataframe(data)

    for col_index, col in enumerate(data.columns):
        pollute_data_mcar_for_one_feature(data=data, feature=col, percent_incomplete=col_index * 0.1, seed=seed)

    return data


def pollute_data_mcar_for_one_feature(data, feature, percent_incomplete=0.2, seed=2023):
    """
    MCAR for the single column labelled `feature`. In place; also returned.
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    _check_dataframe(data)

    mask = np.random.rand(data.shape[0]) < percent_incomplete
    previous_nan = data[feature].isna().to_numpy()
    position = data.columns.get_loc(feature)

    column = data.iloc[:, position].astype(float).to_numpy(copy=True)
    column[mask] = np.nan
    data.isetitem(position, column)

    assert (data[feature].isna().to_numpy() == (previous_nan | mask)).all(), "NaN pattern does not match the mask"
    return data


def pollute_data_mar(data, feature_referenced, feature_dependent, seed=2023, q=0.8):
    """
    MAR: NaN in `feature_dependent` where `feature_referenced` is above its q-quantile (default 0.8).
    In place; also returned.
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    _check_dataframe(data)

    threshold = data[feature_referenced].quantile(q)
    mask = (data[feature_referenced] > threshold).to_numpy()
    previous_nan = data[feature_dependent].isna().to_numpy()

    position = data.columns.get_loc(feature_dependent)
    column = data.iloc[:, position].astype(float).to_numpy(copy=True)
    column[mask] = np.nan
    data.isetitem(position, column)

    assert (data[feature_dependent].isna().to_numpy() == (previous_nan | mask)).all(), "MAR NaN pattern mismatch"
    return data


def pollute_data_mnar(data, feature_mnar, seed=2023, q_low=0.25, q_high=0.75):
    """
    MNAR: NaN in `feature_mnar` where its own value lies in the [q_low, q_high] quantile band (default 0.25-0.75).
    In place; also returned.
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    _check_dataframe(data)

    low, high = data[feature_mnar].quantile([q_low, q_high])
    values = data[feature_mnar]
    mask = ((values >= low) & (values <= high)).to_numpy()
    previous_nan = values.isna().to_numpy()

    position = data.columns.get_loc(feature_mnar)
    column = data.iloc[:, position].astype(float).to_numpy(copy=True)
    column[mask] = np.nan
    data.isetitem(position, column)

    assert (data[feature_mnar].isna().to_numpy() == (previous_nan | mask)).all(), "MNAR NaN pattern mismatch"
    return data

