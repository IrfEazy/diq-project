import heapq

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from library.E_plot_results import plot


def pollute_most_important_features(data, target, percentage, seed=2023):
    # Split the dataset into training and testing sets
    data_train, data_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=seed)

    # Train a regression model (Random Forest, for example)
    model = RandomForestRegressor(random_state=seed)
    model.fit(data_train, y_train)

    # Get feature importance scores
    importance = model.feature_importances_

    # Order features by importance
    feature_importance = sorted(zip(importance, range(data.shape[1])), reverse=True)

    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # Introduce a percentage of missing completely at random in only the first most important feature extracted before
    most_important_features = feature_importance[:3]

    for feature in most_important_features:
        data.iloc[np.random.choice(data.index, int(data.shape[0] * percentage)), feature[1]] = data.iloc[0, feature[1]]

    return data


def pollute_less_important_features(data, target, percentage, seed=2023):
    # Split the dataset into training and testing sets
    data_train, data_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=seed)

    # Train a regression model (Random Forest, for example)
    model = RandomForestRegressor(random_state=seed)
    model.fit(data_train, y_train)

    # Get feature importance scores
    importance = model.feature_importances_

    # Order features by importance
    feature_importance = sorted(zip(importance, range(data.shape[1])), reverse=True)

    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # Introduce a percentage of missing completely at random in only the first most important feature extracted before
    most_important_features = feature_importance[3:]

    for feature in most_important_features:
        data.iloc[np.random.choice(data.index, int(data.shape[0] * percentage)), feature[1]] = data.iloc[0, feature[1]]

    return data


def pollution_first_second_third_experiments(data, percentage, seed=2023):
    """
    Introduces distinctness to a randomly selected column in a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame to be modified.
    percentage : float
        The target distinctness percentage, ranging from 0 to 1.
        Zero means no change, and 1 means all values become unique.
    seed : int
        Seed for reproducibility. If provided, ensures consistent behavior.

    Returns
    -------
    pandas.DataFrame
        A modified DataFrame with introduced distinctness in one of its columns.

    Raises
    ------
    TypeError
        If the input data is not a pandas.DataFrame.

    Example
    -------
    >>> # Sample DataFrame
    >>> df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    >>> # Introduce 50% distinctness to a randomly selected column
    >>> modified_df = pollution_first_second_third_experiments(df, 0.5, seed=42)
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # If the percentage is 0, return the original DataFrame
    if percentage == 0:
        return data

    column = np.random.randint(data.shape[1])
    arr = data[column].to_numpy(copy=True)
    target_distinctness = 1 - percentage
    arr_len = len(arr)
    # Get the unique values from the original array
    n_unique_values = len(list(set(arr)))

    # Repeat the unique values to achieve the target distinctness
    while n_unique_values / arr_len > target_distinctness:
        # Chose a random value from arr
        random_value = np.random.choice(arr)
        # Chose a random index from repeated_values
        random_index = np.random.choice(len(arr))
        # Replace the value at the random index with the random value
        arr[random_index] = random_value

    # Replace the original column with the new one#
    data[column] = arr.copy()
    # Return the polluted DataFrame
    return data


def pollution_fourth_experiment(data, percentage, seed=2023):
    """
    Introduces distinctness to a subset of columns in a pandas DataFrame by modifying their values based on a random
    percentage extracted from a normal distribution.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame to be polluted with distinctness.
    percentage : float
        The central value around which the random percentages are generated. It represents the
        percentage of distinctness to be introduced. The actual percentages are sampled from a normal
        distribution centered around this value.
    seed : int
        Random seed for ensuring consistent behavior.

    Returns
    -------
    pandas.DataFrame
        The polluted DataFrame with introduced distinctness in selected columns.

    Raises
    ------
    TypeError
        If the input data is not a pandas DataFrame.

    Example
    -------
    >>> # Create a DataFrame
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

    >>> # Introduce distinctness to the DataFrame
    >>> polluted_df = pollution_fourth_experiment(df, seed=42, percentage=0.5)
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")

    # Select a random column
    columns = np.random.choice(data.columns, data.shape[1] // 2 + 1)

    # Introduce distinctness to the selected columns
    for column in columns:
        # Chose a random percentage extracted from a normal distribution centered in percentage and small variance
        percentage = np.random.normal(percentage, 0.1)

        # The value of percentage must be between 0 and 1
        if percentage < 0:
            percentage = 0
        elif percentage > 1:
            percentage = 1

        # Introduce distinctness to the selected column
        data[column] = pollution_first_second_third_experiments(pd.DataFrame(data[column].values), percentage)

    # Return the polluted DataFrame
    return data


def pollution_fifth_experiment(data, noise_variance=0, seed=2023):
    """
    Pollutes a pandas DataFrame with random noise.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame to be polluted.
    noise_variance : float
        The variance of the random noise to be added. Default is 0.
    seed : int
        Seed for the random number generator to ensure consistent behavior. Default is 2023.

    Returns
    -------
    pandas.DataFrame
        The polluted DataFrame with added random noise if noise_variance > 0,
        otherwise returns the original DataFrame.

    Raises
    ------
    TypeError
        If the input data is not a pandas DataFrame.

    Examples
    --------
    >>> # Create a sample DataFrame
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    >>> # Pollute the DataFrame with random noise
    >>> polluted_df = pollution_fifth_experiment(df, noise_variance=0.1, seed=42)
    """
    # Set the seed for the random number generator to ensure reproducible results
    np.random.seed(seed)

    # Validate the input data type. If it's not a pandas DataFrame, raise a TypeError
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # If the noise_variance is greater than 0, add random noise to the DataFrame
    if noise_variance > 0:
        # Generate a 2D array of random numbers with the same shape as the DataFrame
        noise = np.random.normal(0, noise_variance, data.shape)
        # Add the generated noise to the DataFrame. This operation is done element-wise,
        # so each element in the DataFrame has a different random number added to it.
        data += noise

    # Return the DataFrame after the noise addition operation
    return data


def pollution_sixth_experiment(data, number_of_variables, seed=2023):
    """
    Randomly transforms a column in a pandas DataFrame into categorical data by binning its values.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame to be modified.
    number_of_variables : int
        Number of categories to bin the selected column into.
    seed : int
        Seed for reproducibility. If not provided, randomness is not controlled.

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame with one randomly selected column transformed into categorical data.

    Raises
    ------
    TypeError
        If the input data is not a pandas DataFrame.

    Example
    -------
    >>> # Create a sample DataFrame
    >>> df = pd.DataFrame({'A': np.random.rand(5), 'B': np.random.rand(5)})

    >>> # Call the function to transform a column into categorical data
    >>> modified_df = pollution_sixth_experiment(df, number_of_variables=3, seed=42)
    """
    # Set the seed for the random number generator to ensure consistent results
    np.random.seed(seed)

    # Validate the input data type. If it's not a pandas DataFrame, raise a TypeError
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # Randomly select a column from the DataFrame
    column = data.sample(axis=1).columns[0]
    # Transform the selected column into categorical data by binning its values into 'number_of_variables' bins
    # The labels=False argument means that the bin identifiers are returned instead of the bin edges
    data[column] = pd.cut(data[column], number_of_variables, labels=False)
    # Return the DataFrame with the transformed column
    return data


def pollution_seventh_experiment(data, percentage, seed=2023):
    """
    Introduces random outliers in a Pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame to which outliers will be introduced.
    percentage : float
        The percentage of data points to be modified as outliers.
        Defaults to 0.01 (1% of data points).
    seed : int
        Seed for reproducibility. If provided, ensures consistent random outcomes.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with randomly introduced outliers.

    Raises
    ------
    TypeError
        If the input data is not a pandas DataFrame.

    Example
    -------
    >>> df = pd.DataFrame(np.random.rand(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    >>> data_with_outliers = pollution_seventh_experiment(df, percentage=0.05, seed=42)
    """
    # Set the seed for the random number generator to ensure consistent results
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # Calculate the number of outliers based on the percentage provided
    n_outliers = int(len(data) * percentage)
    # Randomly select indices for the outliers
    outliers_index = np.random.choice(data.index, size=n_outliers, replace=False)

    # For each outlier index
    for i in outliers_index:
        # Randomly select a feature (column)
        feature = np.random.randint(0, len(data.columns))
        # Replace the value at the selected index and feature with a random integer between 0 and 100
        data.iloc[i, feature] = np.random.randint(0, 100)

    # Return the DataFrame with introduced outliers
    return data


def pollution_eighth_experiment(data, percentage, seed=2023):
    """
    Introduces distinctness or 'pollution' to two randomly selected columns of a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame to be modified.
    percentage : float
        The percentage of distinctness to be introduced. Should be a float between 0 and 1.
    seed : int
        Seed for reproducibility. If provided, sets the random seed for consistent behavior.

    Returns
    -------
    pandas.DataFrame
        A modified DataFrame with introduced pollution.

    Raises
    ------
    TypeError
        If the input 'data' is not a pandas DataFrame.

    Example
    -------
    >>> # Create a sample DataFrame
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>> # Introduce 20% pollution to the DataFrame columns
    >>> polluted_df = pollution_eighth_experiment(df, percentage=0.2, seed=42)
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # If the percentage is 0, return the original DataFrame
    if percentage == 0:
        return data

    # Select two distinct columns from the DataFrame.
    # 'replace=False' ensures that the same column is not selected twice.
    column_high, column_low = np.random.choice(data.columns, 2, replace=False)
    # Call the 'pollution_first_second_third_experiments' function on the first selected column.
    # This function introduces distinctness to the column based on the provided percentage.
    data[column_high] = pollution_first_second_third_experiments(data[column_high], percentage)
    # Call the 'pollution_first_second_third_experiments' function on the second selected column.
    # This function introduces distinctness to the column based on the remaining percentage (1 - percentage).
    data[column_low] = pollution_first_second_third_experiments(data[column_low], 1 - percentage)
    # Return the modified DataFrame
    return data


def pollution_ninth_tenth_experiments(data, target, percentage, informative=True, seed=2023):
    # Initialize the random number generator with a specific seed to ensure reproducibility of results
    np.random.seed(seed)

    # Validate that the input data is a pandas DataFrame. If not, raise a TypeError
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # Divide the dataset into two parts: a training set and a testing set
    data_train, data_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=seed)
    # Instantiate and train a RandomForestRegressor model using the training data
    model = RandomForestRegressor(random_state=seed)
    model.fit(data_train, y_train)
    # Extract the importance of each feature from the trained model
    importance = model.feature_importances_

    # Depending on the 'informative' flag, select the top 3 most or least important features
    if informative:
        features_of_interest = heapq.nlargest(3, zip(importance, range(data.shape[1])))
    else:
        features_of_interest = heapq.nsmallest(3, zip(importance, range(data.shape[1])))

    # For each selected feature, apply the 'pollution_first_second_third_experiments' function
    # This function introduces a certain percentage of distinctness to the feature
    for _, feature in features_of_interest:
        data[feature] = pollution_first_second_third_experiments(pd.DataFrame(data[feature].values), percentage)

    # Return the DataFrame after the pollution process
    return data


def pollute_data_mcar(data, percent_incomplete=0.2, seed=2023):
    """
    Generate a random mask for incompleteness (MCAR) and introduce NaN values in the DataFrame according to the
    generated mask.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame.
    percent_incomplete : float
        The percentage of missing values to be introduced in the DataFrame.
    seed : int
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with NaN values introduced according to the generated mask.
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # Generate a random mask for incompleteness (MCAR)
    mask = np.random.rand(*data.shape) < percent_incomplete

    # MCAR experiment: introduce NaN values in the DataFrame according to the generated mask
    data[mask] = np.nan

    return data


def pollute_data_mcar_for_each_feature_in_incremental_way(data, seed=2023):
    """
    Generate a random mask for incompleteness (MCAR) and introduce NaN values in the DataFrame according to the
    generated mask. The NaN values are introduced for each feature in an incremental way. For example, for a DataFrame
    with 10 features, the first feature will have 10% of missing values, the second feature will have 20% of missing
    values, and so on.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame.
    seed : int
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    """

    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # Generate a random mask for incompleteness (MCAR)
    for col in data.columns:
        pollute_data_mcar_for_one_feature(data=data, feature=col, percent_incomplete=(col * 10))


def pollute_data_mcar_for_one_feature(data, feature, percent_incomplete=0.2, seed=2023):
    """
    Generate a random mask for incompleteness (MCAR) and introduce NaN values in the DataFrame according to the
    generated mask. The NaN values are introduced only for the chosen feature.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame.
    feature : int
        The feature for which NaN values are introduced, identified by the number of the column.
    percent_incomplete : float
        The percentage of missing values to be introduced in the DataFrame.
    seed : int
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with NaN values introduced according to the generated mask.
    """

    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # Generate a random mask for incompleteness (MCAR), introducing NaN values only for one feature
    mask = np.random.rand(*data[feature].shape) < percent_incomplete

    # MCAR experiment: introduce NaN values in the DataFrame according to the generated mask
    data.loc[mask, feature] = np.nan

    return data


def pollute_data_mar(data, feature_referenced, feature_dependent, seed=2023):
    """
    Generate a random mask for incompleteness (MAR) and introduce NaN values in the DataFrame according to the
    generated mask.
    Introduce missing values in a way that depends on the values of other specific features.
    Simulate a non-random incompleteness pattern based on the values of certain features.
    For example, only if a column has a certain range of values for a value then modify the correspondent completeness
    of another column referring to the value.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame.
    feature_referenced : int
        The feature for which we check a certain condition, to modify the completeness of another feature.
    feature_dependent : int
        The feature for which NaN values are introduced, identified by the number of the column.
    seed : int
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    """

    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # Simulate a non-random incompleteness pattern based on the values of certain features
    for element in data.index:
        if data[feature_referenced][element] > 100:
            data.loc[element, feature_dependent] = np.nan


def pollute_data_mnar(data, feature_mnar, seed=2023):
    """
    Generate a random mask for incompleteness (MNAR) and introduce NaN values in the DataFrame according to the
    generated mask.
    Introduce missing values in a way that depends on the values of specific features.
    Simulate a non-random incompleteness pattern based on the values of certain features.
    For example, only if a column has a certain range of values make the correspondent value NaN, creating in this way
    a non-random incompleteness pattern, depending on the value of the feature itself.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame.
    seed : int
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    feature_mnar : int
        The feature for which NaN values are introduced, identified by the number of the column.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with NaN values introduced according to the generated mask.
    feature_MNAR : int
        The feature for which NaN values are introduced, identified by the number of the column.
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # Simulate a non-random incompleteness pattern based on the values of certain features
    for element in data.index:
        if data[feature_mnar][element] > 50:
            if data[feature_mnar][element] < 150:
                data.loc[element, feature_mnar] = np.nan

    return data


def plot_results(x_axis_values, x_label, results, title, algorithms, cleaned_data=False):
    """
    This function generates the plots for the results of the experiments.

    Parameters
    ----------
    x_axis_values : list
        The list of x-axis values.
    x_label : str
        The label of the x-axis.
    results : list
        The list of results.
    title : str
        The title of the plot.
    algorithms : list
        The list of algorithms.
    cleaned_data : bool
        If True, the plots are generated for the cleaned data.
        If False, the plots are generated for the trial data.

    Returns
    -------
    None
    """
    if cleaned_data:
        title_plot = title + ' - Cleaned regression '
        title_zoom = title + ' - Zoomed cleaned regression '
    else:
        title_plot = title + ' - Trial regression '
        title_zoom = title + ' - Zoomed trial regression '

    plot(
        x_axis_values=x_axis_values, x_label=x_label, results=results, title=title_plot + 'performance',
        algorithms=algorithms, plot_type='performance'
    )
    zoomed_plot(
        x_axis_values=x_axis_values, x_label=x_label, results=results, title=title_zoom + 'performance',
        algorithms=algorithms, plot_type='performance', zoom=zoom_data(results, 'mean_perf')
    )

    plot(
        x_axis_values=x_axis_values, x_label=x_label, results=results, title=title_plot + 'distance train-test',
        algorithms=algorithms, plot_type='distance train-test'
    )
    zoomed_plot(
        x_axis_values=x_axis_values, x_label=x_label, results=results, title=title_zoom + 'distance train-test',
        algorithms=algorithms, plot_type='distance train-test', zoom=zoom_data(results, 'distance')
    )

    plot(
        x_axis_values=x_axis_values, x_label=x_label, results=results, title=title_plot + 'speed',
        algorithms=algorithms, plot_type='speed'
    )
    zoomed_plot(
        x_axis_values=x_axis_values, x_label=x_label, results=results, title=title_zoom + 'speed',
        algorithms=algorithms, plot_type='speed', zoom=zoom_data(results, 'speed')
    )


def zoomed_plot(x_axis_values, x_label, results, title, algorithms, plot_type, zoom):
    """
    This function generates the plots for the results of the experiments.
    It is possible to zoom the y-axis to have a better visualization of the results.

    Parameters
    ----------
    x_axis_values : list
        The list of x-axis values.
    x_label : str
        The label of the x-axis.
    results : list
        The list of results.
    title : str
        The title of the plot.
    algorithms : list
        The list of algorithms.
    plot_type : str
        The type of plot to be generated.
    zoom : dict
        The dictionary containing the zoom values for the y-axis.

    Returns
    -------
    None
    """
    plt.ylim(zoom['min'], zoom['max'])  # if you want to fix a limit for the y_axis
    plot(x_axis_values, x_label, results, title, algorithms, plot_type)


def zoom_data(results, attribute):
    """
    This function zooms the y-axis of the plots.
    It removes the values from attributes far away from the median of the values inside attributes.
    It returns a dictionary containing the maximum and minimum values of the zoomed y-axis.

    Parameters
    ----------
    results : list
        The list of results.
    attribute : str
        The attribute to be zoomed.

    Returns
    -------
    dict
        The dictionary containing the maximum and minimum values of the zoomed y-axis.
    """
    # Check if the input is a list, otherwise raise an error
    if not isinstance(results, list):
        raise TypeError("Input results must be a list")

    # Check if the input is a string, otherwise raise an error
    if not isinstance(attribute, str):
        raise TypeError("Input attribute must be a string")

    max_values = []
    min_values = []

    for algorithm in results:
        max_results_list = []
        min_results_list = []

        for result in algorithm:
            max_results_list.append(result[attribute])
            min_results_list.append(result[attribute])

        max_values.append(max(max_results_list))
        min_values.append(min(min_results_list))

    # Remove the values from attributes far away the median of the values inside attributes
    max_values = [x for x in max_values if (x < np.median(max_values) + 1.5 * np.std(max_values))]

    return {"max": np.max(max_values) + np.std(max_values), "min": np.min(min_values) - np.std(max_values)}
