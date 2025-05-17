import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

"""
Tools for splitting the dataset into train and test sets
"""

def rolling_sampling(data: pd.DataFrame, train_window: int = 6):
    """
    Generates rolling window splits for time series training and testing.

    This function creates temporal splits using a rolling window approach for training
    and testing data. It works with a timestamp-indexed Pandas DataFrame and a fixed
    window size.

    Args:
        data: Input Pandas DataFrame with temporal index.
        train_window: Number of time intervals for the training window.
                     Represents how many historical data points are used
                     for training. Default: 6 intervals.

    Returns:
        A tuple containing two lists:
        - splitting_train: List of temporal indices for each training window
        - splitting_test: List of temporal indices for each corresponding test point

    Example:
        >>> df = pd.DataFrame(...) # DataFrame with temporal index
        >>> train_splits, test_splits = rolling_sampling(df, train_window=10)
    """

    list_of_dates = data.index.unique()
    number_of_splits = len(list_of_dates) - train_window

    splitting_train = []  # Initialize as empty lists
    splitting_test = []

    for split in range(number_of_splits):
        # Append date ranges as lists to splitting_train
        splitting_train.append(list_of_dates[split : split + train_window])
        # Append single dates to splitting_test
        splitting_test.append(list_of_dates[split + train_window])

    return splitting_train, splitting_test

"""
Functions for regressing variables using MLR, RIDGE
"""

def run_regression_for_symbol(symbol, group, features, train_window : int = 12):

    group = group.sort_values("timestamp")
    splitting_train, splitting_test = rolling_sampling(group, train_window)

    results = []

    for slice_train, slice_test in zip(splitting_train, splitting_test):
        window_data = group.loc[slice_train.union([slice_test])]

        X = window_data[features]
        y = window_data['return']

        X_train = X.iloc[:-1]
        X_test = X.iloc[[-1]]

        y_train = y.iloc[:-1].values
        y_test = y.iloc[-1]

        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        results.append({
            'symbol': symbol,
            'timestamp': slice_test[1],
            'y_true': y_test,
            'y_pred': y_pred[0],
            'market_timing_returns': y_pred[0] * y_test,
        })

    return results


def run_ridge_for_symbol(symbol, group, features, train_window, shrinkage_list:list):
    group = group.sort_values("timestamp")
    splitting_train, splitting_test = rolling_sampling(group, train_window)

    results = []

    for slice_train, slice_test in zip(splitting_train, splitting_test):
        window_data = group.loc[slice_train.union([slice_test])]
        X = window_data[features]
        y = window_data['return']

        X_train = X.iloc[:-1]
        X_test = X.iloc[[-1]]
        y_train = y.iloc[:-1]
        y_test = y.iloc[-1]

        for alpha in shrinkage_list:
            clf = Ridge(alpha=alpha)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            results.append({
                'symbol': symbol,
                'timestamp': slice_test[1],
                'alpha': alpha,
                'y_true': y_test,
                'y_pred': y_pred[0],
                'market_timing_returns': y_pred[0] * y_test,
            })

    return results

