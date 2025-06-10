import pandas as pd
import numpy as np

def filter_stocks_with_full_coverage(df, timestamp_col='timestamp', symbol_col='symbol'):
    """
    Filter the DataFrame to keep only stocks with full mid_price data from start to end.

    Args:
        df: pandas DataFrame with at least 'timestamp' and 'symbol' columns.
        timestamp_col: name of the timestamp column.
        symbol_col: name of the stock symbol column.

    Returns:
        Filtered DataFrame with only stocks having full time coverage.
    """
    # Get the complete sorted list of unique timestamps expected
    all_timestamps = df[timestamp_col].sort_values().unique()

    # Compute expected count of timestamps (length of full series)
    expected_count = len(all_timestamps)

    # Count number of timestamps per stock
    counts_per_stock = df.groupby(symbol_col)[timestamp_col].nunique()

    # Select stocks with complete coverage
    stocks_with_full_coverage = counts_per_stock[counts_per_stock == expected_count].index

    # Filter original df
    filtered_df = df[df[symbol_col].isin(stocks_with_full_coverage)].copy()

    return filtered_df


def augment_features(df: pd.DataFrame, symbol_col: str, return_col: str) -> pd.DataFrame:

    # df = df.copy()
    # df[time_col] = pd.to_datetime(df[time_col])
    # df = df.sort_values(by=[symbol_col, time_col])

    #window size for 1 hour (6 * 10min)
    #window_size = 6
    #grouped = df.groupby(symbol_col, group_keys=False)

    # Moving average and std (using only current and past data)
    #df['ma_1h'] = grouped[return_col].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    #df['std_1h'] = grouped[return_col].transform(lambda x: x.rolling(window=window_size, min_periods=1).std())

    # Non-linear transformations of returns
    df['return_cos'] = np.cos(df[return_col])
    df['return_sin'] = np.sin(df[return_col])
    df['return_tanh'] = np.tanh(df[return_col])
    df['return_exp'] = np.exp(df[return_col].clip(upper=10))  # avoid overflow
    df['return_sign'] = np.sign(df[return_col])
    df['return_square'] = df[return_col] ** 2
    df['return_log1p'] = np.sign(df[return_col]) * np.log1p(np.abs(df[return_col]))  # preserve sign
    df['return_relu'] = np.maximum(0, df[return_col])

    return df

def compute_hf_features_multiwindow(df, return_col='return', windows=[6,12]):
    """
    Compute high-frequency features across multiple rolling windows.

    Parameters:
        df (pd.DataFrame): DataFrame with a return column.
        return_col (str): Name of the return column.
        windows (list): List of window sizes to apply.

    Returns:
        pd.DataFrame: DataFrame with new columns for each window size.
    """
    df = df.copy()
    for window in windows:
        suffix = f"_{window}"

        # Momentum
        df[f'momentum{suffix}'] = df[return_col].rolling(window).sum()

        # Cumulative volatility (realized volatility)
        df[f'cum_volatility{suffix}'] = np.sqrt(
            df[return_col].rolling(window).apply(lambda x: np.sum(x ** 2), raw=True).fillna(0)
        )

        # Rolling std dev
        df[f'rolling_std{suffix}'] = df[return_col].rolling(window).std().fillna(0)

        # Rolling mean
        df[f'rolling_mean{suffix}'] = df[return_col].rolling(window).mean().fillna(0)

        # Rolling Sharpe
        df[f'rolling_sharpe{suffix}'] = df[f'rolling_mean{suffix}'] / (df[f'rolling_std{suffix}'] + 1e-6).fillna(0)

        # Rolling skewness
        df[f'rolling_skew{suffix}'] = df[return_col].rolling(window).skew().fillna(0)

        # Rolling kurtosis
        df[f'rolling_kurt{suffix}'] = df[return_col].rolling(window).kurt().fillna(0)

    return df

import pandas as pd

def prepare_hf_data(raw_df, name_of_timestamp_column, name_of_symbol_column):
    """
    Processes a high-frequency financial dataset to prepare it for training and testing.

    Args:
        raw_df (pd.DataFrame): Raw input DataFrame.
        name_of_timestamp_column (str): Column name for timestamps.
        name_of_symbol_column (str): Column name for stock symbols.

    Returns:
        pd.DataFrame: Cleaned and chronologically sorted DataFrame indexed by timestamp.
    """

    df = raw_df.copy()

    # Validate required columns
    required_columns = [name_of_timestamp_column, name_of_symbol_column]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"The following required columns are missing from the DataFrame: {missing}")

    # Convert timestamp column to datetime and set as index
    df['timestamp'] = pd.to_datetime(df[name_of_timestamp_column], errors='coerce')
    if df['timestamp'].isna().any():
        raise ValueError("Some timestamp values could not be converted to datetime.")

    df.set_index('timestamp', inplace=True)
    df.drop(columns=[name_of_timestamp_column], inplace=True)

    # Standardize symbol column name
    if 'symbol' not in df.columns:
        df.rename(columns={name_of_symbol_column: 'symbol'}, inplace=True)

    # Extract time-based features (as integers first, then convert to category)
    df['day'] = df.index.day.astype('category')
    df['hour'] = df.index.hour.astype('category')
    df['minute'] = df.index.minute.astype('category')
    df['day_name'] = df.index.day_name()

    # Ensure proper sorting: first by symbol, then chronologically
    df.sort_values(by=['symbol', df.index.name], inplace=True)
    df.drop(columns=["ALL_EX", "SUM_DELTA"], inplace=True)

    return df




