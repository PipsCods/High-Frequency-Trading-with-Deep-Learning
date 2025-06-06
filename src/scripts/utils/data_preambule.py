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
    window_size = 6
    
    grouped = df.groupby(symbol_col, group_keys=False)

    # Moving average and std (using only current and past data)
    df['ma_1h'] = grouped[return_col].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    df['std_1h'] = grouped[return_col].transform(lambda x: x.rolling(window=window_size, min_periods=1).std())

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

def compute_hf_features_multiwindow(df, return_col='return', windows=[3, 6]):
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
            df[return_col].rolling(window).apply(lambda x: np.sum(x ** 2), raw=True)
        )

        # Rolling std dev
        df[f'rolling_std{suffix}'] = df[return_col].rolling(window).std()

        # Rolling mean
        df[f'rolling_mean{suffix}'] = df[return_col].rolling(window).mean()

        # Rolling Sharpe
        df[f'rolling_sharpe{suffix}'] = df[f'rolling_mean{suffix}'] / (df[f'rolling_std{suffix}'] + 1e-6)

        # Rolling skewness
        df[f'rolling_skew{suffix}'] = df[return_col].rolling(window).skew()

        # Rolling kurtosis
        df[f'rolling_kurt{suffix}'] = df[return_col].rolling(window).kurt()

    return df

def prepare_hf_data(raw_df, name_of_timestamp_column, name_of_symbol_column):
    """
    Processes a high-frequency financial dataset to prepare it for training and testing.

    Args:
        raw_df: DataFrame containing the raw data
        name_of_timestamp_column: Name of the column containing the timestamps
        name_of_symbol_column: Name of the column containing the stock symbols

    Returns:
        - Cleaned DataFrame (indexed by Timestamp)
        - List of categorical basic feature names ['symbol', 'day', 'day_name', 'hour', 'minute']
        - List of categorical feature names (additional)
        - List of continuous feature names
        - List of indices (positions) of categorical features in the DataFrame columns (including basic and additional)
        - List of indices (positions) of continuous features in the DataFrame columns
    """

    df = raw_df.copy()

    # Verify that the columns exist in the DataFrame
    if not all(col in raw_df.columns for col in
               [name_of_timestamp_column, name_of_symbol_column]):
        raise ValueError("The specified columns do not exist in the DataFrame.")

    # Convert and set timestamp
    df['timestamp'] = pd.to_datetime(df[name_of_timestamp_column])
    df.set_index('timestamp', inplace=True)
    df.drop(columns=[name_of_timestamp_column], inplace=True)

    # Rename symbol column if needed
    if 'symbol' not in df.columns:
        df.rename(columns={name_of_symbol_column: 'symbol'}, inplace=True)

    # if 'mid_price' not in df.columns:
    #     df.rename(columns={name_of_mid_price_column: 'mid_price'}, inplace=True)

    # Extract time-based features
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_name'] = df.index.day_name()

    # Fill missing values from return computation
    df.fillna(0, inplace=True)

    # Sort chronologically by symbol
    df = df.sort_values(by=['symbol', df.index.name])
    
    df= augment_features(df, symbol_col= "symbol", return_col= "return")

    # Categorical features (known structure)
    basic_cat_features = ['symbol', 'day', 'day_name', 'hour', 'minute']
    cat_features = [col for col in df.select_dtypes(include=['object']).columns if col not in basic_cat_features and col != 'timestamp']

    # Detect continuous features automatically (besides return)
    base = ['return']
    cont_features = [col for col in df.select_dtypes(include=['float64', 'float32', 'int']).columns
                 if col not in base + basic_cat_features]
    df[cont_features] = df[cont_features].astype('float32') # transform to float32 to save memory

    # Reindex column order
    ordered_cols = basic_cat_features + cat_features + cont_features + base
    df = df.reindex(columns=ordered_cols)

    # Concatenate cont_features and base
    cont_features = cont_features + base

    # Compute indices for categorical and continuous features in the reordered DataFrame
    cat_all_features = basic_cat_features + cat_features
    cat_feat_positions = [df.columns.get_loc(col) for col in cat_all_features]
    cont_feat_positions = [df.columns.get_loc(col) for col in cont_features]

    return df, basic_cat_features, cat_features, cont_features, cat_feat_positions, cont_feat_positions



