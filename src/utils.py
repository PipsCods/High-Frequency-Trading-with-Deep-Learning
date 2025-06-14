import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a DataFrame.

    Args:
        file_path: The path to the CSV file.

    Returns:
        A pandas DataFrame containing the loaded data.
    """
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    

def remove_outlier_returns(df: pd.DataFrame, column: str = 'RETURN_NoOVERNIGHT', lower_percentile: float = 0.01, upper_percentile: float = 0.99):
    """
    Removes outlier returns from the specified column in the DataFrame by clipping values
    to the specified lower and upper percentiles.
    """
    if df.empty or column not in df.columns:
        print(f"Warning: Empty DataFrame or '{column}' not found.")
        return df
    if not (0 <= lower_percentile < upper_percentile <= 1):
        print("Warning: Invalid percentiles provided.")
        return df
    
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)
    df_copy = df.copy()
    df_copy[column] = df_copy[column].clip(lower=lower_bound, upper=upper_bound)
    return df_copy


def filter_trading_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the DataFrame to include only actual trading periods.

    Args:
        df: The input DataFrame with a 'RETURN_NoOVERNIGHT' column.

    Returns:
        A new DataFrame containing only rows with non-zero returns.
    """
    # The zeros filled in are not real trading returns.
    df = df.sort_values(['DATE', 'SYMBOL', 'TIME']).reset_index(drop=True)

    # Combine DATE and the TIME object into a single string
    df['DATETIME'] = df['DATE'].astype(str) + ' ' + df['TIME'].astype(str)

    # Convert this new string column to a datetime object
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    # Set this as the DataFrame's index
    df = df.set_index('DATETIME')

    # Sort the index to be in chronological order
    df = df.sort_index()

    # Create a cumulative‐count within each (DATE, SYMBOL) group
    group_idx = df.groupby([df.index.date, 'SYMBOL']).cumcount()

    # Build a mask to keep rows if:
    mask = (group_idx > 0) | ((group_idx == 0) & (df['RETURN_NoOVERNIGHT'] != 0))

    trading_returns_df = df[mask].copy()

    # TO REMOVE
    # symbol_counts = trading_returns_df['SYMBOL'].value_counts()
    # top_symbols = symbol_counts.head(250).index.tolist()
    # trading_returns_df = trading_returns_df[trading_returns_df['SYMBOL'].isin(top_symbols)]

    # Remove outliers
    trading_returns_df = remove_outlier_returns(trading_returns_df, column='RETURN_NoOVERNIGHT')

    if trading_returns_df.empty:
        print("Warning: No non-zero returns found after filtering.")
        
    return trading_returns_df


def data_split(df: pd.DataFrame, split_datetime: str, min_train_size: int = 2, min_test_size: int = 2):
    """
    Splits a DataFrame chronologically at a specified datetime into training and test sets.

    Args:
        df: Input DataFrame with a DatetimeIndex, assumed to be sorted chronologically.
        split_datetime: Datetime string (e.g., '2021-12-27 00:00:00') for splitting.
        min_train_size: Minimum number of rows for training set.
        min_test_size: Minimum number of rows for test set.

    Returns:
        Tuple of (train_df, test_df), or (None, None) if split is invalid.
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        print("Warning: Empty DataFrame or index is not a DatetimeIndex.")
        return None, None

    try:
        split_dt = pd.to_datetime(split_datetime)
    except ValueError:
        print(f"Warning: Invalid split_datetime '{split_datetime}'.")
        return None, None

    train_df = df[df.index < split_dt]
    test_df = df[df.index >= split_dt]

    if len(train_df) < min_train_size or len(test_df) < min_test_size:
        print(f"Warning: Insufficient data for split (train: {len(train_df)}, test: {len(test_df)}).")
        return None, None

    return train_df, test_df

def denormalize_targets(z_values, mean, std):
    return z_values * std + mean

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

def enrich_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Combine DATE + TIME into a proper timestamp column and drop originals."""
    df["datetime"] = pd.to_datetime(df["DATE"].astype(str) + " " + df["TIME"].astype(str))
    df.drop(columns=["TIME", "DATE"], inplace=True)
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
    
    #df= augment_features(df, symbol_col= "symbol", return_col= "return")

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

def encode_categoricals(data: pd.DataFrame, cat_cols: list[str]):
    """Factor‑encode categorical columns and return mapping dictionaries."""
    vocab_maps: dict[str, dict] = {}
    for col in cat_cols:
        codes, uniques = pd.factorize(data[col], sort=False)
        data[col] = codes.astype("int32")
        vocab_maps[col] = {val: idx for idx, val in enumerate(uniques)}
    return data, vocab_maps

def split_and_shift_data(data, date_split: str, target_col='return_raw', group_col='symbol'):

    # Split
    train_df = data[data.index <= date_split].copy()
    test_df = data[data.index > date_split].copy()

    for df in [train_df, test_df]:
        group = df.groupby(group_col)[target_col]
        df['target_return'] = group.shift(-1)
        df['target_return'] = df['target_return']
        df.dropna(subset=['target_return'], inplace=True)

    # Normalize using training statistics
    target_mean = train_df['target_return'].mean()
    target_std = train_df['target_return'].std()

    train_df['target_return'] = (train_df['target_return'] - target_mean) / (target_std + 1e-8)
    test_df['target_return'] = (test_df['target_return'] - target_mean) / (target_std + 1e-8)

    return train_df, test_df, target_mean, target_std

def split_and_normalise(
    data: pd.DataFrame,
    date_split: str,
    cont_features: list[str],
    target_col: str = "return",
):
    """Train/val split by `date_split` and z‑score continuous features."""
    print("Splitting data...")
    train_df, test_df, tgt_mean, tgt_std = split_and_shift_data(data, date_split=date_split, target_col=target_col)


    # train_mean = train_df["target_return"].mean()
    # train_std = train_df["target_return"].std()

    scaler = StandardScaler()
    train_df[cont_features] = scaler.fit_transform(train_df[cont_features])

    test_df[cont_features] = scaler.transform(test_df[cont_features])

    return train_df, test_df, scaler, tgt_mean, tgt_std

