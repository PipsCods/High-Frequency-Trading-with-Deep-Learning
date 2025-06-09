import pandas as pd

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
    # top_symbols = symbol_counts.head(150).index.tolist()
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