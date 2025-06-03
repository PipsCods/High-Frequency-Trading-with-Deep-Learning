import pandas as pd

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

def prepare_hf_data(raw_df, name_of_timestamp_column, name_of_symbol_column, name_of_mid_price_column):
    """
    Processes a high-frequency financial dataset to prepare it for training and testing.

    Args:
        raw_df: DataFrame containing the raw data
        name_of_timestamp_column: Name of the column containing the timestamps
        name_of_symbol_column: Name of the column containing the stock symbols
        name_of_mid_price_column : Name of the column containing the mid-price data

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
               [name_of_timestamp_column, name_of_symbol_column, name_of_mid_price_column]):
        raise ValueError("The specified columns do not exist in the DataFrame.")

    # Convert and set timestamp
    df['timestamp'] = pd.to_datetime(df[name_of_timestamp_column])
    df.set_index('timestamp', inplace=True)
    df.drop(columns=[name_of_timestamp_column], inplace=True)

    # Rename symbol column if needed
    if 'symbol' not in df.columns:
        df.rename(columns={name_of_symbol_column: 'symbol'}, inplace=True)

    if 'mid_price' not in df.columns:
        df.rename(columns={name_of_mid_price_column: 'mid_price'}, inplace=True)

    # Extract time-based features
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_name'] = df.index.day_name()

    # Compute return (by symbol)
    df['return'] = df.groupby('symbol')['mid_price'].pct_change()

    # Fill missing values from return computation
    df.fillna(0, inplace=True)

    # Sort chronologically by symbol
    df = df.sort_values(by=['symbol', df.index.name])

    # Categorical features (known structure)
    basic_cat_features = ['symbol', 'day', 'day_name', 'hour', 'minute']
    cat_features = [col for col in df.select_dtypes(include=['object']).columns if col not in basic_cat_features and col != 'timestamp']

    # Detect continuous features automatically (besides return)
    base = ['mid_price','return']
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
