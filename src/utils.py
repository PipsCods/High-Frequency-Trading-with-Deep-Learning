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
    symbol_counts = trading_returns_df['SYMBOL'].value_counts()
    top_symbols = symbol_counts.head(150).index.tolist()
    trading_returns_df = trading_returns_df[trading_returns_df['SYMBOL'].isin(top_symbols)]

    if trading_returns_df.empty:
        print("Warning: No non-zero returns found after filtering.")
        
    return trading_returns_df