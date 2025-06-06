# Import necessary libraries
import pandas as pd
import numpy as np
from pathlib import Path

# Import utils
from src.utils import load_data

def identify_outlier_stocks(stats_df: pd.DataFrame) -> list:
    """
    Identifies outlier stocks based on predefined thresholds for their statistics.

    Args:
        stats_df: The DataFrame containing statistics for each stock.

    Returns:
        A list of symbols for the outlier stocks.
    """
    print("Identifying outlier stocks based on thresholds...")

    # Define thresholds after reviewing the data
    mean_threshold = 0.01  # Stocks with an avg 10-min return > 1%
    std_threshold = 0.05   # Stocks with a 10-min volatility > 5%
    skew_threshold = 15    # Stocks with skewness more extreme than -15 or +15
    kurt_threshold = 100   # Stocks with kurtosis > 100

    # Find stocks that meet any of the outlier conditions
    outlier_conditions = (
        (stats_df['mean'].abs() > mean_threshold) |
        (stats_df['std'] > std_threshold) |
        (stats_df['skew'].abs() > skew_threshold) |
        (stats_df['kurtosis'] > kurt_threshold) |
        (stats_df['std'] == 0) # Also include stocks with zero volatility
    )

    outlier_stocks_df = stats_df[outlier_conditions]
    outlier_symbols = outlier_stocks_df.index.tolist()

    print(f"Found {len(outlier_symbols)} potential outlier stocks.")
    print("Outlier stocks and their stats:")
    print(outlier_stocks_df)

    return outlier_symbols

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

    # 2) Create a cumulative‐count within each (DATE, SYMBOL) group
    group_idx = df.groupby(['DATE','SYMBOL']).cumcount()

    # 3) Build a mask to keep rows if:
    mask = (group_idx > 0) | ((group_idx == 0) & (df['RETURN_NoOVERNIGHT'] != 0))

    trading_returns_df = df[mask].copy()

    if trading_returns_df.empty:
        print("Warning: No non-zero returns found after filtering.")
        
    return trading_returns_df

def calculate_stats_per_stock(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates descriptive statistics for each stock individually.

    Args:
        returns_df: A DataFrame containing only filtered, non-zero trading returns.

    Returns:
        A DataFrame with statistics (mean, median, std, skew, kurtosis) for each stock.
    """
    if returns_df.empty:
        print("Input DataFrame is empty. Cannot calculate stats.")
        return pd.DataFrame()

    # Use .agg() to calculate multiple statistics at once.
    stats_per_stock = returns_df.groupby('SYMBOL')['RETURN_NoOVERNIGHT'].agg(
        ['mean', 'median', 'std', 'skew']
    )
    # pandas kurt() calculates excess kurtosis (Kurtosis - 3), which is standard.
    stats_per_stock['kurtosis'] = returns_df.groupby('SYMBOL')['RETURN_NoOVERNIGHT'].apply(pd.Series.kurt)
    print("Calculation of stats per stock complete.")
    return stats_per_stock

def calculate_aggregated_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates aggregated statistics by describing the stats_per_stock DataFrame.

    Args:
        stats_df: The DataFrame containing statistics for each stock.

    Returns:
        A DataFrame summarizing the distribution of statistics across all stocks.
    """
    if stats_df.empty:
        print("Input DataFrame is empty. Cannot calculate aggregated stats.")
        return pd.DataFrame()

    # The .describe() method is perfect for summarizing the statistics DataFrame.
    aggregated_stats = stats_df.describe()
    print("Aggregation complete.")
    return aggregated_stats

def run_descriptive_analysis(df: pd.DataFrame):
    """
    Main function to run the entire descriptive analysis pipeline.

    Args:
        df: The initial raw DataFrame.
    """
    # Step 1: Filter the data
    returns_df = filter_trading_returns(df)

    # Step 2: Calculate stats for each stock
    stats_df = calculate_stats_per_stock(returns_df)
    print("\n--- Descriptive Statistics per Stock (First 5) ---")
    print(stats_df.head())
    print("-" * 55)

    # Identify outlier stocks
    # outlier_symbols = identify_outlier_stocks(stats_df)
    
    # Step 3: Aggregate the results
    aggregated_df = calculate_aggregated_stats(stats_df)
    print("\n--- Aggregated Descriptive Statistics ---")
    print(aggregated_df)
    print("-" * 55)

# --- Main execution block ---
if __name__ == '__main__':
    
    BASE_DIR = Path.cwd()
    DATA_PATH = ".." / BASE_DIR / "data" / "processed" / "high_10m.parquet"

    # Load the data
    returns_df = load_data(DATA_PATH)

    # Run the entire analysis by calling the main function
    run_descriptive_analysis(returns_df)