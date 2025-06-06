# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
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

    # 1. Combine DATE and the TIME object into a single string
    #    (Assuming TIME is in a format like 'HH:MM:SS')
    df['DATETIME'] = df['DATE'].astype(str) + ' ' + df['TIME'].astype(str)

    # 2. Convert this new string column to a datetime object
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    # 3. Set this as the DataFrame's index
    df = df.set_index('DATETIME')

    # 4. Optional: Sort the index to be absolutely sure everything is in chronological order
    df = df.sort_index()

    # --- DEBUGGING LINE 1 ---
    # Let's confirm the index is correct at this point.
    print(f"DEBUG: Index type after setting and sorting is: {type(df.index)}")
    # ---

    # Create a cumulative‐count within each (DATE, SYMBOL) group
    group_idx = df.groupby([df.index.date, 'SYMBOL']).cumcount()

    # Build a mask to keep rows if:
    mask = (group_idx > 0) | ((group_idx == 0) & (df['RETURN_NoOVERNIGHT'] != 0))

    trading_returns_df = df[mask].copy()

    # --- DEBUGGING LINE 2 ---
    # Now, let's check the index type of the final DataFrame before it's returned.
    print(f"DEBUG: Index type of the final DataFrame is: {type(trading_returns_df.index)}")
    print("DEBUG: First 5 rows of the final DataFrame:")
    print(trading_returns_df.head())
    # ---

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
    # Calculate stats for each stock
    stats_df = calculate_stats_per_stock(returns_df)
    print("\n--- Descriptive Statistics per Stock (First 5) ---")
    print(stats_df.head())
    print("-" * 55)

    # Identify outlier stocks
    # outlier_symbols = identify_outlier_stocks(stats_df)
    
    # Aggregate the results
    aggregated_df = calculate_aggregated_stats(stats_df)
    print("\n--- Aggregated Descriptive Statistics ---")
    print(aggregated_df)
    print("-" * 55)

def plot_intraday_patterns(df: pd.DataFrame):
    """
    Calculates and plots the aggregated intraday return and volatility patterns.

    Args:
        df: DataFrame with a DatetimeIndex and a 'RETURN_NoOVERNIGHT' column.
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    # Filter for actual trading returns to avoid distortion
    trading_df = df[df['RETURN_NoOVERNIGHT'] != 0].copy()
    if trading_df.empty:
        print("No non-zero trading returns found. Cannot generate plots.")
        return

    print("Calculating intraday statistics...")
    # Group by the time component of the index
    intraday_stats = trading_df.groupby(trading_df.index.time).agg(
        Average_Return=('RETURN_NoOVERNIGHT', 'mean'),
        Average_Volatility=('RETURN_NoOVERNIGHT', 'std')
    )
    # Sort by time to ensure the plot is in chronological order
    intraday_stats = intraday_stats.sort_index()
    print("Calculation complete.")


    print("Generating plots...")
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Aggregated Intraday Patterns', fontsize=16)

    # Plot 1: Average Return by Time Interval
    intraday_stats['Average_Return'].plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
    axes[0].set_title('Average 10-Minute Return by Time of Day')
    axes[0].set_ylabel('Average Return')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1) # Add a line at zero for reference
    axes[0].grid(axis='y', linestyle=':', alpha=0.7)

    # Plot 2: Average Volatility by Time Interval
    intraday_stats['Average_Volatility'].plot(kind='bar', ax=axes[1], color='salmon', edgecolor='black')
    axes[1].set_title('Average 10-Minute Volatility (Std. Dev.) by Time of Day')
    axes[1].set_ylabel('Average Volatility')
    axes[1].grid(axis='y', linestyle=':', alpha=0.7)

    # Improve formatting of the x-axis labels
    axes[1].set_xlabel('Time of Day')
    # Format x-tick labels to show HH:MM
    xtick_labels = [time.strftime('%H:%M') for time in intraday_stats.index]
    axes[1].set_xticklabels(xtick_labels, rotation=45, ha='right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- Main execution block ---
if __name__ == '__main__':
    
    BASE_DIR = Path.cwd()
    DATA_PATH = ".." / BASE_DIR / "data" / "processed" / "high_10m.parquet"

    # 1. Load the raw data
    raw_df = load_data(DATA_PATH)

    # 2. Filter the data and create the DatetimeIndex. 
    # The result is stored back into a variable in the main scope.
    returns_df = filter_trading_returns(raw_df)

    # 3. Run the analysis on the now-correctly-formatted returns_df
    run_descriptive_analysis(returns_df)

    # 4. Plot the intraday patterns, also using the correct returns_df
    plot_intraday_patterns(returns_df)