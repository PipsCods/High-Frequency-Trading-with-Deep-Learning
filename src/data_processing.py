# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import utils
from src.utils import load_data, filter_trading_returns

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

def compute_intraday_patterns(returns_df: pd.DataFrame, save_path: str = None):
    """
    Calculates and plots the aggregated intraday return and volatility patterns.

    Args:
        df: DataFrame with a DatetimeIndex and a 'RETURN_NoOVERNIGHT' column.
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    print("Calculating intraday statistics...")
    # Group by the time component of the index
    intraday_stats = returns_df.groupby(returns_df.index.time).agg(
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
    axes[1].set_title('Average 10-Minute Volatility by Time of Day')
    axes[1].set_ylabel('Average Volatility')
    axes[1].grid(axis='y', linestyle=':', alpha=0.7)

    # Improve formatting of the x-axis labels
    axes[1].set_xlabel('Time of Day')
    # Format x-tick labels to show HH:MM
    xtick_labels = [time.strftime('%H:%M') for time in intraday_stats.index]
    axes[1].set_xticklabels(xtick_labels, rotation=45, ha='right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        output_path = Path(f"{save_path}/intraday_patterns.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        # print(f"Figure saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)

def plot_correlation_heatmap(returns_df: pd.DataFrame, save_path: str = None):
    pivot_df = returns_df.pivot(columns='SYMBOL', values='RETURN_NoOVERNIGHT')
    corr_matrix = pivot_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Stock Returns')
    if save_path:
        output_path = Path(f"{save_path}/correlation_heatmap.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        # print(f"Figure saved to: {output_path}")
    plt.show()

# --- Main execution block ---
if __name__ == '__main__':
    
    BASE_DIR = Path.cwd()
    DATA_PATH = ".." / BASE_DIR / "data" / "processed" / "high_10m.parquet"
    FIGURE_PATH_DIR = ".." / BASE_DIR / "results" / "figures"

    # 1. Load the raw data
    raw_df = load_data(DATA_PATH)

    # 2. Filter the data and create the DatetimeIndex. 
    # The result is stored back into a variable in the main scope.
    returns_df = filter_trading_returns(raw_df)

    # 3. Run the analysis
    run_descriptive_analysis(returns_df)

    # 4. Plot the intraday patterns
    compute_intraday_patterns(returns_df, FIGURE_PATH_DIR)

    # 5. Plot the correlation heatmap
    # plot_correlation_heatmap(returns_df, FIGURE_PATH_DIR)
