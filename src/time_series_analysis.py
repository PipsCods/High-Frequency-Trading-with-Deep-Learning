# Package imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import warnings

# Time series analysis libraries
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from arch import arch_model
# from arch.univariate.base import DataScaleWarning

# Import utils
from src.utils import load_data
from src.data_processing import filter_trading_returns

warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')
# warnings.filterwarnings("ignore", category=DataScaleWarning)

def plot_autocorrelation(series: pd.Series, series_name: str, lags: int = 40):
    """
    Plots the ACF and PACF for a given time series to identify potential model orders.

    Args:
        series: The time series of returns for a single stock.
        series_name: The name/symbol of the stock for titles.
        lags: The number of lags to display on the plots.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'Autocorrelation Analysis for {series_name}', fontsize=16)

    # Plot ACF
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    axes[0].grid(True)

    # Plot PACF
    plot_pacf(series, lags=lags, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('PACF')
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def run_auto_arima(series: pd.Series):
    """
    Finds the best ARIMA model for a time series using an automated search.

    Args:
        series: The time series of returns for a single stock.

    Returns:
        The fitted ARIMA model object.
    """    
    arima_model = auto_arima(
        series,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        d=0,           
        seasonal=False,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    return arima_model

def run_garch_analysis(series: pd.Series):
    """
    Fits a GARCH(1,1) model to analyze the volatility of the series.

    Args:
        series: The time series of returns for a single stock.

    Returns:
        The fitted GARCH model result object.
    """
    garch_model = arch_model(series * 1000, vol='Garch', p=1, q=1, dist='Normal')
    results = garch_model.fit(update_freq=0, disp='off')

    return results

def process_stock(symbol, df):
    """
    Processes a single stock: fits ARIMA and GARCH models.

    Args:
        symbol: Stock symbol.
        df: DataFrame containing all stock data.

    Returns:
        Tuple of (arima_result, garch_result) for the stock.
    """
    stock_series = df[df['SYMBOL'] == symbol]['LOG_RETURN_NoOVERNIGHT'].dropna()
    if len(stock_series) < 50:
        return None, None

    arima_result = None
    garch_result = None

    try:
        arima_model = run_auto_arima(stock_series)
        arima_result = {
            'symbol': symbol,
            'order': arima_model.order,
            'aic': arima_model.aic(),
            'ma.L1_coef': arima_model.params().get('ma.L1'),
            'ma.L1_pval': arima_model.pvalues().get('ma.L1'),
            'ar.L1_coef': arima_model.params().get('ar.L1'),
            'ar.L1_pval': arima_model.pvalues().get('ar.L1'),
        }
    except Exception:
        pass

    try:
        garch_model = run_garch_analysis(stock_series)
        garch_result = {
            'symbol': symbol,
            'alpha[1]_coef': garch_model.params['alpha[1]'],
            'alpha[1]_pval': garch_model.pvalues['alpha[1]'],
            'beta[1]_coef': garch_model.params['beta[1]'],
            'beta[1]_pval': garch_model.pvalues['beta[1]'],
        }
    except Exception:
        pass

    return arima_result, garch_result

def run_full_analysis(df: pd.DataFrame, save_data_path: str = None):
    """
    Runs ARIMA and GARCH analysis on all stocks, collects results,
    and generates summary tables and figures for a report.
    """
    unique_symbols = df['SYMBOL'].unique()
    print(f"Running analysis for {len(unique_symbols)} stocks...")

    # Parallelize stock processing
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(partial(process_stock, df=df), unique_symbols), total=len(unique_symbols)))

    # Collect results
    arima_results_list = [res[0] for res in results if res[0] is not None]
    garch_results_list = [res[1] for res in results if res[1] is not None]

    # Convert to DataFrames
    arima_df = pd.DataFrame(arima_results_list)
    garch_df = pd.DataFrame(garch_results_list)

    # Save results
    if save_data_path is not None:
        arima_path = Path(f"{save_data_path}/arima_results.parquet")
        garch_path = Path(f"{save_data_path}/garch_results.parquet")
        arima_path.parent.mkdir(parents=True, exist_ok=True)
        garch_path.parent.mkdir(parents=True, exist_ok=True)

        arima_df.to_parquet(arima_path)
        garch_df.to_parquet(garch_path)

        print(f"Saved ARIMA results to {arima_path}")
        print(f"Saved GARCH results to {garch_path}")


def generate_outputs(results_path: str, save_tables_path: str = None, save_figures_path: str = None):
    """
    Loads pre-computed analysis results and generates summary tables and figures.
    """    
    # Load data from saved files
    arima_results_path = Path(f"{results_path}/arima_results.parquet")
    garch_results_path = Path(f"{results_path}/garch_results.parquet")

    try:
        arima_df = pd.read_parquet(arima_results_path)
        garch_df = pd.read_parquet(garch_results_path)
    except FileNotFoundError:
        print("Error: Results files not found. Please run `run_full_analysis` first.")
        return

    # == Table 1: ARIMA Model Order Frequency ==
    print("\n--- Table 1: Most Common ARIMA Model Orders ---")
    order_counts = arima_df['order'].value_counts().reset_index()
    order_counts.columns = ['Model Order', 'Frequency']

    if save_tables_path:
        table_path = Path(f"{save_tables_path}/arima_order_frequency.tex")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        order_counts.head(10).to_latex(
            table_path,
            index=False,
            caption="Frequency of Top 10 Most Common ARIMA Model Orders.",
            label="tab:arima_orders",
            column_format="lr" # l for left, r for right-aligned
        )
    else:
        print(order_counts.head(10))


    # == Table 2: GARCH Effect Significance Summary ==
    print("\n--- Table 2: GARCH Effect Significance Summary ---")
    garch_df['alpha_significant'] = garch_df['alpha[1]_pval'] < 0.05
    garch_df['beta_significant'] = garch_df['beta[1]_pval'] < 0.05
    garch_df['both_significant'] = garch_df['alpha_significant'] & garch_df['beta_significant']
    
    alpha_pct = garch_df['alpha_significant'].mean() * 100
    beta_pct = garch_df['beta_significant'].mean() * 100
    both_pct = garch_df['both_significant'].mean() * 100

    # Create a DataFrame for prettier printing and saving
    summary_data = {
        'Metric': [
            'Stocks with significant ARCH term (alpha)',
            'Stocks with significant GARCH term (beta)',
            'Stocks with both terms significant'
        ],
        'Percentage': [
            f"{alpha_pct:.2f}\%",
            f"{beta_pct:.2f}\%",
            f"{both_pct:.2f}\%"
        ]
    }
    garch_summary_df = pd.DataFrame(summary_data)
    print(garch_summary_df)

    if save_tables_path:
        table_path = Path(f"{save_tables_path}/garch_significance_summary.tex")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        garch_summary_df.to_latex(
            table_path,
            index=False,
            caption="Significance of GARCH Model Parameters Across All Stocks.",
            label="tab:garch_summary",
            column_format="lr" # l for left, r for right-aligned
        )


    # == Figure 1: Distribution of ARIMA Model Orders ==
    plt.figure(figsize=(12, 7))
    sns.countplot(y=arima_df['order'].astype(str), order=arima_df['order'].astype(str).value_counts().iloc[:10].index, palette='viridis')
    plt.title('Figure 1: Top 10 Most Common ARIMA Model Orders Across All Stocks', fontsize=16)
    plt.xlabel('Number of Stocks', fontsize=12)
    plt.ylabel('ARIMA(p,d,q) Order', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_figures_path:
        output_path = Path(f"{save_figures_path}/arima_order_dist.png") # Renamed for clarity
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"Figure 1 saved to: {output_path}")
    else:
        plt.show()

    plt.close()


    # == Figure 2: Distribution of Significant GARCH Coefficients ==
    significant_garch_df = garch_df[garch_df['both_significant']].copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Figure 2: Distribution of Significant GARCH Coefficients', fontsize=16)

    # Alpha distribution
    sns.histplot(significant_garch_df['alpha[1]_coef'], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Distribution of ARCH Term (alpha[1])')
    axes[0].set_xlabel('Coefficient Value')
    
    # Beta distribution
    sns.histplot(significant_garch_df['beta[1]_coef'], kde=True, ax=axes[1], color='salmon')
    axes[1].set_title('Distribution of GARCH Term (beta[1])')
    axes[1].set_xlabel('Coefficient Value')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_figures_path:
        output_path = Path(f"{save_figures_path}/garch_coefficients_dist.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"Figure 2 saved to: {output_path}")
    else:
        plt.show()

    plt.close()

# --- Main Execution Block ---
if __name__ == '__main__':
    BASE_DIR = Path.cwd()
    DATA_PATH = ".." / BASE_DIR / "data" / "processed" / "high_10m.parquet"
    SAVE_PATH = ".." / BASE_DIR / "data" / "processed"
    TABLES_PATH = ".." / BASE_DIR / "results" / "tables"
    FIGURES_PATH = ".." / BASE_DIR / "results" / "figures"

    # 1. Load and processed data
    # raw_df = load_data(DATA_PATH)
    # returns_df = filter_trading_returns(raw_df)

    # 2. Run full analysis and generate outputs
    # run_full_analysis(returns_df, SAVE_PATH)

    # 3. Generate summary tables and figures
    generate_outputs(SAVE_PATH, TABLES_PATH, FIGURES_PATH)