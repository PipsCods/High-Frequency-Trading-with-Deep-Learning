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
from src.utils import load_data, filter_trading_returns, data_split

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


def process_stock(symbol_data: tuple, target_col: str, split_datetime: str):
    """
    Fits ARIMA and GARCH models on training data and forecasts on the test set for a single stock.
    Receives pre-filtered stock data.
    """
    symbol, stock_df_for_symbol = symbol_data # Unpack the tuple: (symbol, df_for_that_symbol)

    stock_series = stock_df_for_symbol[target_col].dropna() # This operation is now on a much smaller DataFrame
    if len(stock_series) < 50: # Need enough data to train
        return None

    # Split data
    train_series, test_series = data_split(stock_series.to_frame(), split_datetime)
    if train_series is None or len(test_series) < 2:
        return None
    
    train_series = train_series[target_col]
    test_series = test_series[target_col]
    
    n_test = len(test_series)
    results = {'symbol': symbol} # Ensure symbol is kept for results

    # --- ARIMA Forecasting ---
    try:
        arima_model = auto_arima(
            train_series,
            start_p=1, start_q=1, max_p=3, max_q=3, d=0,
            seasonal=False, trace=False, error_action='ignore',
            suppress_warnings=True, stepwise=True
        )
        # Forecast for the length of the test set
        arima_preds = arima_model.predict(n_periods=n_test)
        results['arima_params'] = {
            'order': arima_model.order,
            'params': arima_model.params().to_dict(),
            'aic': arima_model.aic()
        }
        results['arima_predictions'] = pd.Series(arima_preds, index=test_series.index)
    except Exception as e:
        # print(f"ARIMA failed for {symbol}: {e}")
        results['arima_params'] = None
        results['arima_predictions'] = None

    # --- GARCH Forecasting ---
    try:
        garch_train_series = train_series * 1000
        
        garch_model = arch_model(garch_train_series, lags=1, vol='Garch', p=1, q=1, dist='t')
        garch_fit = garch_model.fit(update_freq=0, disp='off')
        
        forecasts = garch_fit.forecast(horizon=n_test, reindex=False)
        garch_preds = forecasts.mean.iloc[0].values / 1000

        garch_params = garch_fit.params.to_dict()
        garch_pvalues = garch_fit.pvalues.to_dict()
        
        results['garch_results'] = {**garch_params, **{f'{k}_pval': v for k, v in garch_pvalues.items()}}
        results['garch_predictions'] = pd.Series(garch_preds, index=test_series.index)
    except Exception as e:
        # print(f"GARCH failed for {symbol}: {e}")
        results['garch_results'] = None
        results['garch_predictions'] = None
        
    return results


def run_full_analysis(df: pd.DataFrame, target_col: str, split_datetime: str, params_dir: Path, preds_dir: Path):
    """
    Runs ARIMA and GARCH forecasting for all stocks and saves parameters and predictions.
    Data is pre-grouped by symbol for efficiency.
    """
    # Group the DataFrame by SYMBOL once in the main process
    # This creates an iterable of (symbol, DataFrame_for_that_symbol) tuples
    grouped_df = df.groupby('SYMBOL')
    stock_data_iterator = ((symbol, group) for symbol, group in grouped_df)
    total_stocks = len(grouped_df.groups)

    # Create a partial function to pass fixed arguments to the worker
    process_func = partial(process_stock, target_col=target_col, split_datetime=split_datetime)
    
    with Pool(cpu_count() - 1) as pool:
        results_list = list(tqdm(pool.imap(process_func, stock_data_iterator), total=total_stocks))
        
    valid_results = [res for res in results_list if res is not None]
    print(f"Completed TS forecasting for {len(valid_results)} stocks.")

    # Separate and Save Parameters
    arima_params_list = [{'symbol': r['symbol'], **r['arima_params']} for r in valid_results if r['arima_params']]
    garch_results_list = [{'symbol': r['symbol'], **r['garch_results']} for r in valid_results if r.get('garch_results')]
    
    params_dir.mkdir(parents=True, exist_ok=True)
    if arima_params_list:
        pd.DataFrame(arima_params_list).to_parquet(params_dir / "arima_parameters.parquet")
        print(f"ARIMA parameters saved to {params_dir}")
    if garch_results_list:
        pd.DataFrame(garch_results_list).to_parquet(params_dir / "garch_results.parquet")
        print(f"GARCH results saved to {params_dir}")
    
    # Separate and Save Predictions
    preds_dir.mkdir(parents=True, exist_ok=True)
    for model_name in ['arima', 'garch']:
        preds_list = [r[f'{model_name}_predictions'] for r in valid_results if r.get(f'{model_name}_predictions') is not None]
        if preds_list:
            preds_df = pd.concat(preds_list).to_frame(name='predicted_return').reset_index()
            symbol_map = df[['DATETIME', 'SYMBOL']].drop_duplicates().set_index('DATETIME')
            preds_df = preds_df.join(symbol_map, on='DATETIME')
            preds_df.to_parquet(preds_dir / f"{model_name}_predictions.parquet")
            print(f"{model_name.upper()} predictions saved to {preds_dir}")


def generate_outputs(params_dir: Path, tables_dir: Path, figures_dir: Path):
    """
    Loads pre-computed analysis results and generates summary tables and figures.
    """    
    # Load data from saved files
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # ARIMA Analysis
    try:
        arima_df = pd.read_parquet(params_dir / "arima_parameters.parquet")
        
        # Table: ARIMA Model Order Frequency
        order_counts = arima_df['order'].astype(str).value_counts().reset_index()
        order_counts.columns = ['Model Order', 'Frequency']
        order_counts.head(10).to_latex(
            tables_dir / "arima_order_frequency.tex", index=False,
            caption="Frequency of Top 10 Most Common ARIMA Model Orders Across All Stocks.",
            label="tab:arima_orders", column_format="lr"
        )
        # print(f"Table 'arima_order_frequency.tex' saved.")

        # Figure: Distribution of ARIMA Model Orders
        plt.figure(figsize=(12, 7))
        sns.countplot(y=arima_df['order'].astype(str), order=order_counts['Model Order'].head(10), palette='viridis')
        plt.title('Top 10 Most Common ARIMA Model Orders Across All Stocks', fontsize=16)
        plt.xlabel('Number of Stocks', fontsize=12)
        plt.ylabel('ARIMA(p,d,q) Order', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(figures_dir / "arima_order_dist.png")
        plt.close()
        # print(f"Figure 'arima_order_dist.png' saved.")

    except FileNotFoundError:
        print("Skipping ARIMA reports: arima_parameters.parquet not found.")


    # GARCH Analysis
    try:
        # Load the results file which now contains p-values
        garch_df = pd.read_parquet(params_dir / "garch_results.parquet")

        # Table: GARCH Effect Significance Summary
        garch_df['alpha_significant'] = garch_df.get('alpha[1]_pval', 1.0) < 0.05
        garch_df['beta_significant'] = garch_df.get('beta[1]_pval', 1.0) < 0.05
        garch_df['both_significant'] = garch_df['alpha_significant'] & garch_df['beta_significant']
        
        summary_data = {
            'Metric': ['Stocks with significant ARCH term (alpha)', 'Stocks with significant GARCH term (beta)', 'Stocks with both terms significant'],
            'Percentage': [f"{garch_df['alpha_significant'].mean()*100:.2f}\%", f"{garch_df['beta_significant'].mean()*100:.2f}\%", f"{garch_df['both_significant'].mean()*100:.2f}\%"]
        }
        pd.DataFrame(summary_data).to_latex(
            tables_dir / "garch_significance_summary.tex", index=False,
            caption="Significance of GARCH Model Parameters Across All Stocks.",
            label="tab:garch_summary", column_format="lr"
        )
        # print(f"Table 'garch_significance_summary.tex' saved.")

        # Figure: Distribution of Significant GARCH Coefficients
        significant_garch_df = garch_df[garch_df['both_significant']].copy()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Distribution of Significant GARCH Coefficients', fontsize=16)
        sns.histplot(significant_garch_df['alpha[1]'], kde=True, ax=axes[0], color='skyblue').set_title('Distribution of ARCH Term (alpha[1])')
        sns.histplot(significant_garch_df['beta[1]'], kde=True, ax=axes[1], color='salmon').set_title('Distribution of GARCH Term (beta[1])')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(figures_dir / "garch_coefficients_distribution.png")
        plt.close()
        # print(f"Figure 'garch_coefficients_distribution.png' saved.")

    except (FileNotFoundError, KeyError) as e:
        print(f"Skipping GARCH reports. Error: {e}. Check if garch_results.parquet exists.")


# --- Main Execution Block ---
if __name__ == '__main__':
    BASE_DIR = Path.cwd()
    DATA_PATH = BASE_DIR / "data" / "processed" / "high_10m.parquet"
    RESULTS_DIR = BASE_DIR / "results"
    PARAMS_DIR = RESULTS_DIR / "parameters"
    PREDS_DIR = RESULTS_DIR / "predictions"
    TABLES_DIR = RESULTS_DIR / "tables"
    FIGURES_DIR = RESULTS_DIR / "figures" / "time_series"
    
    TARGET_COL = 'RETURN_NoOVERNIGHT'
    SPLIT_DATETIME = '2021-12-27 00:00:00'

    # 1. Load and process data
    raw_df = load_data(DATA_PATH)
    returns_df = filter_trading_returns(raw_df)

    if not returns_df.empty:
        # 2. Run the full analysis (forecasting and parameter saving)
        run_full_analysis(returns_df, TARGET_COL, SPLIT_DATETIME, PARAMS_DIR, PREDS_DIR)
    
        # 3. Generate summary reports from the saved parameters
        generate_outputs(PARAMS_DIR, TABLES_DIR, FIGURES_DIR)
    else:
        print("Halting execution: The initial returns DataFrame is empty after filtering.")