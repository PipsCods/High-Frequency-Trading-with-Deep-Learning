# Package imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from functools import partial
from tqdm import tqdm
import warnings

# Time series analysis libraries
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from arch import arch_model
from arch.univariate.base import DataScaleWarning

# Utils imports
try:
    from .utils import load_data, filter_trading_returns, data_split
except ImportError:
    from utils import load_data, filter_trading_returns, data_split

warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')
warnings.filterwarnings("ignore", category=DataScaleWarning)
warnings.filterwarnings("ignore", message="No supported index is available.")

def plot_autocorrelation(series: pd.Series, series_name: str, lags: int, save_path: Path):
    """
    Plots the ACF and PACF for a given time series and saves the figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'Autocorrelation Analysis for {series_name}', fontsize=16)

    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    axes[0].grid(True)

    plot_pacf(series, lags=lags, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('PACF')
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    # print(f"Autocorrelation plot saved to: {save_path}")


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

    # ARIMA Forecasting
    try:
        arima_train_series = train_series * 100  # Scale to avoid numerical issues with ARIMA

        arima_model = auto_arima(
            arima_train_series.values,
            start_p=1, start_q=1, max_p=3, max_q=3, d=0,
            seasonal=False, trace=False, 
            error_action='ignore',
            suppress_warnings=True, 
            stepwise=True
        )

        param_names = arima_model.arima_res_.param_names
        param_values = arima_model.params()
        params_as_dict = dict(zip(param_names, param_values))
        
        # Forecast for the length of the test set
        results['arima_params'] = {
            'order': arima_model.order,
            'params': params_as_dict,
            'aic': arima_model.aic()
        }

        arima_forecast = arima_model.predict(n_periods=n_test)
        arima_preds = arima_forecast / 100  # Scale back to original units

        results['arima_predictions'] = pd.Series(arima_preds, index=test_series.index, name=symbol)

    except Exception as e:
        print(f"ARIMA failed for {symbol}: {e}")
        results['arima_params'] = None
        results['arima_predictions'] = None

    # GARCH Forecasting
    try:
        garch_train_series = train_series * 100
        
        garch_model = arch_model(garch_train_series, lags=1, vol='Garch', p=1, o=1, q=1, dist='studentst')
        garch_fit = garch_model.fit(update_freq=0, disp='off')
        
        forecasts = garch_fit.forecast(horizon=n_test, reindex=False)
        
        garch_variance_forecast = forecasts.variance.iloc[0].values
        garch_preds = np.sqrt(garch_variance_forecast) / 100

        garch_params = garch_fit.params.to_dict()
        garch_pvalues = garch_fit.pvalues.to_dict()
        
        results['garch_results'] = {**garch_params, **{f'{k}_pval': v for k, v in garch_pvalues.items()}}
        results['garch_predictions'] = pd.Series(garch_preds, index=test_series.index, name=symbol)
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
    grouped_df = df.groupby('SYMBOL')
    stock_data_iterator = ((symbol, group) for symbol, group in grouped_df)
    total_stocks = len(grouped_df.groups)

    # Create a partial function to pass fixed arguments to the worker
    process_func = partial(process_stock, target_col=target_col, split_datetime=split_datetime)
    results_list = []

    with ProcessPool() as pool:
        future = pool.map(process_func, stock_data_iterator, timeout=120)
        results_iterator = future.result()

        with tqdm(total=total_stocks, desc="Processing Stocks") as pbar:
            while True:
                try:
                    result = next(results_iterator)
                    if result is not None:
                        results_list.append(result)
                except StopIteration:
                    break
                except TimeoutError:
                    # print("A stock processing task timed out and was skipped.")
                    pass
                except Exception as e:
                    # print(f"A task failed with an unexpected error: {e}")
                    pass
                finally:
                    pbar.update(1)
            
    print(f"\nCompleted processing for {len(results_list)} stocks (skipped or failed stocks are excluded).")

    valid_results = [res for res in results_list if res is not None]
    print(f"Completed TS forecasting for {len(valid_results)} stocks.")

    # Separate and Save Parameters
    arima_params_list = [{'symbol': r['symbol'], **r['arima_params']} for r in valid_results if r.get('arima_params')]
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
        preds_dict = {
            r['symbol']: r[f'{model_name}_predictions']
            for r in valid_results if r.get(f'{model_name}_predictions') is not None
        }

        if preds_dict:
            # Create a wide DataFrame (symbols as columns)
            concatenated_df = pd.concat(preds_dict, axis=1)
            
            # Pivot from wide to long format. The `stack` method correctly handles this,
            # and we use dropna=False to be safe.
            keys = { 'arima': 'predicted_return', 'garch': 'predicted_volatility' }
            final_df = concatenated_df.stack(dropna=False).to_frame(keys[model_name]).reset_index()
            final_df.rename(columns={'level_0': 'DATETIME', 'level_1': 'SYMBOL'}, inplace=True)
            
            # Drop any remaining NaNs that might result from non-overlapping test sets
            final_df.dropna(inplace=True)

            final_df.to_parquet(preds_dir / f"{model_name}_predictions.parquet", index=False)
            print(f"Successfully saved {len(final_df)} predictions for '{model_name}'.")

    
def generate_outputs(params_dir: Path, figures_dir: Path, tables_dir: Path):
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
        plt.savefig(figures_dir / "garch_coefficients_dist.png")
        plt.close()
        # print(f"Figure 'garch_coefficients_dist.png' saved.")

    except (FileNotFoundError, KeyError) as e:
        print(f"Skipping GARCH reports. Error: {e}. Check if garch_results.parquet exists.")


# --- Main Pipeline Function (for import) ---
def run_timeseries_models_pipeline(data_path: Path, params_dir: Path, preds_dir: Path, figures_dir: Path, split_datetime: str):
    """
    Runs the full time series analysis pipeline including ARIMA and GARCH models.
    """
    print("--- Running Time Series Models Training Pipeline ---")

    # Constants
    TARGET_COL = 'RETURN_NoOVERNIGHT'

    returns_df = filter_trading_returns(load_data(data_path))

    if returns_df.empty:
        print("Halting execution: The initial returns DataFrame is empty after filtering.")
        return

    # Pre-filter out problematic stocks before the main loop
    print(f"Original number of stocks: {returns_df['SYMBOL'].nunique()}")
    
    # Filter 1: Keep only stocks that have data on both sides of the split date
    split_dt_obj = pd.to_datetime(split_datetime)
    stocks_with_test_data = returns_df[returns_df.index >= split_dt_obj]['SYMBOL'].unique()
    returns_df = returns_df[returns_df['SYMBOL'].isin(stocks_with_test_data)]
    print(f"Stocks remaining after filtering for test data: {returns_df['SYMBOL'].nunique()}")

    # Filter 2: Keep only stocks with a minimum number of observations
    returns_df = returns_df.groupby('SYMBOL').filter(lambda x: len(x) > 100)
    print(f"Stocks remaining after filtering for min observations: {returns_df['SYMBOL'].nunique()}")

    # Filter 3: Keep only stocks with non-zero volatility
    volatility = returns_df.groupby('SYMBOL')[TARGET_COL].std()
    stocks_with_vol = volatility[volatility > 1e-8].index # Use a small threshold
    returns_df = returns_df[returns_df['SYMBOL'].isin(stocks_with_vol)]
    print(f"Stocks remaining after filtering for non-zero volatility: {returns_df['SYMBOL'].nunique()}")

    figures_dir.mkdir(parents=True, exist_ok=True)
    symbols_to_plot = returns_df['SYMBOL'].unique()[:2]
    for symbol in symbols_to_plot:
        stock_series = returns_df[returns_df['SYMBOL'] == symbol][TARGET_COL].dropna()
        save_path = figures_dir / f"autocorrelation_{symbol}.png"
        plot_autocorrelation(stock_series, symbol, lags=10, save_path=save_path)

    run_full_analysis(returns_df, TARGET_COL, split_datetime, params_dir, preds_dir)
    
    print("--- Time Series Models Training Pipeline Complete ---")

def generate_summary_reports(params_dir: Path, figures_dir: Path, tables_dir: Path):
    """
    Generates summary reports from the saved parameters and predictions.
    """
    print("--- Generating Summary Tables and Figures ---")
    generate_outputs(params_dir, figures_dir, tables_dir)


# --- Main Execution Block ---
if __name__ == '__main__':
    BASE_DIR = Path.cwd()
    DATA_PATH = BASE_DIR / "data" / "processed" / "high_10m.parquet"
    RESULTS_DIR = BASE_DIR / "results"
    PARAMS_DIR = RESULTS_DIR / "parameters"
    PREDS_DIR = RESULTS_DIR / "predictions"
    TABLES_DIR = RESULTS_DIR / "tables" / "time_series"
    FIGURES_DIR = RESULTS_DIR / "figures" / "time_series"
    
    TARGET_COL = 'RETURN_NoOVERNIGHT'
    SPLIT_DATETIME = '2021-12-27 00:00:00'

    # 1. Load and process data
    returns_df = filter_trading_returns(load_data(DATA_PATH))

    if not returns_df.empty:
        # 2. Run the full analysis (forecasting and parameter saving)
        run_timeseries_models_pipeline(DATA_PATH, PARAMS_DIR, PREDS_DIR, FIGURES_DIR, SPLIT_DATETIME)
    
        # 3. Generate summary reports from the saved parameters
        generate_outputs(PARAMS_DIR, TABLES_DIR, FIGURES_DIR)
    else:
        print("Halting execution: The initial returns DataFrame is empty after filtering.")