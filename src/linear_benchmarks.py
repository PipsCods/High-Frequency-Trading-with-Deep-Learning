# Libraries imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import warnings

# Scikit-learn for modeling and metrics
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Statsmodels for detailed statistical tests and diagnostics
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

# Utils imports
try:
    from .utils import load_data, filter_trading_returns, data_split
except ImportError:
    from utils import load_data, filter_trading_returns, data_split

warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')


def create_lag_features(df: pd.DataFrame, lags: int, target_col: str, min_daily_returns: int = 19) -> pd.DataFrame:
    """
    Creates time-lagged features for each stock, resetting daily,
    and then drops days/stocks with too few valid returns after lag calculation.
    """
    df = df.copy()
    df = df.sort_values(['SYMBOL', 'DATETIME'])
    df['DATETIME_INDEX'] = df.index
    
    for lag in range(1, lags + 1):
        df[f'{target_col}_lag_{lag}'] = df.groupby(['SYMBOL', 'DATE'])[target_col].shift(lag)
    
    # Drop rows where any lag feature is NaN
    feature_cols = [f'{target_col}_lag_{lag}' for lag in range(1, lags + 1)]
    df.dropna(subset=feature_cols, inplace=True)

    # Recalculate daily counts after dropping NaNs
    daily_counts = df.groupby(['SYMBOL', 'DATE']).size().reset_index(name='count')
    df = df.merge(daily_counts, on=['SYMBOL', 'DATE'], how='left')
    
    # Filter and reset index
    df_filtered = df[df['count'] >= min_daily_returns]
    df_filtered = df_filtered.set_index('DATETIME_INDEX').drop(columns='count').sort_index()

    return df_filtered


def fit_single_regression(df_stock: pd.DataFrame, target_col: str, feature_cols: list, model_type: str, split_datetime: str, alpha: float = 0.01):
    """Fits a single regression model for one stock's data and returns results."""
    
    train_df, test_df = data_split(df_stock, split_datetime)
    if train_df is None or test_df is None:
        return None

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Select and initialize the model
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=alpha)
    elif model_type == 'lasso':
        model = Lasso(alpha=alpha)
        # model = LassoCV(cv=5, random_state=2, n_alphas=1000, max_iter=10000)
    else:
        raise ValueError("model_type must be 'linear', 'ridge', or 'lasso'")

    # Fit the model and make predictions
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Create a pandas Series for predictions with the correct index
    predictions_series = pd.Series(y_pred, index=y_test.index, name='predicted_return')

    # Calculate metrics and residuals
    residuals = y_test - y_pred
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = model.score(X_test_scaled, y_test)
    hit_rate = np.mean(np.sign(y_test) == np.sign(y_pred)) * 100

    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'hit_rate': hit_rate,
        'intercept': model.intercept_,
        'coefficients': dict(zip(feature_cols, model.coef_)),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'model_type': model_type,
        'predictions_series': predictions_series,
        'predictions': y_pred,
        'actuals': y_test,
        'residuals': residuals
    }

def process_symbol_regression(symbol_data: tuple, target_col: str, feature_cols: list, model_type: str, split_datetime: str):
    """Worker function that receives a chunk of data for a single symbol."""
    symbol, df_symbol = symbol_data  # Unpack the tuple
    try:
        result = fit_single_regression(df_symbol, target_col, feature_cols, model_type, split_datetime)
        if result:
            result['symbol'] = symbol
            return result
    except Exception:
        return None
    return None

def run_regressions_for_all_stocks(df: pd.DataFrame, target_col: str, feature_cols: list, model_types: list, split_datetime: str):
    """
    Orchestrates running regressions for all stocks in parallel for specified model types.
    """
    all_results = {}
    grouped_df = df.groupby('SYMBOL')
    total_stocks = len(grouped_df)

    for model_type in model_types:
        print(f"\n--- Running {model_type.upper()} regression for {total_stocks} stocks... ---")
        
        # This iterator will yield (symbol, sub_dataframe) tuples.
        stock_data_iterator = ((symbol, group) for symbol, group in grouped_df)

        # Set up partial function for multiprocessing
        process_func = partial(process_symbol_regression, target_col=target_col, feature_cols=feature_cols, model_type=model_type, split_datetime=split_datetime)

        # Run in parallel
        with Pool(cpu_count()) as pool:
            results_list = list(tqdm(pool.imap(process_func, stock_data_iterator), total=total_stocks))

        # Filter out None results (from errors or small data)
        valid_results = [res for res in results_list if res is not None]
        all_results[model_type] = pd.DataFrame(valid_results)
        print(f"Completed processing for {len(valid_results)} stocks.")

    return all_results


def save_model_outputs(results_list: list, model_name: str, params_dir: Path, preds_dir: Path):
    """Saves model parameters/metrics and test set predictions to parquet files."""
    if not results_list:
        print(f"No results to save for {model_name}.")
        return
    
    # These keys hold complex objects (Series/arrays) used only for diagnostics.
    keys_to_exclude = ['predictions_series', 'predictions', 'actuals', 'residuals']

    # Exclude prediction Series from the params DataFrame
    params_data = [{k: v for k, v in res.items() if k not in keys_to_exclude} for res in results_list]
    params_df = pd.DataFrame(params_data)
    
    # Create save paths and save the file
    params_dir.mkdir(parents=True, exist_ok=True)
    params_path = params_dir / f"{model_name}_parameters.parquet"
    params_df.to_parquet(params_path)
    # print(f"Parameters for {model_name} saved to {params_path}")

    # Save Predictions
    all_preds_list = []
    for res in results_list:
        preds_series = res['predictions_series']
        preds_df = preds_series.to_frame()
        preds_df['SYMBOL'] = res['symbol']
        all_preds_list.append(preds_df)
        
    # Concatenate all prediction DataFrames
    full_preds_df = pd.concat(all_preds_list)
    full_preds_df = full_preds_df.reset_index().rename(columns={'DATETIME_INDEX': 'DATETIME', 'index': 'DATETIME'})

    # Create save paths and save the file
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / f"{model_name}_predictions.parquet"
    full_preds_df.to_parquet(preds_path)
    # print(f"Predictions for {model_name} saved to {preds_path}")


def plot_metric_distributions(results_df: pd.DataFrame, model_name: str, save_path_dir: Path = None):
    """Plots histograms and box plots of key performance metrics."""
    metrics = ['mse', 'mae', 'r2', 'hit_rate']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Distribution of Performance Metrics - {model_name}', fontsize=16)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.histplot(results_df[metric], kde=True, ax=axes[i], bins=50)
        axes[i].set_title(f'Distribution of {metric.upper()}')
        if metric == 'hit_rate':
            axes[i].axvline(50, color='red', linestyle='--') # Add 50% line for Hit Rate
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path_dir:
        save_path = save_path_dir / f"{model_name}_metric_distributions.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        # print(f"Saved metric distribution plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_coefficient_distributions(results_df: pd.DataFrame, model_name: str, save_path_dir: Path = None):
    """Plots histograms for the coefficients of each lag feature."""
    coeffs_df = pd.DataFrame(results_df['coefficients'].tolist())
    
    num_coeffs = len(coeffs_df.columns)
    fig, axes = plt.subplots(num_coeffs, 1, figsize=(10, num_coeffs * 4), sharex=True)
    if num_coeffs == 1: axes = [axes] # Ensure axes is iterable
    fig.suptitle(f'Distribution of Coefficients - {model_name}', fontsize=16)

    for i, col in enumerate(coeffs_df.columns):
        sns.histplot(coeffs_df[col], kde=True, ax=axes[i], bins=50)
        axes[i].set_title(f'Distribution of Coefficient: {col}')
        axes[i].axvline(0, color='red', linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path_dir:
        save_path = save_path_dir / f"{model_name}_coefficient_distributions.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        # print(f"Saved coefficient distribution plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_diagnostic_for_stock(stock_result: pd.Series, save_path_dir: Path = None):
    """Generates a full set of diagnostic plots for a single stock's regression results."""
    symbol = stock_result['symbol']
    model_name = stock_result['model_name']

    actuals = stock_result['actuals']
    predictions = stock_result['predictions']
    residuals = stock_result['residuals']

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    fig.suptitle(f'Diagnostic Plots for {symbol} - {model_name}', fontsize=16)

    # Actual vs. Predicted Scatter Plot
    ax1.scatter(actuals, predictions, alpha=0.3)
    ax1.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
    ax1.set_xlabel("Actual Returns")
    ax1.set_ylabel("Predicted Returns")
    ax1.set_title("Actual vs. Predicted Returns")

    # Residuals vs. Predicted Values
    ax2.scatter(predictions, residuals, alpha=0.3)
    ax2.axhline(0, color='r', linestyle='--')
    ax2.set_xlabel("Predicted Returns")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals vs. Predicted Values (Check for Heteroscedasticity)")
    
    # Q-Q Plot of Residuals
    sm.qqplot(residuals, line='45', fit=True, ax=ax3)
    ax3.set_title("Q-Q Plot of Residuals (Check for Normality)")

    # ACF Plot of Residuals
    plot_acf(residuals, lags=30, ax=ax4)
    ax4.set_title("ACF of Residuals (Check for Autocorrelation)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path_dir:
        save_path = save_path_dir / f"{symbol}_{model_name}_diagnostics.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        # print(f"Saved diagnostic plot to {save_path}")
    else:
        plt.show()
    plt.close()

def run_statistical_summary_for_stock(df_stock: pd.DataFrame, target_col: str, feature_cols: list):
    """Runs an OLS regression using statsmodels to get detailed statistical tests."""
    print(f"\n--- Detailed Statistical Summary for {df_stock['SYMBOL'].iloc[0]} ---")
    y = df_stock[target_col]
    X = df_stock[feature_cols]
    X = sm.add_constant(X) # Add constant for intercept term

    model = sm.OLS(y, X).fit()
    print(model.summary())
    print("-" * 70)


# --- Main Pipeline Function (for import) ---
def run_linear_models_pipeline(data_path: Path, params_dir: Path, preds_dir: Path, figures_dir: Path, split_datetime: str):
    """
    Main function to run the linear regression models pipeline.
    Loads data, creates lag features, runs regressions, and generates reports.
    """
    print("--- Running Linear Benchmarks Pipeline ---")

    # Config params for the linear regression models
    NUM_LAGS = 5
    MIN_DAILY_RETURNS = 15
    TARGET_COL = 'LOG_RETURN_NoOVERNIGHT'
    MODELS_TO_RUN = ['linear', 'ridge', 'lasso']

    # Load and process data
    returns_df = filter_trading_returns(load_data(data_path))
    
    # Create lag features
    df_with_features = create_lag_features(returns_df, NUM_LAGS, TARGET_COL, MIN_DAILY_RETURNS)
    feature_cols = [f'{TARGET_COL}_lag_{i}' for i in range(1, NUM_LAGS + 1)]

    # Run regressions for all stocks
    all_results = run_regressions_for_all_stocks(df_with_features, TARGET_COL, feature_cols, MODELS_TO_RUN, split_datetime)

    # Save model outputs and generate plots
    for model_name, results_df in all_results.items():
        if not results_df.empty:
            results_list = results_df.to_dict('records')
            save_model_outputs(results_list, model_name, params_dir, preds_dir)

            # Plot metrics and coefficients
            plot_metric_distributions(results_df, model_name.upper(), save_path_dir=figures_dir)
            plot_coefficient_distributions(results_df, model_name.upper(), save_path_dir=figures_dir)

            # Generate diagnostics for the first 2 stocks
            for result_dict in results_list[:2]:
                result_dict['model_name'] = model_name.upper()
                plot_diagnostic_for_stock(pd.Series(result_dict), save_path_dir=figures_dir)

    print("--- Linear Benchmarks Pipeline Complete ---")

if __name__ == "__main__":
    # --- Configuration ---
    BASE_DIR = Path.cwd()
    DATA_PATH = BASE_DIR / "data" / "processed" / "high_10m.parquet"
    RESULTS_DIR = BASE_DIR / "results"
    FIGURES_DIR = RESULTS_DIR / "figures" / "linear_models"
    PARAMS_DIR = RESULTS_DIR / "parameters"
    PREDS_DIR = RESULTS_DIR / "predictions"

    # Default split date for train/test
    DEFAULT_SPLIT_DATETIME = '2021-12-27 00:00:00'

    # Call the main pipeline function
    run_linear_models_pipeline(
        data_path=DATA_PATH,
        params_dir=PARAMS_DIR,
        preds_dir=PREDS_DIR,
        figures_dir=FIGURES_DIR,
        split_datetime=DEFAULT_SPLIT_DATETIME
    )

    # EXAMPLE_STOCK = 'TWLO'
    # Run Detailed Statistical Summary for an Example Stock
    # if EXAMPLE_STOCK:
    #     example_stock_df = df_with_features[df_with_features['SYMBOL'] == EXAMPLE_STOCK]
    #     if not example_stock_df.empty:
    #         run_statistical_summary_for_stock(example_stock_df, TARGET_COL, feature_cols)