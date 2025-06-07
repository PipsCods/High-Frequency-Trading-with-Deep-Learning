# Package imports
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Time series analysis libraries
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from arch import arch_model

# Import utils
from src.utils import load_data
from src.data_processing import filter_trading_returns

# Suppress warnings for cleaner output
# warnings.filterwarnings("ignore")

# --- Function Definitions ---

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
    print("\n--- Running auto_arima to find the best ARIMA model ---")
    
    # We set d=0 because returns are already a differenced series (price[t] - price[t-1]).
    # We are looking for an ARMA(p,q) model.
    # trace=True will print the results of each model fit.
    arima_model = auto_arima(
        series,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        d=0,             # Crucial for returns data
        seasonal=False,  # Assuming no seasonality for now
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    print("\n--- Auto ARIMA Model Summary ---")
    print(arima_model.summary())
    return arima_model

def run_garch_analysis(series: pd.Series):
    """
    Fits a GARCH(1,1) model to analyze the volatility of the series.

    Args:
        series: The time series of returns for a single stock.

    Returns:
        The fitted GARCH model result object.
    """
    print("\n--- Running GARCH(1,1) analysis for volatility ---")
    
    # A standard GARCH(1,1) model is a good starting point.
    # p=1 refers to the ARCH term (past squared residuals).
    # q=1 refers to the GARCH term (past conditional variances).
    garch_model = arch_model(series * 100, vol='Garch', p=1, q=1, dist='Normal')
    
    # We multiply by 100 to help the optimizer converge better
    results = garch_model.fit(update_freq=10, disp='off')

    print("\n--- GARCH Model Summary ---")
    print(results.summary())
    return results

# --- Main Execution Block ---
if __name__ == '__main__':
    BASE_DIR = Path.cwd()
    DATA_PATH = ".." / BASE_DIR / "data" / "processed" / "high_10m.parquet"

    # 1. Load and processed data
    raw_df = load_data(DATA_PATH)
    returns_df = filter_trading_returns(raw_df)

    # 2. Select a single stock for analysis
    # Time series models are run on a single series, not a whole DataFrame.
    SYMBOL_TO_ANALYZE = 'LOVE'
    
    stock_series = returns_df[returns_df['SYMBOL'] == SYMBOL_TO_ANALYZE]['LOG_RETURN_NoOVERNIGHT'].dropna()

    if stock_series.empty:
        print(f"No data available for symbol {SYMBOL_TO_ANALYZE}. Please choose another.")
    else:
        # 3. Run the analyses
        plot_autocorrelation(stock_series, SYMBOL_TO_ANALYZE)
        run_auto_arima(stock_series)
        run_garch_analysis(stock_series)