"""
PACKAGES
"""
import numpy as np
import pandas as pd

"""
Financial tools
"""
import numpy as np
import pandas as pd

def sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Compute the Sharpe ratio for a return series.

    Parameters:
        returns (array-like): Series or array of strategy returns
        risk_free_rate (float): Risk-free rate (default 0.0)

    Returns:
        float: Sharpe ratio
    """
    returns = pd.Series(returns).dropna()
    excess = returns - risk_free_rate

    if excess.std() == 0:
        return np.nan  # avoid divide-by-zero

    return excess.mean() / excess.std()

"""
Risk classification
"""
def assign_risk_class_every_timestamp(df, window_size=100, q=10):
    """
    Assigns a cross-sectional risk class (1 to q) to each asset at each timestamp,
    based on rolling realized volatility over a fixed-size window of returns.

    This method is well-suited for high-frequency data. Traditional risk classification
    approaches like CAPM often fail in this context due to extremely low variance
    of the market return over short intervals, which causes beta estimates to become
    unstable or explode. In contrast, realized volatility computed directly from asset-level
    return data provides a robust, model-free, and interpretable measure of short-term risk.

    Parameters:
    -----------
    df : pd.DataFrame
        MultiIndexed DataFrame with ['symbol', 'timestamp'] as index and a 'return' column.
    window_size : int
        Number of time steps (e.g., 10-minute intervals) used to compute realized volatility.
    q : int
        Number of risk classes (e.g., 10 for deciles).

    Returns:
    --------
    pd.DataFrame
        Original DataFrame with a new column 'risk_class' that assigns a volatility class
        (1 = lowest volatility, q = highest) to each asset at each timestamp.
    """

    # Ensure correct MultiIndex: ['symbol', 'timestamp']
    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ['symbol', 'timestamp']:
        df = df.reset_index().set_index(['symbol', 'timestamp']).sort_index()

    df = df.sort_index()
    timestamps = df.index.get_level_values('timestamp').unique()

    results = []

    # Loop over each timestamp (starting after the first complete window)
    for current_idx in range(window_size, len(timestamps)):
        window_timestamps = timestamps[current_idx - window_size:current_idx]
        label_timestamp = timestamps[current_idx]

        # Extract data from rolling window
        window_df = df[df.index.get_level_values('timestamp').isin(window_timestamps)]

        # Compute realized volatility per symbol: sum of squared returns
        rv_series = window_df.groupby('symbol')['return'].apply(lambda x: np.sum(x**2))

        # Assign quantile-based risk classes (1 = low risk, q = high risk)
        try:
            risk_class = pd.qcut(rv_series, q=q, labels=False, duplicates='drop') + 1
        except ValueError:
            continue  # skip window if too few unique values

        # Save result
        for symbol in rv_series.index:
            results.append({
                'symbol': symbol,
                'timestamp': label_timestamp,
                'risk_class': risk_class.loc[symbol]
            })

    # Compile and merge
    risk_class_df = pd.DataFrame(results).set_index(['symbol', 'timestamp'])
    df_out = df.merge(risk_class_df, how='left', left_index=True, right_index=True)

    return df_out

