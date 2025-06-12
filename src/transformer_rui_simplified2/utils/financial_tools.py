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
