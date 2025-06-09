import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def first_weights_given_returns(
    returns: np.ndarray,
    prev_weights: np.ndarray,
    tc: float = 0.001
) -> tuple[np.ndarray, float]:
    """
    Compute the weight vector w that maximizes:
        returns^T w  -  tc * ||w - prev_weights||_1
    subject to sum(w) == 1 and w >= 0.

    Parameters
    ----------
    returns : np.ndarray, shape (n,)
        Forecasted returns for each asset.
    prev_weights : np.ndarray, shape (n,)
        Previous weight allocation (must sum to 1, nonnegative).
    tc : float, default=0.001
        Transaction‐cost coefficient (multiplies the L1 distance).

    Returns
    -------
    new_weights : np.ndarray, shape (n,)
        The optimized weights (sum to 1, all ≥ 0).
    total_cost : float
        The realized transaction cost: tc * ||new_weights - prev_weights||_1
    """
    # Ensure inputs are 1-D arrays of equal length
    if returns.ndim != 1 or prev_weights.ndim != 1:
        raise ValueError("`returns` and `prev_weights` must be one‐dimensional arrays.")
    if returns.shape[0] != prev_weights.shape[0]:
        raise ValueError("`returns` and `prev_weights` must have the same length.")
    n = returns.shape[0]
    
    # CVXPY variable for new weights
    weights = cp.Variable(n)
    
    # L1 transaction‐cost term: ||weights - prev_weights||_1
    l1_diff = cp.norm1(weights - prev_weights)
    total_cost_expr = tc * l1_diff
    
    # Objective: maximize returns^T * weights - total_cost_expr
    objective = cp.Maximize(returns @ weights - total_cost_expr)
    
    # Constraints: sum(weights) == 1, weights >= 0
    constraints = [
        cp.sum(weights) == 1,
        weights >= -0.5
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    if weights.value is None:
        raise ValueError(f"Optimization failed. Problem status: {prob.status}. also {weights.value}")
    # Extract the numeric solution
    w_opt = weights.value.flatten()
    # Clip any tiny negatives, then re‐normalize to ensure sum-to-1 exactly
    w_opt = np.maximum(w_opt, 0.0)
    w_opt /= w_opt.sum()
    
    # Compute the actual transaction cost as a float
    total_cost = tc * np.linalg.norm(w_opt - prev_weights, ord=1)
    
    return w_opt, total_cost

def strategy(
    returns: pd.DataFrame,
    prev_weights: np.ndarray = None,
    tc: float = 0.001
) -> tuple[np.ndarray, float]:
    """
    Execute the trading strategy based on forecasted returns and previous weights.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame with forecasted returns for each asset (rows = time, cols = assets).
    prev_weights : np.ndarray
        Previous weight allocation (must sum to 1, nonnegative). If None, starts as zeros.
    tc : float, default=0.001
        Transaction‐cost coefficient.

    Returns
    -------
    new_weights : np.ndarray
        A (T × N) array of optimized weights for each time t (each row sums to 1, all ≥ 0).
    total_cost : float
        The sum of transaction costs incurred over all T periods.
    """
    # Number of periods (T) and number of assets (N)
    T, N = returns.shape
    # If no prev_weights provided, start with a zero vector of size N
    if prev_weights is None or len(prev_weights) == 0:
        w_prev = np.zeros(N)
    else:
        w_prev = prev_weights.copy()

    # Prepare an array to hold weights at each time t, and accumulate costs
    weights_history = np.zeros((T, N))
    total_cost = []

    # Loop over each time index t
    for t in range(T):
        # Extract the t-th row of returns as a 1D array of length N
        returns_array = returns.iloc[t].to_numpy()
        returns_array = np.nan_to_num(returns_array, nan=0.0)

        # Compute new weights and transaction cost at time t
        w_new, cost_t = first_weights_given_returns(returns_array, w_prev, tc)
        
        # Store results
        weights_history[t, :] = w_new
        total_cost.append(cost_t)
        
        # Update w_prev for the next iteration
        w_prev = w_new

    return weights_history, np.array(total_cost)

def model_evaluation(weights:np.ndarray, real_returns:np.ndarray, history_of_total_cost:np.ndarray):
    return (weights*real_returns).sum(axis=1) - (history_of_total_cost)

def cleandata(pred_df,actual_dataset):
    actual_dataset=actual_dataset.drop(columns=['ALL_EX','MID_OPEN','SUM_DELTA'])
    actual_dataset['DATETIME'] = pd.to_datetime(actual_dataset['DATE'].astype(str) + ' ' + actual_dataset['TIME'].astype(str))
    actual_dataset=actual_dataset.drop(columns=['TIME','DATE'])
    actual_dataset= actual_dataset.pivot(index="DATETIME", columns="SYMBOL", values="RETURN")
    predicted_col=pred_df.columns
    actual_dataset=actual_dataset[predicted_col]
    first_date=pred_df.index[0]
    actual_df=actual_dataset.loc[first_date:,:]
    actual_df = actual_df.reset_index(drop=True)
    pred_df=pred_df.reset_index(drop=True)
    actual_df.fillna(0)
    pred_df.fillna(0)
    actual_df = actual_df.loc[actual_df.index.isin(pred_df.index)]
    actual_df = actual_df.loc[pred_df.index]
    actual_df=actual_df.iloc[:,:1000]
    pred_df=pred_df.iloc[:,:1000]
    return pred_df,actual_df

names=['ridge','linear','garch','lasso','arima']
for name in names:
#DATA UPLOADING
    data=pd.read_parquet(f'/Users/emanueledurante/Desktop/predictions/{name}_predictions.parquet')
    pred_df = data.pivot_table(index="DATETIME", columns="SYMBOL", values="predicted_return",aggfunc="mean")
    actual_dataset=pd.read_parquet('/Users/emanueledurante/Desktop/LGMB/lausanne/epfl/MLfinance/High-Frequency-Trading-with-Deep-Learning/data/high_10m.parquet')
    #FUNCTIONS
    returns_strategy=dict()
    pred_df,actual_df=cleandata(pred_df,actual_dataset)
    #PLOTS
    for i,transaction_cost in enumerate([0, 0.0001,0.0005,0.001]):
            
        weights_history, total_cost=strategy(pred_df,tc=transaction_cost)
        returns_strategy[i]=model_evaluation(weights_history,np.nan_to_num(actual_df.values, nan=0.0),total_cost)
    plt.figure(figsize=(8,4))
    plt.plot(returns_strategy[0].cumsum(),label='no fees')
    plt.plot(returns_strategy[1].cumsum(),label='1 basis point')
    plt.plot(returns_strategy[2].cumsum(),label='5 basis point')
    plt.plot(returns_strategy[3].cumsum(),label='10 basis point')
    plt.xlabel('Time step')
    plt.legend()
    plt.ylabel('Cumulative Return')
    plt.title(f'{name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{name}_strategy.png')
    plt.show()
