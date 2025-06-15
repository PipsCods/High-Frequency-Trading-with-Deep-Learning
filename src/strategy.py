# Packages imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from pathlib import Path

# =================================================================
# Helper Functions
# =================================================================
def first_weights_given_returns(
    returns: np.ndarray,
    prev_weights: np.ndarray,
    tc: float = 0.001
) -> tuple[np.ndarray, float]:
    """
    Compute the weight vector w that maximizes:
        returns^T w  -  tc * ||w - prev_weights||_1
    subject to sum(w) == 1 and w >= 0.
    """
    if returns.ndim != 1 or prev_weights.ndim != 1:
        raise ValueError("`returns` and `prev_weights` must be one‐dimensional arrays.")
    if returns.shape[0] != prev_weights.shape[0]:
        raise ValueError("`returns` and `prev_weights` must have the same length.")
    n = returns.shape[0]
    
    weights = cp.Variable(n)
    l1_diff = cp.norm1(weights - prev_weights)
    total_cost_expr = tc * l1_diff
    
    objective = cp.Maximize(returns @ weights - total_cost_expr)
    
    # Original constraints from the provided file
    constraints = [
        cp.sum(weights) == 1,
        weights >= -0.5
    ]
    
    # Original solver from the provided file
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    if weights.value is None:
        raise ValueError(f"Optimization failed. Problem status: {prob.status}. also {weights.value}")

    w_opt = weights.value.flatten()
    w_opt = np.maximum(w_opt, 0.0)
    w_opt /= w_opt.sum()
    
    total_cost = tc * np.linalg.norm(w_opt - prev_weights, ord=1)
    
    return w_opt, total_cost

def strategy(
    returns: pd.DataFrame,
    prev_weights: np.ndarray = None,
    tc: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """
    Execute the trading strategy based on forecasted returns and previous weights.
    """
    T, N = returns.shape
    if prev_weights is None or len(prev_weights) == 0:
        w_prev = np.zeros(N) # Original logic starts with zero-weight portfolio
    else:
        w_prev = prev_weights.copy()

    weights_history = np.zeros((T, N))
    total_cost = []

    for t in range(T):
        returns_array = returns.iloc[t].to_numpy()
        returns_array = np.nan_to_num(returns_array, nan=0.0)

        w_new, cost_t = first_weights_given_returns(returns_array, w_prev, tc)
        
        weights_history[t, :] = w_new
        total_cost.append(cost_t)
        w_prev = w_new

    return weights_history, np.array(total_cost)

def model_evaluation(weights:np.ndarray, real_returns:np.ndarray, history_of_total_cost:np.ndarray):
    """Calculates the final portfolio returns after costs."""
    return (weights*real_returns).sum(axis=1) - (history_of_total_cost)

def cleandata(pred_df, actual_dataset):
    """
    Original cleandata function to align predictions and actuals.
    Note: The original file had a RETURN column which is not in the processed data.
    Using 'RETURN_NoOVERNIGHT' as the logical equivalent.
    """
    actual_dataset = actual_dataset.drop(columns=['ALL_EX', 'MID_OPEN', 'SUM_DELTA'], errors='ignore')
    actual_dataset['DATETIME'] = pd.to_datetime(actual_dataset['DATE'].astype(str) + ' ' + actual_dataset['TIME'].astype(str))
    actual_dataset = actual_dataset.drop(columns=['TIME', 'DATE'])
    
    # Using 'RETURN_NoOVERNIGHT' as it's the primary return column in the data
    actual_dataset = actual_dataset.pivot(index="DATETIME", columns="SYMBOL", values="RETURN_NoOVERNIGHT")
    
    predicted_col = pred_df.columns
    actual_dataset = actual_dataset[predicted_col]
    
    first_date = pred_df.index[0]
    actual_df = actual_dataset.loc[first_date:, :]
    
    actual_df = actual_df.reset_index(drop=True)
    pred_df = pred_df.reset_index(drop=True)
    
    actual_df.fillna(0, inplace=True)
    pred_df.fillna(0, inplace=True)
    
    actual_df = actual_df.loc[actual_df.index.isin(pred_df.index)]
    actual_df = actual_df.loc[pred_df.index]
    
    return pred_df, actual_df

# =================================================================
# Main Pipeline Function
# =================================================================
def run_strategy_pipeline(preds_dir: Path, transformer_preds_dir: Path, processed_data_path: Path, figures_dir: Path):
    """
    Main function to run the backtesting pipeline, preserving original logic.
    """
    # This logic is taken directly from the original script to ensure identical results
    # It uses a specific transformer experiment to define the universe of stocks for all benchmarks.
    try:
        specific_transformer_file = transformer_preds_dir / '100_time_cross-sectional_1_prediction.csv'
        data_transformer = pd.read_csv(specific_transformer_file)
        data_transformer = data_transformer.pivot(index="index", columns="stock", values="actual")
        stocks = data_transformer.columns
    except FileNotFoundError:
        print(f"Warning: Specific transformer file not found at {specific_transformer_file}. Cannot run benchmark backtests.")
        stocks = None

    # --- Part 1: Backtest Benchmark Models ---
    if stocks is not None:
        print("\n--- Backtesting Benchmark Models ---")
        actual_dataset = pd.read_parquet(processed_data_path)
        names = ['ridge', 'linear', 'lasso', 'arima']
        for name in names:
            print(f"Processing benchmark: {name}")
            try:
                data = pd.read_parquet(preds_dir / f'{name}_predictions.parquet')
            except FileNotFoundError:
                print(f"  Prediction file for {name} not found. Skipping.")
                continue

            pred_df = data.pivot_table(index="DATETIME", columns="SYMBOL", values="predicted_return", aggfunc="mean")
            
            # Using original cleandata function and stock filtering logic
            pred_df, actual_df = cleandata(pred_df.copy(), actual_dataset.copy())
            
            missing = [t for t in stocks if t not in actual_df.columns]
            if missing:
                print(f"  Dropping {len(missing)} missing tickers.")
            
            valid_stocks = [t for t in stocks if t in actual_df.columns]
            actual_df = actual_df[valid_stocks]
            pred_df = pred_df[valid_stocks]

            returns_strategy = {}
            for i, transaction_cost in enumerate([0, 0.0001, 0.0005, 0.001]):
                weights_history, total_cost = strategy(pred_df, tc=transaction_cost)
                returns_strategy[i] = model_evaluation(weights_history, np.nan_to_num(actual_df.values, nan=0.0), total_cost)

            plt.figure(figsize=(8, 4))
            plt.plot(returns_strategy[0].cumsum(), label='no fees')
            plt.plot(returns_strategy[1].cumsum(), label='1 basis point')
            plt.plot(returns_strategy[2].cumsum(), label='5 basis point')
            plt.plot(returns_strategy[3].cumsum(), label='10 basis point')
            plt.plot(actual_df.mean(axis=1).cumsum(), label='market', linestyle='--')
            plt.xlabel('Time step')
            plt.legend()
            plt.ylabel('Cumulative Return')
            plt.title(f'{name}')
            plt.grid(True)
            plt.tight_layout()
            
            output_path = figures_dir / f"{name}_strategy.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            plt.close() # Close plot to free memory
            print(f"  Saved plot to {output_path}")

    # --- Part 2: Backtest Transformer Models ---
    print("\n--- Backtesting Transformer Models ---")
    transformer_model_names = [
        '100_cross-sectional_None_0.01_prediction', '100_cross-sectional_None_0.5_prediction',
        '100_cross-sectional_None_1_prediction', '100_cross-sectional_time_0.01_prediction',
        '100_cross-sectional_time_0.5_prediction', '100_cross-sectional_time_1_prediction',
        '100_time_cross-sectional_0.01_prediction', '100_time_cross-sectional_0.5_prediction',
        '100_time_cross-sectional_1_prediction', '100_time_None_0.01_prediction',
        '100_time_None_0.5_prediction', '100_time_None_1_prediction'
    ]

    for model_name in transformer_model_names:
        print(f"Processing transformer: {model_name}")
        try:
            data = pd.read_csv(transformer_preds_dir / f'{model_name}.csv')
        except FileNotFoundError:
            print(f"  Prediction file for {model_name} not found. Skipping.")
            continue
        
        pred_df = data.pivot(index="index", columns="stock", values="pred")
        actual_df = data.pivot(index="index", columns="stock", values="actual")
        
        returns_strategy = {}
        for i, transaction_cost in enumerate([0, 0.0001, 0.0005, 0.001]):
            weights_history, total_cost = strategy(pred_df, tc=transaction_cost)
            returns_strategy[i] = model_evaluation(weights_history, actual_df.values, total_cost)

        plt.figure(figsize=(8, 4))
        plt.plot(returns_strategy[0].cumsum(), label='no fees')
        plt.plot(returns_strategy[1].cumsum(), label='1 basis point')
        plt.plot(returns_strategy[2].cumsum(), label='5 basis point')
        plt.plot(returns_strategy[3].cumsum(), label='10 basis point')
        plt.plot(actual_df.mean(axis=1).cumsum(), label='market', linestyle='--')
        plt.xlabel('Time step')
        plt.legend()
        plt.ylabel('Cumulative Return')
        plt.title(f'{model_name}')
        plt.grid(True)
        plt.tight_layout()

        output_path = figures_dir / f"{model_name}_strategy.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"  Saved plot to {output_path}")

    print("\n--- Strategy Backtesting Pipeline Complete ---")