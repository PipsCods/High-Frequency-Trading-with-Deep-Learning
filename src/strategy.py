import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from pathlib import Path
from tqdm import tqdm

def get_optimal_weights(returns: np.ndarray, prev_weights: np.ndarray, tc: float) -> tuple[np.ndarray, float]:
    """
    Computes the optimal portfolio weights for a single period.

    Args:
        returns: Forecasted returns for each asset.
        prev_weights: The weight allocation from the previous period.
        tc: Transaction cost coefficient.

    Returns:
        A tuple containing the new optimized weights and the realized transaction cost.
    """
    n = returns.shape[0]
    weights = cp.Variable(n)
    
    # Objective: Maximize returns minus transaction costs
    l1_diff = cp.norm1(weights - prev_weights)
    objective = cp.Maximize(returns @ weights - (tc * l1_diff))
    
    # Constraints: Fully invested, no short-selling (long-only)
    constraints = [cp.sum(weights) == 1, weights >= 0]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS) # Using ECOS solver, which is good for this type of problem
    
    # If solver fails, hold the previous position
    if prob.status != cp.OPTIMAL:
        # print(f"Warning: Solver failed. Status: {prob.status}. Holding previous weights.")
        return prev_weights, 0.0

    w_opt = weights.value.flatten()
    w_opt = np.maximum(w_opt, 0.0) # Clip tiny negatives
    w_opt /= w_opt.sum() # Re-normalize to ensure sum-to-1
    
    cost = tc * np.linalg.norm(w_opt - prev_weights, ord=1)
    return w_opt, cost

def run_backtest(predictions_df: pd.DataFrame, tc: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Executes the trading strategy over a history of forecasted returns.

    Args:
        predictions_df: DataFrame of forecasted returns (rows=time, cols=assets).
        tc: Transaction cost coefficient.

    Returns:
        A tuple of (weights_history, cost_history).
    """
    T, N = predictions_df.shape
    w_prev = np.full(N, 1/N) # Start with an equally weighted portfolio
    
    weights_history = np.zeros((T, N))
    cost_history = np.zeros(T)

    for t in range(T):
        returns_t = predictions_df.iloc[t].to_numpy(na_value=0.0)
        w_new, cost_t = get_optimal_weights(returns_t, w_prev, tc)
        
        weights_history[t, :] = w_new
        cost_history[t] = cost_t
        w_prev = w_new
        
    return weights_history, cost_history

# --- Data Handling and Alignment ---
def load_and_align_data(pred_path: Path, actual_returns_df: pd.DataFrame):
    """
    Loads predictions and aligns them with actual returns on a common datetime index.
    """
    # Load predictions based on file type
    if pred_path.suffix == '.csv': # Transformer predictions
        preds = pd.read_csv(pred_path)
        pred_df = preds.pivot(index="index", columns="stock", values="pred")
        pred_df.index = pd.to_datetime(pred_df.index)
        pred_df.index.name = "DATETIME"
    else: # Benchmark predictions
        preds = pd.read_parquet(pred_path)
        pred_df = preds.pivot_table(index="DATETIME", columns="SYMBOL", values="predicted_return", aggfunc="mean")

    # Align with actual returns
    actuals_ts = actual_returns_df.pivot_table(index="DATETIME", columns="SYMBOL", values="RETURN_NoOVERNIGHT")
    
    # Find common stocks and time range
    common_stocks = pred_df.columns.intersection(actuals_ts.columns)
    if len(common_stocks) == 0:
        return None, None
        
    pred_df = pred_df[common_stocks]
    actual_df = actuals_ts[common_stocks]
    
    common_index = pred_df.index.intersection(actual_df.index)
    
    pred_df = pred_df.loc[common_index].fillna(0)
    actual_df = actual_df.loc[common_index].fillna(0)
    
    if pred_df.empty or actual_df.empty:
        return None, None

    return pred_df, actual_df

# --- Plotting and Evaluation ---
def plot_cumulative_returns(tc_results: dict, market_returns: pd.Series, model_name: str, figures_dir: Path):
    """
    Plots and saves the cumulative returns of the strategy vs. the market.
    """
    plt.style.use('seaborn-v0_8-grid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot strategy returns for different transaction costs
    for tc, returns in tc_results.items():
        label = f'Strategy (TC={tc*10000:.0f} bps)' if tc > 0 else 'Strategy (No Fees)'
        ax.plot(np.cumsum(returns), label=label, linewidth=2)

    # Plot market benchmark (equal weight)
    ax.plot(np.cumsum(market_returns), label='Market Benchmark (Equal Weight)', linestyle='--', color='black', linewidth=2)

    ax.set_title(f'Cumulative Returns for Model: {model_name}', fontsize=16)
    ax.set_xlabel('Time Steps (10-minute intervals)', fontsize=12)
    ax.set_ylabel('Cumulative Log Return', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    output_path = figures_dir / f"{model_name}_strategy_performance.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved performance plot to {output_path}")

# --- Main Pipeline Function ---
def run_strategy_pipeline(preds_dir: Path, processed_data_path: Path, figures_dir: Path):
    """
    Main function to run the backtesting pipeline for all available models.
    """
    print("\n--- Running Trading Strategy & Backtesting Pipeline ---")
    
    # Find all prediction files from benchmarks and transformer experiments
    benchmark_files = list(preds_dir.glob("*_predictions.parquet"))
    transformer_files = list((preds_dir.parent / "transformer_experiments").glob("*_prediction.csv"))
    all_pred_files = benchmark_files + transformer_files

    if not all_pred_files:
        print("Error: No prediction files found. Please run model training stages first.")
        return

    # Load actual returns data once
    try:
        raw_df = pd.read_parquet(processed_data_path)
        raw_df['DATETIME'] = pd.to_datetime(raw_df['DATE'].astype(str) + ' ' + raw_df['TIME'].astype(str))
    except Exception as e:
        print(f"Error loading processed data from {processed_data_path}: {e}")
        return

    transaction_costs = [0, 0.0001, 0.0005, 0.001]
    
    for pred_path in tqdm(all_pred_files, desc="Backtesting Models"):
        model_name = pred_path.stem.replace('_predictions', '').replace('_prediction', '')
        
        pred_df, actual_df = load_and_align_data(pred_path, raw_df)

        if pred_df is None or actual_df is None:
            print(f"Skipping {model_name}: No common data after alignment.")
            continue
        
        market_returns = actual_df.mean(axis=1)
        tc_results = {}

        for tc in transaction_costs:
            weights, costs = run_backtest(pred_df, tc=tc)
            strategy_returns = (weights * actual_df.to_numpy()).sum(axis=1) - costs
            tc_results[tc] = strategy_returns
        
        plot_cumulative_returns(tc_results, market_returns, model_name, figures_dir)

    print("\n--- Trading Strategy & Backtesting Pipeline Complete ---")