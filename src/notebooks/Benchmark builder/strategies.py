"""
Generating the returns values
Author: Rui Azevedo
Date: 16.05.2025

This script loads lagged return data, compute returns (simple and cumsum), compute performance values,
and print it for each of the strategy, plot if True.
"""

# =========================== #
#         PACKAGES            #
# =========================== #

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from utils import financial_tools
from tool_kit import *

# =========================== #
#         ARGUMENTS           #
# =========================== #

def parse_args():
    parser = argparse.ArgumentParser(description="Managed Returns Benchmark Script")
    parser.add_argument('--data_path', type=str, required=True, help='Path to processed parquet file')
    parser.add_argument('--model', type=str, required=True, help = 'Model to use')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--shrinkage_list', type=float, nargs='+', default=[0.01], help='List of alphas for Ridge')
    parser.add_argument('--plot', type=bool, help='Indicate true if you want to plot the results')
    return parser.parse_args()

# =========================== #
#         BUY & HOLD          #
# =========================== #

def main():
    args = parse_args()
    print("Loading data...")
    df = pd.read_parquet(args.data_path)

    print("Running Buy & Hold model...")
    bh_ret, bh_cum, bh_sharpe = buy_and_hold(df)
    print(f"Buy & Hold - SR: {bh_sharpe:.4f}")
    print(20 * "=")

    if args.model == 'ols':
        ols_df = pd.read_parquet(args.model_path + 'OLS_regression_results.parquet')
        rmt, rmt_cum, rmt_sr, ls, ls_cum, ls_sr = evaluate_strategy_from_predictions(ols_df)
        print(f"Market timing returns - SR: {rmt_sr:.4f}")
        print(f"Long short returns - SR: {ls_sr:.4f}")
        print(f"R2 score: {r2_score(ols_df['y_true'], ols_df['y_pred']):.4f}")

        if args.plot:
            # Order
            bh_cum = bh_cum.sort_index()
            rmt_cum = rmt_cum.sort_index()
            ls_cum = ls_cum.sort_index()

            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(bh_cum, label="Buy & Hold strategy", linestyle="--")
            plt.plot(rmt_cum, label="Market timing returns", linewidth=2)
            plt.plot(ls_cum, label="Long short returns", linewidth=2)

            plt.xlabel("Time")
            plt.ylabel("Cumulative Return")
            plt.title("Strategies vs. Market Benchmark")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            pass

    elif args.model == 'ridge':
        ridge_df = pd.read_parquet(args.model_path + 'ridge_regression_results.parquet')

        results_per_alpha = {}

        for alpha in args.shrinkage_list:
            df_alpha = ridge_df[ridge_df['alpha'] == float(alpha)]
            rmt, rmt_cum, rmt_sr, ls, ls_cum, ls_sr = evaluate_strategy_from_predictions(df_alpha)
            r2 = r2_score(df_alpha['y_true'], df_alpha['y_pred'])

            if alpha == 0:
                print(f"OLS - Market Timing Sharpe: {rmt_sr:.4f}, Long-Short Sharpe: {ls_sr:.4f}, R²: {r2:.4f}")
            else:
                print(f"[α={alpha}] Market Timing Sharpe: {rmt_sr:.4f}, Long-Short Sharpe: {ls_sr:.4f}, R²: {r2:.4f}")

            results_per_alpha[alpha] = {
                'rmt_cum': rmt_cum.sort_index(),
                'ls_cum': ls_cum.sort_index()
            }

        if args.plot and results_per_alpha:
            plt.figure(figsize=(12, 6))
            plt.plot(bh_cum, label="Buy & Hold strategy", linestyle="--")

            for alpha, result in results_per_alpha.items():
                plt.plot(result['rmt_cum'], label=f"Market Timing (α={alpha})")
                plt.plot(result['ls_cum'], label=f"Long-Short (α={alpha})", linestyle=":")

            plt.xlabel("Time")
            plt.ylabel("Cumulative Return")
            plt.title("Ridge Strategies vs. Market Benchmark")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


    else:
        raise Exception("Model not implemented")

def buy_and_hold(df: pd.DataFrame):
    """
    Simulate a Buy & Hold strategy across December using simple return accumulation.

    Args:
        df: A DataFrame indexed by ['symbol', 'timestamp'], containing a 'return' column.

    Returns:
        market_returns: equal-weighted return at each timestamp
        cumulative_returns: additive cumulative return (via cumsum)
        sharpe: Sharpe ratio of market_returns
    """
    df = df.reset_index()
    df = df[['symbol', 'timestamp', 'return']]

    # Sort to ensure correct time sequence
    df = df.sort_values('timestamp')

    # Equal-weighted average return at each timestamp
    market_returns = df.groupby('timestamp')['return'].mean()

    # Cumulative return via simple addition (not compounding)
    cumulative_returns = market_returns.cumsum()

    # Sharpe ratio
    sharpe = financial_tools.sharpe_ratio(market_returns)

    return market_returns, cumulative_returns, sharpe


# =========================== #
#         REGRESSION          #
# =========================== #

def evaluate_strategy_from_predictions(results: pd.DataFrame):

    # Extract the returns of each of the strategies
    returns_market_timing = results.groupby('timestamp')['market_timing_returns'].mean()
    returns_long_short = compute_dollar_neutral_long_short(results)

    # Cumulative return
    returns_market_timing_cum = returns_market_timing.cumsum()

    returns_long_short = returns_long_short.sort_index()
    returns_long_short_cum = returns_long_short['long_short_returns'].cumsum()

    # Sharpe ratio
    sr_mt = financial_tools.sharpe_ratio(returns_market_timing)
    sr_ls = financial_tools.sharpe_ratio(returns_long_short['long_short_returns'])


    return returns_market_timing, returns_market_timing_cum,sr_mt, returns_long_short, returns_long_short_cum, sr_ls

def compute_dollar_neutral_long_short(df, k=5):
    """
    Compute dollar-neutral long-short strategy.
    For each timestamp:
    - Long top-k predicted assets
    - Short bottom-k predicted assets
    - Use price to allocate equal capital to both sides
    """
    results = []

    for ts, group in df.groupby('timestamp'):
        group = group.dropna(subset=['y_pred', 'y_true', 'price'])

        if len(group) < 2 * k:
            continue

        # Sort predictions
        sorted_group = group.sort_values(by='y_pred', ascending=False)

        long = sorted_group.head(k).copy()
        short = sorted_group.tail(k).copy()

        # Calculate total price (capital) needed for long and short legs
        total_long_price = long['price'].sum()
        total_short_price = short['price'].sum()

        # Dollar-neutral weights (half capital to each side)
        long['weight'] = 0.5 * (long['price'] / total_long_price)
        short['weight'] = 0.5 * (short['price'] / total_short_price)

        # Calculate weighted returns
        long_ret = (long['y_true'] * long['weight']).sum()
        short_ret = (short['y_true'] * short['weight']).sum()

        results.append({
            'timestamp': ts,
            'long_short_returns': long_ret - short_ret,
            'long_return': long_ret,
            'short_return': short_ret
        })

    return pd.DataFrame(results).set_index('timestamp')