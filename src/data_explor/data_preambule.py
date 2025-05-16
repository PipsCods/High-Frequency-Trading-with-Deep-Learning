"""
Market Return Lag Feature Engineering Script
Author: Rui Azevedo
Date: 16.05.2025

This script loads high-frequency stock data, creates lag features, and selects a random subset of stocks
for machine learning benchmarking (e.g., regression tasks). The final DataFrame is MultiIndexed by ['symbol', 'timestamp'].
"""

# =========================== #
#         PACKAGES            #
# =========================== #

import numpy as np
import pandas as pd
import argparse



# =========================== #
#        CONFIGURATION        #
# =========================== #

def parse_args():
    parser = argparse.ArgumentParser(description="Loading data script")
    parser.add_argument('--data_path', type=str, required=True, help='Path to processed parquet file')
    parser.add_argument('--num_lags', type=int, required=True, help='Lags to introduce')
    parser.add_argument('--num_random_stocks', type=int, required=True, help='Number of random stocks to analyse')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output file')
    parser.add_argument('--seed', type=int, required=True, help='Seed')
    return parser.parse_args()

# =========================== #
#            MAIN             #
# =========================== #

def main():
    args = parse_args()
    print("Loading data...")
    df = load_data(args.data_path)

    print("Creating lag features...")
    df = create_lag_features(df, args.num_lags)
    df = df.dropna()

    print(f"Sampling {args.num_random_stocks} random stocks...")
    df = sample_random_stocks(df, args.num_random_stocks, args.seed)

    print("Data ready for modeling.")
    print(df.head())

    print(f"Saving to {args.output_path}")
    df.to_parquet(args.output_path)

# =========================== #
#         FUNCTIONS           #
# =========================== #

def load_data(file_path):
    """Load and preprocess raw data from Parquet file."""
    df = pd.read_parquet(file_path)

    # Standardize column names to lowercase
    df = df.rename(columns=str.lower)

    # Create timestamp
    df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))

    # Drop old date/time
    df = df.drop(['date', 'time'], axis=1, errors='ignore')

    # Set MultiIndex: (symbol, timestamp)
    df = df.set_index(['symbol', 'timestamp'])

    # Sort for groupby operations
    df = df.sort_index()

    return df

def create_lag_features(df, lags, target_col='return'):
    """Add lag features per symbol."""
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f'return_lag_{lag}'] = df.groupby(level=0)[target_col].shift(lag)
    return df

def sample_random_stocks(df, num_stocks, seed=42):
    """Sample random stocks from the dataset."""
    symbols = df.index.get_level_values('symbol').unique()
    rng = np.random.default_rng(seed)
    selected = rng.choice(symbols, num_stocks, replace=False)
    return df.loc[selected]

if __name__ == "__main__":
    main()