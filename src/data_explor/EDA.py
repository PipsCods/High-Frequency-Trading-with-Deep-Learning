import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from pmdarima import auto_arima
import os
import sys

def describe_series(series):
    print("Description of", series.name, f"{series.describe()}:", sep="\n")
    print(f"Skew: {series.skew()}")
    print(f"Kurtosis: {series.kurtosis()}")

def plot_density_logvslin(lin_ret: pd.DataFrame, log_ret: pd.DataFrame):
    plt.figure(figsize=(12,6))
    sns.kdeplot(lin_ret.values.flatten(), color='blue', label='Linear Returns', linewidth=2, fill=True, alpha=0.3)
    sns.kdeplot(log_ret.values.flatten(), color='orange', label='Log Returns', linewidth=2, fill=True, alpha=0.3)
    plt.title("Density Plot: Linear vs Log Returns", fontsize=14, pad=20)
    plt.xlabel('Returns', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    sns.despine()
    plt.axvline(x=np.mean(lin_ret), color='blue', linestyle='--', alpha=0.7)
    plt.axvline(x=np.mean(log_ret), color='orange', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_histogram_distribution(series, n_bins=500, left_limit = -0.06, right_limit = 0.06):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.histplot(series, kde=True, bins=n_bins, color="steelblue")
    plt.title("Distribution of Return")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.xlim(left_limit, right_limit) 
    plt.show()

def plot_boxplot(series):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.boxplot(x = series, color="orange")
    plt.title("Box Plot of Return")
    plt.show()

#TODO: IS THAT USEFUL??

def plot_violin(series):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.violinplot(x=series, color="purple")
    plt.title("Violin Plot of Return")
    plt.show()

def plot_ecdf(series):
    sns.ecdfplot(series)
    plt.title("ECDF of RETURN")
    plt.xlabel("RETURN")
    plt.ylabel("Cumulative Probability")
    plt.show()

def plot_qq(series):
    plt.figure(figsize=(10, 5))
    stats.probplot(series, dist="norm", plot=plt)
    plt.title("Q-Q Plot of RETURN")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.grid()
    plt.show()

def select_quantiles(series, lower_bound=0.01, upper_bound=0.99):
    quantile_values = series.quantile([lower_bound, upper_bound])
    return quantile_values

# def select_winsorized(series, lower_percentile=0.01, upper_percentile=0.99):
#     lower_bound = select_quantiles(series, lower_percentile, upper_percentile).iloc[0]
#     upper_bound = select_quantiles(series, lower_percentile, upper_percentile).iloc[1]
#     selected_df = series[(series>= lower_bound) & (series <= upper_bound)]
#     #winsorized_series = series.clip(lower=lower_bound, upper=upper_bound)
#     return selected_df

def select_winsorized(df, column, lower_percentile=0.01, upper_percentile=0.99):
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)
    df_copy = df.copy()
    df_copy[column] = df_copy[column].clip(lower=lower_bound, upper=upper_bound)
    
    return df_copy


def distribution_correlation(returns: pd.DataFrame):
    corr_matrix = returns.corr()
    all_corrs = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
    sns.histplot(all_corrs, bins=50)
    plt.title('Distribution of Pairwise Stock Correlations')

