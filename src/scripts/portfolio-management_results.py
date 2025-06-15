import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper_portfolio_management import first_weights_given_returns,strategy,model_evaluation,cleandata

data_transformer=pd.read_csv('results/transformer_experiments/100_time_cross-sectional_1_prediction.csv')
data_transformer = data_transformer.pivot(index="index", columns="stock", values="actual")
stocks=data_transformer.columns
names=['ridge','linear','lasso','arima']
for name in names:
#DATA UPLOADING
    data=pd.read_parquet(f'results/predictions/{name}_predictions.parquet')
    pred_df = data.pivot_table(index="DATETIME", columns="SYMBOL", values="predicted_return",aggfunc="mean")
    actual_dataset=pd.read_parquet('data/high_10m.parquet')
    #INITIALIZING FUNCTIONS
    returns_strategy=dict()
    pred_df,actual_df=cleandata(pred_df,actual_dataset)
    
    missing = [t for t in stocks if t not in actual_df.columns]
    if missing:
        print("dropping missing tickers:", missing)
    stocks = [t for t in stocks if t in actual_df.columns]      


    actual_df = actual_df[stocks]      
    pred_df   = pred_df[stocks]

    actual_df=actual_df[stocks]
    pred_df=pred_df[stocks]
    #RUNNING ANALYSIS AND PLOTS
    for i,transaction_cost in enumerate([0, 0.0001,0.0005,0.001]):
            
        weights_history, total_cost=strategy(pred_df,tc=transaction_cost)
        returns_strategy[i]=model_evaluation(weights_history,np.nan_to_num(actual_df.values, nan=0.0),total_cost)
    plt.figure(figsize=(8,4))
    plt.plot(returns_strategy[0].cumsum(),label='no fees')
    plt.plot(returns_strategy[1].cumsum(),label='1 basis point')
    plt.plot(returns_strategy[2].cumsum(),label='5 basis point')
    plt.plot(returns_strategy[3].cumsum(),label='10 basis point')
    plt.plot(actual_df.mean(axis=1).cumsum(),label='market',linestyle='--')
    plt.xlabel('Time step')
    plt.legend()
    plt.ylabel('Cumulative Return')
    plt.title(f'{name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{name}_strategy.png')
    plt.show()

returns_strategy=dict()
for j in ['100_cross-sectional_None_0.01_prediction',
          '100_cross-sectional_None_0.5_prediction',
          '100_cross-sectional_None_1_prediction',
          '100_cross-sectional_time_0.01_prediction',
          '100_cross-sectional_time_0.5_prediction',
          '100_cross-sectional_time_1_prediction',
          '100_time_cross-sectional_0.01_prediction',
          '100_time_cross-sectional_0.5_prediction',
          '100_time_cross-sectional_1_prediction',
          '100_time_None_0.01_prediction',
          '100_time_None_0.5_prediction',
          '100_time_None_1_prediction']:
    data=pd.read_csv(f'results/transformer_experiments/{j}.csv')
    pred_df = data.pivot(index="index", columns="stock", values="pred")
    actual_df = data.pivot(index="index", columns="stock", values="actual")
    for i,transaction_cost in enumerate([0, 0.0001,0.0005,0.001]):
            
        weights_history, total_cost=strategy(pred_df,tc=transaction_cost)
        returns_strategy[i]=model_evaluation(weights_history,actual_df.values,total_cost)

    plt.figure(figsize=(8,4))
    plt.plot(returns_strategy[0].cumsum(),label='no fees')
    plt.plot(returns_strategy[1].cumsum(),label='1 basis point')
    plt.plot(returns_strategy[2].cumsum(),label='5 basis point')
    plt.plot(returns_strategy[3].cumsum(),label='10 basis point')
    plt.plot(actual_df.mean(axis=1).cumsum(),label='market',linestyle='--')
    plt.xlabel('Time step')
    plt.legend()
    plt.ylabel('Cumulative Return')
    plt.title(f'{j}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{j}.png')
    plt.show()