import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper_portfolio_management import first_weights_given_returns,strategy,model_evaluation,cleandata

names=['ridge','linear','garch','lasso','arima']
for name in names:
#DATA UPLOADING
    data=pd.read_parquet(f'/Users/emanueledurante/Desktop/predictions/{name}_predictions.parquet')
    pred_df = data.pivot_table(index="DATETIME", columns="SYMBOL", values="predicted_return",aggfunc="mean")
    actual_dataset=pd.read_parquet('/Users/emanueledurante/Desktop/LGMB/lausanne/epfl/MLfinance/High-Frequency-Trading-with-Deep-Learning/data/high_10m.parquet')
    #INITIALIZING FUNCTIONS
    returns_strategy=dict()
    pred_df,actual_df=cleandata(pred_df,actual_dataset)
    actual_df=actual_df.iloc[:,:1000]
    pred_df=pred_df.iloc[:,:1000]
    #RUNNING ANALYSIS AND PLOTS
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
