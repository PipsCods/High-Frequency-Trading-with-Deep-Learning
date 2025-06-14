import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper_portfolio_management import strategy,model_evaluation,cleandata,build_combined_strategy
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
          '100_time_None_1_prediction'
          ]:

    data_transformer=pd.read_csv(f'/Users/emanueledurante/Desktop/LGMB/lausanne/epfl/MLfinance/High-Frequency-Trading-with-Deep-Learning/data/predictions_transformers/{j} 2.csv')
    data_transformer = data_transformer.pivot(index="index", columns="stock", values="pred").fillna(0)
    stocks=data_transformer.columns
    name='arima'
    #DATA UPLOADING
    data=pd.read_parquet(f'/Users/emanueledurante/Desktop/predictions/{name}_predictions.parquet')
    pred_df = data.pivot_table(index="DATETIME", columns="SYMBOL", values="predicted_return",aggfunc="mean")
    actual_dataset=pd.read_parquet('/Users/emanueledurante/Desktop/LGMB/lausanne/epfl/MLfinance/High-Frequency-Trading-with-Deep-Learning/data/high_10m.parquet')
    #INITIALIZING FUNCTIONS
    returns_strategy=dict()
    pred_df,actual_df=cleandata(pred_df,actual_dataset)
    actual_df=actual_df[pred_df.columns.intersection(stocks)]
    pred_df=pred_df[pred_df.columns.intersection(stocks)]
    actual_df=actual_df[pred_df.columns.intersection(stocks)]
    pred_df=pred_df[pred_df.columns.intersection(stocks)]

    data_transformer=data_transformer[stocks.intersection(pred_df.columns)]
    if len(actual_df.index)<len(data_transformer.index):
        data_transformer=data_transformer.loc[len(data_transformer.index)-len(actual_df.index):]
    else:
        actual_df=actual_df.loc[-len(data_transformer.index)+len(actual_df.index):]
        pred_df=pred_df.loc[-len(data_transformer.index)+len(actual_df.index):]

    #RUNNING ANALYSIS
    weights_transfomer,total_cost=strategy(data_transformer,tc=0)
    returns_transformer_real=model_evaluation(weights_transfomer,actual_df,total_cost)
    plt.figure()
    plt.plot(returns_transformer_real.cumsum())
    plt.show()
    info=build_combined_strategy(pred_df,data_transformer,actual_df)
    print("=== MODEL OUTPUT SUMMARY ===")
    print(f"Alpha_hat:       {info['alpha_hat']:.4f}")
    print(f"Beta_hat:        {info['beta_hat']:.4f}")
    print(f"Sigma_alpha:     {info['sigma_alpha']:.4f}")
    print(f"SR_alpha:        {info['SR_alpha']:.4f}")
    print(f"SR_base:         {info['SR_base']:.4f}")
    print(f"SR_combined:     {info['SR_combined']:.4f}")
    print(f"SR_expost_comb:  {info['SR_expost_comb']:.4f}")
    print()

    print("=== FRACTIONS ===")
    print(f"f_base (scalar): {info['f_base']:.4f}")
    print(f"f_alpha (scalar):{info['f_alpha']:.4f}")
    print()

    print("=== LABELS ===")
    print(f"Label base:      {info['label_base']}")
    print(f"Label lean:      {info['label_lean']}")
    print()

    print("=== RETURNS (head) ===")
    print(info['returns_combined'].head())
    print()

    print("=== WEIGHTS COMBINED (head) ===")
    print(info['W_combined_hist'].head())
