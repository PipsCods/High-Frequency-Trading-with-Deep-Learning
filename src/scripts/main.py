# =================
# Libraries
# =================
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader

from utils.utils import df_to_transformer_input
from utils.data_preambule import prepare_hf_data
from utils.split_by_date_and_shift import split_and_shift_data
from utils.utils import tensor_to_dataset
from utils.utils import sanity_check
from utils.data_preambule import filter_stocks_with_full_coverage

from model.model_init import ModelPipeline

import matplotlib.pyplot as plt
# =================
# Main
# =================

if __name__ == '__main__':
    pd.set_option('future.no_silent_downcasting', True)  # ignore errors about replace

    folder = "../../data/high_10m.parquet"
    data = pd.read_parquet(folder)
    to_drop = data[data.isnull().any(axis=1)]['SYMBOL'].unique()
    df_cleaned = data[~data['SYMBOL'].isin(to_drop)]
    data['datetime'] = pd.to_datetime(data['DATE'].astype(str) + ' ' + data['TIME'].astype(str))
    data = data.drop(columns=['TIME', 'DATE'])
    timestamp_col = 'datetime'
    symbol_col = 'SYMBOL'
    mid_price_col = 'MID_OPEN'
    return_col = "RETURN_SiOVERNIGHT"
    date_split = '2021-12-05'

    data = data[data[timestamp_col] <= '2021-12-10']

    # keep only the stocks that have full data on the time window
    data = filter_stocks_with_full_coverage(data, timestamp_col, symbol_col)
    print(f"Sample of stocks : {len(data[symbol_col].unique())}")

    """# Synthetic dataset test
    
    folder = "../../data/Synthetic_HF_Dataset.csv"
    data = pd.read_csv(folder)
    timestamp_col = 'Timestamp'
    symbol_col = 'Stock_Symbol'
    mid_price_col = 'Mid_Price'
    return_col = "RETURN_SiOVERNIGHT"
    date_split = '2023-01-02'"""

    # We create a function that prepares the data for the model. This function will:
    #  - Extract the timestamp, stock symbol, and mid_price columns
    #  - Encode the timestamp and stock symbol columns using a one-hot encoding
    #  - Compute the returns for each stock
    #  - Separate basic_cat_features (included in all models), categorical_features (not necessary in all models)
    #    and continuous_features (not necessary in all models)
    #  - Reindex the columns according to the following structure : basic_cat_features + categorical_features +
    #    continuous_features (note that returns are at the end)
    #  - Return the prepared data and the positions of the basic_cat_features, categorical_features,
    #    and continuous_features in the data also it returns the lists of the different types of features.
    breakpoint()

    data, basic_cat_features, cat_features, cont_features, cat_feat_positions, cont_feat_positions = (
        prepare_hf_data(data, name_of_timestamp_column= timestamp_col, name_of_symbol_column=symbol_col, name_of_return_column= return_col)
    )

    breakpoint()

# ===================================================================================================================

# ========================
# INPUT FACTORING
# ========================

    # =================
    # Encoding the non-numerical data that can be encoded
    # =================
    print("Encoding the non-numerical data...")
    vocab_maps = {}
    for col in cat_features + basic_cat_features:
        codes, uniques = pd.factorize(data[col], sort=False)
        data[col] = codes.astype('int32')
        vocab_maps[col] = {val: idx for idx, val in enumerate(uniques)}

    symbol_reverse = {idx: val for val, idx in vocab_maps['symbol'].items()}

    breakpoint()
    # =================
    # Do the splitting --> We choose a date of split, the train will be done on one side and the test on the other side
    # =================
    print("Splitting the data...")
    train_df, test_df = split_and_shift_data(data, date_split=date_split, target_col= "return")

    # =================
    # Parameters for the dataloader
    # =================
    LAGS = 6  # 1HOUR
    BATCH_SIZE = 4

    breakpoint()
    # Normalize the data
    print("Normalizing the data...")
    #data['return_raw'] = data['return']
    scaler = StandardScaler()
    train_df[cont_features] = scaler.fit_transform(train_df[cont_features])
    test_df[cont_features] = scaler.transform(test_df[cont_features])

    #data[cont_features] = scaler.fit_transform(data[cont_features])
    # =================
    # Building the input dictionary
    # =================
    breakpoint()

    print("Building training signals and targets...")
    train_signals, train_targets = df_to_transformer_input(
        df=train_df,
        basic_cat_features=basic_cat_features,
        cat_features=cat_features,
        cont_features=cont_features,  #
        seq_len=LAGS
    )

    print("Building test signals and targets...")
    test_signals, test_targets = df_to_transformer_input(
        df=test_df,
        basic_cat_features=basic_cat_features,
        cat_features=cat_features,
        cont_features=cont_features,
        seq_len=LAGS
    )

    breakpoint()
    # =================
    # Building the dataloaders
    # =================
    print("Building the dataloaders...")
    train_dataset = tensor_to_dataset(train_signals, train_targets)
    test_dataset = tensor_to_dataset(test_signals, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    """# sanity check
    sanity_check(train_loader)
    breakpoint()"""

    breakpoint()
# ===================================================================================================================

# =================
# MODEL
# =================

    # =================
    # Building the embedding dictionary
    # =================
    print("Building the embedding dictionary...")
    # Basic features
    basic_embed_dims = {
        'symbol': 8,
        'day': 4,
        'day_name': 6,
    }
    vocab_sizes_basic = {f"{feat}_vocab_size": len(vocab_maps[feat].keys())
                         for feat in vocab_maps if feat in basic_embed_dims}
    vocab_sizes_basic['day_vocab_size'] = data['day'].max() + 1

    # Other features
    embed_dims = {feat: 4 for feat in cat_features}
    #  set embedded dims to 4 for all categorical features

    vocab_sizes = {f"{feat}_vocab_size": len(vocab_maps[feat].keys())
                   for feat in vocab_maps if feat not in basic_cat_features}


# ==============================

    # =================
    # Configuration of the model parameters and the optimizer parameters
    # =================

    # HYPER PARAMETERS
    lags = LAGS
    batch_size = BATCH_SIZE
    d_model = 128
    num_heads = 8
    num_layers = 1
    dropout = 0
    expansion_factor = 2
    num_epochs = 50
    output_dim = 1  # since we are predicting a vector of returns for each timestamp through all stocks.
    lr = 1e-3  # learning rate
    wrapper = 'time' # 'time' or 'cross-section' or None
    initial_attention = 'time' # 'time' or 'cross-section'
    loss_method = 'mse'  # mse or mssr (mssr produces better results)

    config = {
        'basic_embed_dims': basic_embed_dims,
        'embed_dims': embed_dims,
        'vocab_sizes_basic': vocab_sizes_basic,
        'vocab_sizes': vocab_sizes,
        'num_cont_features': len(cont_features),
        'd_model': d_model,
        'seq_len': lags,
        'num_layers': num_layers,
        'expansion_factor': expansion_factor,
        'n_heads': num_heads,
        'dropout': dropout,
        'output_dim': output_dim,
        'lr': lr,
        'cat_feat_positions': cat_feat_positions,
        'cont_feat_positions': cont_feat_positions,
        'wrapper': wrapper,
        'loss_method': loss_method,
        'initial_attention': initial_attention
    }

    # =================
    # Model initialization and training
    # =================
    print("WoW moment...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    # Instantiate model
    pipeline = ModelPipeline(config)
    pipeline.to(device)

    # ===================
    # Checkpoint for the good model
    # ===================

    best_test_loss = float('inf')
    best_model_state = None
    best_predictions = None
    best_targets = None

    """#===================
    # Sanity check
    #===================
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = pipeline.forward(inputs)
            print("Output shape:", outputs.shape)  # Expected: [B, S] or [B, S, 1]
            print("Sample outputs:", outputs[0, :5])  # First batch, first 5 stocks predictions
            print("Sample targets:", targets[0, :5])  # Corresponding targets
            breakpoint()"""

    # ===================
    # Training and evaluation
    # ===================

    # Training loop
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        train_loss = pipeline.train_epoch(train_loader, device)
        test_loss, predictions, targets = pipeline.evaluate_epoch(test_loader, device, track_best=True)

        pred_signs = torch.sign(predictions)
        true_signs = torch.sign(targets)
        directional_accuracy = (pred_signs == true_signs).float().mean().item()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Dir. Acc: {directional_accuracy:.4f}")

        # Save the best model and predictions
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = {
                'encoder': pipeline.encoder.state_dict(),
                'predictor': pipeline.predictor.state_dict()
            }
            best_predictions = predictions
            best_targets = targets

    # Save the best model
    torch.save(best_model_state, "transformer_model_8b_time_attention.pth")
    print(f"Best Test Loss: {best_test_loss:.4f}")

    # ===================
    # Results visualization
    # ===================

    # Plot marketing returns

    # Assume these are already available after evaluation or training:
    predictions = pipeline.best_predictions.cpu().numpy()  # shape: [num_samples, num_stocks]
    targets = pipeline.best_targets.cpu().numpy()  # shape: [num_samples, num_stocks]

    # Reconvert the names of the stocks back to their original names
    pred_df = pd.DataFrame(predictions, columns = vocab_maps['symbol'])
    target_df = pd.DataFrame(targets, columns = vocab_maps['symbol'])

    # Compute market timing returns
    market_timing_returns = predictions * targets  # shape: [num_samples, num_stocks]
    portfolio_returns = market_timing_returns.mean(axis=1)  # shape: [num_samples]
    cumulative_returns = portfolio_returns.cumsum()

    # Compute Sharpe ratio
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std()

    # Easy benchmark
    buy_and_hold = targets.mean(axis=1).cumsum()

    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    plt.xlabel('Time Steps')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Returns')
    plt.title('Cumulative Market Timing Returns Over Time')
    plt.plot(cumulative_returns, label='Market Timing')
    plt.plot(buy_and_hold, label='Buy and Hold')
    plt.suptitle(f'SR:{sharpe_ratio:.4f}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot confusion matrix

    pred_sign = torch.sign(torch.from_numpy(predictions))
    true_sign = torch.sign(torch.from_numpy(targets))

    tp = ((pred_sign == 1) & (true_sign == 1)).sum()
    tn = ((pred_sign == -1) & (true_sign == -1)).sum()
    fp = ((pred_sign == 1) & (true_sign == -1)).sum()
    fn = ((pred_sign == -1) & (true_sign == 1)).sum()

    print(f"TPR: {tp / (tp + fn):.4f}, TNR: {tn / (tn + fp):.4f}",
          f"FPR: {fp / (fp + tn):.4f}, FNR: {fn / (fn + tp):.4f}")