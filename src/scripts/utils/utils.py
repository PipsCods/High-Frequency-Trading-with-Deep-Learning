import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

# =================
# Functions
# =================
def timestamp_extract(df):
    df.index = pd.to_datetime(df.index)

    df['day'] = df.index.day
    df['day_name'] = df.index.day_name()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute

    return df

def df_to_transformer_input_fast(
    df: pd.DataFrame,
    seq_len: int
):
    """
    Vectorised version of df_to_transformer_input.
    Returns:
        X : (batch, seq_len, num_symbols, num_features)  torch.float32
        y : (batch, num_symbols)                         torch.float32
    """

    cube_df = (
        df
        .set_index('symbol', append= True)
        .unstack('symbol')
        .sort_index(axis=1, level=0)        # symbols outer-level in α order
    )
    target_df = cube_df['target_return']

    num_times     = cube_df.shape[0]
    symbols_lvl   = cube_df.columns.levels[1]      # first level  → symbols
    features_lvl  = cube_df.columns.levels[0]      # second level → features
    num_symbols   = len(symbols_lvl)
    num_features  = len(features_lvl)

    cube = cube_df.to_numpy(dtype=np.float32).reshape(
        num_times, num_symbols, num_features
    )

    X_windows = np.lib.stride_tricks.sliding_window_view(
        cube, window_shape=seq_len, axis=0
    )[:-1]                              # (T-seq_len, L, N, F)

    target_df = cube_df['target_return']
    target_df = target_df[symbols_lvl]
    y_vec = target_df.to_numpy(dtype=np.float32)[seq_len:]        # (T-seq_len, N)

    good = (~np.isnan(X_windows).any(axis=(1, 2, 3))) & (~np.isnan(y_vec).any(axis=1))

    X = torch.from_numpy(X_windows[good]).contiguous()    # (B, L, N, F)
    X = np.transpose(X, (0, 3, 1, 2))

    y = torch.from_numpy(y_vec[good]).contiguous()        # (B, N)
    breakpoint()
    return X, y

def df_to_transformer_input(df, basic_cat_features, cat_features, cont_features, seq_len):
    # Reset the index
    tmp = df.copy()
    tmp.reset_index(inplace=True)

    # Extract unique timestamps and symbols
    all_times = sorted(tmp['timestamp'].unique())
    all_symbols = sorted(tmp['symbol'].unique())

    # Map timestamps to indices
    map_timestamp = {t: i for i, t in enumerate(all_times)}
    # Map symbols to indices for clarity (if symbol is already encoded as integer index, you can skip this)
    map_symbol = {s: i for i, s in enumerate(all_symbols)}

    num_times = len(all_times)
    num_symbols = len(all_symbols)
    num_features = len(basic_cat_features) + len(cat_features) + len(cont_features)

    # Initialize tensor with NaNs
    input_tensor = np.full((num_times, num_symbols, num_features + 1 ), np.nan, dtype=np.float32) #TO CHECK

    # Fill tensor: [time, stock, features]
    for idx, row in tmp.iterrows():
        t_idx = map_timestamp[row['timestamp']]
        s_idx = map_symbol[row['symbol']]

        basic_vals = [row[col] for col in basic_cat_features]
        cat_vals = [row[col] for col in cat_features]
        cont_vals = [row[col] for col in cont_features]
        target = [row["target_return"]] #TO-CHECK
        

        all_vals = basic_vals + cat_vals + cont_vals + target
        input_tensor[t_idx, s_idx, :] = np.array(all_vals, dtype=np.float32)

    X_list = []
    y_list = []

    # Build sequences: iterate over time indices where a full sequence and target exist
    for t in range(seq_len, num_times - 1):  # predict t+1
        # Sequence: [seq_len, num_symbols, num_features]
        X_seq = input_tensor[t - seq_len:t, :, :-1] # slicing does not account the last t #TOCHECK

        # Target returns at t+1 for all stocks: shape [num_symbols]
        y_ret = input_tensor[t, :, -1]  # last feature is target return

        # Validate no NaNs in inputs or targets
        if not np.isnan(X_seq).any() and not np.isnan(y_ret).any():
            X_list.append(X_seq)
            y_list.append(y_ret)

    # Stack sequences: shape [batch_size, seq_len, num_symbols, num_features]
    X = torch.tensor(np.stack(X_list), dtype=torch.float32)

    # Stack targets: shape [batch_size, num_symbols]
    y = torch.tensor(np.stack(y_list), dtype=torch.float32)
    breakpoint()
    return X, y


def tensor_to_dataset(signals_tensor, target_tensor):
    return TensorDataset(signals_tensor, target_tensor)

def sanity_check(train_loader):
    for inputs, targets in train_loader:
        print("Input shape:", inputs.shape)  # Expect [B, T, S, features]
        print("Target shape:", targets.shape)  # Expect [B, S] or [B, S, 1]
        print("Input dtype:", inputs.dtype)
        print("Target dtype:", targets.dtype)
        break  # just check the first batch


class ReadyToTransformerDataset(Dataset):
    def __init__(self, df, basic_cat_features, cat_features, cont_features, seq_len, target_return):
        self.seq_len = seq_len
        self.samples = []

        tmp = df.copy()
        tmp.reset_index(inplace=True)

        all_times = sorted(tmp['timestamp'].unique())
        all_symbols = sorted(tmp['symbol'].unique())

        map_timestamp = {t: i for i, t in enumerate(all_times)}
        map_symbol = {s: i for i, s in enumerate(all_symbols)}

        num_times = len(all_times)
        num_symbols = len(all_symbols)
        num_features = len(basic_cat_features) + len(cat_features) + len(cont_features)

        input_tensor = np.full((num_times, num_symbols, num_features), np.nan, dtype=np.float32)
        target_tensor = np.full((num_times, num_symbols), np.nan, dtype=np.float32)

        for _, row in tmp.iterrows():
            t_idx = map_timestamp[row['timestamp']]
            s_idx = map_symbol[row['symbol']]

            all_vals = [row[col] for col in basic_cat_features + cat_features + cont_features]
            input_tensor[t_idx, s_idx, :] = np.array(all_vals, dtype=np.float32)
            target_tensor[t_idx, s_idx] = row[target_return]

        for t in range(seq_len, num_times - 1):
            X_seq = input_tensor[t - seq_len:t, :, :]
            y_ret = target_tensor[t, :]

            if not np.isnan(X_seq).any() and not np.isnan(y_ret).any():
                self.samples.append((X_seq, y_ret))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)