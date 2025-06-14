import numpy as np
import torch
from torch.utils.data import Dataset


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