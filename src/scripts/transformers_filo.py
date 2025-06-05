import torch 
import torch.nn as nn
import pandas as pd 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from sklearn.preprocessing import StandardScaler
from transformer.MSSRLoss import MSSRLoss



class HFTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int, symbol2id : dict, features: str):
        self.seq_len = seq_len
        self.symbol2id = symbol2id
        self.samples = []

        df= df.sort_values(["SYMBOL", "DATETIME"])
        
        #Give total minute count inside the day where the price is observed (ex. 600 : 10:00 in the morning)
        total_minutes = df["DATETIME"].dt.hour * 60 + df["DATETIME"].dt.minute
        df["time_id"] = (total_minutes // 10).astype(int) # Gives a numeric id to the time of the day (ex 600 -> 60)

        for sym, group in df.groupby("SYMBOL"):
            group = group.reset_index(drop=True)
            returns = group[features].values
            time_id = group["time_id"].values

            sym_id = self.symbol2id[sym]
            
            #slide window
            for i in range(self.seq_len, len(group)- 1): 
                feat = returns[i - self.seq_len : i]
                ti = time_id[i - self.seq_len : i]
                target = returns[i]
                self.samples.append((feat, ti, sym_id, target))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        feat, ti, sym_id, target = self.samples[idx]
        features = torch.tensor(feat, dtype= torch.float32).unsqueeze(-1) #(seq_len, 1)
        time_ids = torch.tensor(ti, dtype= torch.long) #(seq_len,)
        sym_ids = torch.tensor(sym_id, dtype= torch.long)
        targets = torch.tensor(target, dtype= torch.float32).unsqueeze(-1)
        
        return features, time_ids, sym_ids, targets
        

class ResidualBlock(nn.Module): 
    def __init__(self, channels, kernel_size, dilation, dropout = 0.1):
        super().__init__()
        padding = (kernel_size -1 )* dilation
        self.conv = nn.Conv1d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size= kernel_size,
            dilation= dilation,
            padding = padding
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x): 
        residual = x
        out = x.transpose(1,2)
        out = self.conv(out)
        out = out[:, :, :x.size(1)]          
        out = out.transpose(1, 2)
        out = self.layer_norm(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out + residual
    
class HFTModel(nn.Module):
    def __init__(
            self,
            num_symbols : int,
            dim_features : int, 
            seq_len : int = 30,
            d_model : int = 64,
            n_blocks : int = 4, 
            kernel_size : int = 3, 
            n_heads : int = 4,
            dropout : float = 0.1
    ):
        super().__init__()
        #embeddings
        self.symbol_embed= nn.Embedding(num_symbols, d_model)
        self.time_embed = nn.Embedding(24 * 6, d_model) #number of 10min period in a day
        #input projection
        self.input_proj = nn.Linear(dim_features, d_model)
        # Temporal convolutional network (TCN)
        self.tcn = nn.Sequential(*[
            ResidualBlock(d_model, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_blocks)
        ])
        # Self-attention layer for global context
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)  # Predict next 10-min return
        )
    def forward(self, features: torch.Tensor, time_ids : torch.LongTensor, sym_ids : torch.LongTensor) -> torch.Tensor:

        x = self.input_proj(features)

        sym_emb = self.symbol_embed(sym_ids)
        sym_emb = sym_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x = x + sym_emb

        t_emb = self.time_embed(time_ids)
        x = x + t_emb

        x = self.tcn(x)

        attn_out, _ = self.attn(x ,x ,x )
        pooled = attn_out.mean(dim = 1)
        preds = self.head(pooled)

        return preds
    

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    a = []
    for feat, tid, sid, trg in tqdm(loader, desc= "Training", leave = False):
        feat = feat.to(device)
        tid = tid.to(device)
        sid = sid.to(device)
        trg = trg.to(device)

        preds = model(feat, tid, sid)
        loss = loss_fn(preds, trg)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= 1.0)
        optimizer.step()
        a.append(loss.item())
        total_loss += loss.item() * feat.size(0)

    return np.mean(a)#total_loss/len(loader.dataset)


@torch.no_grad()
def eval(model, loader, loss_fn, device): 
    model.eval()
    total_loss = 0.0
    a = []
    for feat, tid, sid, trg in tqdm(loader, desc= "Evaluating", leave = False):
        feat = feat.to(device)
        tid = tid.to(device)
        sid = sid.to(device)
        trg = trg.to(device)

        preds = model(feat, tid, sid)
        loss = loss_fn(preds, trg)
        a.append(loss.item())
        total_loss += loss.item() * feat.size(0)

    return np.mean(a)#total_loss / len(loader.dataset) 
    
# def Standardization(df_train: pd.DataFrame, df_test : pd.DataFrame, ret_col : str):
#     symbol_stat = df_train.groupby("SYMBOL")[ret_col].agg(
#         mu = 'mean',
#         sigma = 'std'
#     ).reset_index()

#     df_train = df_train.merge(symbol_stat, on = "SYMBOL", how = "left")
#     df_test = df_test.merge(symbol_stat, on = "SYMBOL", how= "left")

#     df_train["Scaled_RET"] = (df_train[ret_col] - df_train['mu']) / df_train["sigma"]
#     df_test["Scaled_RET"] = (df_test[ret_col] - df_test['mu']) / df_test["sigma"]

#     return df_train, df_test

if __name__ == "__main__": 
    #Config
    seq_len = 30   #5 hours of data for prediction
    batch_size = 512
    num_epochs = 200
    learning_rate = 3e-4
    weight_decay = 1e-2

    #Datacleaning 
    df = pd.read_parquet("../../data/high_10m.parquet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("RUNNING ON : " , device)

    df["DATETIME"] = pd.to_datetime(df["DATE"].astype(str) + ' '+ df["TIME"])
    full_data = df["DATETIME"].nunique()
    print("the number of stocks with full data are : ",np.sum((df["SYMBOL"].value_counts() == full_data)))

    vc = (df["SYMBOL"].value_counts() == full_data)
    symbols_keep = list(vc[vc.values].index)
    df_cn = df[df["SYMBOL"].isin(symbols_keep)]
    #select the last week as test and take the rest for train 
    df_train = df_cn[df_cn["DATE"] < '2021-12-27 00:00:00'].copy()
    df_test = df_cn[df_cn["DATE"] >= '2021-12-27 00:00:00'].copy()

    scaler = StandardScaler()
    df_train["Scaled_RET"] = scaler.fit_transform(df_train[["RETURN_NoOVERNIGHT"]]) #RETURN_NoOVERNIGHT
    df_test["Scaled_RET"] = scaler.transform(df_test[["RETURN_NoOVERNIGHT"]]) # RETURN_NoOVERNIGHT

    unique_sym = df_train["SYMBOL"].unique()
    symbol2id = {sym : i for i, sym in enumerate(unique_sym)}

    train_dataset = HFTDataset(df_train, seq_len= seq_len, symbol2id= symbol2id, features= "Scaled_RET")
    test_dataset = HFTDataset(df_test, seq_len= seq_len, symbol2id= symbol2id, features= "Scaled_RET")
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True, num_workers= 0, pin_memory= True)
    test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle= False, num_workers = 0, pin_memory=True)

    model = HFTModel(
        num_symbols= len(unique_sym), 
        dim_features= 1,
        seq_len= seq_len 
    ).to(device)

    optimizer = torch.optim.AdamW(params= model.parameters(), lr = learning_rate, weight_decay= weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer = optimizer,
        max_lr= learning_rate,
        total_steps= num_epochs * len(train_loader)
    )
    loss_fn = MSSRLoss(leverage_penalty= 1e-4)

    print("Start Training")
    best_val_loss = float('inf')
    train_losses = []
    test_losses = []
    c = 0

    for epoch in range(1, num_epochs +1):
        start = time.time()
        train_loss = train_epoch(model=model, loader= train_loader, optimizer= optimizer, loss_fn= loss_fn, device= device)
        test_loss = eval(model= model, loader= test_loader, loss_fn= loss_fn, device= device)
        scheduler.step()
        end = time.time()
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch:02d} — Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Time: {end - start:.2f}s")
        
        if (test_loss < best_val_loss) & (c > 10):
            best_val_loss = test_loss
            torch.save(model.state_dict(), "../models/best_model.pth")
        c += 1 #we only start counting and saving models from the 10 epoch forward
    print(f"Best Test Loss: {best_val_loss:.6f}")
    