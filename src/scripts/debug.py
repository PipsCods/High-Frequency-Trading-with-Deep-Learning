import os
import pandas as pd
import numpy as np 
import torch 
import matplotlib.pyplot as plt

from utils.utils import denormalize_targets, filter_top_risky_stocks_static, volatility_filter2
from utils.data_preambule import filter_stocks_with_full_coverage, compute_hf_features_multiwindow
from main import load_raw_data, encode_categoricals, split_and_normalise, \
    build_dataloaders, build_feature_frames, train_and_evaluate, enrich_datetime

from model.model_init import ModelPipeline

TOT_STOCK = 100
BASELINE = "cross-sectional"
WRAPPER = "time"
ALPHA = 0.1

RAW_PATH = os.path.join("..", "..", "data", "high_10m.parquet")
EPOCHS = 500
SEQ_LEN = 12
BATCH_SIZE = 16

CUTOFF_DATE = "2021-12-17"

MODEL_DIM = 128
NUM_LAYERS = 3
EXPANSION_FACT = 1
NUM_HEADS = 8
DROPOUT = 0
OUTPUT_DIM = 1
LEARNING_RATE = 1e-3









def build_config(
    basic_embed_dims: dict,
    embed_dims: dict,
    vocab_sizes_basic: dict,
    vocab_sizes_other: dict,
    cont_positions: list[int],
    cat_positions: list[int],
    total_steps: int,
):
    return {
        "basic_embed_dims": basic_embed_dims,
        "embed_dims": embed_dims,
        "vocab_sizes_basic": vocab_sizes_basic,
        "vocab_sizes": vocab_sizes_other,
        "num_cont_features": len(cont_positions),
        "total_steps" : total_steps,
        "d_model": MODEL_DIM,
        "seq_len": SEQ_LEN,
        "num_layers": NUM_LAYERS,
        "expansion_factor": EXPANSION_FACT,
        "n_heads": NUM_HEADS,
        "dropout": DROPOUT,
        "output_dim": OUTPUT_DIM,
        "lr": LEARNING_RATE,
        "cat_feat_positions": cat_positions,
        "cont_feat_positions": cont_positions,
        "wrapper": WRAPPER,
        "loss_method": "custom", #mse or custom or huber
        "alpha" : ALPHA,
        "initial_attention": BASELINE,
    }



if __name__ == "__main__": 
    df = load_raw_data(RAW_PATH) 
    df["return"] = df["RETURN_SiOVERNIGHT"]
    df.drop(columns=["RETURN_SiOVERNIGHT"], inplace=True)
    df = enrich_datetime(df)
    df = filter_stocks_with_full_coverage(df, "datetime", "SYMBOL")
    df = compute_hf_features_multiwindow(df, "return")

    # df2= filter_top_risky_stocks_static(
    #     df,
    #     cutoff_date=CUTOFF_DATE,
    #     window=20,
    #     quantiles=100,
    #     top_n=TOT_STOCK,
    #     MOST_VOLATILE_STOCKS=True,
    # )
    df = volatility_filter2(df, cutoff_date= CUTOFF_DATE, top_n= TOT_STOCK)

    breakpoint()
    df.drop(columns=["ALL_EX", "SUM_DELTA", "index", "risk_quantile"], inplace=True)
    #trys = df[["datetime", "SYMBOL", "return"]]
    (data,
     basic_cat_features,
     cat_features,
     cont_features,
     cat_positions,
     cont_positions) = build_feature_frames(
        df, "datetime", "SYMBOL", "return"
    )
    data, vocab_maps = encode_categoricals(
        data, cat_cols=cat_features + basic_cat_features
    )
    
    train_df, test_df, _, tgt_mean, tgt_std = split_and_normalise(
        data, CUTOFF_DATE, cont_features
    )

    train_loader, test_loader = build_dataloaders(
        train_df,
        test_df,
        basic_cat_features,
        cat_features,
        cont_features,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
    )
    basic_embed_dims = {"symbol": 16, "day": 4, "day_name": 6}
    embed_dims       = {feat: 4 for feat in cat_features}
    vocab_sizes_basic = {f"{f}_vocab_size": len(vocab_maps[f]) for f in basic_embed_dims}
    vocab_sizes_basic["day_vocab_size"] = data["day"].max() + 1
    vocab_sizes_other = {
        f"{f}_vocab_size": len(vocab_maps[f])
        for f in vocab_maps if f not in basic_cat_features
    }
    total_steps = len(train_loader) * EPOCHS

    cfg = build_config(
        basic_embed_dims     = basic_embed_dims,
        embed_dims           = embed_dims,
        vocab_sizes_basic    = vocab_sizes_basic,
        vocab_sizes_other    = vocab_sizes_other,
        cont_positions       = cont_positions,
        cat_positions        = cat_positions,
        total_steps          = total_steps
    )

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = ModelPipeline(cfg).to(device)

    _, best_loss, history = train_and_evaluate(
        pipeline, train_loader, test_loader, device, EPOCHS
    )

    # inputs, target = next(iter(train_loader))
    # for step in range(500):
    #     pipeline.optimizer.zero_grad()
    #     outputs = pipeline.forward(inputs.to(device))
    #     loss = pipeline.loss_fn(outputs, target.to(device))
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
    #     pipeline.optimizer.step()
    #     #pipeline.scheduler.step()
    #     print(f"step {step:03d}: {loss.item():.4f}")

    
    # for step in range(500):
    #     c = 0
    #     losses = []
    #     for inputs,target in train_loader:
    #         pipeline.optimizer.zero_grad()
    #         outputs = pipeline.forward(inputs.to(device))
    #         loss = pipeline.loss_fn(outputs, target.to(device))
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
    #         pipeline.optimizer.step()
    #         #pipeline.scheduler.step()
    #         losses.append(loss.item())
    #         c += 1
    #         if c > 9: 
    #             break
    #     print(f"step {step:03d} average_loss: {np.mean(losses):.4f}")





