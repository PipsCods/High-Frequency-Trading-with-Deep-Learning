import os
import pandas as pd
import numpy as np 
import torch 
import matplotlib.pyplot as plt

from utils.utils import denormalize_targets, filter_top_risky_stocks_static
from utils.data_preambule import filter_stocks_with_full_coverage, compute_hf_features_multiwindow
from main import load_raw_data, build_config, encode_categoricals, split_and_normalise, \
    build_dataloaders, build_feature_frames, train_and_evaluate, enrich_datetime

from model.model_init import ModelPipeline

TOT_STOCK = 100
BASELINE = "cross-sectional"
WRAPPER = "time"
ALPHA = 0.01

RAW_PATH = os.path.join("..", "..", "data", "high_10m.parquet")
EPOCHS = 30
SEQ_LEN = 12
BATCH_SIZE = 32

CUTOFF_DATE = "2021-12-27"

MODEL_DIM = 128
NUM_LAYERS = 8
EXPANSION_FACT = 2
NUM_HEADS = 16
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



def train_and_debugg(pipeline: ModelPipeline, dataloader, device, max_batches = 5, grad_clip = 1.0):
    pipeline.train_mode()
    total_loss = 0.0
    batch_count = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        pipeline.optimizer.zero_grad()

        outputs = pipeline.forward(inputs)             # forward pass
        loss = pipeline.loss_fn(outputs, targets)      # compute loss
        loss.backward()                                # backprop

        # gather gradient norms
        grad_norms = {
            name: param.grad.data.norm(2).item()
            for name, param in pipeline.named_parameters()
            if param.grad is not None
        }
        sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1])

        # print smallest & largest three
        print(f"[Batch {batch_count+1}/{max_batches}] loss = {loss.item():.4f}")
        print("  ↓  Smallest grads")
        for name, g in sorted_grads[:3]:
            print(f"    {name:<40s} {g:.3e}")
        print("  ↑  Largest grads")
        for name, g in sorted_grads[-3:]:
            print(f"    {name:<40s} {g:.3e}")

        # gradient clipping & optimizer step
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), max_norm=grad_clip)
        pipeline.optimizer.step()
        pipeline.scheduler.step()

        total_loss += loss.item()
        batch_count += 1
        if batch_count >= max_batches:
            break

    avg_loss = total_loss / batch_count
    print(f"[Debug] avg loss over {batch_count} batches: {avg_loss:.4f}\n")
    return avg_loss



if __name__ == "__main__": 
    df = load_raw_data(RAW_PATH)  
    df["return"] = df["RETURN_SiOVERNIGHT"]
    df.drop(columns=["RETURN_SiOVERNIGHT"], inplace=True)
    df = enrich_datetime(df)
    df = filter_stocks_with_full_coverage(df, "datetime", "SYMBOL")
    df = compute_hf_features_multiwindow(df, "return")

    df = filter_top_risky_stocks_static(
        df,
        cutoff_date="2021-12-27",
        window=20,
        quantiles=100,
        top_n=TOT_STOCK,
        MOST_VOLATILE_STOCKS=True,
    )
    df.drop(columns=["ALL_EX", "SUM_DELTA", "index", "risk_quantile"], inplace=True)
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
        data, "2021-12-27", cont_features
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

    # _, best_loss, history = train_and_evaluate(
    #     pipeline, train_loader, test_loader, device, EPOCHS
    # )

    inputs, target = next(iter(train_loader))
    for step in range(100):
        pipeline.optimizer.zero_grad()
        outputs = pipeline.forward(inputs.to(device))
        loss = pipeline.loss_fn(outputs, target.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
        pipeline.optimizer.step()
        pipeline.scheduler.step()
        print(f"step {step:03d}: {loss.item():.4f}")

    





