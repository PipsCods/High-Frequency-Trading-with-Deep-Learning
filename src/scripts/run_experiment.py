import os
import pandas as pd
import numpy as np
import random 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from utils.utils import df_to_transformer_input, tensor_to_dataset, df_to_transformer_input_fast, \
    ReadyToTransformerDataset, denormalize_targets
from utils.utils import filter_top_risky_stocks_static, volatility_filter2
from utils.data_preambule import prepare_hf_data, filter_stocks_with_full_coverage, compute_hf_features_multiwindow
from utils.split_by_date_and_shift import split_and_shift_data
from main import load_raw_data, enrich_datetime, restrict_time_window, build_feature_frames, encode_categoricals, split_and_normalise, \
    build_dataloaders, train_and_evaluate
from model.model_init import ModelPipeline


# TOT_STOCK = 100
# BASELINE = "cross-sectional"
# WRAPPER = "time"
# ALPHA = 0.1

RAW_PATH = os.path.join("..", "..", "data", "high_10m.parquet")
EPOCHS = 500
SEQ_LEN = 12
BATCH_SIZE = 16

CUTOFF_DATE = "2021-12-23"

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
    wrapper : str | None, 
    alpha : float, 
    baseline: str,
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
        "wrapper": wrapper,
        "loss_method": "custom", #mse or custom or huber
        "alpha" : alpha,
        "initial_attention": baseline,
    }


def run_single_experiment(
    tot_stocks: int,
    baseline: str,
    wrapper: str | None,
    alpha: float,
) -> tuple:
    """
    Train once with the provided hyper-parameters and return both *scalar*
    metrics and several pd.Series objects (actual, predicted, per-epoch
    losses & accuracy).
    """
    df = load_raw_data(RAW_PATH) 
    df["return"] = df["RETURN_SiOVERNIGHT"]
    df.drop(columns=["RETURN_SiOVERNIGHT", "RETURN_NoOVERNIGHT"], inplace=True)
    df = enrich_datetime(df)
    df = filter_stocks_with_full_coverage(df, "datetime", "SYMBOL")
    df = compute_hf_features_multiwindow(df, "return")

    random.seed(42)
    stocks = random.sample(list(df["SYMBOL"].unique()), tot_stocks)
    df = df[df["SYMBOL"].isin(stocks)]
    #df = volatility_filter2(df, cutoff_date= CUTOFF_DATE, top_n= tot_stocks)
    # df = filter_top_risky_stocks_static(
    #     df,
    #     cutoff_date=CUTOFF_DATE,
    #     window=20,
    #     quantiles=100,
    #     top_n=tot_stocks,
    #     MOST_VOLATILE_STOCKS=True,
    # )


    df.drop(columns=["ALL_EX", "SUM_DELTA"], inplace=True)
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
        total_steps          = total_steps,
        wrapper              = wrapper,
        alpha                = alpha,
        baseline             = baseline
        
    )

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = ModelPipeline(cfg).to(device)

    _, best_loss, history = train_and_evaluate(
        pipeline, train_loader, test_loader, device, EPOCHS
    )

    preds   = denormalize_targets(
        pipeline.best_predictions.cpu().numpy(), tgt_mean, tgt_std
    )
    targets = denormalize_targets(
        pipeline.best_targets.cpu().numpy(), tgt_mean, tgt_std
    )

    preds_df = pd.DataFrame(preds, columns = vocab_maps['symbol'])
    actual_df = pd.DataFrame(targets, columns = vocab_maps['symbol'])

    long_actual = actual_df.reset_index().melt(id_vars= "index", var_name= "stock", value_name= "actual")
    long_pred = preds_df.reset_index().melt(id_vars= "index", var_name= "stock", value_name= "pred")

    combined = pd.merge(long_actual, long_pred, on= ["index", "stock"]).set_index("index")
    
    # Scalars
    sign_acc_final = (np.sign(preds) == np.sign(targets)).mean()
    mt_ret   = preds * targets
    portfolio= mt_ret.mean(axis=1)
    sharpe   = portfolio.mean() / (portfolio.std() + 1e-9)

    metrics = {
        "best_loss"      : best_loss,
        "sign_acc_final" : sign_acc_final,
        "sharpe"         : sharpe,
    }

    # predictions = {
    #     "predicted"      : pd.DataFrame(preds, columns = vocab_maps['symbol']),
    #     "actual"         : pd.DataFrame(targets, columns = vocab_maps['symbol']),
    # }
    # -------------------------------------------------
    #  Build & return a dict with Series objects too --
    # -------------------------------------------------
    



    return metrics, history, combined



def run_experiments():
    from pathlib import Path

    combos = [
        #TOT_STOCKS = 100, baseline = cross-sectional, wrapper = time/None
        *[
            (100, "cross-sectional", w, a)
            for w in ("time", None)
            for a in (1, 0.5, 0.01)
        ],
        #TOT_STOCKS = 100, baseline = time, wrapper = cross-sectional/None
        *[
            (100, "time", w, a)
            for w in ("cross-sectional", None)
            for a in (1, 0.5, 0.01)
        ],
        #TOT_STOCKS = 800, baseline = time, wrapper = cross-sectional/None
        *[
            (500, "time", w, a)
            for w in ("cross-sectional", None)
            for a in (1, 0.5, 0.1)
        ],
        # *[
        #     (500, "time", None, a)
        #     for a in (1, 0.5, 0.01)
        # ],
    ]

    results_path = Path("../../data/results2/transformer_hft_metrics.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume logic unchanged
    if results_path.exists():
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame()

    seen = {
        (int(r.TOT_STOCKS), r.BASELINE, r.WRAPPER if pd.notna(r.WRAPPER) else None, float(r.alpha))
        for _, r in results_df.iterrows()
    }

    for tot, baseline, wrapper, a in combos:
        key = (tot, baseline, wrapper, a)
        if key in seen:
            print(f"Skipping already-finished run {key}")
            continue

        print(f"\n[RUN] stocks={tot} | baseline={baseline} | wrapper={wrapper} | α={a}")
        try:
            metrics, losses, predictions = run_single_experiment(
                tot_stocks = tot,
                baseline   = baseline,
                wrapper    = wrapper,
                alpha      = a,
            )
        except Exception as exc:
            print(f" run {key} failed – {exc}")
            continue

        # one tidy row, incl. the Series
        row = {
            "TOT_STOCKS" : tot,
            "BASELINE"   : baseline,
            "WRAPPER"    : wrapper,
            "alpha"      : a,
            **metrics,          # <-- adds best_loss, sharpe, and *all* Series
        }

        metrics_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

        name_exp = "_".join([str(k) for k in key])

        metrics_df.to_csv(results_path, index=False)  # <= persist immediately

        losses_df = pd.DataFrame(losses)
        losses_df.to_csv(f"../../data/results2/{name_exp}_losses.csv")

        predictions.to_csv(f"../../data/results2/{name_exp}_prediction.csv")

        print("saved")

    print("\nAll requested experiments attempted. Results at:", results_path)


if __name__ == "__main__": 
    run_experiments()
