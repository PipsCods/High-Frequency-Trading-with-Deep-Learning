import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from utils.utils import df_to_transformer_input, tensor_to_dataset, df_to_transformer_input_fast, \
    ReadyToTransformerDataset, denormalize_targets
from utils.utils import filter_top_risky_stocks_static
from utils.data_preambule import prepare_hf_data, filter_stocks_with_full_coverage, compute_hf_features_multiwindow
from utils.split_by_date_and_shift import split_and_shift_data
from main import load_raw_data, enrich_datetime, restrict_time_window, build_feature_frames, encode_categoricals, split_and_normalise, \
    build_dataloaders, build_config, train_and_evaluate
from model.model_init import ModelPipeline


def run_single_experiment(
    tot_stocks: int,
    baseline: str,
    wrapper: str | None,
    alpha: float,
    *,
    raw_path: str = os.path.join("..", "..", "data", "high_10m.parquet"),
    n_epochs: int = 30,
    seq_len: int = 12,
    batch_size: int = 32,
) -> tuple:
    """
    Train once with the provided hyper-parameters and return both *scalar*
    metrics and several pd.Series objects (actual, predicted, per-epoch
    losses & accuracy).
    """
    df = load_raw_data(raw_path)
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
        top_n=tot_stocks,
        MOST_VOLATILE_STOCKS=True,
    )

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
        seq_len=seq_len,
        batch_size=batch_size,
    )

    basic_embed_dims = {"symbol": 8, "day": 4, "day_name": 6}
    embed_dims       = {feat: 4 for feat in cat_features}
    vocab_sizes_basic = {f"{f}_vocab_size": len(vocab_maps[f]) for f in basic_embed_dims}
    vocab_sizes_basic["day_vocab_size"] = data["day"].max() + 1
    vocab_sizes_other = {
        f"{f}_vocab_size": len(vocab_maps[f])
        for f in vocab_maps if f not in basic_cat_features
    }

    cfg = build_config(
        basic_embed_dims     = basic_embed_dims,
        embed_dims           = embed_dims,
        vocab_sizes_basic    = vocab_sizes_basic,
        vocab_sizes_other    = vocab_sizes_other,
        cont_positions       = cont_positions,
        cat_positions        = cat_positions,
        lags                 = seq_len,
        alpha                = alpha,
        wrapper              = wrapper,
        initial_attention    = baseline,
    )

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = ModelPipeline(cfg)
    pipeline.to(device)

    _, best_loss, history = train_and_evaluate(
        pipeline, train_loader, test_loader, device, n_epochs
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
    import itertools
    from datetime import datetime
    from pathlib import Path

    combos = [
        #TOT_STOCKS = 200, baseline = cross-sectional, wrapper = time/None
        # *[
        #     (100, "cross-sectional", w, a)
        #     for w in ("time", None)
        #     for a in (1, 0.5, 0.01)
        # ],
        # #TOT_STOCKS = 200, baseline = time, wrapper = cross-sectional/None
        # *[
        #     (100, "time", w, a)
        #     for w in ("cross-sectional", None)
        #     for a in (1, 0.5, 0.01)
        # ],
        # #TOT_STOCKS = 800, baseline = time, wrapper = cross-sectional/None
        # *[
        #     (800, "time", w, a)
        #     for w in ("cross-sectional", None)
        #     for a in (1, 0.5, 0.01)
        # ],
        *[
            (500, "time", None, a)
            for a in (1, 0.5, 0.01)
        ],
    ]

    results_path = Path("../../data/results/transformer_hft_metrics.csv")
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
        losses_df.to_csv(f"../../data/results/{name_exp}_losses.csv")

        predictions.to_csv(f"../../data/results/{name_exp}_prediction.csv")

        print("saved")

    print("\nAll requested experiments attempted. Results at:", results_path)


if __name__ == "__main__": 
    run_experiments()
