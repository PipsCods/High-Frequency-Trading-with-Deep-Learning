import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#from src.scripts.transformer.CustomLoss import CustomLoss
# -----------------------------------------------------------------------------
# Project‑specific helpers (assumed to live in your project package)
# -----------------------------------------------------------------------------
from utils.utils import df_to_transformer_input, tensor_to_dataset, df_to_transformer_input_fast, \
    ReadyToTransformerDataset, denormalize_targets
from utils.utils import filter_top_risky_stocks_static
from utils.data_preambule import prepare_hf_data, filter_stocks_with_full_coverage, compute_hf_features_multiwindow
from utils.split_by_date_and_shift import split_and_shift_data

from model.model_init import ModelPipeline


# =============================================================================
# 1. DATA LOADING & CLEANING
# =============================================================================

def load_raw_data(path: str) -> pd.DataFrame:
    """Load the 10‑minute parquet file and basic‑clean null rows."""
    print("Loading raw data...")
    df = pd.read_parquet(path)

    # drop symbols with *any* missing values to keep sequences contiguous
    bad_symbols = df[df.isnull().any(axis=1)]["SYMBOL"].unique()
    df = df[~df["SYMBOL"].isin(bad_symbols)].copy()

    return df


def enrich_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Combine DATE + TIME into a proper timestamp column and drop originals."""
    df["datetime"] = pd.to_datetime(df["DATE"].astype(str) + " " + df["TIME"].astype(str))
    df.drop(columns=["TIME", "DATE"], inplace=True)
    return df


def restrict_time_window(df: pd.DataFrame, end_date: str) -> pd.DataFrame:
    """Cut the dataframe to observations **up to** `end_date` (inclusive)."""
    return df[df["datetime"] <= end_date]


# =============================================================================
# 2.FEATURE ENGINEERING
# =============================================================================

def build_feature_frames(
    df: pd.DataFrame,
    timestamp_col: str,
    symbol_col: str,
    return_col: str,
):
    """Run project‑specific helpers that create lagged / encoded features.

    Returns
    -------
    data: pd.DataFrame – fully prepared sample‑level dataframe
    basic_cat_features, cat_features, cont_features: lists of str
    cat_positions, cont_positions: list[int]
    """
    print("Building features ...")
    (
        data,
        basic_cat_features,
        cat_features,
        cont_features,
        cat_positions,
        cont_positions,
    ) = prepare_hf_data(
        df,
        name_of_timestamp_column=timestamp_col,
        name_of_symbol_column=symbol_col,
    )

    return (
        data,
        basic_cat_features,
        cat_features,
        cont_features,
        cat_positions,
        cont_positions,
    )


# =============================================================================
# 3. ENCODING & SPLITTING
# =============================================================================

def encode_categoricals(data: pd.DataFrame, cat_cols: list[str]):
    """Factor‑encode categorical columns and return mapping dictionaries."""
    vocab_maps: dict[str, dict] = {}
    for col in cat_cols:
        codes, uniques = pd.factorize(data[col], sort=False)
        data[col] = codes.astype("int32")
        vocab_maps[col] = {val: idx for idx, val in enumerate(uniques)}
    return data, vocab_maps


def split_and_normalise(
    data: pd.DataFrame,
    date_split: str,
    cont_features: list[str],
    target_col: str = "return",
):
    """Train/val split by `date_split` and z‑score continuous features."""
    print("Splitting data...")
    train_df, test_df, tgt_mean, tgt_std = split_and_shift_data(data, date_split=date_split, target_col=target_col)


    # train_mean = train_df["target_return"].mean()
    # train_std = train_df["target_return"].std()

    scaler = StandardScaler()
    train_df[cont_features] = scaler.fit_transform(train_df[cont_features])

    test_df[cont_features] = scaler.transform(test_df[cont_features])

    return train_df, test_df, scaler, tgt_mean, tgt_std


# =============================================================================
# 4.DATALOADERS
# =============================================================================
#486, 3846, 20, 6
#486, 3846

def build_dataloaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    basic_cat: list[str],
    cat: list[str],
    cont: list[str],
    seq_len: int,
    batch_size: int,
):
    print("Building the dataloaders for training...")
    # train_signals, train_targets = df_to_transformer_input(
    #     df=train_df,
    #     basic_cat_features=basic_cat,
    #     cat_features=cat,
    #     cont_features=cont,
    #     seq_len=seq_len,
    # )

    # test_signals, test_targets = df_to_transformer_input(
    #     df=test_df,
    #     basic_cat_features=basic_cat,
    #     cat_features=cat,
    #     cont_features=cont,
    #     seq_len=seq_len,
    # )

    # train_signals, train_targets = df_to_transformer_input_fast(
    #     df=train_df,
    #     seq_len=seq_len,
    # )

    # test_signals, test_targets = df_to_transformer_input_fast(
    #     df=test_df,
    #     seq_len=seq_len,
    # )
    train_dataset = ReadyToTransformerDataset(
        df=train_df,
        basic_cat_features=basic_cat,
        cat_features=cat,
        cont_features=cont,
        seq_len=seq_len,
        target_return= "target_return"
    )
    print("Building dataloaders for testing...")
    test_dataset =ReadyToTransformerDataset(
        df=test_df,
        basic_cat_features=basic_cat,
        cat_features=cat,
        cont_features=cont,
        seq_len=seq_len,
        target_return= "target_return"
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print("Done")
    return train_loader, test_loader


# =============================================================================
# 5.MODEL CONFIGURATION
# =============================================================================

def build_config(
    basic_embed_dims: dict,
    embed_dims: dict,
    vocab_sizes_basic: dict,
    vocab_sizes_other: dict,
    cont_positions: list[int],
    cat_positions: list[int],
    lags: int,
    alpha: float,
    wrapper,
    initial_attention
):
    return {
        "basic_embed_dims": basic_embed_dims,
        "embed_dims": embed_dims,
        "vocab_sizes_basic": vocab_sizes_basic,
        "vocab_sizes": vocab_sizes_other,
        "num_cont_features": len(cont_positions),
        "d_model": 128,
        "seq_len": lags,
        "num_layers": 8,
        "expansion_factor": 2,
        "n_heads": 16,
        "dropout": 0.05,
        "output_dim": 1,
        "lr": 1e-7,
        "cat_feat_positions": cat_positions,
        "cont_feat_positions": cont_positions,
        "wrapper": wrapper,
        "loss_method": "custom", #mse or custom or huber
        "alpha" : alpha,
        "initial_attention": initial_attention,
    }


# =============================================================================
# 6.TRAIN / EVAL LOOP
# =============================================================================

def train_and_evaluate(
    pipeline: ModelPipeline,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 50,
):
    best_state, best_loss = None, float("inf")
    history = {"train": [], "test": [], "acc": []}

    for epoch in range(1, num_epochs + 1):
        train_loss = pipeline.train_epoch(train_loader, device)
        test_loss, preds, targets = pipeline.evaluate_epoch(test_loader, device, track_best=True)

        accuracy = (torch.sign(preds) == torch.sign(targets)).float().mean().item()
        history["train"].append(train_loss)
        history["test"].append(test_loss)
        history["acc"].append(accuracy)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | Train {train_loss:.4f} | "
            f"Test {test_loss:.4f} | Directional Acc {accuracy:.4f}"
        )

        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {
                "encoder": pipeline.encoder.state_dict(),
                "predictor": pipeline.predictor.state_dict(),
            }
            pipeline.best_predictions = preds
            pipeline.best_targets = targets

    return best_state, best_loss, history


# =============================================================================
# 7.VISUALISATION
# =============================================================================

def plot_performance(preds: torch.Tensor, targets: torch.Tensor):
    preds_np, targets_np = preds.cpu().numpy(), targets.cpu().numpy()

    mt_returns = preds_np * targets_np
    portfolio = mt_returns.mean(axis=1)
    cumulative = portfolio.cumsum()
    buy_hold = targets_np.mean(axis=1).cumsum()
    sharpe = portfolio.mean() / portfolio.std()

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative, label="3")
    plt.plot(buy_hold, label="Buy & Hold")
    plt.title(f"Cumulative Returns (SR={sharpe:.2f})")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Return")
    plt.legend(); plt.grid(True); plt.show()


def plot_training_history(history: dict[str, list[float]]) -> None:
    """
    Parameters
    ----------
    history : dict
        Expected keys:
            - "train_loss"   : list[float]
            - "test_loss"    : list[float]
            - "sign_accuracy": list[float]   (or "sign_acc", see below)

    Notes
    -----
    * If you used a different key for sign accuracy (e.g. "sign_acc"),
      just replace the lookup line.
    * Shows three lines on the same axis with a legend.
    """
    # ------- defensive look-ups -----------------------------------------
    train_loss   = history.get("train_loss", [])
    test_loss    = history.get("test_loss", [])
    sign_acc_key = "sign_accuracy" if "sign_accuracy" in history else "sign_acc"
    sign_acc     = history.get(sign_acc_key, [])

    n_epochs = max(len(train_loss), len(test_loss), len(sign_acc))
    epochs   = range(1, n_epochs + 1)

    # ------- plot -------------------------------------------------------
    plt.figure(figsize=(8, 4))
    if train_loss:
        plt.plot(epochs, train_loss,  label="Train loss")
    if test_loss:
        plt.plot(epochs, test_loss,   label="Test loss")
    if sign_acc:
        plt.plot(epochs, sign_acc,    label="Sign-accuracy")
    plt.xlabel("Epoch")
    plt.title("Training history")
    plt.legend()
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 8.MAIN DRIVER
# =============================================================================

def main():
    pd.set_option("future.no_silent_downcasting", True)

    # ------------------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------------------
    RAW_PATH = os.path.join("..", "..", "data", "high_10m.parquet")
    END_DATE = "2021-12-10"
    DATE_SPLIT = "2021-12-27"
    TIMESTAMP_COL = "datetime"
    SYMBOL_COL = "SYMBOL"
    RETURN_COL = "RETURN_SiOVERNIGHT"
    LAGS = 12
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    MOST_VOLATILE_STOCKS = True
    ALPHA = 1
    WRAPPER = "time"
    BASELINE = "time"
    TOT_STOCKS = 800


    # ------------------------------------------------------------------
    # LOAD & CLEAN
    # ------------------------------------------------------------------
    df = load_raw_data(RAW_PATH)
    # Compute return (by symbol)
    df['return'] = df[RETURN_COL]
    df.drop(columns= [RETURN_COL], inplace = True )

    #df = df[df["DATE"] <= END_DATE]
    df = enrich_datetime(df)

    #df = restrict_time_window(df, END_DATE)
    df = filter_stocks_with_full_coverage(df, TIMESTAMP_COL, SYMBOL_COL)

    print("Creating some financial variables...")
    # Creating some financial variables
    df = compute_hf_features_multiwindow(df, 'return')

    # ------------------------------------------------------------------
    # FILTER NUMBER OF STOCKS
    # ------------------------------------------------------------------
    df = filter_top_risky_stocks_static(df, cutoff_date=DATE_SPLIT, window=20, quantiles=100, top_n=TOT_STOCKS,
                                          MOST_VOLATILE_STOCKS=MOST_VOLATILE_STOCKS)

    # ------------------------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------------------------
    (
        data,
        basic_cat_features,
        cat_features,
        cont_features,
        cat_positions,
        cont_positions,
    ) = build_feature_frames(df, TIMESTAMP_COL, SYMBOL_COL, RETURN_COL)

    # ------------------------------------------------------------------
    # ENCODE CATEGORICALS
    # ------------------------------------------------------------------
    data, vocab_maps = encode_categoricals(data, cat_cols=cat_features + basic_cat_features)
    symbol_reverse = {idx: val for val, idx in vocab_maps["symbol"].items()}

    # ------------------------------------------------------------------
    # SPLIT + NORMALISE
    # ------------------------------------------------------------------

    train_df, test_df, _ , tgt_mean, tgt_std= split_and_normalise(data, DATE_SPLIT, cont_features)


    # ------------------------------------------------------------------
    # DATALOADERS
    # ------------------------------------------------------------------
    train_loader, test_loader = build_dataloaders(
        train_df,
        test_df,
        basic_cat_features,
        cat_features,
        cont_features,
        seq_len=LAGS,
        batch_size=BATCH_SIZE,
    )

    # ------------------------------------------------------------------
    # MODEL CONFIG
    # ------------------------------------------------------------------
    basic_embed_dims = {"symbol": 8, "day": 4, "day_name": 6}
    embed_dims = {feat: 4 for feat in cat_features}

    vocab_sizes_basic = {f"{feat}_vocab_size": len(vocab_maps[feat]) for feat in basic_embed_dims}
    vocab_sizes_basic["day_vocab_size"] = data["day"].max() + 1  # continuous day int
    vocab_sizes_other = {
        f"{feat}_vocab_size": len(vocab_maps[feat]) for feat in vocab_maps if feat not in basic_cat_features
    }

    config = build_config(
        basic_embed_dims=basic_embed_dims,
        embed_dims=embed_dims,
        vocab_sizes_basic=vocab_sizes_basic,
        vocab_sizes_other=vocab_sizes_other,
        cont_positions=cont_positions,
        cat_positions=cat_positions,
        lags=LAGS,
        alpha=ALPHA,
        wrapper=WRAPPER,
        initial_attention=BASELINE
    )

    # Test if the loss function is working properly
    """# After pipeline is created
    pipeline = ModelPipeline(config)
    pipeline.to("cpu")  # for easier debugging

    # Get one batch
    batch = next(iter(train_loader))
    x_batch, y_batch = batch
    x_batch = x_batch.to("cpu")
    y_batch = y_batch.to("cpu")

    # Set model to eval mode to disable dropout etc.
    pipeline.eval_mode()

    # Forward pass
    with torch.no_grad():
        out = pipeline(x_batch)

    # Show results
    print("\n=== Manual Debug ===")
    print("Predictions (1st sample):", out[0].view(-1).cpu().numpy())
    print("Targets     (1st sample):", y_batch[0].view(-1).cpu().numpy())

    # Compute MSSR loss
    loss_fn = MSSRLoss()
    loss = loss_fn(out, y_batch)
    print("MSE Loss:", loss.item())
    breakpoint()"""

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    pipeline = ModelPipeline(config)
    pipeline.to(device)

    best_state, best_loss, history = train_and_evaluate(
        pipeline, train_loader, test_loader, device, NUM_EPOCHS
    )

    #torch.save(best_state, "transformer_model_best.pth")
    print(f"Best Test Loss: {best_loss:.4f}")

    preds, targets = pipeline.best_predictions.cpu().numpy(), pipeline.best_targets.cpu().numpy()

    # Denormalize
    preds = denormalize_targets(preds, tgt_mean, tgt_std)
    targets = denormalize_targets(targets, tgt_mean, tgt_std)

    history_df = pd.DataFrame(history)

    pred_df = pd.DataFrame(preds, columns = vocab_maps['symbol'])
    target_df = pd.DataFrame(targets, columns = vocab_maps['symbol'])


    # history_df.to_pickle("results/model4/losses.pickle")
    # pred_df.to_pickle("results/model4/predictions.pickle")
    # target_df.to_pickle("results/model4/targets.pickle")


    breakpoint()
    # ------------------------------------------------------------------
    # VISUALISE
    # ------------------------------------------------------------------
    plot_performance(pipeline.best_predictions, pipeline.best_targets)


if __name__ == "__main__":
    main()
