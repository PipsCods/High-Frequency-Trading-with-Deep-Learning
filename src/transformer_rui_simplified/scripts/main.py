import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import random
from types import SimpleNamespace


#from src.scripts.transformer.CustomLoss import CustomLoss
# -----------------------------------------------------------------------------
# Project‑specific helpers (assumed to live in your project package)
# -----------------------------------------------------------------------------
from utils.utils import df_to_transformer_input, tensor_to_dataset, df_to_transformer_input_fast, \
    ReadyToTransformerDataset, denormalize_targets
from utils.utils import assign_risk_class_by_cumulative_std
from utils.data_preambule import prepare_hf_data, filter_stocks_with_full_coverage, compute_hf_features_multiwindow, \
    augment_features
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
    """Enrich the initial dataframe with categorical features related to time, with continuous features based on
    financial variables and with continuous features based on non-linear functions.
    Returns
    -------
    data: pd.DataFrame – fully prepared sample‑level dataframe
    """
    print("Building features ...")

    #  First, we prepare the hf data by setting timestamp as an index
    #  and creating some time related categorical variables

    tmp = df.copy()
    tmp = prepare_hf_data(tmp, name_of_timestamp_column=timestamp_col, name_of_symbol_column=symbol_col)
    # Secondly, generate financial features
    tmp = compute_hf_features_multiwindow(tmp, return_col)
    # Finally, generate highly non-linear features
    tmp = augment_features(tmp, 'symbol', return_col)

    return tmp


# =============================================================================
# 3. ENCODING & SPLITTING
# =============================================================================

def encode_categoricals(data: pd.DataFrame, cat_cols: list[str]):
    """Factor‑encode categorical columns and return mapping dictionaries."""
    vocab_maps: dict[str, dict] = {}
    for col in cat_cols:
        codes, uniques = pd.factorize(data[col], sort=False)
        data[col] = codes.astype(np.int64)
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

    # Winsorize
    for col in cont_features:
        q_low_train = train_df[col].quantile(0.01)
        q_high_train = train_df[col].quantile(0.99)
        q_low_test = test_df[col].quantile(0.01)
        q_high_test = test_df[col].quantile(0.99)
        train_df[col] = train_df[col].clip(lower=q_low_train, upper=q_high_train)
        test_df[col] = test_df[col].clip(lower=q_low_test, upper=q_high_test)

    scaler = RobustScaler()

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
        seq_len: int,
        batch_size: int,
        categorical_variables: list[str],
        continuous_variables: list[str],
):
    print("Building the dataloaders for training...")
    train_dataset = ReadyToTransformerDataset(
        df=train_df,
        seq_len=seq_len,
        categorical_variables=categorical_variables,
        continuous_variables=continuous_variables,
        target_return="target_return",
    )

    print("Building dataloaders for testing...")
    test_dataset = ReadyToTransformerDataset(
        df=test_df,
        seq_len=seq_len,
        categorical_variables=categorical_variables,
        continuous_variables=continuous_variables,
        target_return="target_return",
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print("Done")
    return train_loader, test_loader


# =============================================================================
# 5.MODEL CONFIGURATION
# =============================================================================

def build_config(
        vocab_sizes: dict[str, int],
        embed_dims: dict[str, int],
        continuous_variables: list[str],
        cat_feat_positions: list[int],
        cont_feat_positions: list[int],
        wrapper: str = None,
        baseline_attention: str = 'time',
        d_model: int = 256,
        seq_len: int = 12,
        alpha: float = 0.1,
        num_layers: int = 2,
        expansion_factor: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-3,
        total_steps: int = None,
):
    return {
        "vocab_sizes": vocab_sizes,
        "embed_dims": embed_dims,
        "num_cont_features": len(continuous_variables),
        "continuous_variables": continuous_variables,
        "cat_feat_positions": cat_feat_positions,
        "cont_feat_positions": cont_feat_positions,
        "d_model": d_model,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "expansion_factor": expansion_factor,
        "n_heads": num_heads,
        "dropout": dropout,
        "output_dim": 1,
        "lr": lr,
        "wrapper": wrapper,
        "loss_method": "custom",  # options: "mse", "custom", "huber"
        "alpha": alpha,
        "baseline_attention": baseline_attention,
        "total_steps": total_steps
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
    plt.legend();
    plt.grid(True);
    plt.show()


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
    train_loss = history.get("train_loss", [])
    test_loss = history.get("test_loss", [])
    sign_acc_key = "sign_accuracy" if "sign_accuracy" in history else "sign_acc"
    sign_acc = history.get(sign_acc_key, [])

    n_epochs = max(len(train_loss), len(test_loss), len(sign_acc))
    epochs = range(1, n_epochs + 1)

    # ------- plot -------------------------------------------------------
    plt.figure(figsize=(8, 4))
    if train_loss:
        plt.plot(epochs, train_loss, label="Train loss")
    if test_loss:
        plt.plot(epochs, test_loss, label="Test loss")
    if sign_acc:
        plt.plot(epochs, sign_acc, label="Sign-accuracy")
    plt.xlabel("Epoch")
    plt.title("Training history")
    plt.legend()
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 8.MAIN DRIVER
# =============================================================================

def main(config: SimpleNamespace):
    pd.set_option("future.no_silent_downcasting", True)

    # ------------------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------------------
    RAW_PATH = config.RAW_PATH
    END_DATE = config.END_DATE
    CUTOFF_DATE = config.CUTOFF_DATE
    TIMESTAMP_COL = config.TIMESTAMP_COL
    SYMBOL_COL = config.SYMBOL_COL
    RETURN_COL = config.RETURN_COL
    SEQ_LEN = config.SEQ_LEN
    BATCH_SIZE = config.BATCH_SIZE
    EPOCHS = config.EPOCHS
    MOST_VOLATILE_STOCKS = config.MOST_VOLATILE_STOCKS
    ALPHA = config.ALPHA
    WRAPPER = config.WRAPPER
    BASELINE = config.BASELINE
    TOT_STOCKS = config.TOT_STOCKS
    MODEL_DIM = config.MODEL_DIM
    NUM_LAYERS = config.NUM_LAYERS
    EXPANSION_FACT = config.EXPANSION_FACT
    NUM_HEADS = config.NUM_HEADS
    DROPOUT = config.DROPOUT
    LEARNING_RATE = config.LEARNING_RATE

    # ------------------------------------------------------------------
    # LOAD & CLEAN
    # ------------------------------------------------------------------
    df = load_raw_data(RAW_PATH)
    # Compute return (by symbol)
    df['return'] = df[RETURN_COL]
    return_col = 'return'
    df.drop(columns=[RETURN_COL], inplace=True)

    #df = df[df["DATE"] <= END_DATE]
    df = enrich_datetime(df)

    #df = restrict_time_window(df, END_DATE)
    df = filter_stocks_with_full_coverage(df, TIMESTAMP_COL, SYMBOL_COL)

    # Select N random stocks
    stocks = random.sample(list(df[SYMBOL_COL].unique()), TOT_STOCKS)
    df = df[df[SYMBOL_COL].isin(stocks)]

    # ------------------------------------------------------------------
    # FILTER NUMBER OF STOCKS
    # ------------------------------------------------------------------
    df = assign_risk_class_by_cumulative_std(df, cutoff_date=CUTOFF_DATE, quantiles=100, top_n=TOT_STOCKS,
                                        MOST_VOLATILE_STOCKS=MOST_VOLATILE_STOCKS)

    # ------------------------------------------------------------------
    # DATA PREPARATION & FEATURE ENGINEERING
    # ------------------------------------------------------------------
    data = build_feature_frames(df, TIMESTAMP_COL, SYMBOL_COL, return_col)

    # ------------------------------------------------------------------
    # ENCODE CATEGORICAL VARIABLES AND RECOVER MAPPING DICTIONARY
    # ------------------------------------------------------------------
    categorical_variables = list(data.select_dtypes(include=['category', 'object']))
    continuous_variables = list(data.select_dtypes(include=['number']))

    data, vocab_maps = encode_categoricals(data, cat_cols=categorical_variables)

    # verify if all columns have been take into account
    selected_columns = categorical_variables + continuous_variables
    if len(selected_columns) != len(data.columns):
        unaccounted = [col for col in data.columns if col not in selected_columns]
        print("Unaccounted columns:", unaccounted)
        raise ValueError("Some variables were not encoded/selected!")

    # ------------------------------------------------------------------
    # SPLIT + NORMALISE
    # ------------------------------------------------------------------

    train_df, test_df, _, tgt_mean, tgt_std = split_and_normalise(data, CUTOFF_DATE, continuous_variables)
    # ------------------------------------------------------------------
    # DATALOADERS
    # ------------------------------------------------------------------
    train_loader, test_loader = build_dataloaders(
        train_df,
        test_df,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        categorical_variables=categorical_variables,
        continuous_variables=continuous_variables,
    )

    # ------------------------------------------------------------------
    # VOCAB SIZES & EMBED_DIMS
    # ------------------------------------------------------------------

    vocab_sizes = {col: data[col].max() + 1 for col in categorical_variables}
    embed_dims = {cat: min(50, vocab_sizes[cat] // 2) for cat in categorical_variables}  # Rule of thumb


    all_features = categorical_variables + continuous_variables

    # ------------------------------------------------------------------
    # GET POSITIONS
    # ------------------------------------------------------------------

    cat_feat_positions = [all_features.index(f) for f in categorical_variables]
    cont_feat_positions = [all_features.index(f) for f in continuous_variables]

    # ------------------------------------------------------------------
    # MODEL CONFIG
    # ------------------------------------------------------------------
    TOTAL_STEPS = len(train_loader) * EPOCHS

    config = build_config(
        vocab_sizes=vocab_sizes,
        embed_dims=embed_dims,
        continuous_variables=continuous_variables,
        cat_feat_positions=cat_feat_positions,
        cont_feat_positions=cont_feat_positions,
        seq_len=SEQ_LEN,
        alpha=ALPHA,
        wrapper=WRAPPER,
        baseline_attention=BASELINE,
        d_model=MODEL_DIM,
        num_layers=NUM_LAYERS,
        expansion_factor=EXPANSION_FACT,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        lr=LEARNING_RATE,
        total_steps=TOTAL_STEPS
    )

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    # Clean the GPU before running
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    pipeline = ModelPipeline(config)
    pipeline.to(device)

    _, best_loss, history = train_and_evaluate(
        pipeline, train_loader, test_loader, device, EPOCHS
    )

    #torch.save(best_state, "transformer_model_best.pth")
    print(f"Best Test Loss: {best_loss:.4f}")

    # ------------------------------------------------------------------
    # SAVING RESULTS
    # ------------------------------------------------------------------

    preds = denormalize_targets(
        pipeline.best_predictions.cpu().numpy(), tgt_mean, tgt_std
    )*100
    targets = denormalize_targets(
        pipeline.best_targets.cpu().numpy(), tgt_mean, tgt_std
    )*100

    preds_df = pd.DataFrame(preds, columns=vocab_maps['symbol'])
    actual_df = pd.DataFrame(targets, columns=vocab_maps['symbol'])

    long_actual = actual_df.reset_index().melt(id_vars="index", var_name="stock", value_name="actual")
    long_pred = preds_df.reset_index().melt(id_vars="index", var_name="stock", value_name="pred")

    combined = pd.merge(long_actual, long_pred, on=["index", "stock"]).set_index("index")

    # Scalars
    sign_acc_final = (np.sign(preds) == np.sign(targets)).mean()
    mt_ret = preds * targets
    portfolio = mt_ret.mean(axis=1)
    sharpe = portfolio.mean() / (portfolio.std() + 1e-9)

    metrics = {
        "best_loss": best_loss,
        "sign_acc_final": sign_acc_final,
        "sharpe": sharpe,
    }

    return metrics, history, combined

if __name__ == "__main__":
    main()
