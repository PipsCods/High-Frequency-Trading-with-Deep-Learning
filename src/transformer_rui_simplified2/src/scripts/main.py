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

from transformer.CustomLoss import CustomLoss
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
from utils.utils import set_seed

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
        target_regime = "target_regime"
    )

    print("Building dataloaders for testing...")
    test_dataset = ReadyToTransformerDataset(
        df=test_df,
        seq_len=seq_len,
        categorical_variables=categorical_variables,
        continuous_variables=continuous_variables,
        target_return="target_return",
        target_regime="target_regime"
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
        wrapper: str,
        baseline_attention: str,
        d_model: int,
        seq_len: int,
        weight_decay : float,
        alpha: float,
        zeta: float,
        norm: float,
        num_layers: int,
        expansion_factor: int,
        num_heads: int,
        dropout: float,
        lr: float,
        total_steps: int,
        loss_method,
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
        "loss_method": loss_method,
        "alpha": alpha,
        "zeta": zeta,
        "norm": norm,
        "baseline_attention": baseline_attention,
        "total_steps": total_steps,
        "weight_decay": weight_decay,
    }


# =============================================================================
# 6.TRAIN / EVAL LOOP
# =============================================================================

def train_and_evaluate(
    pipeline,
    train_loader,
    test_loader,
    device,
    num_epochs: int = 50,
):
    best_state, best_loss = None, float("inf")
    history = {
        "train": [],
        "test": [],
        "return_loss": [],
        "regime_loss": [],
        "regime_acc": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_loss = pipeline.train_epoch(train_loader, device)

        # Evaluate
        test_loss, preds, targets = pipeline.evaluate_epoch(test_loader, device, track_best=True)

        # Unpack dicts
        pred_returns = preds["returns"]        # [B*S, ]
        pred_regimes_logits = preds["regimes"] # [B*S, ] raw logits
        target_returns = targets["returns"]    # [B*S, ]
        target_regimes = targets["regimes"]    # [B*S, ] binary 0/1

        # Apply sigmoid to regime logits to get probabilities
        pred_regimes_probs = torch.sigmoid(pred_regimes_logits)

        # Binary regime predictions and accuracy
        binary_preds = (pred_regimes_probs > 0.5).float()
        regime_acc = (binary_preds == target_regimes).float().mean().item()

        # Individual task losses
        loss_return = pipeline.loss_fn_returns(pred_returns, target_returns).item()
        loss_regime = pipeline.loss_fn_regime(pred_regimes_logits, target_regimes).item()

        # Log metrics
        history["train"].append(train_loss)
        history["test"].append(test_loss)
        history["return_loss"].append(loss_return)
        history["regime_loss"].append(loss_regime)
        history["regime_acc"].append(regime_acc)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train {train_loss:.4f} | Test {test_loss:.4f} | "
        )
        if epoch % 20 == 0:
            print(f"ReturnLoss {loss_return:.4f} | RegimeLoss {loss_regime:.4f} | Regime Acc {regime_acc:.4f}")

        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {
                "encoder": pipeline.encoder.state_dict(),
                "predictor_returns": pipeline.predictor_returns.state_dict(),
                "predictor_regime": pipeline.predictor_regime.state_dict(),
            }
            if pipeline.wrapper is not None:
                best_state["wrapper"] = pipeline.wrapper.state_dict()

            # Store best outputs for future metrics / plots
            pipeline.best_predictions = {
                "returns": pred_returns.detach().cpu(),
                "regimes": pred_regimes_probs.detach().cpu(),  # save probs not logits
            }
            pipeline.best_targets = {
                "returns": target_returns.detach().cpu(),
                "regimes": target_regimes.detach().cpu(),
            }

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
    # 8.COLD EVALUATION
    # =============================================================================

def evaluate_strategies(pred_returns, target_returns, pred_regimes=None):
    """
    Computes and plots the cumulative returns of 3 strategies:
    (1) Buy and Hold, (2) Return Only, (3) Return x Regime

    Args:
        pred_returns (np.ndarray): [T, S] predicted returns
        target_returns (np.ndarray): [T, S] actual returns
        pred_regimes (np.ndarray or None): [T, S] predicted regimes (probabilities in [0,1])

    Returns:
        sharpe_dict (dict): Sharpe ratios of the 3 strategies
    """

    # Buy & Hold → equally weighted portfolio of actual returns
    buy_hold_ret = target_returns.mean(axis=1)  # [T]

    # Return-Only → pred_return * actual_return
    return_only_ret = (pred_returns * target_returns).mean(axis=1)

    # Return x Regime → apply regime as filter (default: always 1 if None)
    if pred_regimes is None:
        pred_regimes = np.ones_like(pred_returns)
    regime_signal = (pred_regimes > 0.5).astype(int)
    absolute_pred_returns = np.abs(pred_returns)
    combined_signal = absolute_pred_returns * regime_signal
    return_regime_ret = (combined_signal * target_returns).mean(axis=1)

    # Standardized returns
    eps = 1e-9
    buy_hold_ret_standard = buy_hold_ret / (buy_hold_ret.std() + eps)
    return_only_ret_standard = return_only_ret / (return_only_ret.std() + eps)
    return_regime_ret_standard = return_regime_ret / (return_regime_ret.std() + eps)

    # Cumulative returns
    cumulative_returns = pd.DataFrame({
        "Buy & Hold": np.cumsum(buy_hold_ret_standard),
        "Return Only": np.cumsum(return_only_ret_standard),
        "Return × Regime": np.cumsum(return_regime_ret_standard),
    })

    # Sharpe ratios
    sharpe_dict = {}
    for col in cumulative_returns.columns:
        daily_ret = cumulative_returns[col].diff().dropna()
        sharpe = daily_ret.mean() / (daily_ret.std() + 1e-9)
        sharpe_dict[col] = sharpe

    # Plot
    plt.figure(figsize=(12, 6))
    for col in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[col], label=f"{col} (Sharpe={sharpe_dict[col]:.2f})")

    plt.title("Cumulative Returns of Strategies")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return (risk-adjusted)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return sharpe_dict, cumulative_returns

# =============================================================================
# 9.MAIN DRIVER
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
    ZETA = config.ZETA
    NORM = config.NORM
    WRAPPER = config.WRAPPER
    BASELINE = config.BASELINE
    TOT_STOCKS = config.TOT_STOCKS
    MODEL_DIM = config.MODEL_DIM
    NUM_LAYERS = config.NUM_LAYERS
    EXPANSION_FACT = config.EXPANSION_FACT
    NUM_HEADS = config.NUM_HEADS
    DROPOUT = config.DROPOUT
    LEARNING_RATE = config.LEARNING_RATE
    LOSS_METHOD = config.LOSS_METHOD
    WEIGHT_DECAY = config.WEIGHT_DECAY

    # ------------------------------------------------------------------
    # LOAD & CLEAN
    # ------------------------------------------------------------------
    set_seed(42)
    df = load_raw_data(RAW_PATH)
    # Compute return (by symbol)
    df['return'] = df[RETURN_COL]
    return_col = 'return'
    df.drop(columns=[RETURN_COL], inplace=True)

    #df = df[df["DATE"] <= END_DATE]
    df = enrich_datetime(df)

    #df = restrict_time_window(df, END_DATE)
    df = filter_stocks_with_full_coverage(df, TIMESTAMP_COL, SYMBOL_COL)


    # ------------------------------------------------------------------
    # CREATE RISK CLASS AND FILTER THE STOCKS
    # ------------------------------------------------------------------
    # Select N random stocks
    stocks = random.sample(list(df[SYMBOL_COL].unique()), TOT_STOCKS)
    df = df[df[SYMBOL_COL].isin(stocks)]

    # Create risk class
    df = assign_risk_class_by_cumulative_std(df, cutoff_date=CUTOFF_DATE, quantiles=50)

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
        zeta = ZETA,
        norm = NORM,
        wrapper=WRAPPER,
        baseline_attention=BASELINE,
        d_model=MODEL_DIM,
        num_layers=NUM_LAYERS,
        expansion_factor=EXPANSION_FACT,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        lr=LEARNING_RATE,
        total_steps=TOTAL_STEPS,
        loss_method=LOSS_METHOD,
        weight_decay = WEIGHT_DECAY,
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

    # torch.save(best_state, "transformer_model_best.pth")
    print(f"Best Test Loss: {best_loss:.4f}")

    # ------------------------------------------------------------------
    # SAVING RESULTS
    # ------------------------------------------------------------------

    # Extract and denormalize returns
    pred_returns = pipeline.best_predictions["returns"].cpu().numpy()
    target_returns = pipeline.best_targets["returns"].cpu().numpy()

    pred_returns = denormalize_targets(pred_returns, tgt_mean, tgt_std)
    target_returns = denormalize_targets(target_returns, tgt_mean, tgt_std)

    # Extract regime predictions and targets (already in [0, 1])
    pred_regimes = pipeline.best_predictions["regimes"].cpu().numpy()
    target_regimes = pipeline.best_targets["regimes"].cpu().numpy()

    # Wrap as DataFrames
    returns_pred_df = pd.DataFrame(pred_returns, columns=vocab_maps["symbol"])
    returns_actual_df = pd.DataFrame(target_returns, columns=vocab_maps["symbol"])

    regime_pred_df = pd.DataFrame(pred_regimes, columns=vocab_maps["symbol"])
    regime_actual_df = pd.DataFrame(target_regimes, columns=vocab_maps["symbol"])

    # Long format for merging
    long_r_actual = returns_actual_df.reset_index().melt(id_vars="index", var_name="stock", value_name="actual_return")
    long_r_pred = returns_pred_df.reset_index().melt(id_vars="index", var_name="stock", value_name="pred_return")

    long_s_actual = regime_actual_df.reset_index().melt(id_vars="index", var_name="stock", value_name="actual_regime")
    long_s_pred = regime_pred_df.reset_index().melt(id_vars="index", var_name="stock", value_name="pred_regime")

    # Combine everything
    combined = (
        long_r_actual
        .merge(long_r_pred, on=["index", "stock"])
        .merge(long_s_actual, on=["index", "stock"])
        .merge(long_s_pred, on=["index", "stock"])
        .set_index("index")
    )

    # Compute Sharpe
    sharpe_ratios, cumulative_returns = evaluate_strategies(pred_returns, target_returns, pred_regimes)
    sign_acc_final = (np.sign(pred_returns) == np.sign(target_returns)).mean()

    # Final metrics
    metrics = {
        "best_loss": best_loss,
        "sign_acc_final": sign_acc_final,
        "sharpe": sharpe_ratios,
    }
    print(pred_returns.mean(), pred_returns.std())
    print(pred_regimes.mean(), pred_regimes.std())

    return metrics, history, combined, cumulative_returns

if __name__ == "__main__":

    for weight in [1e-3, 1e-2, 1e-1, 0]:
        config_dict = {
            "RAW_PATH": "../../data/high_10m.parquet",
            "END_DATE": None,
            "CUTOFF_DATE": "2021-12-23",
            "TIMESTAMP_COL": "datetime",
            "SYMBOL_COL": "SYMBOL",
            "RETURN_COL": "RETURN_SiOVERNIGHT",

            "SEQ_LEN": 12,
            "BATCH_SIZE": 16,
            "EPOCHS": 50,

            "MOST_VOLATILE_STOCKS": False,

            "MODEL_DIM": 128,
            "NUM_LAYERS": 3, # best 3 or 2
            "EXPANSION_FACT": 1,
            "NUM_HEADS": 8,  # was 8

            "DROPOUT": 0.1,
            "LEARNING_RATE": 1e-3,

            "ALPHA": 0.66,
            "ZETA": 0, #shrink factor bests : 1e-3
            "NORM": 2, # 1 for lasso, 0 for none, 2 for ridge

            "BASELINE": "time",
            "WRAPPER": "cross-sectional",
            "LOSS_METHOD": CustomLoss(gamma=0.9),
            "TOT_STOCKS": 100,  # at least 200
            "WEIGHT_DECAY" : weight
        }
        config = SimpleNamespace(**config_dict)

        metrics, losses, predictions, cumulative_returns = main(config)
        plt.figure()
        for x in losses:
            plt.plot(range(50),losses[x], label=x)
        plt.legend()
        plt.title(f"Losses for weight {weight}")
        plt.show()
