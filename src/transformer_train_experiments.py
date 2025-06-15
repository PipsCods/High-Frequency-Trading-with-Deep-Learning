# Package imports
import os 
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader

# Utils imports
try:
    from .utils import denormalize_targets, filter_stocks_with_full_coverage, compute_hf_features_multiwindow, \
    load_data , enrich_datetime, prepare_hf_data, encode_categoricals, split_and_normalise
    from .models.dataset import ReadyToTransformerDataset
    from .models.model_init import ModelPipeline

except ImportError:
    from utils import denormalize_targets, filter_stocks_with_full_coverage, compute_hf_features_multiwindow, \
    load_data , enrich_datetime, prepare_hf_data, encode_categoricals, split_and_normalise
    from models.dataset import ReadyToTransformerDataset
    from models.model_init import ModelPipeline


# Split date configuration
DEFAULT_SPLIT_DATETIME = '2021-12-27 00:00:00'

# Hyperparameters
MODEL_DIM = 128
NUM_LAYERS = 3
EXPANSION_FACT = 1
NUM_HEADS = 8
DROPOUT = 0
OUTPUT_DIM = 1
LEARNING_RATE = 1e-3

EPOCHS = 500
SEQ_LEN = 12
BATCH_SIZE = 16

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

def run_single_experiment(
        tot_stocks: int,
        baseline : str,
        wrapper: str | None,
        alpha: float,
        data_path: str,
)-> tuple: 
    df = load_data(data_path)
    df["return"] = df["RETURN_NoOvernight"]
    df.drop(columns=["RETURN_SiOVERNIGHT", "RETURN_NoOVERNIGHT"], inplace=True)
    df = enrich_datetime(df)
    df = filter_stocks_with_full_coverage(df, "datetime", "SYMBOL")
    df = compute_hf_features_multiwindow(df, "return")

    random.seed(42)
    stocks = random.sample(list(df["SYMBOL"].unique()), tot_stocks)
    df = df[df["SYMBOL"].isin(stocks)]

    df.drop(columns=["ALL_EX", "SUM_DELTA"], inplace=True)

    (data,
     basic_cat_features,
     cat_features,
     cont_features,
     cat_positions,
     cont_positions) = prepare_hf_data(
        df, "datetime", "SYMBOL"
    )
    data, vocab_maps = encode_categoricals(
        data, cat_cols=cat_features + basic_cat_features
    )

    train_df, test_df, _, tgt_mean, tgt_std = split_and_normalise(
        data, DEFAULT_SPLIT_DATETIME, cont_features
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

    return metrics, history, combined


def run_experiments(data_path: str, result_path:str):

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

    metrics_path = os.path.join(result_path, "transformer_hft_metrics.csv")
    # results_path = Path("../../data/results2/transformer_hft_metrics.csv")
    # results_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume logic unchanged
    if metrics_path.exists():
        results_df = pd.read_csv(metrics_path)
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
                data_path  = data_path
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

        metrics_df.to_csv(metrics_path, index=False)  # <= persist immediately

        losses_df = pd.DataFrame(losses)
        # losses_df.to_csv(f"../../data/results2/{name_exp}_losses.csv")
        losses_df.to_csv(os.path.join(result_path, f"{name_exp}_losses.csv"))

        #predictions.to_csv(f"../../data/results2/{name_exp}_prediction.csv")
        predictions.to_csv(os.path.join(result_path, f"{name_exp}_prediction.csv"))
        print("saved")

    print("\nAll requested experiments attempted. Results at:", result_path)

if __name__ == "__main__":
    run_experiments()
    # Uncomment the line below to run a single experiment with specific parameters
    # run_single_experiment(tot_stocks=100, baseline="cross-sectional", wrapper="time", alpha=1)