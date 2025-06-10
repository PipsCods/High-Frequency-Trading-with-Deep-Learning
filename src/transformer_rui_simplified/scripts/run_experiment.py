from types import SimpleNamespace
from main import main
import itertools
from datetime import datetime
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    # Fixed parameters
    RAW_PATH = "../../data/high_10m.parquet"
    END_DATE = None
    CUTOFF_DATE = "2021-12-23"
    TIMESTAMP_COL = "datetime"
    SYMBOL_COL = "SYMBOL"
    RETURN_COL = "RETURN_SiOVERNIGHT"
    SEQ_LEN = 12
    BATCH_SIZE = 16
    EPOCHS = 500
    MOST_VOLATILE_STOCKS = False
    MODEL_DIM = 128
    NUM_LAYERS = 3
    EXPANSION_FACT = 1
    NUM_HEADS = 8
    DROPOUT = 0
    ALPHA = 1

    combos = [
        #TOT_STOCKS = 200, baseline = cross-sectional, wrapper = time
        *[
            (300, "cross-sectional", w, lr)
            for w in ("time")
            for lr in (1e-6, 1e-5, 1e-4)
        ],
        #TOT_STOCKS = 100, baseline = time, wrapper = cross-sectional/None
        *[
            (300, "time", w, lr)
            for w in ("cross-sectional")
            for lr in (1e-6, 1e-5, 1e-4)
        ],
        #TOT_STOCKS = 800, baseline = time, wrapper = cross-sectional/None
        # *[
        #     (500, "time", w, lr)
        #     for w in ("cross-sectional", None)
        #     for lr in (1e-8, 1e-5, 1e-3)
        # ],
        # *[
        #     (500, "time", None, a)
        #     for a in (1, 0.5, 0.01)
        # ],
    ]

    results_path = Path("../../data/results_rui/transformer_hft_metrics.csv")
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

    for tot, baseline, wrapper, lr in combos:

        config = SimpleNamespace(
            RAW_PATH=RAW_PATH,
            END_DATE=END_DATE,
            CUTOFF_DATE=CUTOFF_DATE,
            TIMESTAMP_COL=TIMESTAMP_COL,
            SYMBOL_COL=SYMBOL_COL,
            RETURN_COL=RETURN_COL,
            SEQ_LEN=SEQ_LEN,
            BATCH_SIZE=BATCH_SIZE,
            EPOCHS=EPOCHS,
            MOST_VOLATILE_STOCKS=MOST_VOLATILE_STOCKS,
            MODEL_DIM=MODEL_DIM,
            NUM_LAYERS=NUM_LAYERS,
            EXPANSION_FACT=EXPANSION_FACT,
            NUM_HEADS=NUM_HEADS,
            DROPOUT=DROPOUT,
            LEARNING_RATE=lr,
            ALPHA=ALPHA,
            WRAPPER=wrapper,
            BASELINE=baseline,
            TOT_STOCKS=tot,
        )

        key = (tot, baseline, wrapper, lr)
        if key in seen:
            print(f"Skipping already-finished run {key}")
            continue

        print(f"\n[RUN] stocks={tot} | baseline={baseline} | wrapper={wrapper} | LR={lr}")

        try:
            metrics, losses, predictions = main(config)

        except Exception as exc:
            print(f" run {key} failed – {exc}")
            continue

        row = {
            "TOT_STOCKS": tot,
            "BASELINE": baseline,
            "WRAPPER": wrapper,
            "LEARNING_RATE": lr,
            **metrics,  # <-- adds best_loss, sharpe, and *all* Series
        }

        metrics_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

        name_exp = "_".join([str(k) for k in key])

        metrics_df.to_csv(results_path, index=False)

        losses_df = pd.DataFrame(losses)

        losses_df.to_csv(f"../../data/results_rui/{name_exp}_losses.csv")
        predictions.to_csv(f"../../data/results_rui/{name_exp}_prediction.csv")
        print("saved")

    print("\nAll requested experiments attempted. Results at:", results_path)
