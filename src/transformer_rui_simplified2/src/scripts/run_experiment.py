from types import SimpleNamespace
from main import main
import itertools
from datetime import datetime
from pathlib import Path
import pandas as pd

from transformer.CustomLoss import CustomLoss

if __name__ == "__main__":
    # Fixed parameters
    RAW_PATH = "../../data/high_10m.parquet"
    END_DATE = None
    CUTOFF_DATE = "2021-12-23"
    TIMESTAMP_COL = "datetime"
    SYMBOL_COL = "SYMBOL"
    RETURN_COL = "RETURN_SiOVERNIGHT"
    SEQ_LEN = 12
    BATCH_SIZE = 32
    EPOCHS = 200
    MOST_VOLATILE_STOCKS = False
    MODEL_DIM = 128
    NUM_LAYERS = 3
    EXPANSION_FACT = 1
    NUM_HEADS = 8
    DROPOUT = 0.1
    LEARNING_RATE = 1e-3
    ZETA = 1e-7
    GAMMA = 0.9
    NORM = 2
    LOSS_METHOD = CustomLoss(gamma=GAMMA)

    combos = [
        #TOT_STOCKS = 300, baseline = cross-sectional, wrapper = time
        *[
            (200, "cross-sectional", w, a)  # Note that if there isn't enough stocks, it will trigger an error
            for w in ("time", None)
            for a in (0.33, 0.66)
        ],
        #TOT_STOCKS = 300, baseline = time, wrapper = cross-sectional/None
        *[
            (200, "time", w, a)
            for w in ("cross-sectional", None)
            for a in (0.33, 0.66)
        ]
        #TOT_STOCKS = 800, baseline = time, wrapper = cross-sectional/None
         # *[
         #     (500, "time", w, a, z)
         #     for w in ("cross-sectional", None)
         #     for a in (0.01, 0.5, 1)
         # ],
        # *[
        #     (500, "time", None, a)
        #     for a in (1, 0.5, 0.01)
        # ],
    ]

    results_path = Path("results/transformer_hft_metrics.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    for tot, baseline, wrapper, alpha in combos:

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
            LEARNING_RATE=LEARNING_RATE,
            ALPHA=alpha,
            LOSS_METHOD=LOSS_METHOD,
            NORM=NORM,
            ZETA=ZETA,
            WRAPPER=wrapper,
            BASELINE=baseline,
            TOT_STOCKS=tot,
        )

        if results_path.exists():
            results_df = pd.read_csv(results_path)
        else:
            results_df = pd.DataFrame()

        seen = {
            (int(r.TOT_STOCKS), r.BASELINE, r.WRAPPER if pd.notna(r.WRAPPER) else None, float(r.ALPHA))
            for _, r in results_df.iterrows()
        }

        key = (tot, baseline, wrapper, alpha)
        if key in seen:
            print(f"Skipping already-finished run {key}")
            continue

        print(f"\n[RUN] stocks={tot} | baseline={baseline} | wrapper={wrapper} | α={alpha}")

        try:
            metrics, losses, predictions, cumulative_returns = main(config)

        except Exception as exc:
            print(f" run {key} failed – {exc}")
            continue

        row = {
            "TOT_STOCKS": tot,
            "BASELINE": baseline,
            "WRAPPER": wrapper,
            "ALPHA": alpha,
            **metrics,  # <-- adds best_loss, sharpe, and *all* Series
        }

        metrics_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

        name_exp = "_".join([str(k) for k in key])

        metrics_df.to_csv(results_path, index=False)

        losses_df = pd.DataFrame(losses)

        losses_df.to_csv(f"results/{name_exp}_losses.csv")
        predictions.to_csv(f"results/{name_exp}_prediction.csv")
        cumulative_returns.to_csv(f"results/{name_exp}_standardized_strategies_cumsum.csv")
        print("saved")

    print("\nAll requested experiments attempted. Results at:", results_path)
