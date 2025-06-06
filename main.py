# main.py
import argparse
from src.data_processing import process_data
# from src.training import train_model
# from src.evaluation import evaluate_model
# from src.backtesting import run_backtest

def main():
    parser = argparse.ArgumentParser(description="A modular pipeline for high-frequency return prediction.")

    # --- Add flags for each pipeline stage ---
    parser.add_argument("--process-data", action="store_true", help="Run the data processing pipeline.")
    parser.add_argument("--train", action="store_true", help="Run the model training pipeline.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained model's performance.")
    # parser.add_argument("--backtest", action="store_true", help="Run a trading backtest using the model.")

    # --- Add arguments for file paths and hyperparameters ---
    parser.add_argument("--raw-data-path", type=str, default="data/raw/high_10m/*.csv.gz", help="Path to raw data.")
    parser.add_argument("--processed-data-path", type=str, default="data/processed/high_10m.parquet", help="Path for processed data.")
    # parser.add_argument("--model-path", type=str, default="results/models/best_model.pth", help="Path to save/load the model.")
    # parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    
    args = parser.parse_args()

    # --- Execute the selected stage(s) ---
    if args.process_data:
        print("\n--- STAGE: DATA PROCESSING ---")
        process_data(raw_path=args.raw_data_path, processed_path=args.processed_data_path)

    # if args.train:
    #     print("\n--- STAGE: MODEL TRAINING ---")
    #     train_model(processed_path=args.processed_data_path, model_path=args.model_path, epochs=args.epochs)

    # if args.evaluate:
    #     print("\n--- STAGE: MODEL EVALUATION ---")
    #     evaluate_model(processed_path=args.processed_data_path, model_path=args.model_path)

    # if args.backtest:
    #     print("\n--- STAGE: TRADING BACKTEST ---")
    #     run_backtest(processed_path=args.processed_data_path, model_path=args.model_path)

    # If no flags were provided, show help message.
    if not any([args.process_data, args.train, args.evaluate, args.backtest]):
        print("No stage selected. Please specify a stage to run (e.g., --train). Use --help for more info.")
        parser.print_help()


if __name__ == "__main__":
    main()