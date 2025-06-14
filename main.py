# Packages and modules for the pipeline
import argparse
from pathlib import Path

# Import custom modules for each stage of the pipeline
from src.data_analysis import run_data_analysis_pipeline
from src.linear_benchmarks import run_linear_models_pipeline
from src.time_series_analysis import run_timeseries_models_pipeline, generate_summary_reports
from src.transformer_train_experiments import run_experiments
# from src.training import train_model
# from src.evaluation import evaluate_model
# from src.strategy import run_strategy

def main():
    parser = argparse.ArgumentParser(description="A modular pipeline for high-frequency return prediction.")

    # --- Define consistent paths ---
    BASE_DIR = Path.cwd()
    PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "high_10m.parquet"
    RESULTS_DIR = BASE_DIR / "results"
    FIGURES_DIR = RESULTS_DIR / "figures"
    TABLES_DIR = RESULTS_DIR / "tables"
    PARAMS_DIR = RESULTS_DIR / "parameters"
    PREDS_DIR = RESULTS_DIR / "predictions"

    # Split date configuration
    DEFAULT_SPLIT_DATETIME = '2021-12-27 00:00:00'

    # --- Add flags for each pipeline stage ---
    parser.add_argument("--data-analysis", action="store_true", help="Run the data exploration and analysis pipeline.")
    parser.add_argument("--train-benchmarks", action="store_true", help="Run all benchmark model training pipelines.")
    parser.add_argument("--evaluate-benchmarks", action="store_true", help="Generate reports for benchmark models.")

    parser.add_argument("--train_transformer", action="store_true", help="Run the model training pipeline.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained model's performance.")
    parser.add_argument("--strategy", action="store_true", help="Run a trading strategy using the model.")

    # --- Add arguments for file paths and hyperparameters ---
    # parser.add_argument("--model-path", type=str, default="results/models/best_model.pth", help="Path to save/load the model.")
    # parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")

    # Add split date as a configurable command-line argument
    parser.add_argument("--split-date", type=str, default=DEFAULT_SPLIT_DATETIME, help=f"The train/test split date (YYYY-MM-DD HH:MM:SS). Default: {DEFAULT_SPLIT_DATETIME}")
    
    args = parser.parse_args()

    # --- Execute the selected stage(s) ---
    if args.data_analysis:
        print("\n--- STAGE: DATA ANALYSIS & EDA ---")
        run_data_analysis_pipeline(PROCESSED_DATA_PATH, FIGURES_DIR / "data_analysis", TABLES_DIR / "data_analysis")

    if args.train_benchmarks:
        print("\n--- STAGE: BENCHMARK MODEL TRAINING ---")
        run_linear_models_pipeline(PROCESSED_DATA_PATH, PARAMS_DIR, PREDS_DIR, FIGURES_DIR / "linear_models", split_datetime=args.split_date)
        run_timeseries_models_pipeline(PROCESSED_DATA_PATH, PARAMS_DIR, PREDS_DIR, FIGURES_DIR / "time_series", split_datetime=args.split_date)

    if args.evaluate_benchmarks:
        print("\n--- STAGE: BENCHMARK MODEL EVALUATION ---")
        generate_summary_reports(PARAMS_DIR, FIGURES_DIR / "time_series", TABLES_DIR / "time_series")

    if args.train_transformer:
        print("\n--- STAGE: MODEL TRAINING ---")
        run_experiments(data_path=PROCESSED_DATA_PATH, result_path = RESULTS_DIR / "transformer_experiments")
        
    #     train_model(processed_path=args.processed_data_path, model_path=args.model_path, epochs=args.epochs)

    if args.evaluate:
        print("\n--- STAGE: MODEL EVALUATION ---")
        pass
    #     evaluate_model(processed_path=args.processed_data_path, model_path=args.model_path)

    if args.strategy:
        print("\n--- STAGE: TRADING STRATEGY ---")
        pass
    #     run_strategy(processed_path=args.processed_data_path, model_path=args.model_path)

    # If no flags were provided, show help message.
    if not any([args.data_analysis, args.train_benchmarks, args.evaluate_benchmarks, args.train_transformer, args.evaluate, args.strategy]):
        print("No stage selected. Please specify a stage to run (e.g., --train). Use --help for more info.")
        parser.print_help()


if __name__ == "__main__":
    main()