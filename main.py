# Import libraries
import argparse
from pathlib import Path

# Import custom modules for each stage of the pipeline
from src.data_loading import main as process_raw_data
from src.data_analysis import run_data_analysis_pipeline
from src.linear_benchmarks import run_linear_models_pipeline
from src.time_series_analysis import run_timeseries_models_pipeline, generate_summary_reports
from src.transformer_train_experiments import run_experiments
from src.strategy import run_strategy_pipeline

def main():
    """Main controller to run specified stages of the ML pipeline."""
    parser = argparse.ArgumentParser(
        description="A modular pipeline for high-frequency return prediction."
    )

    # --- Define consistent paths ---
    BASE_DIR = Path.cwd()
    PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "high_10m.parquet"
    RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "high_10m" / "*.csv.gz"
    RESULTS_DIR = BASE_DIR / "results"
    FIGURES_DIR = RESULTS_DIR / "figures"
    TABLES_DIR = RESULTS_DIR / "tables"
    PARAMS_DIR = RESULTS_DIR / "parameters"
    PREDS_DIR = RESULTS_DIR / "predictions"

    # --- Define pipeline stage flags ---
    parser.add_argument("--load-data", action="store_true", help="Load and preprocess the raw data.")
    parser.add_argument("--data-analysis", action="store_true", help="Run the data exploration and analysis pipeline.")
    parser.add_argument("--train-benchmarks", action="store_true", help="Run all benchmark model training pipelines.")
    parser.add_argument("--evaluate-benchmarks", action="store_true", help="Generate reports for benchmark models.")
    parser.add_argument("--train-transformer", action="store_true", help="Run transformer model training and evaluation experiments.")
    parser.add_argument("--strategy", action="store_true", help="Run trading strategy backtest on all model predictions.")

    # --- Define configurable parameters ---
    DEFAULT_SPLIT_DATETIME = '2021-12-27 00:00:00'
    parser.add_argument(
        "--split-date",
        type=str,
        default=DEFAULT_SPLIT_DATETIME,
        help=f"The train/test split date (YYYY-MM-DD HH:MM:SS). Default: {DEFAULT_SPLIT_DATETIME}"
    )
    
    args = parser.parse_args()

    # --- Create a list of selected stages ---
    stages_to_run = [arg for arg, value in vars(args).items() if value is True]

    # If no stage flags were provided, show a help message and exit.
    if not stages_to_run:
        print("No stage selected. Please specify at least one stage to run (e.g., --load-data).")
        parser.print_help()
        return

    # --- Execute the selected stage(s) ---
    if args.load_data:
        print("\n--- STAGE: DATA LOADING & PREPROCESSING ---")
        process_raw_data(data_path=RAW_DATA_DIR)

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
        print("\n--- STAGE: TRANSFORMER EXPERIMENTS ---")
        run_experiments(data_path=PROCESSED_DATA_PATH, result_path=RESULTS_DIR / "transformer_experiments")

    if args.strategy:
        print("\n--- STAGE: TRADING STRATEGY BACKTESTING ---")
        run_strategy_pipeline(preds_dir=PREDS_DIR, processed_data_path=PROCESSED_DATA_PATH, figures_dir=FIGURES_DIR / "strategy")
    
    print("\nPipeline execution complete for selected stages.")


if __name__ == "__main__":
    main()