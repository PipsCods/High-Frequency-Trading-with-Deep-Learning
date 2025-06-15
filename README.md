# Machine Learning for High-Frequency Return Prediction

## Description
This project aims to predict future high-frequency stock returns using past returns. The goal is to build, train, and evaluate a deep learning model capable of forecasting the next 10-minute return for a cross-section of stocks. The project includes data preprocessing, model training, performance evaluation, and a simulated trading backtest to assess the strategy's potential, accounting for factors like transaction costs.

This repository is submitted as part of the “Machine-Learning in Finance” course project.

## Project Structure
The codebase is organized to be modular, clean, and reproducible, as per the project guidelines:

```
High-Frequency-Trading-with-Deep-Learning/
|
├── data/
│   ├── raw/                # Original, untouched 10-minute frequency data
│   └── processed/          # Processed data ready for modeling
|
├── notebooks/              # Jupyter notebooks for exploration and prototyping
|
├── results/
│   ├── models/                 # Saved model weights
│   ├── figures/                # All plots and charts for the final report
│   ├── tables/                 # All LaTeX tables for the final report
│   ├── parameters/             # Saved parameters for benchmark models
│   └── predictions/            # Saved out-of-sample predictions from benchmarks
│
├── src/
│   ├── __init__.py
│   ├── data_analysis.py        # EDA and descriptive statistics pipeline
│   ├── linear_benchmarks.py    # OLS, Ridge, and Lasso model pipeline
│   ├── time_series_analysis.py # ARIMA and GARCH model pipeline
│   ├── training.py             # Logic for the 'train' stage
│   ├── evaluation.py           # Logic for the 'evaluate' stage
│   ├── strategy.py             # Logic for the 'strategy' stage
│   └── utils.py                # Helper functions used across modules
|
├── .gitattributes
├── .gitignore
├── main.py                 # Main controller to run pipeline stages
├── run.sh                  # Shell script to execute the full end-to-end pipeline
├── requirements.txt        # Required Python libraries
└── README.md               # This file
```

## Authors
* [Teammate 1 Name]  
<!-- * Emanuele Durante [emanuele.durante@epfl.ch](mailto:emanuele.durante@epfl.ch) -->
* [Teammate 3 Name]  
<!-- * Filippo Passerini [filippo.passerini@epfl.ch](mailto:filippo.passerini@epfl.ch) -->
<!-- * Letizia Seveso [letizia.seveso@epfl.ch](mailto:letizia.seveso@epfl.ch) -->
<!-- * Alex Martinez [alex.martinezdefrancisco@epfl.ch](mailto:alex.martinezdefrancisco@epfl.ch) -->

## Quickstart
Follow these steps to set up the project environment and run the pipeline:

### 1. Clone the Repository
```bash
git clone [your-repository-url]
cd high_frequency_project
```

### 2. Create and Activate a Virtual Environment
Using venv:
```bash
python -m venv venv
source venv/bin/activate   # On Windows: `venv\Scripts\activate`
```

Using `conda` is recommended:
```bash
conda create -n ml_finance python=3.11
conda activate ml_finance
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Process the Data
* Insert the raw data file into the "data/raw/high_10m" directory. all files inside that director should be in the same format "*.csv.gz" in order to be processed correctly.
```
python main.py --load-data
```

This command will process the raw data files, generating the necessary processed data files in the "data/processed" directory. The processed data will be used for training and evaluation of the models.


### 5. Run a pipeline stage
The project is controlled via `main.py`, which allows you to run each stage of the pipeline independently using flags.

Run Data Analysis & EDA:
* This generates descriptive statistics, tables, and plots about the dataset.
```bash
python main.py --data-analysis
```

Train Benchmark Models:
* This runs the OLS, Ridge, Lasso, ARIMA, and GARCH models for all stocks, saving their parameters and predictions.
```bash
python main.py --train-benchmarks
```

Evaluate Benchmark Models:
* This uses the saved parameters to generate summary tables and figures for the benchmark models.
```bash
python main.py --evaluate-benchmarks
```

Train Transformer Model:
* This trains 15 experiment on our transformer model, saving the best model weights and all evaluation metrics.
```bash
python main.py --train-transformer
```


Run Trading Strategy:
```bash
python main.py --strategy
```

For a full list of commands and arguments, you can use the help flag:
```bash
python main.py --help
```

### 5. Run End-to-End Pipeline with `run.sh`
To ensure full reproducibility and execute the entire pipeline from data processing to the final backtest, use the provided shell script. This is the recommended method for generating the final results for the report.

```bash
sh run.sh
```

This script will execute all necessary stages in the correct sequence (e.g., data analysis, benchmark training, transformer training, evaluation, etc.).

## License

This project is licensed under the [MIT License](LICENSE).