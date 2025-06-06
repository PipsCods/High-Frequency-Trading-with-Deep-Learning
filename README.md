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
│   ├── models/             # Saved model weights
│   └── figures/            # Plots and charts for the final report
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # Logic for the 'process-data' stage
│   ├── training.py         # Logic for the 'train' stage
│   ├── evaluation.py       # Logic for the 'evaluate' stage
│   ├── strategy.py         # Logic for the 'strategy' stage
│   └── utils.py            # Helper functions (e.g., custom loss, metrics)
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
* [Teammate 4 Name]  
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

Using conda:
```bash
conda create -n ml_finance python=3.9
conda activate ml_finance
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run a pipeline stage
The project is controlled via main.py, which allows you to run each stage of the pipeline independently using flags.

Process data:
```bash
python main.py --process-data
```

Train Model:
```bash
python main.py --train --epochs 50
```

Evaluate Model:
```bash
python main.py --evaluate
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

This script will execute all stages (`--process-data`, `--train`, `--evaluate`, `--strategy`) in the correct sequence.

## License

This project is licensed under the [MIT License](LICENSE).